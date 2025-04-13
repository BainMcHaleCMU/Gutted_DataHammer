"""
Manager Agent

This module defines a manager agent that coordinates between other agents in the workflow.
It plans analysis steps, delegates tasks, and synthesizes results.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback
from datetime import datetime
import pandas as pd
import json

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

# Set up logging
logger = logging.getLogger(__name__)


async def create_analysis_plan(ctx: Context, user_question: str) -> str:
    """
    Create a structured analysis plan based on the user's question or requirements.

    Args:
        ctx: Context object containing state information
        user_question: The user's original question or requirements

    Returns:
        A string describing the created plan
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Cannot create plan - no DataFrame found in context"

        # Extract dataset properties for planning
        row_count = df.shape[0]
        column_count = df.shape[1]
        column_names = df.columns.tolist()

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()

        # Create a plan based on the question and data properties
        plan = {
            "original_question": user_question,
            "timestamp": datetime.now().isoformat(),
            "data_properties": {
                "rows": row_count,
                "columns": column_count,
                "column_types": {
                    "numeric": numeric_columns,
                    "categorical": categorical_columns,
                    "datetime": datetime_columns,
                },
            },
            "steps": [],
        }

        # Determine analysis steps based on the question and data
        # 1. Always start with a general data summary
        plan["steps"].append(
            {
                "step_number": 1,
                "agent": "ExplorationAgent",
                "action": "generate_data_summary",
                "status": "pending",
                "reason": "Need to understand basic dataset properties before detailed analysis",
            }
        )

        # 2. Depending on question, add appropriate analysis steps
        if any(
            term in user_question.lower()
            for term in ["correlation", "relationship", "compare", "related"]
        ):
            if len(numeric_columns) > 1:
                plan["steps"].append(
                    {
                        "step_number": 2,
                        "agent": "ExplorationAgent",
                        "action": "compute_correlations",
                        "status": "pending",
                        "reason": "User is interested in relationships between variables",
                    }
                )

        if any(
            term in user_question.lower() for term in ["outlier", "anomaly", "unusual"]
        ):
            if numeric_columns:
                plan["steps"].append(
                    {
                        "step_number": len(plan["steps"]) + 1,
                        "agent": "ExplorationAgent",
                        "action": "detect_outliers",
                        "parameters": {
                            "column_name": numeric_columns[0]
                        },  # Default to first numeric column
                        "status": "pending",
                        "reason": "User is interested in outliers/anomalies",
                    }
                )

        # If no specific analysis detected, default to comprehensive analysis
        if len(plan["steps"]) <= 1:
            plan["steps"].append(
                {
                    "step_number": 2,
                    "agent": "ExplorationAgent",
                    "action": "analyze_dataframe",
                    "status": "pending",
                    "reason": "Providing comprehensive analysis based on user request",
                }
            )

        # Add final step to synthesize all findings
        plan["steps"].append(
            {
                "step_number": len(plan["steps"]) + 1,
                "agent": "ManagerAgent",
                "action": "synthesize_results",
                "status": "pending",
                "reason": "Consolidate all analysis results into a coherent response",
            }
        )

        # Store the plan in the context
        current_state["analysis_plan"] = plan
        await ctx.set("state", current_state)

        # Generate a response summarizing the plan
        plan_summary = f"Created analysis plan with {len(plan['steps'])} steps:\n"
        for step in plan["steps"]:
            plan_summary += f"- Step {step['step_number']}: {step['action']} using {step['agent']}\n"

        return plan_summary

    except Exception as e:
        logger.error(f"Error creating analysis plan: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to create analysis plan: {str(e)}"


async def delegate_to_agent(
    ctx: Context, agent_name: str, action: str, **parameters
) -> str:
    """
    Delegate a task to a specific agent.

    Args:
        ctx: Context object containing state information
        agent_name: Name of the agent to delegate to
        action: The action to perform
        **parameters: Additional parameters for the action

    Returns:
        A string describing the delegation result
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Update execution log
        if "execution_log" not in current_state:
            current_state["execution_log"] = []

        current_state["execution_log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "delegation",
                "agent": agent_name,
                "task": action,
                "parameters": parameters,
            }
        )

        # Store the task information for handoff
        current_state["current_delegation"] = {
            "agent": agent_name,
            "action": action,
            "parameters": parameters,
            "status": "pending",
        }

        await ctx.set("state", current_state)

        return f"Task '{action}' delegated to agent '{agent_name}'. The agent will now take over."

    except Exception as e:
        logger.error(f"Error delegating task: {str(e)}")
        return f"Failed to delegate task: {str(e)}"


async def track_progress(ctx: Context) -> str:
    """
    Track the progress of the current analysis plan.

    Args:
        ctx: Context object containing state information

    Returns:
        A string describing the current progress
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        plan = current_state.get("analysis_plan")

        if not plan:
            return "No analysis plan found in context"

        completed_steps = sum(
            1 for step in plan["steps"] if step["status"] == "completed"
        )
        total_steps = len(plan["steps"])
        progress_percentage = (
            (completed_steps / total_steps * 100) if total_steps > 0 else 0
        )

        # Generate progress summary
        progress_summary = f"Progress: {completed_steps}/{total_steps} steps completed ({progress_percentage:.1f}%)\n\n"

        for step in plan["steps"]:
            status_symbol = (
                "✅"
                if step["status"] == "completed"
                else "⏳" if step["status"] == "in_progress" else "🔄"
            )
            progress_summary += f"{status_symbol} Step {step['step_number']}: {step['action']} ({step['agent']}) - {step['status']}\n"

        return progress_summary

    except Exception as e:
        logger.error(f"Error tracking progress: {str(e)}")
        return f"Failed to track progress: {str(e)}"


async def update_step_status(
    ctx: Context, step_number: int, status: str, result: Optional[str] = None
) -> str:
    """
    Update the status of a step in the analysis plan.

    Args:
        ctx: Context object containing state information
        step_number: The step number to update
        status: New status (completed, in_progress, failed, pending)
        result: Optional result information

    Returns:
        A string confirming the update
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        plan = current_state.get("analysis_plan")

        if not plan:
            return "No analysis plan found in context"

        # Find and update the step
        found = False
        for step in plan["steps"]:
            if step["step_number"] == step_number:
                step["status"] = status
                if result:
                    step["result"] = result
                found = True
                break

        if not found:
            return f"Step {step_number} not found in the analysis plan"

        # Update the plan in the context
        current_state["analysis_plan"] = plan
        await ctx.set("state", current_state)

        return f"Updated status of step {step_number} to '{status}'"

    except Exception as e:
        logger.error(f"Error updating step status: {str(e)}")
        return f"Failed to update step status: {str(e)}"


async def synthesize_results(ctx: Context) -> str:
    """
    Synthesize all analysis results into a coherent response.

    Args:
        ctx: Context object containing state information

    Returns:
        A comprehensive summary of all findings
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        plan = current_state.get("analysis_plan")
        observations = current_state.get("Observations", [])

        if not plan:
            return "No analysis plan found in context"

        # Check if all necessary steps are completed
        incomplete_steps = [
            step
            for step in plan["steps"]
            if step["status"] != "completed" and step["action"] != "synthesize_results"
        ]

        if incomplete_steps:
            incomplete_steps_list = ", ".join(
                [
                    f"Step {step['step_number']}: {step['action']}"
                    for step in incomplete_steps
                ]
            )
            return f"Cannot synthesize yet. Waiting for steps to complete: {incomplete_steps_list}"

        # Extract key observations and insights
        data_summary = current_state.get("data_summary", {})
        column_analyses = current_state.get("column_analyses", {})
        correlation_analysis = current_state.get("correlation_analysis", {})

        # Generate a comprehensive report
        report = []

        # Add dataset overview
        if data_summary:
            row_count = data_summary.get("row_count", "unknown")
            col_count = data_summary.get("column_count", "unknown")
            report.append(
                f"## Dataset Overview\n"
                f"The dataset contains {row_count} rows and {col_count} columns.\n"
            )

            # Missing values info
            missing_info = data_summary.get("missing_values", {})
            if missing_info:
                missing_pct = missing_info.get("percentage", 0)
                report.append(f"Missing values: {missing_pct:.2f}% of all entries.\n")

            # Duplicates info
            duplicate_info = data_summary.get("duplicates", {})
            if duplicate_info and duplicate_info.get("has_duplicates"):
                dup_count = duplicate_info.get("count", 0)
                report.append(f"Found {dup_count} duplicate rows in the dataset.\n")

        # Add key observations
        report.append("## Key Observations\n")
        if observations:
            for idx, obs in enumerate(observations[:10]):  # Limit to top 10
                report.append(f"{idx+1}. {obs}\n")
        else:
            report.append("No significant observations recorded.\n")

        # Add column insights
        if column_analyses:
            report.append("## Column Insights\n")
            for col_name, analysis in list(column_analyses.items())[
                :5
            ]:  # Limit to 5 columns
                col_type = analysis.get("type", "unknown")
                stats = analysis.get("stats", {})

                if col_type == "numeric":
                    min_val = stats.get("min", "N/A")
                    max_val = stats.get("max", "N/A")
                    mean_val = stats.get("mean", "N/A")
                    outlier_info = stats.get("outliers", {})
                    outlier_count = outlier_info.get("count", 0)

                    report.append(
                        f"**{col_name}** (Numeric): Range from {min_val} to {max_val}, "
                        f"average {mean_val:.2f} with {outlier_count} outliers.\n"
                    )

                elif col_type == "categorical":
                    unique_vals = stats.get("unique_values", "N/A")
                    missing_pct = stats.get("missing_percentage", 0)
                    most_common = stats.get("most_common", [])[:3]  # Top 3

                    report.append(
                        f"**{col_name}** (Categorical): {unique_vals} unique values, "
                        f"most common: {', '.join(most_common)}, "
                        f"{missing_pct:.2f}% missing.\n"
                    )

        # Add correlation insights
        if correlation_analysis:
            report.append("## Relationships Between Variables\n")
            high_correlations = correlation_analysis.get("high_correlations", [])

            if high_correlations:
                for corr in high_correlations[:5]:  # Limit to top 5
                    col1 = corr.get("column1", "")
                    col2 = corr.get("column2", "")
                    corr_val = corr.get("correlation", 0)

                    strength = (
                        "strong positive" if corr_val > 0.7 else "strong negative"
                    )
                    report.append(
                        f"**{col1}** and **{col2}** have a {strength} correlation ({corr_val:.2f}).\n"
                    )
            else:
                report.append("No strong correlations found between variables.\n")

        # Add conclusion
        report.append("## Conclusion\n")
        report.append("The analysis has been completed successfully. ")

        original_question = plan.get("original_question", "")
        if original_question:
            report.append(f"Based on your request to '{original_question}', ")

            # Add specific conclusions based on the question content
            if (
                "correlation" in original_question.lower()
                or "relationship" in original_question.lower()
            ):
                if high_correlations:
                    cols = [
                        f"{corr['column1']}-{corr['column2']}"
                        for corr in high_correlations[:2]
                    ]
                    report.append(
                        f"the most significant relationships were found between {' and '.join(cols)}. "
                    )
                else:
                    report.append(
                        "no strong relationships were found between variables. "
                    )

            if (
                "outlier" in original_question.lower()
                or "anomaly" in original_question.lower()
            ):
                outlier_cols = []
                for col_name, analysis in column_analyses.items():
                    if analysis.get("type") == "numeric":
                        outlier_count = (
                            analysis.get("stats", {})
                            .get("outliers", {})
                            .get("count", 0)
                        )
                        if outlier_count > 0:
                            outlier_cols.append((col_name, outlier_count))

                if outlier_cols:
                    top_outlier = max(outlier_cols, key=lambda x: x[1])
                    report.append(
                        f"the most significant outliers were found in column {top_outlier[0]} ({top_outlier[1]} outliers). "
                    )
                else:
                    report.append("no significant outliers were detected. ")

        # Finalize report
        report.append(
            "\nThis analysis provides an overview of the dataset's key characteristics and patterns."
        )

        # Join all report sections
        full_report = "\n".join(report)

        # Store the synthesized report in the context
        current_state["synthesized_report"] = full_report
        await ctx.set("state", current_state)

        return full_report

    except Exception as e:
        logger.error(f"Error synthesizing results: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to synthesize results: {str(e)}"


async def determine_completion(ctx: Context) -> str:
    """
    Determine if the analysis is complete and provide a final status.

    Args:
        ctx: Context object containing state information

    Returns:
        A string indicating if the analysis is complete and next steps
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        plan = current_state.get("analysis_plan")

        if not plan:
            return "No analysis plan found in context"

        # Check steps status
        steps = plan["steps"]
        total_steps = len(steps)
        completed_steps = sum(1 for step in steps if step["status"] == "completed")
        pending_steps = [step for step in steps if step["status"] == "pending"]
        in_progress_steps = [step for step in steps if step["status"] == "in_progress"]

        if completed_steps == total_steps:
            if "synthesized_report" in current_state:
                current_state["analysis_status"] = "completed"
                await ctx.set("state", current_state)
                return "✅ Analysis complete. All steps have been executed and results have been synthesized."
            else:
                return "Almost complete. All steps executed but final synthesis needed."
        else:
            # Determine next step
            if pending_steps:
                next_step = min(pending_steps, key=lambda x: x["step_number"])
                return (
                    f"⏳ Analysis in progress: {completed_steps}/{total_steps} steps completed. "
                    f"Next step is #{next_step['step_number']}: {next_step['action']} using {next_step['agent']}."
                )
            elif in_progress_steps:
                step_info = ", ".join(
                    [
                        f"#{step['step_number']}: {step['action']}"
                        for step in in_progress_steps
                    ]
                )
                return f"⏳ Analysis in progress: Waiting for steps to complete: {step_info}"
            else:
                # This shouldn't happen - all steps are neither completed, pending, nor in progress
                return "⚠️ Analysis status unclear. Please check individual steps for errors."

    except Exception as e:
        logger.error(f"Error determining completion: {str(e)}")
        return f"Failed to determine completion status: {str(e)}"


async def execute_next_step(ctx: Context) -> str:
    """
    Execute the next pending step in the analysis plan.

    Args:
        ctx: Context object containing state information

    Returns:
        A string describing the step execution
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        plan = current_state.get("analysis_plan")

        if not plan:
            return "No analysis plan found in context"

        # Find the next pending step
        pending_steps = [step for step in plan["steps"] if step["status"] == "pending"]

        if not pending_steps:
            return "No pending steps found in the analysis plan"

        next_step = min(pending_steps, key=lambda x: x["step_number"])

        # Mark the step as in progress
        for step in plan["steps"]:
            if step["step_number"] == next_step["step_number"]:
                step["status"] = "in_progress"
                break

        current_state["analysis_plan"] = plan
        await ctx.set("state", current_state)

        # Execute the step based on agent and action
        agent_name = next_step["agent"]
        action = next_step["action"]
        parameters = next_step.get("parameters", {})

        # Delegate to the appropriate agent
        delegation_result = await delegate_to_agent(
            ctx, agent_name, action, **parameters
        )

        return f"Executing step #{next_step['step_number']}: {action} with {agent_name}.\n{delegation_result}"

    except Exception as e:
        logger.error(f"Error executing next step: {str(e)}")
        return f"Failed to execute next step: {str(e)}"


def make_manager_agent(llm) -> FunctionAgent:
    """
    Create a manager agent with planning and delegation tools.

    Args:
        llm: Language model to use for the agent

    Returns:
        FunctionAgent for managing the analysis workflow
    """
    tools = [
        FunctionTool.from_defaults(fn=create_analysis_plan),
        # FunctionTool.from_defaults(fn=delegate_to_agent),
        FunctionTool.from_defaults(fn=track_progress),
        FunctionTool.from_defaults(fn=update_step_status),
        FunctionTool.from_defaults(fn=synthesize_results),
        FunctionTool.from_defaults(fn=determine_completion),
        FunctionTool.from_defaults(fn=execute_next_step),
    ]

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        name="ManagerAgent",
        description="A manager agent that coordinates data analysis with streamlined plans for faster results.",
        system_prompt="""
        You are a data analysis manager assistant. Your role is to coordinate a concise analysis process by:
        
        1. Creating a focused, minimal analysis plan based on the user's question
        2. Delegating only the most essential tasks to specialized agents
        3. Tracking progress efficiently
        4. Synthesizing results into a brief, clear response
        
        When a user asks a question about their data:
        1. Create a minimal plan with create_analysis_plan (limit to 2-3 steps maximum)
        2. Execute steps quickly using execute_next_step
        3. Skip unnecessary analysis steps that don't directly answer the user's question
        4. When all steps are complete, synthesize the results in a concise format
        5. Describe the report to the user in 200 words or less in markdown format.
        
        The ExplorationAgent handles data analysis tasks:
        - Quick data summaries
        - Targeted analysis of relevant columns only
        
        The ReportingAgent creates brief reports with key findings only.
        
        Prioritize speed and relevance over comprehensive analysis. Focus only on what directly
        answers the user's question and skip exploratory steps that aren't essential.
        
        Keep the user informed with short, direct updates on progress.
        """,
        can_handoff_to=["DataLoadingAgent", "ExplorationAgent", "ReportingAgent"],
    )

    return agent
