"""
Reporting Agent

This module defines a reporting agent that specializes in creating formatted reports
from data insights. It can generate markdown formatted reports with various sections
like executive summaries, data overviews, and detailed analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback
import pandas as pd
from datetime import datetime
import json

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

# Set up logging
logger = logging.getLogger(__name__)


async def create_executive_summary(ctx: Context) -> str:
    """
    Create an executive summary for the report based on analysis results.

    Args:
        ctx: Context object containing state information

    Returns:
        Markdown formatted executive summary
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get important information from the state
        observations = current_state.get("Observations", [])
        data_summary = current_state.get("data_summary", {})
        user_question = current_state.get("User Question", "")
        file_name = current_state.get("File Name", "")

        # Extract key dataset information
        row_count = data_summary.get("row_count", "unknown")
        column_count = data_summary.get("column_count", "unknown")

        # Create the executive summary
        summary_parts = [
            "## Executive Summary\n",
            f"This report provides an analysis of the dataset *{file_name}* containing {row_count} rows and {column_count} columns.",
        ]

        # Add user question if available
        if user_question:
            summary_parts.append(
                f"\nThe analysis addresses the question: **{user_question}**"
            )

        # Add key findings from observations (limited to top 3)
        if observations:
            summary_parts.append("\n\n**Key Findings:**")
            for i, obs in enumerate(observations[:3]):
                summary_parts.append(f"\n- {obs}")

            if len(observations) > 3:
                summary_parts.append(
                    f"\n\nAdditional insights are detailed in the following sections."
                )

        # Join all parts
        executive_summary = "".join(summary_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["executive_summary"] = executive_summary
        await ctx.set("state", current_state)

        return executive_summary
    except Exception as e:
        logger.error(f"Error creating executive summary: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to create executive summary: {str(e)}"


async def format_data_overview(ctx: Context) -> str:
    """
    Create a formatted data overview section with key statistics.

    Args:
        ctx: Context object containing state information

    Returns:
        Markdown formatted data overview section
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get data summary
        data_summary = current_state.get("data_summary", {})
        if not data_summary:
            return "No data summary available to format."

        # Extract key information
        row_count = data_summary.get("row_count", "unknown")
        column_count = data_summary.get("column_count", "unknown")
        columns = data_summary.get("columns", [])
        column_types = data_summary.get("column_types", {})
        missing_values = data_summary.get("missing_values", {})
        duplicates = data_summary.get("duplicates", {})

        # Create data overview markdown
        overview_parts = [
            "## Data Overview\n",
            f"**Dataset Dimensions:** {row_count} rows × {column_count} columns\n\n",
        ]

        # Add missing values info
        missing_count = missing_values.get("count", 0)
        missing_pct = missing_values.get("percentage", 0)
        overview_parts.append(
            f"**Missing Values:** {missing_count} ({missing_pct:.2f}% of all entries)\n\n"
        )

        # Add duplicates info
        duplicate_count = duplicates.get("count", 0)
        duplicate_pct = duplicates.get("percentage", 0)
        overview_parts.append(
            f"**Duplicate Rows:** {duplicate_count} ({duplicate_pct:.2f}% of all rows)\n\n"
        )

        # Add column type breakdown
        numeric_columns = column_types.get("numeric", [])
        categorical_columns = column_types.get("categorical", [])
        datetime_columns = column_types.get("datetime", [])

        overview_parts.append("**Column Types:**\n")
        overview_parts.append(f"- Numeric: {len(numeric_columns)} columns\n")
        overview_parts.append(f"- Categorical: {len(categorical_columns)} columns\n")
        overview_parts.append(f"- Datetime: {len(datetime_columns)} columns\n\n")

        # Create a table of columns (limit to 10 for readability)
        if columns:
            overview_parts.append("**Column Preview:**\n\n")
            overview_parts.append("| # | Column Name | Type |\n")
            overview_parts.append("|---|------------|------|\n")

            for i, col in enumerate(columns[:10]):
                col_type = (
                    "Numeric"
                    if col in numeric_columns
                    else (
                        "Categorical"
                        if col in categorical_columns
                        else "Datetime" if col in datetime_columns else "Other"
                    )
                )
                overview_parts.append(f"| {i+1} | {col} | {col_type} |\n")

            if len(columns) > 10:
                overview_parts.append(f"\n*...and {len(columns) - 10} more columns*\n")

        # Join all parts
        data_overview = "".join(overview_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["data_overview"] = data_overview
        await ctx.set("state", current_state)

        return data_overview
    except Exception as e:
        logger.error(f"Error formatting data overview: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to format data overview: {str(e)}"


async def format_correlation_section(ctx: Context) -> str:
    """
    Create a formatted section for correlation analysis.

    Args:
        ctx: Context object containing state information

    Returns:
        Markdown formatted correlation section
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get correlation analysis
        correlation_analysis = current_state.get("correlation_analysis", {})
        if not correlation_analysis:
            return "No correlation analysis available to format."

        # Extract key information
        high_correlations = correlation_analysis.get("high_correlations", [])

        # Create correlation section
        corr_parts = [
            "## Correlation Analysis\n",
            "This section highlights the relationships between numeric variables in the dataset.\n\n",
        ]

        if high_correlations:
            # Create a table of high correlations
            corr_parts.append("**Strong Correlations:**\n\n")
            corr_parts.append("| Feature 1 | Feature 2 | Correlation |\n")
            corr_parts.append("|-----------|-----------|------------|\n")

            # Sort by absolute correlation value
            sorted_correlations = sorted(
                high_correlations,
                key=lambda x: abs(x.get("correlation", 0)),
                reverse=True,
            )

            for corr in sorted_correlations[:10]:  # Limit to top 10
                col1 = corr.get("column1", "")
                col2 = corr.get("column2", "")
                corr_val = corr.get("correlation", 0)

                # Format correlation with color hint
                if corr_val > 0:
                    corr_str = f"**{corr_val:.3f}** (positive)"
                else:
                    corr_str = f"**{corr_val:.3f}** (negative)"

                corr_parts.append(f"| {col1} | {col2} | {corr_str} |\n")

            corr_parts.append("\n")

            # Add interpretation
            corr_parts.append("**Interpretation:**\n\n")
            corr_parts.append(
                "- A correlation close to 1 indicates a strong positive relationship\n"
            )
            corr_parts.append(
                "- A correlation close to -1 indicates a strong negative relationship\n"
            )
            corr_parts.append(
                "- A correlation close to 0 indicates little to no linear relationship\n\n"
            )

            # Add specific insights based on strongest correlation
            if sorted_correlations:
                strongest = sorted_correlations[0]
                col1 = strongest.get("column1", "")
                col2 = strongest.get("column2", "")
                corr_val = strongest.get("correlation", 0)

                if abs(corr_val) > 0.8:
                    strength = "very strong"
                elif abs(corr_val) > 0.6:
                    strength = "strong"
                else:
                    strength = "moderate"

                direction = "positive" if corr_val > 0 else "negative"

                corr_parts.append(
                    f"The {strength} {direction} correlation between **{col1}** and **{col2}** "
                )

                if corr_val > 0:
                    corr_parts.append(
                        f"indicates that as **{col1}** increases, **{col2}** tends to increase as well.\n"
                    )
                else:
                    corr_parts.append(
                        f"indicates that as **{col1}** increases, **{col2}** tends to decrease.\n"
                    )
        else:
            corr_parts.append(
                "No strong correlations (|r| > 0.7) were found between any numeric columns in the dataset.\n"
            )

        # Join all parts
        correlation_section = "".join(corr_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["correlation_analysis"] = correlation_section
        await ctx.set("state", current_state)

        return correlation_section
    except Exception as e:
        logger.error(f"Error formatting correlation section: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to format correlation section: {str(e)}"


async def format_column_insights(ctx: Context, max_columns: int = 5) -> str:
    """
    Create a formatted section with insights about individual columns.

    Args:
        ctx: Context object containing state information
        max_columns: Maximum number of columns to include (default: 5)

    Returns:
        Markdown formatted column insights section
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get column analyses
        column_analyses = current_state.get("column_analyses", {})
        if not column_analyses:
            return "No column analyses available to format."

        # Create column insights section
        insights_parts = [
            "## Column Insights\n",
            "This section provides detailed analysis of key columns in the dataset.\n\n",
        ]

        # Add insights for numeric columns first
        numeric_columns = {
            col: analysis
            for col, analysis in column_analyses.items()
            if analysis.get("type") == "numeric"
        }

        if numeric_columns:
            insights_parts.append("### Numeric Columns\n\n")

            count = 0
            for col, analysis in numeric_columns.items():
                if count >= max_columns:
                    break

                stats = analysis.get("stats", {})

                # Skip if no meaningful stats
                if not stats:
                    continue

                insights_parts.append(f"#### {col}\n\n")

                # Add basic statistics
                min_val = stats.get("min")
                max_val = stats.get("max")
                mean_val = stats.get("mean")
                median_val = stats.get("median")
                missing = stats.get("missing", 0)
                missing_pct = stats.get("missing_percentage", 0)

                insights_parts.append(f"**Range:** {min_val} to {max_val}\n\n")
                insights_parts.append(f"**Central Tendency:**\n")
                insights_parts.append(f"- Mean: {mean_val:.2f}\n")
                insights_parts.append(f"- Median: {median_val:.2f}\n\n")

                if missing > 0:
                    insights_parts.append(
                        f"**Missing Values:** {missing} ({missing_pct:.2f}%)\n\n"
                    )

                # Add outlier information if available
                outliers = stats.get("outliers", {})
                if outliers:
                    outlier_count = outliers.get("count", 0)
                    outlier_pct = outliers.get("percentage", 0)

                    if outlier_count > 0:
                        insights_parts.append(
                            f"**Outliers:** {outlier_count} values ({outlier_pct:.2f}%) "
                        )
                        insights_parts.append(
                            f"outside the range [{outliers.get('lower_bound', 'N/A'):.2f}, {outliers.get('upper_bound', 'N/A'):.2f}]\n\n"
                        )

                # Add a horizontal rule between columns
                insights_parts.append("---\n\n")
                count += 1

        # Add insights for categorical columns
        categorical_columns = {
            col: analysis
            for col, analysis in column_analyses.items()
            if analysis.get("type") == "categorical"
        }

        if categorical_columns:
            insights_parts.append("### Categorical Columns\n\n")

            count = 0
            for col, analysis in categorical_columns.items():
                if count >= max_columns:
                    break

                stats = analysis.get("stats", {})

                # Skip if no meaningful stats
                if not stats:
                    continue

                insights_parts.append(f"#### {col}\n\n")

                # Add basic statistics
                unique_values = stats.get("unique_values", 0)
                most_common = stats.get("most_common", [])
                most_common_counts = stats.get("most_common_counts", [])
                missing = stats.get("missing", 0)
                missing_pct = stats.get("missing_percentage", 0)

                insights_parts.append(f"**Unique Values:** {unique_values}\n\n")

                if missing > 0:
                    insights_parts.append(
                        f"**Missing Values:** {missing} ({missing_pct:.2f}%)\n\n"
                    )

                # Add table of most common values
                if most_common and most_common_counts:
                    insights_parts.append("**Most Common Values:**\n\n")
                    insights_parts.append("| Value | Count | Percentage |\n")
                    insights_parts.append("|-------|-------|------------|\n")

                    total = sum(most_common_counts) if most_common_counts else 0

                    for i, (value, count) in enumerate(
                        zip(most_common[:5], most_common_counts[:5])
                    ):
                        pct = (count / total * 100) if total > 0 else 0
                        insights_parts.append(f"| {value} | {count} | {pct:.1f}% |\n")

                # Add a horizontal rule between columns
                insights_parts.append("---\n\n")
                count += 1

        # Join all parts
        column_insights = "".join(insights_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["column_insights"] = column_insights
        await ctx.set("state", current_state)

        return column_insights
    except Exception as e:
        logger.error(f"Error formatting column insights: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to format column insights: {str(e)}"


async def suggest_visualizations(ctx: Context) -> str:
    """
    Suggest visualizations based on the data analysis.

    Args:
        ctx: Context object containing state information

    Returns:
        Markdown formatted visualization recommendations
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get necessary data
        data_summary = current_state.get("data_summary", {})
        column_analyses = current_state.get("column_analyses", {})
        correlation_analysis = current_state.get("correlation_analysis", {})

        # Check if we have enough information to suggest visualizations
        if not data_summary:
            return "Insufficient data to suggest visualizations."

        # Create visualizations section
        viz_parts = [
            "## Recommended Visualizations\n",
            "Based on the data analysis, these visualizations would provide further insights:\n\n",
        ]

        # Get column types
        column_types = data_summary.get("column_types", {})
        numeric_columns = column_types.get("numeric", [])
        categorical_columns = column_types.get("categorical", [])
        datetime_columns = column_types.get("datetime", [])

        suggestions = []

        # If we have numeric columns, suggest histograms
        if numeric_columns:
            # Find interesting numeric columns (with outliers or skew)
            interesting_cols = []
            for col, analysis in column_analyses.items():
                if analysis.get("type") == "numeric":
                    stats = analysis.get("stats", {})
                    outliers = stats.get("outliers", {})
                    if outliers and outliers.get("count", 0) > 0:
                        interesting_cols.append(col)

            # If we found interesting columns, suggest them first
            cols_to_suggest = (
                interesting_cols[:2] if interesting_cols else numeric_columns[:2]
            )

            for col in cols_to_suggest:
                suggestions.append(
                    {
                        "type": "Histogram",
                        "target": col,
                        "description": f"A histogram of **{col}** would reveal the distribution and identify any skewness or unusual patterns.",
                    }
                )

        # If we have categorical columns, suggest bar charts
        if categorical_columns:
            for col in categorical_columns[:2]:
                suggestions.append(
                    {
                        "type": "Bar Chart",
                        "target": col,
                        "description": f"A bar chart of **{col}** frequencies would show the distribution of categories.",
                    }
                )

        # If we have datetime columns and numeric columns, suggest time series
        if datetime_columns and numeric_columns:
            suggestions.append(
                {
                    "type": "Time Series Plot",
                    "target": f"{datetime_columns[0]} vs {numeric_columns[0]}",
                    "description": f"A time series plot of **{numeric_columns[0]}** over **{datetime_columns[0]}** would reveal trends and seasonal patterns.",
                }
            )

        # If we have correlation analysis, suggest scatter plots
        if correlation_analysis:
            high_correlations = correlation_analysis.get("high_correlations", [])
            if high_correlations:
                # Suggest scatter plot for highest correlation
                top_corr = high_correlations[0]
                col1 = top_corr.get("column1", "")
                col2 = top_corr.get("column2", "")
                corr_val = top_corr.get("correlation", 0)

                suggestions.append(
                    {
                        "type": "Scatter Plot",
                        "target": f"{col1} vs {col2}",
                        "description": f"A scatter plot of **{col1}** vs **{col2}** would visualize their strong correlation (r={corr_val:.2f}) and help identify any non-linear patterns.",
                    }
                )

        # If we have multiple numeric columns, suggest a correlation heatmap
        if len(numeric_columns) > 3:
            suggestions.append(
                {
                    "type": "Correlation Heatmap",
                    "target": "All numeric columns",
                    "description": "A correlation heatmap of all numeric variables would provide an overview of relationships between variables.",
                }
            )

        # If we have both numeric and categorical columns, suggest box plots
        if numeric_columns and categorical_columns:
            suggestions.append(
                {
                    "type": "Box Plot",
                    "target": f"{numeric_columns[0]} by {categorical_columns[0]}",
                    "description": f"Box plots of **{numeric_columns[0]}** grouped by **{categorical_columns[0]}** would show differences between categories and identify outliers.",
                }
            )

        # Create a list of visualization suggestions
        if suggestions:
            for i, sugg in enumerate(suggestions):
                viz_parts.append(f"### {i+1}. {sugg['type']}: {sugg['target']}\n")
                viz_parts.append(f"{sugg['description']}\n\n")
        else:
            viz_parts.append(
                "No specific visualizations can be recommended based on the current analysis.\n"
            )

        # Join all parts
        viz_section = "".join(viz_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["visualization_recommendations"] = viz_section
        await ctx.set("state", current_state)

        return viz_section
    except Exception as e:
        logger.error(f"Error suggesting visualizations: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to suggest visualizations: {str(e)}"


async def create_conclusion(ctx: Context) -> str:
    """
    Create a conclusion section summarizing key findings and next steps.

    Args:
        ctx: Context object containing state information

    Returns:
        Markdown formatted conclusion section
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Get observations and user question
        observations = current_state.get("Observations", [])
        user_question = current_state.get("User Question", "")

        # Create conclusion section
        conclusion_parts = ["## Conclusion\n"]

        # Address the original user question
        if user_question:
            conclusion_parts.append(
                f'This analysis addressed the question: "{user_question}"\n\n'
            )

        # Summarize key findings
        if observations:
            conclusion_parts.append("### Key Findings\n\n")

            top_observations = observations[:5]
            for obs in top_observations:
                conclusion_parts.append(f"- {obs}\n")

            conclusion_parts.append("\n")

        # Add recommendations
        conclusion_parts.append("### Recommendations\n\n")

        # Generate recommendations based on the analysis
        data_summary = current_state.get("data_summary", {})
        column_analyses = current_state.get("column_analyses", {})

        recommendations = []

        # Check for missing values
        missing_values = data_summary.get("missing_values", {})
        missing_pct = missing_values.get("percentage", 0)
        if missing_pct > 5:
            recommendations.append(
                "Address missing values in the dataset. Consider imputation strategies or removing columns with excessive missing data."
            )

        # Check for duplicate rows
        duplicates = data_summary.get("duplicates", {})
        if duplicates.get("has_duplicates", False):
            recommendations.append("Remove duplicate rows to ensure analysis accuracy.")

        # Check for outliers
        has_outliers = False
        for col, analysis in column_analyses.items():
            if analysis.get("type") == "numeric":
                stats = analysis.get("stats", {})
                outliers = stats.get("outliers", {})
                if outliers and outliers.get("count", 0) > 0:
                    has_outliers = True
                    break

        if has_outliers:
            recommendations.append(
                "Investigate outliers in numeric columns. Consider their impact on your analysis and whether they represent actual data or errors."
            )

        # Add general recommendations
        recommendations.append(
            "Consider feature engineering to create new variables that might better capture the underlying patterns in the data."
        )

        recommendations.append(
            "Apply appropriate statistical tests to validate any hypotheses suggested by this exploratory analysis."
        )

        # Add recommendations to the conclusion
        for rec in recommendations:
            conclusion_parts.append(f"- {rec}\n")

        conclusion_parts.append("\n### Next Steps\n\n")
        conclusion_parts.append("For further analysis, consider:\n\n")
        conclusion_parts.append(
            "1. Creating the visualizations recommended in this report\n"
        )
        conclusion_parts.append("2. Performing more advanced statistical modeling\n")
        conclusion_parts.append(
            "3. Collecting additional data if necessary to address gaps identified in this analysis\n"
        )

        # Join all parts
        conclusion_section = "".join(conclusion_parts)

        # Store in state
        current_state["report_sections"] = current_state.get("report_sections", {})
        current_state["report_sections"]["conclusion"] = conclusion_section
        await ctx.set("state", current_state)

        return conclusion_section
    except Exception as e:
        logger.error(f"Error creating conclusion: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to create conclusion: {str(e)}"


async def generate_complete_report(ctx: Context) -> str:
    """
    Generate a complete report by assembling all sections.

    Args:
        ctx: Context object containing state information

    Returns:
        Complete markdown formatted report
    """
    try:
        # Get current state
        current_state = await ctx.get("state")
        file_name = current_state.get("File Name", "Dataset")

        # Check if we have report sections
        report_sections = current_state.get("report_sections", {})
        if not report_sections:
            # Generate sections if they don't exist
            await create_executive_summary(ctx)
            await format_data_overview(ctx)
            await format_correlation_section(ctx)
            await format_column_insights(ctx)
            await suggest_visualizations(ctx)
            await create_conclusion(ctx)

            # Get updated state with all sections
            current_state = await ctx.get("state")
            report_sections = current_state.get("report_sections", {})

        # Create report header
        current_date = datetime.now().strftime("%Y-%m-%d")

        header = [
            f"# Data Analysis Report: {file_name}\n",
            f"*Generated on {current_date}*\n\n",
            "---\n\n",
        ]

        # Create table of contents
        toc = ["## Table of Contents\n\n"]

        section_order = [
            "executive_summary",
            "data_overview",
            "correlation_analysis",
            "column_insights",
            "visualization_recommendations",
            "conclusion",
        ]

        section_titles = {
            "executive_summary": "Executive Summary",
            "data_overview": "Data Overview",
            "correlation_analysis": "Correlation Analysis",
            "column_insights": "Column Insights",
            "visualization_recommendations": "Recommended Visualizations",
            "conclusion": "Conclusion",
        }

        # Add sections to TOC if they exist
        toc_items = []
        for section_id in section_order:
            if section_id in report_sections:
                toc_items.append(
                    f"- [{section_titles.get(section_id, section_id)}](#{section_id.replace('_', '-')})"
                )

        toc.extend([f"{item}\n" for item in toc_items])
        toc.append("\n---\n\n")

        # Assemble report sections in order
        sections = []
        for section_id in section_order:
            if section_id in report_sections:
                sections.append(report_sections[section_id])
                sections.append("\n\n---\n\n")  # Add separator between sections

        # Remove the last separator
        if sections:
            sections = sections[:-1]

        # Add footer
        footer = [
            "\n\n---\n\n",
            "*This report was automatically generated by DataHammer Analytics.*\n",
        ]

        # Combine all parts
        full_report = "".join(header + toc + sections + footer)

        # Store the complete report in the context
        current_state["final_report"] = full_report
        current_state["report_generated_timestamp"] = datetime.now().isoformat()
        await ctx.set("state", current_state)

        return "Complete report generated successfully. See 'final_report' in the context for the full markdown document."
    except Exception as e:
        logger.error(f"Error generating complete report: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Failed to generate complete report: {str(e)}"


async def get_report_content(ctx: Context) -> str:
    """
    Retrieve the generated report content.

    Args:
        ctx: Context object containing state information

    Returns:
        The complete report content in markdown format
    """
    try:
        # Get current state
        current_state = await ctx.get("state")

        # Check if report exists
        if "final_report" in current_state:
            return current_state["final_report"]
        else:
            # Try to generate it if it doesn't exist
            await generate_complete_report(ctx)

            # Get updated state
            current_state = await ctx.get("state")

            if "final_report" in current_state:
                return current_state["final_report"]
            else:
                return "No report has been generated yet. Please run generate_complete_report first."
    except Exception as e:
        logger.error(f"Error retrieving report content: {str(e)}")
        return f"Failed to retrieve report content: {str(e)}"


def make_reporting_agent(llm) -> FunctionAgent:
    """
    Create a reporting agent with markdown formatting tools.

    Args:
        llm: Language model to use for the agent

    Returns:
        FunctionAgent for creating markdown reports
    """
    tools = [
        FunctionTool.from_defaults(fn=create_executive_summary),
        FunctionTool.from_defaults(fn=format_data_overview),
        FunctionTool.from_defaults(fn=format_correlation_section),
        FunctionTool.from_defaults(fn=format_column_insights),
        FunctionTool.from_defaults(fn=suggest_visualizations),
        FunctionTool.from_defaults(fn=create_conclusion),
        FunctionTool.from_defaults(fn=generate_complete_report),
        FunctionTool.from_defaults(fn=get_report_content),
    ]

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        name="ReportingAgent",
        description="A reporting agent that creates formatted markdown reports from data analysis results.",
        system_prompt="""
        You are a data reporting assistant that creates professional markdown reports from analysis results.
        
        Your capabilities:
        - Creating executive summaries, data overviews, correlation analyses
        - Formatting column insights, suggesting visualizations
        - Creating conclusions with recommendations
        - Assembling complete reports
        
        When working:
        - Check whether to generate specific sections or complete reports
        - Use appropriate tools (create_executive_summary, format_data_overview, etc.)
        - Structure content logically with clear markdown formatting
        
        Ensure reports are clear, concise, well-structured, and focus on actionable insights.
        
        After completion, hand off to the ManagerAgent.
        """,
        can_handoff_to=["ManagerAgent"],
    )

    return agent
