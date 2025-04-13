"""
Data Exploration Agent

This module defines a data exploration agent that can be used with LlamaIndex agent workflows.
It provides functionality for analyzing and visualizing data.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np
import traceback

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

# Set up logging
logger = logging.getLogger(__name__)


async def analyze_numeric_column(ctx: Context, column_name: str) -> str:
    """
    Analyze a numeric column and return statistics.

    Args:
        ctx: Context object containing the DataFrame
        column_name: Name of the numeric column to analyze

    Returns:
        String summarizing the analysis results
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in DataFrame"

        series = df[column_name]
        # Convert to numeric if not already
        numeric_series = pd.to_numeric(series, errors="coerce")
        missing_count = numeric_series.isna().sum()

        # Only compute statistics if we have valid data
        if len(numeric_series.dropna()) > 0:
            stats = {
                "min": (
                    float(numeric_series.min())
                    if not pd.isna(numeric_series.min())
                    else None
                ),
                "max": (
                    float(numeric_series.max())
                    if not pd.isna(numeric_series.max())
                    else None
                ),
                "mean": (
                    float(numeric_series.mean())
                    if not pd.isna(numeric_series.mean())
                    else None
                ),
                "median": (
                    float(numeric_series.median())
                    if not pd.isna(numeric_series.median())
                    else None
                ),
                "std": (
                    float(numeric_series.std())
                    if not pd.isna(numeric_series.std())
                    else None
                ),
                "missing": int(missing_count),
                "missing_percentage": float(missing_count / len(series) * 100),
                "quartiles": {
                    "25%": (
                        float(numeric_series.quantile(0.25))
                        if not pd.isna(numeric_series.quantile(0.25))
                        else None
                    ),
                    "50%": (
                        float(numeric_series.quantile(0.5))
                        if not pd.isna(numeric_series.quantile(0.5))
                        else None
                    ),
                    "75%": (
                        float(numeric_series.quantile(0.75))
                        if not pd.isna(numeric_series.quantile(0.75))
                        else None
                    ),
                },
            }

            # Detect potential outliers using IQR method
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_series[
                (numeric_series < lower_bound) | (numeric_series > upper_bound)
            ]

            stats["outliers"] = {
                "count": int(len(outliers)),
                "percentage": (
                    float(len(outliers) / len(numeric_series.dropna()) * 100)
                    if len(numeric_series.dropna()) > 0
                    else 0.0
                ),
                "lower_bound": float(lower_bound) if not pd.isna(lower_bound) else None,
                "upper_bound": float(upper_bound) if not pd.isna(upper_bound) else None,
            }

            # Store the analysis in the context
            if "column_analyses" not in current_state:
                current_state["column_analyses"] = {}

            current_state["column_analyses"][column_name] = {
                "type": "numeric",
                "stats": stats,
            }

            # Store an observation about the column
            if "Observations" not in current_state:
                current_state["Observations"] = []

            current_state["Observations"].append(
                f"Column '{column_name}' is numeric with values ranging from {stats['min']} to {stats['max']}, "
                f"mean {stats['mean']:.2f}, with {stats['missing']} missing values ({stats['missing_percentage']:.2f}%)"
            )

            # Update the context
            await ctx.set("state", current_state)

            return f"Analyzed numeric column '{column_name}'. Mean: {stats['mean']:.2f}, Range: {stats['min']} to {stats['max']}, Missing: {stats['missing_percentage']:.2f}%, Outliers: {stats['outliers']['count']}"
        else:
            error_msg = f"Column '{column_name}' has no valid numeric data"
            if "Observations" not in current_state:
                current_state["Observations"] = []
            current_state["Observations"].append(error_msg)
            await ctx.set("state", current_state)
            return error_msg
    except Exception as e:
        logger.error(f"Error analyzing numeric column: {str(e)}")
        return f"Error analyzing column '{column_name}': {str(e)}"


async def analyze_categorical_column(ctx: Context, column_name: str) -> str:
    """
    Analyze a categorical column and return statistics.

    Args:
        ctx: Context object containing the DataFrame
        column_name: Name of the categorical column to analyze

    Returns:
        String summarizing the analysis results
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in DataFrame"

        series = df[column_name]
        missing_count = series.isna().sum()
        valid_series = series.dropna()

        if len(valid_series) > 0:
            # Get value counts
            value_counts = valid_series.value_counts()

            # Get most common values (up to 10)
            most_common = value_counts.head(10).index.tolist()
            most_common_counts = value_counts.head(10).tolist()

            # Convert to native Python types for JSON serialization
            most_common = [str(val) for val in most_common]

            stats = {
                "unique_values": int(len(value_counts)),
                "most_common": most_common,
                "most_common_counts": most_common_counts,
                "missing": int(missing_count),
                "missing_percentage": float(missing_count / len(series) * 100),
            }

            # Store the analysis in the context
            if "column_analyses" not in current_state:
                current_state["column_analyses"] = {}

            current_state["column_analyses"][column_name] = {
                "type": "categorical",
                "stats": stats,
            }

            # Store an observation about the column
            if "Observations" not in current_state:
                current_state["Observations"] = []

            current_state["Observations"].append(
                f"Column '{column_name}' is categorical with {stats['unique_values']} unique values, "
                f"most common: '{most_common[0]}' (appears {most_common_counts[0]} times), "
                f"with {stats['missing']} missing values ({stats['missing_percentage']:.2f}%)"
            )

            # Update the context
            await ctx.set("state", current_state)

            most_common_str = ", ".join(most_common[:3])
            return f"Analyzed categorical column '{column_name}'. Unique values: {stats['unique_values']}, Most common: {most_common_str}, Missing: {stats['missing_percentage']:.2f}%"
        else:
            error_msg = f"Column '{column_name}' has no valid categorical data"
            if "Observations" not in current_state:
                current_state["Observations"] = []
            current_state["Observations"].append(error_msg)
            await ctx.set("state", current_state)
            return error_msg
    except Exception as e:
        logger.error(f"Error analyzing categorical column: {str(e)}")
        return f"Error analyzing column '{column_name}': {str(e)}"


async def compute_correlations(ctx: Context) -> str:
    """
    Compute correlations between numeric columns in the DataFrame.

    Args:
        ctx: Context object containing the DataFrame

    Returns:
        String summarizing the correlation analysis
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        if not numeric_df.empty and numeric_df.shape[1] > 1:
            # Compute Pearson correlation
            pearson_corr = numeric_df.corr(method="pearson")

            # Convert to dictionary format
            pearson_dict = {}
            for col1 in pearson_corr.columns:
                pearson_dict[col1] = {}
                for col2 in pearson_corr.columns:
                    if col1 != col2:  # Skip self-correlations
                        value = pearson_corr.loc[col1, col2]
                        if not pd.isna(value):
                            pearson_dict[col1][col2] = float(value)

            # Find highly correlated features
            high_correlations = []
            for i, col1 in enumerate(pearson_corr.columns):
                for j, col2 in enumerate(pearson_corr.columns):
                    if i < j:  # Only look at upper triangle
                        corr_value = pearson_corr.loc[col1, col2]
                        if abs(corr_value) > 0.7:  # Threshold for high correlation
                            high_correlations.append(
                                {
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": float(corr_value),
                                }
                            )

            # Store the correlation analysis in the context
            current_state["correlation_analysis"] = {
                "pearson": pearson_dict,
                "high_correlations": high_correlations,
            }

            # Store observations about high correlations
            if "Observations" not in current_state:
                current_state["Observations"] = []

            for hc in high_correlations[:5]:  # Limit to top 5
                current_state["Observations"].append(
                    f"Strong correlation ({hc['correlation']:.2f}) detected between '{hc['column1']}' and '{hc['column2']}'"
                )

            # Update the context
            await ctx.set("state", current_state)

            correlation_summary = f"Computed correlations between {len(numeric_df.columns)} numeric columns."
            if high_correlations:
                correlation_summary += (
                    f" Found {len(high_correlations)} strong correlations."
                )
                # Add details about top 3
                for i, hc in enumerate(high_correlations[:3]):
                    correlation_summary += f"\n- {hc['column1']} and {hc['column2']}: {hc['correlation']:.2f}"
            else:
                correlation_summary += " No strong correlations found."

            return correlation_summary
        else:
            return "Not enough numeric columns for correlation analysis"
    except Exception as e:
        logger.error(f"Error computing correlations: {str(e)}")
        return f"Error computing correlations: {str(e)}"


async def generate_data_summary(ctx: Context) -> str:
    """
    Generate a summary of the DataFrame including basic statistics and data quality metrics.

    Args:
        ctx: Context object containing the DataFrame

    Returns:
        String summarizing the DataFrame
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        row_count = df.shape[0]
        column_count = df.shape[1]
        columns = df.columns.tolist()

        # Calculate missing values
        missing_values = df.isna().sum().sum()
        missing_percentage = (
            (missing_values / (row_count * column_count) * 100)
            if row_count * column_count > 0
            else 0
        )

        # Check for duplicates
        duplicate_count = df.duplicated().sum()

        # Analyze column types
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()

        summary = {
            "row_count": int(row_count),
            "column_count": int(column_count),
            "columns": columns,
            "column_types": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "datetime": datetime_columns,
            },
            "missing_values": {
                "count": int(missing_values),
                "percentage": float(missing_percentage),
            },
            "duplicates": {
                "count": int(duplicate_count),
                "percentage": (
                    float(duplicate_count / row_count * 100) if row_count > 0 else 0
                ),
                "has_duplicates": duplicate_count > 0,
            },
        }

        # Store the summary in the context
        current_state["data_summary"] = summary

        # Store observations about the dataset
        if "Observations" not in current_state:
            current_state["Observations"] = []

        current_state["Observations"].append(
            f"Dataset has {row_count} rows and {column_count} columns ({len(numeric_columns)} numeric, "
            f"{len(categorical_columns)} categorical, {len(datetime_columns)} datetime)"
        )

        if missing_percentage > 0:
            current_state["Observations"].append(
                f"Dataset contains {missing_percentage:.2f}% missing values"
            )

        if duplicate_count > 0:
            current_state["Observations"].append(
                f"Dataset contains {duplicate_count} duplicate rows ({(duplicate_count / row_count * 100):.2f}%)"
            )

        # Update the context
        await ctx.set("state", current_state)

        # Return a string summary
        summary_text = f"Dataset Summary: {row_count} rows, {column_count} columns "
        summary_text += f"({len(numeric_columns)} numeric, {len(categorical_columns)} categorical, {len(datetime_columns)} datetime). "

        if missing_percentage > 0:
            summary_text += f"Missing values: {missing_percentage:.2f}%. "

        if duplicate_count > 0:
            summary_text += f"Duplicate rows: {duplicate_count} ({(duplicate_count / row_count * 100):.2f}%). "

        return summary_text
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        return f"Error generating data summary: {str(e)}"


async def detect_outliers(ctx: Context, column_name: str) -> str:
    """
    Detect outliers in a numeric column using the IQR method.

    Args:
        ctx: Context object containing the DataFrame
        column_name: Name of the numeric column to analyze

    Returns:
        String summarizing the outlier detection
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in DataFrame"

        series = df[column_name]
        # Convert to numeric if not already
        numeric_series = pd.to_numeric(series, errors="coerce")

        if len(numeric_series.dropna()) > 0:
            # Calculate IQR
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1

            # Define outlier boundaries
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Find outliers
            outliers = numeric_series[
                (numeric_series < lower_bound) | (numeric_series > upper_bound)
            ]
            outlier_indices = outliers.index.tolist()

            outlier_info = {
                "column": column_name,
                "outlier_count": int(len(outliers)),
                "outlier_percentage": float(
                    len(outliers) / len(numeric_series.dropna()) * 100
                ),
                "lower_bound": float(lower_bound) if not pd.isna(lower_bound) else None,
                "upper_bound": float(upper_bound) if not pd.isna(upper_bound) else None,
                "outlier_indices": (
                    outlier_indices[:10]
                    if len(outlier_indices) > 10
                    else outlier_indices
                ),
                "outlier_values": (
                    outliers.tolist()[:10] if len(outliers) > 10 else outliers.tolist()
                ),
            }

            # Store the outlier analysis in the context
            if "outlier_analyses" not in current_state:
                current_state["outlier_analyses"] = {}

            current_state["outlier_analyses"][column_name] = outlier_info

            # Store an observation about the outliers
            if "Observations" not in current_state:
                current_state["Observations"] = []

            current_state["Observations"].append(
                f"Column '{column_name}' has {len(outliers)} outliers ({outlier_info['outlier_percentage']:.2f}% of non-null values)"
            )

            # Update the context
            await ctx.set("state", current_state)

            # Return a string summary
            outlier_summary = f"Detected {len(outliers)} outliers ({outlier_info['outlier_percentage']:.2f}%) in column '{column_name}'"
            if len(outliers) > 0:
                sample_values = [
                    f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                    for val in outliers.tolist()[:5]
                ]
                outlier_summary += (
                    f"\nOutlier values (sample): {', '.join(sample_values)}"
                    if sample_values
                    else ""
                )
                outlier_summary += (
                    f"\nOutlier boundaries: < {lower_bound:.2f} or > {upper_bound:.2f}"
                )

            return outlier_summary
        else:
            return f"Column '{column_name}' has insufficient valid numeric data to detect outliers"
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return f"Error detecting outliers in column '{column_name}': {str(e)}"


async def analyze_dataframe(ctx: Context) -> str:
    """
    Perform comprehensive analysis on the DataFrame.

    Args:
        ctx: Context object containing the DataFrame

    Returns:
        String summarizing the analysis results
    """
    try:
        # Get current state and DataFrame
        current_state = await ctx.get("state")
        df = current_state.get("DataFrame")

        if df is None:
            return "Error: No DataFrame found in context"

        # Generate data summary
        await generate_data_summary(ctx)

        # Get updated state after summary generation
        current_state = await ctx.get("state")
        summary = current_state.get("data_summary", {})

        # Extract column types
        numeric_columns = summary.get("column_types", {}).get("numeric", [])
        categorical_columns = summary.get("column_types", {}).get("categorical", [])

        # Analyze numeric columns (limit to first 10 for performance)
        for col in numeric_columns[:10]:
            await analyze_numeric_column(ctx, col)

        # Analyze categorical columns (limit to first 10)
        for col in categorical_columns[:10]:
            await analyze_categorical_column(ctx, col)

        # Compute correlations if we have multiple numeric columns
        if len(numeric_columns) > 1:
            await compute_correlations(ctx)

        # Get the updated state with all analyses
        current_state = await ctx.get("state")

        # Generate insights
        insights = []
        observations = current_state.get("Observations", [])

        # Generate a comprehensive analysis report
        analysis_report = f"Completed comprehensive analysis of the dataset with {summary.get('row_count', 0)} rows and {summary.get('column_count', 0)} columns.\n\n"

        analysis_report += "Key Findings:\n"
        # Add top observations (limit to avoid overwhelming response)
        for idx, obs in enumerate(observations[:7]):
            analysis_report += f"{idx+1}. {obs}\n"

        if len(observations) > 7:
            analysis_report += f"... and {len(observations) - 7} more observations\n"

        # Store analysis completion status in context
        current_state["analysis_completed"] = True
        current_state["analysis_timestamp"] = pd.Timestamp.now().isoformat()
        await ctx.set("state", current_state)

        return analysis_report
    except Exception as e:
        logger.error(f"Error analyzing DataFrame: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        return f"Error analyzing DataFrame: {str(e)}"


def make_data_exploration_agent(llm) -> FunctionAgent:
    """
    Create a data exploration agent with analysis tools.

    Args:
        llm: Language model to use for the agent

    Returns:
        FunctionAgent for data exploration
    """
    tools = [
        FunctionTool.from_defaults(fn=analyze_numeric_column),
        FunctionTool.from_defaults(fn=analyze_categorical_column),
        FunctionTool.from_defaults(fn=compute_correlations),
        FunctionTool.from_defaults(fn=generate_data_summary),
        FunctionTool.from_defaults(fn=detect_outliers),
        FunctionTool.from_defaults(fn=analyze_dataframe),
    ]

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        name="ExplorationAgent",
        description="A data exploration agent that analyzes datasets and provides insights.",
        system_prompt="""
        You are a data exploration assistant that analyzes datasets and provides insights. Your goal is to help the user understand their data.
        
        You can:
        1. Generate summaries of datasets
        2. Analyze numeric and categorical columns
        3. Compute correlations between features
        4. Detect outliers and data quality issues
        5. Provide insights and recommendations
        
        Always make sure to explain your findings in clear, concise terms. When analyzing data, focus on the most important patterns and anomalies.
        When a user asks you to analyze their data, start with generate_data_summary to understand the dataset, then analyze_dataframe for a comprehensive analysis.
        
        For specific questions about columns, use the specialized tools like analyze_numeric_column, analyze_categorical_column, or detect_outliers.
        
        If asked about visualizations, describe what would be helpful to visualize, even though you can't create the visualizations directly.
        """,
    )

    return agent
