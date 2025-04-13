"""
Data Loading Agent

This module defines a data loading agent that can be used with LlamaIndex agent workflows.
It provides functionality for loading data from various sources like CSV, Excel, JSON, etc.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import logging
import time
import pandas as pd
import numpy as np
import json
import sqlite3
import requests
from io import StringIO
import traceback

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM

# Set up logging
logger = logging.getLogger(__name__)

# Maximum file size to load (in bytes) - 100MB default
MAX_FILE_SIZE = 100 * 1024 * 1024


# Helper functions
def _validate_file_path(file_path: str, max_file_size: int = MAX_FILE_SIZE) -> None:
    """
    Validate that a file path is accessible and within size limits.

    Args:
        file_path: Path to the file
        max_file_size: Maximum allowed file size in bytes

    Raises:
        FileNotFoundError: If the file doesn't exist or isn't accessible
        ValueError: If the file is too large
    """
    # Handle URLs
    if file_path.startswith(("http://", "https://")):
        try:
            # Just check if the URL is valid, don't download the file
            response = requests.head(file_path, timeout=5)
            if response.status_code != 200:
                raise FileNotFoundError(f"URL not accessible: {file_path}")

            # Check file size if content-length is provided
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_file_size:
                raise ValueError(
                    f"File too large (>{max_file_size/1024/1024}MB): {file_path}"
                )
        except requests.RequestException as e:
            raise FileNotFoundError(f"Error accessing URL: {file_path}, {str(e)}")
    else:
        # Handle local files
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError(f"File not readable: {file_path}")

        if os.path.getsize(file_path) > max_file_size:
            raise ValueError(
                f"File too large (>{max_file_size/1024/1024}MB): {file_path}"
            )


def _generate_dataframe_metadata(
    df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate schema and metadata for a DataFrame.

    Args:
        df: The DataFrame to analyze

    Returns:
        Tuple containing:
            - schema: The inferred schema
            - metadata: Additional metadata
    """
    if df.empty:
        return {"columns": [], "types": [], "row_count": 0, "is_empty": True}, {
            "is_empty": True
        }

    # Get column names and types
    columns = df.columns.tolist()
    types = []

    for col in columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            types.append("int")
        elif pd.api.types.is_float_dtype(dtype):
            types.append("float")
        elif pd.api.types.is_bool_dtype(dtype):
            types.append("bool")
        elif pd.api.types.is_datetime64_dtype(dtype):
            types.append("datetime")
        else:
            types.append("string")

    # Calculate basic statistics
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}

    if numeric_columns:
        stats["numeric"] = {
            col: {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": (
                    float(df[col].median()) if not pd.isna(df[col].median()) else None
                ),
                "null_count": int(df[col].isna().sum()),
            }
            for col in numeric_columns
        }

    # Check for missing values
    missing_values = {
        col: int(df[col].isna().sum()) for col in columns if df[col].isna().any()
    }

    schema = {
        "columns": columns,
        "types": types,
        "row_count": len(df),
        "column_count": len(columns),
    }

    metadata = {
        "stats": stats,
        "missing_values": missing_values,
        "sample": df.head(5).to_dict(orient="records"),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    return schema, metadata


def _infer_json_structure(data: Any) -> Dict[str, Any]:
    """
    Infer the structure of JSON data.

    Args:
        data: The JSON data to analyze

    Returns:
        Dict containing structure information
    """
    if isinstance(data, dict):
        return {"type": "object", "keys": list(data.keys()), "key_count": len(data)}
    elif isinstance(data, list):
        return {
            "type": "array",
            "length": len(data),
            "sample_type": type(data[0]).__name__ if data else "unknown",
        }
    else:
        return {"type": type(data).__name__}


# Functions that will be exposed as tools
def load_csv(file_path: str, **options) -> Dict[str, Any]:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file
        **options: Additional options for pandas.read_csv
            - delimiter: Column delimiter (default: ',')
            - encoding: File encoding (default: 'utf-8')
            - header: Row to use as header (default: 0)
            - skiprows: Number of rows to skip (default: 0)
            - usecols: List of columns to use

    Returns:
        Dict containing information about the loaded data
    """
    _validate_file_path(file_path)

    # Extract options with defaults
    delimiter = options.get("delimiter", ",")
    encoding = options.get("encoding", "utf-8")
    header = options.get("header", 0)
    skiprows = options.get("skiprows", 0)
    usecols = options.get("usecols", None)

    try:
        # Handle remote files
        if file_path.startswith(("http://", "https://")):
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(
                StringIO(response.text),
                delimiter=delimiter,
                encoding=encoding,
                header=header,
                skiprows=skiprows,
                usecols=usecols,
                on_bad_lines="warn",
            )
        else:
            # Load the CSV file
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                header=header,
                skiprows=skiprows,
                usecols=usecols,
                on_bad_lines="warn",
            )

        # Generate schema and metadata
        schema, metadata = _generate_dataframe_metadata(df)

        return {
            "data": df.to_dict(orient="records"),  # Convert to serializable format
            "schema": schema,
            "metadata": metadata,
            "source_type": "csv",
            "source_path": file_path,
        }
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to load CSV file: {str(e)}")


def load_excel(file_path: str, **options) -> Dict[str, Any]:
    """
    Load data from an Excel file.

    Args:
        file_path: Path to the Excel file
        **options: Additional options for pandas.read_excel
            - sheet_name: Name or index of sheet to load (default: 0)
            - header: Row to use as header (default: 0)
            - skiprows: Number of rows to skip (default: 0)
            - usecols: List of columns to use

    Returns:
        Dict containing information about the loaded data
    """
    _validate_file_path(file_path)

    # Extract options with defaults
    sheet_name = options.get("sheet_name", 0)
    header = options.get("header", 0)
    skiprows = options.get("skiprows", 0)
    usecols = options.get("usecols", None)

    try:
        # Handle remote files
        if file_path.startswith(("http://", "https://")):
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()

            # Save to temporary file
            temp_file = f"/tmp/temp_excel_{int(time.time())}.xlsx"
            with open(temp_file, "wb") as f:
                f.write(response.content)

            try:
                df = pd.read_excel(
                    temp_file,
                    sheet_name=sheet_name,
                    header=header,
                    skiprows=skiprows,
                    usecols=usecols,
                )
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            # Load the Excel file
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header,
                skiprows=skiprows,
                usecols=usecols,
            )

        # Generate schema and metadata
        schema, metadata = _generate_dataframe_metadata(df)

        return {
            "data": df.to_dict(orient="records"),  # Convert to serializable format
            "schema": schema,
            "metadata": metadata,
            "source_type": "excel",
            "source_path": file_path,
        }
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to load Excel file: {str(e)}")


def load_json(file_path: str, **options) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file
        **options: Additional options
            - encoding: File encoding (default: 'utf-8')
            - orient: Expected JSON format for pandas (default: 'records')
            - lines: Whether JSON is in lines format (default: False)

    Returns:
        Dict containing information about the loaded data
    """
    _validate_file_path(file_path)

    # Extract options with defaults
    encoding = options.get("encoding", "utf-8")
    orient = options.get("orient", "records")
    lines = options.get("lines", False)

    try:
        # Handle remote files
        if file_path.startswith(("http://", "https://")):
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()

            if lines:
                # JSON Lines format
                df = pd.read_json(StringIO(response.text), lines=True, orient=orient)
                schema, metadata = _generate_dataframe_metadata(df)
                return {
                    "data": df.to_dict(orient="records"),
                    "schema": schema,
                    "metadata": metadata,
                    "source_type": "json",
                    "source_path": file_path,
                }
            else:
                # Regular JSON
                data = response.json()
        else:
            # Load the JSON file
            with open(file_path, "r", encoding=encoding) as f:
                if lines:
                    # JSON Lines format
                    df = pd.read_json(f, lines=True, orient=orient)
                    schema, metadata = _generate_dataframe_metadata(df)
                    return {
                        "data": df.to_dict(orient="records"),
                        "schema": schema,
                        "metadata": metadata,
                        "source_type": "json",
                        "source_path": file_path,
                    }
                else:
                    # Regular JSON
                    data = json.load(f)

        # Try to convert to DataFrame if it's a list of records
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            schema, metadata = _generate_dataframe_metadata(df)
            return {
                "data": df.to_dict(orient="records"),
                "schema": schema,
                "metadata": metadata,
                "source_type": "json",
                "source_path": file_path,
            }
        else:
            # Return as-is with basic schema
            return {
                "data": data,
                "schema": _infer_json_structure(data),
                "metadata": {"type": "json", "size": len(json.dumps(data))},
                "source_type": "json",
                "source_path": file_path,
            }
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to load JSON file: {str(e)}")


def analyze_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze loaded data and provide insights.

    Args:
        data: Dictionary containing loaded data information

    Returns:
        Dict containing analysis results
    """
    try:
        # Extract data
        data_type = data.get("source_type", "unknown")
        schema = data.get("schema", {})
        metadata = data.get("metadata", {})

        # Generate insights
        insights = []

        # Check for missing values
        missing_values = metadata.get("missing_values", {})
        if missing_values:
            for col, count in missing_values.items():
                insights.append(f"Column '{col}' has {count} missing values")

        # Check for numeric columns stats
        if "stats" in metadata and "numeric" in metadata["stats"]:
            for col, stats in metadata["stats"]["numeric"].items():
                insights.append(
                    f"Column '{col}' has values ranging from {stats.get('min')} to {stats.get('max')}"
                )
                insights.append(f"Column '{col}' has mean value of {stats.get('mean')}")

        # Check for duplicated rows
        if "duplicated_rows" in metadata and metadata["duplicated_rows"] > 0:
            insights.append(
                f"Found {metadata['duplicated_rows']} duplicated rows in the data"
            )

        # Generate suggestions
        suggestions = []

        # Suggest actions based on data quality
        if missing_values:
            suggestions.append(
                "Consider handling missing values by imputation or removal"
            )

        if "duplicated_rows" in metadata and metadata["duplicated_rows"] > 0:
            suggestions.append("Consider removing duplicated rows for analysis")

        return {
            "data_type": data_type,
            "row_count": schema.get("row_count", 0),
            "column_count": schema.get("column_count", 0),
            "insights": insights,
            "suggestions": suggestions,
        }
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": f"Failed to analyze data: {str(e)}"}


def analyze_data_source(
    file_path: str, file_type: str = None, **options
) -> Dict[str, Any]:
    """
    Analyze a data source and load data accordingly.

    Args:
        file_path: Path or URL to the data source
        file_type: Type of file to load (csv, excel, json)
        **options: Additional options for loading

    Returns:
        Dict containing results of the data loading operation
    """
    # Detect file type if not provided
    if not file_type:
        if file_path.lower().endswith(".csv"):
            file_type = "csv"
        elif file_path.lower().endswith((".xls", ".xlsx", ".xlsm")):
            file_type = "excel"
        elif file_path.lower().endswith(".json"):
            file_type = "json"
        else:
            raise ValueError(f"Could not determine file type for {file_path}")

    # Load the data based on file type
    try:
        if file_type == "csv":
            data = load_csv(file_path, **options)
        elif file_type == "excel":
            data = load_excel(file_path, **options)
        elif file_type == "json":
            data = load_json(file_path, **options)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Run analysis on the loaded data
        analysis_results = analyze_data(data)

        return {
            "message": "Data loading and analysis completed",
            "loaded_data": data,
            "analysis": analysis_results,
        }
    except Exception as e:
        logger.error(f"Error analyzing data source: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to analyze data source: {str(e)}")


def analyze_uploaded_file(
    file_content: bytes, file_name: str, **options
) -> Dict[str, Any]:
    """
    Analyze an uploaded file and load data accordingly.

    Args:
        file_content: Content of the uploaded file
        file_name: Name of the uploaded file
        **options: Additional options for loading

    Returns:
        Dict containing results of the data loading operation
    """
    # Determine file type from extension
    file_type = None
    if file_name.lower().endswith(".csv"):
        file_type = "csv"
    elif file_name.lower().endswith((".xls", ".xlsx", ".xlsm")):
        file_type = "excel"
    elif file_name.lower().endswith(".json"):
        file_type = "json"
    else:
        raise ValueError(f"Unsupported file type for {file_name}")

    # Save to temporary file
    temp_file = f"/tmp/upload_{int(time.time())}_{file_name}"
    try:
        with open(temp_file, "wb") as f:
            f.write(file_content)

        # Use analyze_data_source function to process the file
        return analyze_data_source(temp_file, file_type, **options)
    except Exception as e:
        logger.error(f"Error analyzing uploaded file: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to analyze uploaded file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def make_data_loading_agent(llm) -> FunctionAgent:
    tools = [
        # FunctionTool.from_defaults(fn=load_csv),
        # FunctionTool.from_defaults(fn=load_excel),
        # FunctionTool.from_defaults(fn=load_json),
        # FunctionTool.from_defaults(fn=analyze_data),
        # FunctionTool.from_defaults(fn=analyze_data_source),
        # FunctionTool.from_defaults(fn=analyze_uploaded_file),
    ]

    agent = FunctionAgent(
        # tools=tools,
        llm=llm,
        name="DataLoadingAgent",
        description="A data loading agent that can load and analyze data from various sources.",
        system_prompt="""
        You are a data loading assistant that can help load and analyze data from various sources. Your goal is to help the user.
        You can load data from files (CSV, Excel, JSON), and then analyze the data to provide insights.
        
        When loading data, consider:
        1. The type of file format
        2. The path or URL of the data source
        3. Any specific options needed (like encoding, delimiter, etc.)
        
        After loading data, analyze it to find patterns, statistics, and potential issues.
        """,
        can_handoff_to=["ExplorationAgent"],
    )

    return agent
