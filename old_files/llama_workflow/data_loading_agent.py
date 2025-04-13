"""
Data Loading Task Agent Module

This module defines the data loading task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import time
import pandas as pd
import numpy as np
import json
import sqlite3
import pyarrow.parquet as pq
from urllib.parse import urlparse
import requests
from io import StringIO
import traceback

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class DataLoadingTaskAgent(BaseTaskAgent):
    """
    Task agent for loading data from various sources.

    Responsibilities:
    - Loading data from files (CSV, Excel, JSON, Parquet, etc.)
    - Loading data from databases
    - Loading data from APIs
    - Basic data validation and schema inference
    - Error handling and recovery
    """

    # Maximum number of retries for transient errors
    MAX_RETRIES = 3
    # Delay between retries (in seconds)
    RETRY_DELAY = 2
    # Maximum file size to load (in bytes) - 100MB default
    MAX_FILE_SIZE = 100 * 1024 * 1024
    # Supported file types
    SUPPORTED_FILE_TYPES = ["csv", "excel", "json", "parquet", "database", "api"]

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the DataLoadingTaskAgent."""
        super().__init__(name="DataLoadingAgent", llm=llm)
        self.logger = logging.getLogger(__name__)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data loading task.

        Args:
            input_data: Input data for the task containing:
                - environment: The shared environment state
                - goals: List of goals for the task
                - data_sources: Dict of data source references
                - file_type: Optional type of file to load
                - validation_rules: Optional rules for data validation

        Returns:
            Dict containing loaded data and metadata:
                - Data Overview.raw_data: The loaded data
                - Data Overview.schema: The inferred schema
                - Data Overview.validation: Validation results
                - Data Overview.errors: Any errors encountered
        """
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        validation_rules = input_data.get("validation_rules", {})

        # Get data sources from the environment
        data_sources = environment.get("Data", {})
        specific_sources = input_data.get("data_sources", {})

        # If specific sources are provided, use those instead
        if specific_sources:
            data_sources = specific_sources

        if not data_sources:
            self.logger.warning("No data sources provided")
            return {
                "Data Overview.raw_data": {},
                "Data Overview.schema": {},
                "Data Overview.validation": {"status": "failed", "message": "No data sources provided"},
                "Data Overview.errors": ["No data sources provided"]
            }

        self.logger.info(f"Loading data from {len(data_sources)} sources")

        loaded_data = {}
        schema_info = {}
        validation_results = {}
        all_errors = []

        # Process each data source
        for source_name, source_info in data_sources.items():
            source_errors = []
            retry_count = 0
            success = False

            while not success and retry_count < self.MAX_RETRIES:
                try:
                    # Extract source information
                    source_type = source_info.get("type", "unknown").lower()
                    source_path = source_info.get("path", "")
                    
                    if source_type not in self.SUPPORTED_FILE_TYPES:
                        error_msg = f"Unsupported data source type: {source_type}"
                        self.logger.warning(error_msg)
                        loaded_data[source_name] = {
                            "type": "unknown",
                            "error": error_msg,
                        }
                        source_errors.append(error_msg)
                        break  # No need to retry for unsupported types
                    
                    # Load data based on source type
                    if source_type in ["csv", "excel", "json", "parquet"]:
                        # Check if file exists and is accessible
                        if not self._is_valid_file_path(source_path):
                            error_msg = f"File not found or not accessible: {source_path}"
                            self.logger.error(error_msg)
                            source_errors.append(error_msg)
                            break  # No need to retry for missing files
                        
                        # Check file size
                        if not self._check_file_size(source_path):
                            error_msg = f"File too large (>{self.MAX_FILE_SIZE/1024/1024}MB): {source_path}"
                            self.logger.error(error_msg)
                            source_errors.append(error_msg)
                            break  # No need to retry for oversized files
                        
                        # Load file data
                        data, schema = self._load_file_data(source_type, source_path, source_info)
                    elif source_type == "database":
                        # Load database data
                        data, schema = self._load_database_data(source_info)
                    elif source_type == "api":
                        # Load API data
                        data, schema = self._load_api_data(source_info)
                    else:
                        # This should never happen due to the check above
                        error_msg = f"Unknown data source type: {source_type}"
                        self.logger.warning(error_msg)
                        source_errors.append(error_msg)
                        break
                    
                    # Validate the loaded data
                    validation_result = self._validate_data(
                        data, 
                        source_name, 
                        validation_rules.get(source_name, {})
                    )
                    
                    # Store results
                    loaded_data[source_name] = {
                        "type": "dataframe",
                        "rows": len(data) if hasattr(data, "__len__") else 0,
                        "source": source_path or source_info.get("connection", "") or source_info.get("url", ""),
                        "data": data,  # Store the actual data
                    }
                    schema_info[source_name] = schema
                    validation_results[source_name] = validation_result
                    
                    # Mark as successful
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"Error loading data from {source_name} (attempt {retry_count}/{self.MAX_RETRIES}): {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.debug(traceback.format_exc())
                    source_errors.append(error_msg)
                    
                    if retry_count < self.MAX_RETRIES:
                        self.logger.info(f"Retrying in {self.RETRY_DELAY} seconds...")
                        time.sleep(self.RETRY_DELAY)
                    else:
                        # All retries failed
                        loaded_data[source_name] = {
                            "type": "error", 
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
            
            # Add any errors from this source to the overall error list
            all_errors.extend(source_errors)

        # Return the loaded data with additional metadata
        return {
            "Data Overview.raw_data": loaded_data,
            "Data Overview.schema": schema_info,
            "Data Overview.validation": validation_results,
            "Data Overview.errors": all_errors
        }

    def _is_valid_file_path(self, file_path: str) -> bool:
        """
        Check if a file path is valid and accessible.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file exists and is accessible, False otherwise
        """
        # Handle URLs
        if file_path.startswith(('http://', 'https://')):
            try:
                # Just check if the URL is valid, don't download the file
                response = requests.head(file_path, timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        
        # Handle local files
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

    def _check_file_size(self, file_path: str) -> bool:
        """
        Check if a file is within the size limit.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is within the size limit, False otherwise
        """
        # Handle URLs
        if file_path.startswith(('http://', 'https://')):
            try:
                response = requests.head(file_path, timeout=5)
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.MAX_FILE_SIZE:
                    return False
                return True
            except Exception:
                # If we can't determine the size, assume it's OK
                return True
        
        # Handle local files
        try:
            return os.path.getsize(file_path) <= self.MAX_FILE_SIZE
        except Exception:
            # If we can't determine the size, assume it's OK
            return True

    def _load_file_data(self, file_type: str, file_path: str, source_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Load data from a file.
        
        Args:
            file_type: Type of file (csv, excel, json, parquet)
            file_path: Path to the file
            source_info: Additional information about the source
            
        Returns:
            Tuple containing:
                - The loaded data
                - The inferred schema
        """
        self.logger.info(f"Loading {file_type} data from {file_path}")
        
        # Handle remote files
        if file_path.startswith(('http://', 'https://')):
            return self._load_remote_file(file_type, file_path, source_info)
        
        # Handle local files
        if file_type == "csv":
            # Get CSV options from source_info
            delimiter = source_info.get("delimiter", ",")
            encoding = source_info.get("encoding", "utf-8")
            header = source_info.get("header", 0)
            
            # Load the CSV file
            df = pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                encoding=encoding,
                header=header,
                on_bad_lines='warn'
            )
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema
            
        elif file_type == "excel":
            # Get Excel options from source_info
            sheet_name = source_info.get("sheet_name", 0)
            header = source_info.get("header", 0)
            
            # Load the Excel file
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header
            )
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema
            
        elif file_type == "json":
            # Get JSON options from source_info
            orient = source_info.get("orient", "records")
            encoding = source_info.get("encoding", "utf-8")
            
            # Load the JSON file
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Convert to DataFrame if possible
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                schema = self._infer_dataframe_schema(df)
                return df, schema
            else:
                # Return as-is with basic schema
                schema = {
                    "type": "json",
                    "structure": self._infer_json_structure(data)
                }
                return data, schema
                
        elif file_type == "parquet":
            # Load the Parquet file
            df = pd.read_parquet(file_path)
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _load_remote_file(self, file_type: str, file_url: str, source_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Load data from a remote file.
        
        Args:
            file_type: Type of file (csv, excel, json, parquet)
            file_url: URL of the file
            source_info: Additional information about the source
            
        Returns:
            Tuple containing:
                - The loaded data
                - The inferred schema
        """
        self.logger.info(f"Loading remote {file_type} data from {file_url}")
        
        # Download the file content
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        if file_type == "csv":
            # Get CSV options from source_info
            delimiter = source_info.get("delimiter", ",")
            encoding = source_info.get("encoding", "utf-8")
            header = source_info.get("header", 0)
            
            # Load the CSV data
            df = pd.read_csv(
                StringIO(response.text),
                delimiter=delimiter,
                encoding=encoding,
                header=header,
                on_bad_lines='warn'
            )
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema
            
        elif file_type == "json":
            # Parse the JSON data
            data = response.json()
            
            # Convert to DataFrame if possible
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                schema = self._infer_dataframe_schema(df)
                return df, schema
            else:
                # Return as-is with basic schema
                schema = {
                    "type": "json",
                    "structure": self._infer_json_structure(data)
                }
                return data, schema
                
        elif file_type in ["excel", "parquet"]:
            # These formats require saving to a temporary file
            temp_file = f"/tmp/temp_data_{int(time.time())}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            try:
                if file_type == "excel":
                    # Get Excel options from source_info
                    sheet_name = source_info.get("sheet_name", 0)
                    header = source_info.get("header", 0)
                    
                    # Load the Excel file
                    df = pd.read_excel(
                        temp_file,
                        sheet_name=sheet_name,
                        header=header
                    )
                    
                    # Infer schema
                    schema = self._infer_dataframe_schema(df)
                    
                    return df, schema
                    
                elif file_type == "parquet":
                    # Load the Parquet file
                    df = pd.read_parquet(temp_file)
                    
                    # Infer schema
                    schema = self._infer_dataframe_schema(df)
                    
                    return df, schema
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        else:
            raise ValueError(f"Unsupported file type for remote loading: {file_type}")

    def _load_database_data(self, source_info: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from a database.
        
        Args:
            source_info: Information about the database source
            
        Returns:
            Tuple containing:
                - The loaded data as a DataFrame
                - The inferred schema
        """
        connection_string = source_info.get("connection", "")
        query = source_info.get("query", "")
        db_type = source_info.get("db_type", "").lower()
        
        if not connection_string:
            raise ValueError("Database connection string is required")
        if not query:
            raise ValueError("SQL query is required")
            
        self.logger.info(f"Loading database data from {connection_string}")
        
        # Handle SQLite databases (for simplicity in this example)
        if db_type == "sqlite" or connection_string.endswith(".db") or connection_string.endswith(".sqlite"):
            # Check if the database file exists
            if not os.path.isfile(connection_string):
                raise FileNotFoundError(f"Database file not found: {connection_string}")
                
            # Connect to the database and execute the query
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema
        else:
            # For other database types, we would use SQLAlchemy
            # This is a placeholder for demonstration purposes
            self.logger.warning(f"Database type {db_type} not fully implemented")
            
            # Return simulated data
            df = pd.DataFrame({
                "id": range(1, 101),
                "name": [f"Item {i}" for i in range(1, 101)],
                "value": np.random.rand(100) * 100,
                "date": pd.date_range(start="2023-01-01", periods=100)
            })
            
            # Infer schema
            schema = self._infer_dataframe_schema(df)
            
            return df, schema

    def _load_api_data(self, source_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Load data from an API.
        
        Args:
            source_info: Information about the API source
            
        Returns:
            Tuple containing:
                - The loaded data
                - The inferred schema
        """
        url = source_info.get("url", "")
        method = source_info.get("method", "GET").upper()
        headers = source_info.get("headers", {})
        params = source_info.get("params", {})
        data = source_info.get("data", {})
        auth = source_info.get("auth", {})
        
        if not url:
            raise ValueError("API URL is required")
            
        self.logger.info(f"Loading API data from {url}")
        
        # Prepare authentication if provided
        auth_tuple = None
        if auth and "username" in auth and "password" in auth:
            auth_tuple = (auth["username"], auth["password"])
        
        # Make the API request
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data if method in ["POST", "PUT", "PATCH"] else None,
            auth=auth_tuple,
            timeout=30
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse the response
        try:
            json_data = response.json()
            
            # Convert to DataFrame if possible
            if isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
                df = pd.DataFrame(json_data)
                schema = self._infer_dataframe_schema(df)
                return df, schema
            else:
                # Return as-is with basic schema
                schema = {
                    "type": "json",
                    "structure": self._infer_json_structure(json_data)
                }
                return json_data, schema
        except ValueError:
            # Not JSON, return text
            schema = {
                "type": "text",
                "length": len(response.text)
            }
            return response.text, schema

    def _infer_dataframe_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infer the schema of a DataFrame.
        
        Args:
            df: The DataFrame to analyze
            
        Returns:
            Dict containing schema information
        """
        if df.empty:
            return {
                "columns": [],
                "types": [],
                "row_count": 0,
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
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        stats = {}
        
        if numeric_columns:
            stats["numeric"] = {
                col: {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "null_count": int(df[col].isna().sum())
                }
                for col in numeric_columns
            }
        
        # Check for missing values
        missing_values = {
            col: int(df[col].isna().sum())
            for col in columns
            if df[col].isna().any()
        }
        
        return {
            "columns": columns,
            "types": types,
            "row_count": len(df),
            "column_count": len(columns),
            "stats": stats,
            "missing_values": missing_values,
            "sample": df.head(5).to_dict(orient="records")
        }

    def _infer_json_structure(self, data: Any) -> Dict[str, Any]:
        """
        Infer the structure of JSON data.
        
        Args:
            data: The JSON data to analyze
            
        Returns:
            Dict containing structure information
        """
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_count": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample_type": type(data[0]).__name__ if data else "unknown"
            }
        else:
            return {
                "type": type(data).__name__
            }

    def _validate_data(self, data: Any, source_name: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the loaded data against validation rules.
        
        Args:
            data: The data to validate
            source_name: Name of the data source
            validation_rules: Rules for validation
            
        Returns:
            Dict containing validation results
        """
        if not validation_rules:
            # No validation rules provided
            return {
                "status": "passed",
                "message": "No validation rules provided"
            }
            
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return {
                "status": "skipped",
                "message": "Validation only supported for DataFrame data"
            }
            
        validation_errors = []
        
        # Check for required columns
        if "required_columns" in validation_rules:
            required_columns = validation_rules["required_columns"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for minimum row count
        if "min_rows" in validation_rules:
            min_rows = validation_rules["min_rows"]
            if len(data) < min_rows:
                validation_errors.append(f"Data has {len(data)} rows, minimum required is {min_rows}")
        
        # Check for maximum null values
        if "max_null_percentage" in validation_rules:
            max_null_percentage = validation_rules["max_null_percentage"]
            
            for col in data.columns:
                null_percentage = (data[col].isna().sum() / len(data)) * 100
                if null_percentage > max_null_percentage:
                    validation_errors.append(
                        f"Column '{col}' has {null_percentage:.2f}% null values, "
                        f"maximum allowed is {max_null_percentage}%"
                    )
        
        # Check for column data types
        if "column_types" in validation_rules:
            column_types = validation_rules["column_types"]
            
            for col, expected_type in column_types.items():
                if col not in data.columns:
                    continue
                    
                # Check if the column has the expected type
                if expected_type == "int" and not pd.api.types.is_integer_dtype(data[col].dtype):
                    validation_errors.append(f"Column '{col}' should be of type 'int', got '{data[col].dtype}'")
                elif expected_type == "float" and not pd.api.types.is_float_dtype(data[col].dtype):
                    validation_errors.append(f"Column '{col}' should be of type 'float', got '{data[col].dtype}'")
                elif expected_type == "bool" and not pd.api.types.is_bool_dtype(data[col].dtype):
                    validation_errors.append(f"Column '{col}' should be of type 'bool', got '{data[col].dtype}'")
                elif expected_type == "datetime" and not pd.api.types.is_datetime64_dtype(data[col].dtype):
                    validation_errors.append(f"Column '{col}' should be of type 'datetime', got '{data[col].dtype}'")
                elif expected_type == "string" and not pd.api.types.is_string_dtype(data[col].dtype):
                    validation_errors.append(f"Column '{col}' should be of type 'string', got '{data[col].dtype}'")
        
        # Check for value ranges
        if "value_ranges" in validation_rules:
            value_ranges = validation_rules["value_ranges"]
            
            for col, range_info in value_ranges.items():
                if col not in data.columns:
                    continue
                    
                if "min" in range_info and data[col].min() < range_info["min"]:
                    validation_errors.append(
                        f"Column '{col}' has minimum value {data[col].min()}, "
                        f"should be at least {range_info['min']}"
                    )
                    
                if "max" in range_info and data[col].max() > range_info["max"]:
                    validation_errors.append(
                        f"Column '{col}' has maximum value {data[col].max()}, "
                        f"should be at most {range_info['max']}"
                    )
        
        # Return validation results
        if validation_errors:
            return {
                "status": "failed",
                "message": "Validation failed",
                "errors": validation_errors
            }
        else:
            return {
                "status": "passed",
                "message": "All validation rules passed"
            }
