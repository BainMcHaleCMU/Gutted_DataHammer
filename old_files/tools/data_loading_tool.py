"""
Data Loading Tool

This module defines the DataLoadingTool class for loading data from various sources.
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


class DataLoadingTool:
    """
    Tool for loading data from various sources.
    
    This tool is primarily used by the DataLoadingAgent to load
    data from files, databases, APIs, and other sources.
    """
    
    # Maximum file size to load (in bytes) - 100MB default
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self):
        """Initialize the Data Loading Tool."""
        self.logger = logging.getLogger(__name__)
    
    def load_csv(self, file_path: str, **options) -> Dict[str, Any]:
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
            Dict containing:
                - data: The loaded DataFrame
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        self._validate_file_path(file_path)
        
        # Extract options with defaults
        delimiter = options.get("delimiter", ",")
        encoding = options.get("encoding", "utf-8")
        header = options.get("header", 0)
        skiprows = options.get("skiprows", 0)
        usecols = options.get("usecols", None)
        
        try:
            # Handle remote files
            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(
                    StringIO(response.text),
                    delimiter=delimiter,
                    encoding=encoding,
                    header=header,
                    skiprows=skiprows,
                    usecols=usecols,
                    on_bad_lines='warn'
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
                    on_bad_lines='warn'
                )
            
            # Generate schema and metadata
            schema, metadata = self._generate_dataframe_metadata(df)
            
            return {
                "data": df,
                "schema": schema,
                "metadata": metadata,
                "source_type": "csv",
                "source_path": file_path
            }
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load CSV file: {str(e)}")
    
    def load_excel(self, file_path: str, **options) -> Dict[str, Any]:
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
            Dict containing:
                - data: The loaded DataFrame
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        self._validate_file_path(file_path)
        
        # Extract options with defaults
        sheet_name = options.get("sheet_name", 0)
        header = options.get("header", 0)
        skiprows = options.get("skiprows", 0)
        usecols = options.get("usecols", None)
        
        try:
            # Handle remote files
            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file
                temp_file = f"/tmp/temp_excel_{int(time.time())}.xlsx"
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                try:
                    df = pd.read_excel(
                        temp_file,
                        sheet_name=sheet_name,
                        header=header,
                        skiprows=skiprows,
                        usecols=usecols
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
                    usecols=usecols
                )
            
            # Generate schema and metadata
            schema, metadata = self._generate_dataframe_metadata(df)
            
            return {
                "data": df,
                "schema": schema,
                "metadata": metadata,
                "source_type": "excel",
                "source_path": file_path
            }
        except Exception as e:
            self.logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load Excel file: {str(e)}")
    
    def load_json(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            **options: Additional options
                - encoding: File encoding (default: 'utf-8')
                - orient: Expected JSON format for pandas (default: 'records')
                - lines: Whether JSON is in lines format (default: False)
            
        Returns:
            Dict containing:
                - data: The loaded data (DataFrame or dict/list)
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        self._validate_file_path(file_path)
        
        # Extract options with defaults
        encoding = options.get("encoding", "utf-8")
        orient = options.get("orient", "records")
        lines = options.get("lines", False)
        
        try:
            # Handle remote files
            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path, timeout=30)
                response.raise_for_status()
                
                if lines:
                    # JSON Lines format
                    df = pd.read_json(StringIO(response.text), lines=True, orient=orient)
                    schema, metadata = self._generate_dataframe_metadata(df)
                    return {
                        "data": df,
                        "schema": schema,
                        "metadata": metadata,
                        "source_type": "json",
                        "source_path": file_path
                    }
                else:
                    # Regular JSON
                    data = response.json()
            else:
                # Load the JSON file
                with open(file_path, 'r', encoding=encoding) as f:
                    if lines:
                        # JSON Lines format
                        df = pd.read_json(f, lines=True, orient=orient)
                        schema, metadata = self._generate_dataframe_metadata(df)
                        return {
                            "data": df,
                            "schema": schema,
                            "metadata": metadata,
                            "source_type": "json",
                            "source_path": file_path
                        }
                    else:
                        # Regular JSON
                        data = json.load(f)
            
            # Try to convert to DataFrame if it's a list of records
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                schema, metadata = self._generate_dataframe_metadata(df)
                return {
                    "data": df,
                    "schema": schema,
                    "metadata": metadata,
                    "source_type": "json",
                    "source_path": file_path
                }
            else:
                # Return as-is with basic schema
                return {
                    "data": data,
                    "schema": self._infer_json_structure(data),
                    "metadata": {
                        "type": "json",
                        "size": len(json.dumps(data))
                    },
                    "source_type": "json",
                    "source_path": file_path
                }
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load JSON file: {str(e)}")
    
    def load_parquet(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Load data from a Parquet file.
        
        Args:
            file_path: Path to the Parquet file
            **options: Additional options for pandas.read_parquet
                - columns: List of columns to load
            
        Returns:
            Dict containing:
                - data: The loaded DataFrame
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        self._validate_file_path(file_path)
        
        # Extract options with defaults
        columns = options.get("columns", None)
        
        try:
            # Handle remote files
            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file
                temp_file = f"/tmp/temp_parquet_{int(time.time())}.parquet"
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                try:
                    df = pd.read_parquet(temp_file, columns=columns)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # Load the Parquet file
                df = pd.read_parquet(file_path, columns=columns)
            
            # Generate schema and metadata
            schema, metadata = self._generate_dataframe_metadata(df)
            
            return {
                "data": df,
                "schema": schema,
                "metadata": metadata,
                "source_type": "parquet",
                "source_path": file_path
            }
        except Exception as e:
            self.logger.error(f"Error loading Parquet file {file_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load Parquet file: {str(e)}")
    
    def load_database(self, connection_string: str, query: str, **options) -> Dict[str, Any]:
        """
        Load data from a database.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            **options: Additional options
                - db_type: Type of database (sqlite, mysql, postgres, etc.)
                - params: Parameters for the query
            
        Returns:
            Dict containing:
                - data: The loaded DataFrame
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        if not connection_string:
            raise ValueError("Database connection string is required")
        if not query:
            raise ValueError("SQL query is required")
        
        # Extract options with defaults
        db_type = options.get("db_type", "").lower()
        params = options.get("params", {})
        
        try:
            # Handle SQLite databases
            if db_type == "sqlite" or connection_string.endswith(".db") or connection_string.endswith(".sqlite"):
                # Check if the database file exists
                if not os.path.isfile(connection_string):
                    raise FileNotFoundError(f"Database file not found: {connection_string}")
                
                # Connect to the database and execute the query
                conn = sqlite3.connect(connection_string)
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
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
            
            # Generate schema and metadata
            schema, metadata = self._generate_dataframe_metadata(df)
            
            return {
                "data": df,
                "schema": schema,
                "metadata": metadata,
                "source_type": "database",
                "source_connection": connection_string,
                "query": query
            }
        except Exception as e:
            self.logger.error(f"Error loading data from database {connection_string}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load database data: {str(e)}")
    
    def load_api(self, url: str, **options) -> Dict[str, Any]:
        """
        Load data from an API.
        
        Args:
            url: API endpoint URL
            **options: Additional options
                - method: HTTP method (default: 'GET')
                - headers: HTTP headers
                - params: Query parameters
                - data: Request body for POST/PUT
                - auth: Authentication (username, password)
                - timeout: Request timeout in seconds (default: 30)
            
        Returns:
            Dict containing:
                - data: The loaded data (DataFrame or dict/list)
                - schema: Initial data schema
                - metadata: Additional information about the data
        """
        if not url:
            raise ValueError("API URL is required")
        
        # Extract options with defaults
        method = options.get("method", "GET").upper()
        headers = options.get("headers", {})
        params = options.get("params", {})
        data = options.get("data", {})
        auth = options.get("auth", {})
        timeout = options.get("timeout", 30)
        
        try:
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
                timeout=timeout
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            try:
                json_data = response.json()
                
                # Try to convert to DataFrame if it's a list of records
                if isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
                    df = pd.DataFrame(json_data)
                    schema, metadata = self._generate_dataframe_metadata(df)
                    return {
                        "data": df,
                        "schema": schema,
                        "metadata": metadata,
                        "source_type": "api",
                        "source_url": url
                    }
                else:
                    # Return as-is with basic schema
                    return {
                        "data": json_data,
                        "schema": self._infer_json_structure(json_data),
                        "metadata": {
                            "type": "json",
                            "size": len(json.dumps(json_data))
                        },
                        "source_type": "api",
                        "source_url": url
                    }
            except ValueError:
                # Not JSON, return text
                return {
                    "data": response.text,
                    "schema": {"type": "text"},
                    "metadata": {
                        "type": "text",
                        "size": len(response.text),
                        "content_type": response.headers.get("content-type", "")
                    },
                    "source_type": "api",
                    "source_url": url
                }
        except Exception as e:
            self.logger.error(f"Error loading data from API {url}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"Failed to load API data: {str(e)}")
    
    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate that a file path is accessible and within size limits.
        
        Args:
            file_path: Path to the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist or isn't accessible
            ValueError: If the file is too large
        """
        # Handle URLs
        if file_path.startswith(('http://', 'https://')):
            try:
                # Just check if the URL is valid, don't download the file
                response = requests.head(file_path, timeout=5)
                if response.status_code != 200:
                    raise FileNotFoundError(f"URL not accessible: {file_path}")
                
                # Check file size if content-length is provided
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.MAX_FILE_SIZE:
                    raise ValueError(f"File too large (>{self.MAX_FILE_SIZE/1024/1024}MB): {file_path}")
            except requests.RequestException as e:
                raise FileNotFoundError(f"Error accessing URL: {file_path}, {str(e)}")
        else:
            # Handle local files
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not os.access(file_path, os.R_OK):
                raise FileNotFoundError(f"File not readable: {file_path}")
            
            if os.path.getsize(file_path) > self.MAX_FILE_SIZE:
                raise ValueError(f"File too large (>{self.MAX_FILE_SIZE/1024/1024}MB): {file_path}")
    
    def _generate_dataframe_metadata(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            return {
                "columns": [],
                "types": [],
                "row_count": 0,
                "is_empty": True
            }, {
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
        
        schema = {
            "columns": columns,
            "types": types,
            "row_count": len(df),
            "column_count": len(columns)
        }
        
        metadata = {
            "stats": stats,
            "missing_values": missing_values,
            "sample": df.head(5).to_dict(orient="records"),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicated_rows": int(df.duplicated().sum())
        }
        
        return schema, metadata
    
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