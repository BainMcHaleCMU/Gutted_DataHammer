"""
Exploration Task Agent Module

This module defines the exploration task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import json

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class ExplorationTaskAgent(BaseTaskAgent):
    """
    Task agent for exploring and understanding data.

    Responsibilities:
    - Data profiling
    - Statistical analysis
    - Feature discovery
    - Correlation analysis
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the ExplorationTaskAgent."""
        super().__init__(name="ExplorationAgent", llm=llm)

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the input data for the task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(input_data, dict):
            return False, f"Input data must be a dictionary, got {type(input_data)}"
            
        environment = input_data.get("environment")
        if not environment:
            return False, "Missing 'environment' in input data"
            
        data_reference = input_data.get("data_reference")
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        
        if not data_reference and not raw_data:
            return False, "No data reference or raw data provided"
            
        return True, ""

    def get_dataset(self, input_data: Dict[str, Any], dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Attempt to retrieve a dataset from various possible sources.
        
        Args:
            input_data: Input data for the task
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            DataFrame if found, None otherwise
        """
        # Try to get from data_reference (could be a DataFrame or dict of DataFrames)
        data_reference = input_data.get("data_reference")
        if data_reference is not None:
            if isinstance(data_reference, dict) and dataset_name in data_reference:
                df = data_reference[dataset_name]
                if isinstance(df, pd.DataFrame):
                    return df
            elif isinstance(data_reference, pd.DataFrame) and dataset_name == "default":
                return data_reference
                
        # Try to get from environment
        environment = input_data.get("environment", {})
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        
        if dataset_name in raw_data:
            dataset_info = raw_data[dataset_name]
            if isinstance(dataset_info, dict) and "data" in dataset_info:
                data = dataset_info["data"]
                if isinstance(data, pd.DataFrame):
                    return data
                elif isinstance(data, dict):
                    try:
                        return pd.DataFrame.from_dict(data)
                    except Exception as e:
                        self.logger.error(f"Failed to convert dict to DataFrame: {str(e)}")
                elif isinstance(data, list):
                    try:
                        return pd.DataFrame(data)
                    except Exception as e:
                        self.logger.error(f"Failed to convert list to DataFrame: {str(e)}")
        
        return None

    def analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a numeric column and return statistics.
        
        Args:
            series: Pandas Series containing numeric data
            
        Returns:
            Dict of statistics
        """
        stats = {}
        try:
            # Handle potential non-numeric values
            numeric_series = pd.to_numeric(series, errors='coerce')
            missing_count = numeric_series.isna().sum()
            
            # Only compute statistics if we have valid data
            if len(numeric_series.dropna()) > 0:
                stats = {
                    "min": float(numeric_series.min()) if not pd.isna(numeric_series.min()) else None,
                    "max": float(numeric_series.max()) if not pd.isna(numeric_series.max()) else None,
                    "mean": float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else None,
                    "median": float(numeric_series.median()) if not pd.isna(numeric_series.median()) else None,
                    "std": float(numeric_series.std()) if not pd.isna(numeric_series.std()) else None,
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100),
                    "quartiles": {
                        "25%": float(numeric_series.quantile(0.25)) if not pd.isna(numeric_series.quantile(0.25)) else None,
                        "50%": float(numeric_series.quantile(0.5)) if not pd.isna(numeric_series.quantile(0.5)) else None,
                        "75%": float(numeric_series.quantile(0.75)) if not pd.isna(numeric_series.quantile(0.75)) else None
                    }
                }
                
                # Detect potential outliers using IQR method
                q1 = numeric_series.quantile(0.25)
                q3 = numeric_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
                
                stats["outliers"] = {
                    "count": int(len(outliers)),
                    "percentage": float(len(outliers) / len(numeric_series.dropna()) * 100) if len(numeric_series.dropna()) > 0 else 0.0,
                    "lower_bound": float(lower_bound) if not pd.isna(lower_bound) else None,
                    "upper_bound": float(upper_bound) if not pd.isna(upper_bound) else None
                }
            else:
                stats = {
                    "error": "No valid numeric data",
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100)
                }
        except Exception as e:
            self.logger.error(f"Error analyzing numeric column: {str(e)}")
            stats = {"error": str(e)}
            
        return stats

    def analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a categorical column and return statistics.
        
        Args:
            series: Pandas Series containing categorical data
            
        Returns:
            Dict of statistics
        """
        stats = {}
        try:
            missing_count = series.isna().sum()
            valid_series = series.dropna()
            
            if len(valid_series) > 0:
                # Get value counts
                value_counts = valid_series.value_counts()
                
                # Get most common values (up to 10)
                most_common = value_counts.head(10).index.tolist()
                
                # Convert to native Python types for JSON serialization
                most_common = [str(val) for val in most_common]
                
                stats = {
                    "unique_values": int(len(value_counts)),
                    "most_common": most_common,
                    "most_common_counts": value_counts.head(10).tolist(),
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100)
                }
            else:
                stats = {
                    "error": "No valid categorical data",
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100)
                }
        except Exception as e:
            self.logger.error(f"Error analyzing categorical column: {str(e)}")
            stats = {"error": str(e)}
            
        return stats

    def analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a datetime column and return statistics.
        
        Args:
            series: Pandas Series containing datetime data
            
        Returns:
            Dict of statistics
        """
        stats = {}
        try:
            # Try to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(series):
                datetime_series = pd.to_datetime(series, errors='coerce')
            else:
                datetime_series = series
                
            missing_count = datetime_series.isna().sum()
            valid_series = datetime_series.dropna()
            
            if len(valid_series) > 0:
                min_date = valid_series.min()
                max_date = valid_series.max()
                
                stats = {
                    "min_date": min_date.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(min_date) else None,
                    "max_date": max_date.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(max_date) else None,
                    "range_days": (max_date - min_date).days if not pd.isna(min_date) and not pd.isna(max_date) else None,
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100)
                }
            else:
                stats = {
                    "error": "No valid datetime data",
                    "missing": int(missing_count),
                    "missing_percentage": float(missing_count / len(series) * 100)
                }
        except Exception as e:
            self.logger.error(f"Error analyzing datetime column: {str(e)}")
            stats = {"error": str(e)}
            
        return stats

    def detect_column_type(self, series: pd.Series) -> str:
        """
        Detect the type of a column based on its content.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            String indicating the column type: 'numeric', 'datetime', 'categorical', or 'unknown'
        """
        # Check if already a datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
            
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
            
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().sum() / len(series) > 0.7:  # If more than 70% can be converted to numeric
            return "numeric"
            
        # Try to convert to datetime
        datetime_series = pd.to_datetime(series, errors='coerce')
        if datetime_series.notna().sum() / len(series) > 0.7:  # If more than 70% can be converted to datetime
            return "datetime"
            
        # Default to categorical
        return "categorical"

    def compute_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute correlations between numeric columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict of correlation matrices
        """
        correlations = {}
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                # Compute Pearson correlation
                pearson_corr = numeric_df.corr(method='pearson')
                
                # Convert to dictionary format
                pearson_dict = {}
                for col1 in pearson_corr.columns:
                    pearson_dict[col1] = {}
                    for col2 in pearson_corr.columns:
                        if col1 != col2:  # Skip self-correlations
                            value = pearson_corr.loc[col1, col2]
                            if not pd.isna(value):
                                pearson_dict[col1][col2] = float(value)
                
                correlations["pearson"] = pearson_dict
                
                # Find highly correlated features
                high_correlations = []
                for i, col1 in enumerate(pearson_corr.columns):
                    for j, col2 in enumerate(pearson_corr.columns):
                        if i < j:  # Only look at upper triangle
                            corr_value = pearson_corr.loc[col1, col2]
                            if abs(corr_value) > 0.7:  # Threshold for high correlation
                                high_correlations.append({
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": float(corr_value)
                                })
                
                correlations["high_correlations"] = high_correlations
        except Exception as e:
            self.logger.error(f"Error computing correlations: {str(e)}")
            correlations["error"] = str(e)
            
        return correlations

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data exploration task.

        Args:
            input_data: Input data for the task

        Returns:
            Dict containing exploration results
        """
        self.logger.info("Starting data exploration")
        
        # Validate input
        is_valid, error_message = self.validate_input(input_data)
        if not is_valid:
            self.logger.error(f"Invalid input: {error_message}")
            return {
                "Data Overview.summary": {"error": error_message},
                "Data Overview.statistics": {}
            }
        
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])

        # Get raw data from the environment
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        schema = data_overview.get("schema", {})

        # Initialize results
        summary = {}
        statistics = {}

        # If raw_data is empty, try to use data_reference directly
        if not raw_data and "data_reference" in input_data:
            data_reference = input_data["data_reference"]
            if isinstance(data_reference, pd.DataFrame):
                raw_data = {"default": {"data": data_reference}}
            elif isinstance(data_reference, dict):
                raw_data = {k: {"data": v} for k, v in data_reference.items() if isinstance(v, pd.DataFrame)}

        # Process each dataset
        dataset_names = list(raw_data.keys())
        if not dataset_names:
            dataset_names = ["default"]
            
        for dataset_name in dataset_names:
            try:
                self.logger.info(f"Processing dataset: {dataset_name}")
                
                # Get the dataset
                df = self.get_dataset(input_data, dataset_name)
                
                if df is None:
                    self.logger.warning(f"Dataset {dataset_name} not found or not a DataFrame")
                    summary[dataset_name] = {"error": "Dataset not found or not a DataFrame"}
                    continue
                    
                if not isinstance(df, pd.DataFrame):
                    self.logger.warning(f"Dataset {dataset_name} is not a DataFrame, type: {type(df)}")
                    summary[dataset_name] = {"error": f"Expected DataFrame, got {type(df)}"}
                    continue

                # Get basic dataset info
                row_count = df.shape[0]
                column_count = df.shape[1]
                columns = df.columns.tolist()
                
                # Calculate missing values
                missing_values = df.isna().sum().sum()
                missing_percentage = (missing_values / (row_count * column_count) * 100) if row_count * column_count > 0 else 0
                
                # Check for duplicates
                duplicate_count = df.duplicated().sum()
                has_duplicates = duplicate_count > 0
                
                # Generate summary for this dataset
                summary[dataset_name] = {
                    "row_count": int(row_count),
                    "column_count": int(column_count),
                    "columns": columns,
                    "missing_values": {
                        "count": int(missing_values),
                        "percentage": float(missing_percentage)
                    },
                    "duplicates": {
                        "count": int(duplicate_count),
                        "percentage": float(duplicate_count / row_count * 100) if row_count > 0 else 0,
                        "has_duplicates": has_duplicates
                    }
                }

                # Generate statistics for this dataset
                statistics[dataset_name] = {
                    "numeric_columns": {},
                    "categorical_columns": {},
                    "datetime_columns": {},
                }
                
                # Analyze each column
                for col in columns:
                    try:
                        series = df[col]
                        col_type = self.detect_column_type(series)
                        
                        if col_type == "numeric":
                            statistics[dataset_name]["numeric_columns"][col] = self.analyze_numeric_column(series)
                        elif col_type == "datetime":
                            statistics[dataset_name]["datetime_columns"][col] = self.analyze_datetime_column(series)
                        else:  # categorical
                            statistics[dataset_name]["categorical_columns"][col] = self.analyze_categorical_column(series)
                    except Exception as e:
                        self.logger.error(f"Error analyzing column {col}: {str(e)}")
                        traceback_str = traceback.format_exc()
                        self.logger.debug(f"Traceback: {traceback_str}")
                
                # Compute correlations
                correlations = self.compute_correlations(df)
                summary[dataset_name]["correlations"] = correlations
                
            except Exception as e:
                self.logger.error(f"Error exploring dataset {dataset_name}: {str(e)}")
                traceback_str = traceback.format_exc()
                self.logger.debug(f"Traceback: {traceback_str}")
                summary[dataset_name] = {"error": str(e), "traceback": traceback_str}

        # Return the exploration results
        return {
            "Data Overview.summary": summary,
            "Data Overview.statistics": statistics,
        }
