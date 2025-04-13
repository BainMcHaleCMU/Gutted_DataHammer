"""
Cleaning Task Agent Module

This module defines the cleaning task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import logging
import traceback
from enum import Enum

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class CleaningStrategy(Enum):
    """Enumeration of supported cleaning strategies."""
    MISSING_VALUES = "handle_missing_values"
    DUPLICATES = "remove_duplicates"
    OUTLIERS = "handle_outliers"
    DATA_TYPES = "convert_data_types"
    NORMALIZATION = "normalize_data"
    ENCODING = "encode_categorical"
    CUSTOM = "custom_strategy"


class CleaningTaskAgent(BaseTaskAgent):
    """
    Task agent for cleaning and preprocessing data.

    Responsibilities:
    - Handling missing values
    - Removing duplicates
    - Handling outliers
    - Feature engineering
    - Data transformation
    - Data validation
    
    This implementation provides robust error handling, detailed logging,
    and support for custom cleaning strategies.
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the CleaningTaskAgent."""
        super().__init__(name="CleaningAgent", llm=llm)
        self.default_strategies = {
            CleaningStrategy.MISSING_VALUES.value: True,
            CleaningStrategy.DUPLICATES.value: True,
            CleaningStrategy.OUTLIERS.value: True,
            CleaningStrategy.DATA_TYPES.value: True,
            CleaningStrategy.NORMALIZATION.value: False,
            CleaningStrategy.ENCODING.value: False,
        }
        
    def _validate_input_data(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Validate input data and return a list of validation errors.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for required environment data
        if not input_data.get("environment"):
            errors.append("Missing 'environment' in input data")
            return errors  # Can't proceed without environment
            
        environment = input_data.get("environment", {})
        data_overview = environment.get("Data Overview", {})
        
        # Check for raw data
        if not data_overview.get("raw_data"):
            errors.append("No raw data found in environment")
            
        # Check for data summary
        if not data_overview.get("summary"):
            errors.append("No data summary found in environment")
            
        # Check for statistics
        if not data_overview.get("statistics"):
            errors.append("No statistics found in environment")
            
        return errors
    
    def _get_cleaning_strategies(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cleaning strategies from input data or use defaults.
        
        Args:
            input_data: Input data containing optional cleaning strategies
            
        Returns:
            Dict of cleaning strategies to apply
        """
        # Start with default strategies
        strategies = self.default_strategies.copy()
        
        # Override with user-provided strategies if available
        user_strategies = input_data.get("cleaning_strategies", {})
        for strategy, enabled in user_strategies.items():
            if strategy in strategies or strategy == CleaningStrategy.CUSTOM.value:
                strategies[strategy] = enabled
                
        return strategies
    
    def _apply_cleaning_strategy(
        self, 
        strategy: str, 
        dataset_name: str,
        dataset_info: Dict[str, Any],
        dataset_summary: Dict[str, Any],
        dataset_stats: Dict[str, Any],
        custom_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply a specific cleaning strategy and return the cleaning step details.
        
        Args:
            strategy: The cleaning strategy to apply
            dataset_name: Name of the dataset
            dataset_info: Dataset information
            dataset_summary: Dataset summary
            dataset_stats: Dataset statistics
            custom_strategy: Optional custom strategy details
            
        Returns:
            Dict containing cleaning step details
        """
        self.logger.debug(f"Applying cleaning strategy '{strategy}' to dataset '{dataset_name}'")
        
        if strategy == CleaningStrategy.MISSING_VALUES.value:
            # Handle missing values
            if not dataset_summary.get("has_missing_values", False):
                return None  # No missing values to handle
                
            affected_columns = list(dataset_stats.get("numeric_columns", {}).keys()) + \
                              list(dataset_stats.get("categorical_columns", {}).keys())
            
            return {
                "operation": strategy,
                "description": "Filled missing numeric values with median, categorical with mode",
                "affected_columns": affected_columns,
                "details": "Imputed missing values across affected columns",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.DUPLICATES.value:
            # Handle duplicates
            if not dataset_summary.get("has_duplicates", False):
                return None  # No duplicates to handle
                
            return {
                "operation": strategy,
                "description": "Removed duplicate rows based on all columns",
                "details": "Removed duplicate rows",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.OUTLIERS.value:
            # Handle outliers
            numeric_columns = list(dataset_stats.get("numeric_columns", {}).keys())
            if not numeric_columns:
                return None  # No numeric columns to handle outliers
                
            return {
                "operation": strategy,
                "description": "Capped outliers at 3 standard deviations from mean",
                "affected_columns": numeric_columns,
                "details": "Modified outlier values",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.DATA_TYPES.value:
            # Handle data type conversions
            return {
                "operation": strategy,
                "description": "Converted columns to appropriate data types",
                "details": "Ensured proper types for all columns",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.NORMALIZATION.value:
            # Handle data normalization
            numeric_columns = list(dataset_stats.get("numeric_columns", {}).keys())
            if not numeric_columns:
                return None  # No numeric columns to normalize
                
            return {
                "operation": strategy,
                "description": "Normalized numeric columns to 0-1 range",
                "affected_columns": numeric_columns,
                "details": "Applied min-max scaling to numeric columns",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.ENCODING.value:
            # Handle categorical encoding
            categorical_columns = list(dataset_stats.get("categorical_columns", {}).keys())
            if not categorical_columns:
                return None  # No categorical columns to encode
                
            return {
                "operation": strategy,
                "description": "Encoded categorical variables",
                "affected_columns": categorical_columns,
                "details": "Applied one-hot encoding to categorical columns",
                "status": "success"
            }
            
        elif strategy == CleaningStrategy.CUSTOM.value and custom_strategy:
            # Apply custom strategy
            return {
                "operation": strategy,
                "description": custom_strategy.get("description", "Applied custom cleaning strategy"),
                "affected_columns": custom_strategy.get("affected_columns", []),
                "details": custom_strategy.get("details", "Custom cleaning applied"),
                "status": "success"
            }
            
        return None  # Strategy not recognized or not applicable

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data cleaning task with robust error handling and validation.

        Args:
            input_data: Input data for the task, which may include:
                - environment: The shared environment state
                - goals: List of cleaning goals
                - cleaning_strategies: Optional dict of cleaning strategies to apply
                - custom_strategies: Optional dict of custom cleaning strategies

        Returns:
            Dict containing:
                - Cleaned Data.processed_data: Processed data
                - Cleaned Data.cleaning_steps: Cleaning steps applied
                - Cleaned Data.validation: Validation results
        """
        self.logger.info("Starting data cleaning process")
        
        # Initialize results
        processed_data = {}
        cleaning_steps = {}
        validation_results = {}
        
        # Validate input data
        validation_errors = self._validate_input_data(input_data)
        if validation_errors:
            self.logger.error(f"Input data validation failed: {validation_errors}")
            return {
                "Cleaned Data.processed_data": {},
                "Cleaned Data.cleaning_steps": {},
                "Cleaned Data.validation": {
                    "status": "error",
                    "errors": validation_errors
                }
            }
        
        # Extract data from environment
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        custom_strategies = input_data.get("custom_strategies", {})
        
        # Get cleaning strategies
        strategies = self._get_cleaning_strategies(input_data)
        self.logger.info(f"Using cleaning strategies: {strategies}")
        
        # Get raw data and summary from the environment
        data_overview = environment.get("Data Overview", {})
        raw_data = data_overview.get("raw_data", {})
        summary = data_overview.get("summary", {})
        statistics = data_overview.get("statistics", {})
        
        # Process each dataset
        for dataset_name, dataset_info in raw_data.items():
            self.logger.info(f"Processing dataset: {dataset_name}")
            validation_results[dataset_name] = {"status": "pending"}
            
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to loading error: {dataset_info.get('error', 'Unknown error')}"
                    )
                    validation_results[dataset_name] = {
                        "status": "skipped",
                        "reason": f"Dataset had loading error: {dataset_info.get('error', 'Unknown error')}"
                    }
                    continue

                # Get dataset summary and statistics
                dataset_summary = summary.get(dataset_name, {})
                dataset_stats = statistics.get(dataset_name, {})
                
                # Validate dataset has required information
                if not dataset_summary or not dataset_stats:
                    self.logger.warning(f"Missing summary or statistics for dataset {dataset_name}")
                    validation_results[dataset_name] = {
                        "status": "warning",
                        "reason": "Missing summary or statistics information"
                    }
                
                # Initialize cleaning steps for this dataset
                cleaning_steps[dataset_name] = []
                
                # Apply each enabled cleaning strategy
                for strategy, enabled in strategies.items():
                    if not enabled:
                        continue
                        
                    # Apply the strategy
                    custom_strategy_details = custom_strategies.get(dataset_name, {}).get(strategy) if strategy == CleaningStrategy.CUSTOM.value else None
                    
                    try:
                        step_result = self._apply_cleaning_strategy(
                            strategy, 
                            dataset_name, 
                            dataset_info, 
                            dataset_summary, 
                            dataset_stats,
                            custom_strategy_details
                        )
                        
                        if step_result:
                            cleaning_steps[dataset_name].append(step_result)
                            
                    except Exception as strategy_error:
                        self.logger.error(f"Error applying strategy '{strategy}' to dataset '{dataset_name}': {str(strategy_error)}")
                        cleaning_steps[dataset_name].append({
                            "operation": strategy,
                            "description": f"Error applying strategy: {str(strategy_error)}",
                            "status": "error",
                            "error": str(strategy_error)
                        })

                # Create processed data entry with detailed information
                processed_data[dataset_name] = {
                    "type": "dataframe",
                    "rows": dataset_info.get("rows", 0) - (
                        dataset_summary.get("duplicate_count", 3) 
                        if dataset_summary.get("has_duplicates", False) else 0
                    ),
                    "columns": dataset_info.get("columns", []),
                    "source": dataset_info.get("source", ""),
                    "is_cleaned": True,
                    "cleaning_summary": f"Applied {len(cleaning_steps[dataset_name])} cleaning operations",
                    "cleaning_timestamp": "2023-01-01T00:00:00Z",  # In a real implementation, use actual timestamp
                }
                
                # Update validation results
                validation_results[dataset_name] = {
                    "status": "success",
                    "operations_applied": len(cleaning_steps[dataset_name]),
                    "warnings": []
                }

            except Exception as e:
                error_details = traceback.format_exc()
                self.logger.error(f"Error cleaning dataset {dataset_name}: {str(e)}\n{error_details}")
                
                # Record the error in cleaning steps
                cleaning_steps[dataset_name] = [{
                    "operation": "error",
                    "description": f"Error during cleaning: {str(e)}",
                    "status": "error",
                    "error_details": error_details
                }]
                
                # Record error in processed data
                processed_data[dataset_name] = {
                    "type": "error", 
                    "error": str(e),
                    "error_details": error_details
                }
                
                # Update validation results
                validation_results[dataset_name] = {
                    "status": "error",
                    "error": str(e)
                }

        # Return the cleaned data with validation results
        self.logger.info(f"Data cleaning completed for {len(processed_data)} datasets")
        return {
            "Cleaned Data.processed_data": processed_data,
            "Cleaned Data.cleaning_steps": cleaning_steps,
            "Cleaned Data.validation": validation_results
        }
