"""
Cleaning Agent

This module defines the CleaningAgent class for preprocessing and cleaning data.
"""

from typing import Any, Dict, List, Optional
import logging
import traceback

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import CleaningTaskAgent
from ..llama_workflow.cleaning_agent import CleaningStrategy


class CleaningAgent(BaseAgent):
    """
    Agent responsible for preprocessing and cleaning data.
    
    The Cleaning Agent:
    - Implements strategies for handling missing values
    - Handles outliers
    - Performs data type conversions
    - Addresses data sparsity issues
    - Documents cleaning steps applied
    - Validates data quality before and after cleaning
    """
    
    def __init__(self):
        """Initialize the Cleaning Agent."""
        super().__init__(name="CleaningAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = CleaningTaskAgent()
    
    def _validate_data_reference(self, data_reference: Any) -> bool:
        """
        Validate that the data reference is valid.
        
        Args:
            data_reference: Reference to the data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not data_reference:
            return False
            
        # Add additional validation as needed
        return True
    
    def _prepare_cleaning_strategies(self, user_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare cleaning strategies based on user input and defaults.
        
        Args:
            user_strategies: User-provided cleaning strategies
            
        Returns:
            Dict of prepared cleaning strategies
        """
        # Start with default strategies
        strategies = {
            CleaningStrategy.MISSING_VALUES.value: True,
            CleaningStrategy.DUPLICATES.value: True,
            CleaningStrategy.OUTLIERS.value: True,
            CleaningStrategy.DATA_TYPES.value: True,
        }
        
        # Override with user-provided strategies
        if user_strategies:
            for strategy, enabled in user_strategies.items():
                if strategy in strategies or strategy == CleaningStrategy.CUSTOM.value:
                    strategies[strategy] = enabled
        
        return strategies
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality with improved error handling and validation.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to clean
                - cleaning_strategies: Optional dict of cleaning strategies to apply
                - custom_strategies: Optional dict of custom cleaning strategies
                
        Returns:
            Dict containing:
                - cleaned_data: Reference to cleaned data
                - cleaning_steps: List of cleaning steps applied
                - validation: Validation results
                - suggestions: List of suggested next steps
                - status: Overall status of the cleaning operation
        """
        self.logger.info("Starting cleaning agent")
        
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference and "loaded_data" in environment:
            data_reference = environment["loaded_data"]
        
        # Validate data reference
        if not self._validate_data_reference(data_reference):
            error_msg = "Invalid or missing data reference"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "cleaned_data": None,
                "cleaning_steps": {},
                "validation": {"status": "error", "errors": [error_msg]},
                "suggestions": ["Provide valid data reference or load data first"]
            }
        
        # Prepare cleaning strategies
        cleaning_strategies = self._prepare_cleaning_strategies(kwargs.get("cleaning_strategies", {}))
        custom_strategies = kwargs.get("custom_strategies", {})
        
        self.logger.info(f"Cleaning data with strategies: {cleaning_strategies}")
        
        # Use the task agent to clean the data
        task_input = {
            "environment": environment,
            "goals": ["Clean and preprocess data for analysis"],
            "data_reference": data_reference,
            "cleaning_strategies": cleaning_strategies,
            "custom_strategies": custom_strategies
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            processed_data = result.get("Cleaned Data.processed_data", {})
            cleaning_steps = result.get("Cleaned Data.cleaning_steps", {})
            validation_results = result.get("Cleaned Data.validation", {})
            
            # Check for errors in validation results
            has_errors = False
            for dataset_name, validation in validation_results.items():
                if validation.get("status") == "error":
                    has_errors = True
                    break
            
            # Generate suggestions based on cleaning results
            suggestions = []
            if has_errors:
                suggestions.append("Review error details and fix data issues")
                suggestions.append("Check data format and try again with modified strategies")
            else:
                suggestions.append("Run ExplorationAgent again on cleaned data")
                suggestions.append("Proceed to data analysis with cleaned data")
            
            # Determine overall status
            status = "error" if has_errors else "success"
            
            return {
                "status": status,
                "cleaned_data": processed_data,
                "cleaning_steps": cleaning_steps,
                "validation": validation_results,
                "suggestions": suggestions
            }
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error cleaning data: {str(e)}\n{error_details}")
            
            return {
                "status": "error",
                "error": str(e),
                "error_details": error_details,
                "cleaned_data": None,
                "cleaning_steps": {},
                "validation": {"status": "error", "errors": [str(e)]},
                "suggestions": [
                    "Check data format and try again",
                    "Review error details and fix underlying issues",
                    "Try with different cleaning strategies"
                ]
            }