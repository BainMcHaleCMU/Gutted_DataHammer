"""
Data Loading Agent

This module defines the DataLoadingAgent class for ingesting data from various sources.
"""

from typing import Any, Dict, List, Optional
import logging
import pandas as pd
import json
import os
import traceback

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import DataLoadingTaskAgent


class DataLoadingAgent(BaseAgent):
    """
    Agent responsible for ingesting data from various sources.
    
    The Data Loading Agent:
    - Parses files (PDF, CSV, Excel, JSON, Parquet, etc.)
    - Connects to databases
    - Loads data from APIs
    - Loads data into standard formats (e.g., Pandas DataFrame)
    - Handles initial loading errors and retries
    - Validates data against specified rules
    - Reports loaded data location/reference and initial schema
    """
    
    def __init__(self):
        """Initialize the Data Loading Agent."""
        super().__init__(name="DataLoadingAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = DataLoadingTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_sources: Dict of data source references
                - file_type: Optional type of file to load
                - validation_rules: Optional rules for data validation
                
        Returns:
            Dict containing:
                - loaded_data: Reference to loaded data
                - schema: Initial data schema
                - validation_results: Results of data validation
                - errors: Any errors encountered during loading
                - suggestions: List of suggested next steps
        """
        # Extract data sources from kwargs or environment
        data_sources = kwargs.get("data_sources", {})
        if not data_sources and "data_sources" in environment:
            data_sources = environment["data_sources"]
        
        # Extract other parameters
        file_type = kwargs.get("file_type")
        validation_rules = kwargs.get("validation_rules", {})
        
        if not data_sources:
            self.logger.warning("No data sources provided")
            return {
                "error": "No data sources provided",
                "loaded_data": {},
                "schema": {},
                "validation_results": {},
                "suggestions": ["Provide data sources to load"]
            }
        
        self.logger.info(f"Loading data from {len(data_sources)} sources")
        
        # Use the task agent to load the data
        task_input = {
            "environment": environment,
            "goals": ["Load and prepare data for analysis"],
            "data_sources": data_sources,
            "file_type": file_type,
            "validation_rules": validation_rules
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            loaded_data = result.get("Data Overview.raw_data", {})
            schema = result.get("Data Overview.schema", {})
            validation_results = result.get("Data Overview.validation", {})
            errors = result.get("Data Overview.errors", [])
            
            # Generate suggestions based on results
            suggestions = []
            
            # Check if there were any errors
            if errors:
                suggestions.append("Review and fix data loading errors")
                
                # Add specific suggestions based on error types
                for error in errors:
                    if "File not found" in error:
                        suggestions.append("Check file paths and ensure files exist")
                    elif "Unsupported data source type" in error:
                        suggestions.append("Use supported data source types: csv, excel, json, parquet, database, api")
                    elif "Database connection" in error:
                        suggestions.append("Verify database connection parameters")
            
            # Check validation results
            validation_failed = False
            for source_name, validation in validation_results.items():
                if validation.get("status") == "failed":
                    validation_failed = True
                    suggestions.append(f"Fix validation issues in {source_name}")
            
            # Add next step suggestions
            if not errors and not validation_failed:
                suggestions.append("Run ExplorationAgent to analyze the loaded data")
                suggestions.append("Run CleaningAgent if data preprocessing is needed")
            
            # Store loaded data in environment for other agents to use
            if "Data" not in environment:
                environment["Data"] = {}
            
            for source_name, source_data in loaded_data.items():
                if "error" not in source_data:
                    environment["Data"][source_name] = source_data
            
            return {
                "loaded_data": loaded_data,
                "schema": schema,
                "validation_results": validation_results,
                "errors": errors,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "loaded_data": {},
                "schema": {},
                "validation_results": {},
                "suggestions": [
                    "Check data source paths and formats",
                    "Verify that required dependencies are installed",
                    "Check for network connectivity issues if loading from remote sources"
                ]
            }