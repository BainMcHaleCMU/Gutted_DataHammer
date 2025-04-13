"""
Exploration Agent

This module defines the ExplorationAgent class for initial data analysis and profiling.
"""

from typing import Any, Dict, List, Optional
import logging

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import ExplorationTaskAgent


class ExplorationAgent(BaseAgent):
    """
    Agent responsible for performing initial data analysis and profiling.
    
    The Exploration Agent:
    - Calculates descriptive statistics
    - Identifies data types
    - Assesses missing values
    - Performs initial outlier detection
    - Computes correlations
    - Requests visualizations via the Orchestrator
    """
    
    def __init__(self):
        """Initialize the Exploration Agent."""
        super().__init__(name="ExplorationAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = ExplorationTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to explore
                
        Returns:
            Dict containing:
                - data_overview: Dict with exploration results
                - visualization_requests: List of visualization requests
                - suggestions: List of suggested next steps
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference and "loaded_data" in environment:
            data_reference = environment["loaded_data"]
        
        self.logger.info(f"Exploring data: {data_reference}")
        
        # Use the task agent to explore the data
        task_input = {
            "environment": environment,
            "goals": ["Explore and understand the data structure and quality"],
            "data_reference": data_reference
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            summary = result.get("Data Overview.summary", {})
            statistics = result.get("Data Overview.statistics", {})
            
            # Create data overview
            data_overview = {
                "statistics": statistics,
                "missing_values": summary.get("missing_values", {"count": 0, "percentage": 0}),
                "outliers": summary.get("outliers", {"count": 0, "indices": []}),
                "correlations": summary.get("correlations", {})
            }
            
            # Generate visualization requests
            visualization_requests = [
                {"type": "histogram", "data": "column_x"},
                {"type": "correlation_matrix", "data": "all_numeric"}
            ]
            
            # Add suggestions for next steps
            suggestions = ["Run CleaningAgent to handle missing values and outliers"]
            
            return {
                "data_overview": data_overview,
                "visualization_requests": visualization_requests,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error exploring data: {str(e)}")
            return {
                "error": str(e),
                "data_overview": {
                    "statistics": {},
                    "missing_values": {"count": 0, "percentage": 0},
                    "outliers": {"count": 0, "indices": []},
                    "correlations": {}
                },
                "visualization_requests": [],
                "suggestions": ["Check data format and try again"]
            }