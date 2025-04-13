"""
Visualization Agent

This module defines the VisualizationAgent class for generating visual representations.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import os
import json

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import VisualizationTaskAgent


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for generating visual representations of data and results.
    
    The Visualization Agent:
    - Receives requests for specific plot types
    - Formulates Python code using plotting libraries
    - Uses CodeActAgent to execute plotting code
    - Returns plot references and descriptions
    """
    
    def __init__(self):
        """Initialize the Visualization Agent."""
        super().__init__(name="VisualizationAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = VisualizationTaskAgent()
        
        # Create visualizations directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
    
    def _validate_plot_params(self, plot_type: str, plot_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize plot parameters based on plot type.
        
        Args:
            plot_type: Type of plot to generate
            plot_params: Dict of parameters for the plot
            
        Returns:
            Dict of validated and normalized plot parameters
        """
        validated_params = plot_params.copy() if plot_params else {}
        
        # Validate based on plot type
        if plot_type == "histogram":
            # Ensure bins is an integer
            if "bins" in validated_params:
                try:
                    validated_params["bins"] = int(validated_params["bins"])
                except (ValueError, TypeError):
                    validated_params["bins"] = 10
            else:
                validated_params["bins"] = 10
                
        elif plot_type == "scatter_plot":
            # Ensure trend_line is a boolean
            if "trend_line" in validated_params:
                validated_params["trend_line"] = bool(validated_params["trend_line"])
            else:
                validated_params["trend_line"] = True
                
        elif plot_type == "box_plot":
            # Ensure features is a list
            if "features" in validated_params and not isinstance(validated_params["features"], list):
                if isinstance(validated_params["features"], str):
                    validated_params["features"] = [validated_params["features"]]
                else:
                    validated_params["features"] = []
                    
        elif plot_type == "bar_chart":
            # Ensure horizontal is a boolean
            if "horizontal" in validated_params:
                validated_params["horizontal"] = bool(validated_params["horizontal"])
            else:
                validated_params["horizontal"] = False
                
        elif plot_type == "line_chart":
            # Ensure markers is a boolean
            if "markers" in validated_params:
                validated_params["markers"] = bool(validated_params["markers"])
            else:
                validated_params["markers"] = True
                
        elif plot_type == "heatmap":
            # Ensure features is a list
            if "features" in validated_params and not isinstance(validated_params["features"], list):
                if isinstance(validated_params["features"], str):
                    validated_params["features"] = [validated_params["features"]]
                else:
                    validated_params["features"] = []
        
        return validated_params
    
    def _find_data_reference(self, environment: Dict[str, Any], data_reference: Optional[str] = None) -> Optional[str]:
        """
        Find a valid data reference in the environment.
        
        Args:
            environment: The shared environment state
            data_reference: Optional reference to the data to visualize
            
        Returns:
            A valid data reference or None if not found
        """
        # If data_reference is provided, check if it exists in the environment
        if data_reference and data_reference in environment:
            return data_reference
            
        # Check common data locations
        common_data_keys = [
            "cleaned_data", 
            "loaded_data", 
            "processed_data", 
            "data",
            "Cleaned Data",
            "Loaded Data",
            "Data"
        ]
        
        for key in common_data_keys:
            if key in environment and environment[key]:
                return key
                
        # Check for nested data structures
        for key, value in environment.items():
            if isinstance(value, dict) and "data" in value:
                return f"{key}.data"
            elif isinstance(value, dict) and "dataset" in value:
                return f"{key}.dataset"
                
        # No valid data reference found
        return None
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to visualize
                - plot_type: Type of plot to generate
                - plot_params: Dict of parameters for the plot
                - output_format: Format for the output (png, svg, html)
                - title: Title for the visualization
                - description: Description for the visualization
                
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        data_reference = self._find_data_reference(environment, data_reference)
        
        if not data_reference:
            self.logger.error("No valid data reference found in environment")
            return {
                "error": "No valid data reference found in environment",
                "plot_reference": None,
                "plot_description": "Failed to create visualization due to missing data reference"
            }
        
        # Extract and validate plot parameters
        plot_type = kwargs.get("plot_type", "histogram")
        plot_params = self._validate_plot_params(plot_type, kwargs.get("plot_params", {}))
        
        # Extract additional parameters
        output_format = kwargs.get("output_format", "png")
        title = kwargs.get("title")
        description = kwargs.get("description")
        
        # Add title and description to plot_params if provided
        if title:
            plot_params["title"] = title
        if description:
            plot_params["description"] = description
        
        self.logger.info(f"Creating visualization: {plot_type} for {data_reference}")
        
        # Use the task agent to create visualizations
        task_input = {
            "environment": environment,
            "goals": ["Create informative visualizations"],
            "data_reference": data_reference,
            "plot_type": plot_type,
            "plot_params": plot_params,
            "output_format": output_format
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Check for errors
            if "Visualization.error" in result:
                error_message = result.get("Visualization.error", "Unknown error")
                self.logger.error(f"Error from task agent: {error_message}")
                return {
                    "error": error_message,
                    "plot_reference": None,
                    "plot_description": f"Failed to create {plot_type} visualization: {error_message}"
                }
            
            # Extract the results
            plot_reference = result.get("Visualization.plot_path")
            plot_description = result.get("Visualization.description", f"{plot_type} visualization")
            plot_details = result.get("Visualization.details")
            
            # Ensure we have a plot reference
            if not plot_reference:
                plot_reference = f"visualizations/{data_reference.replace('.', '_')}_{plot_type}.{output_format}"
            
            # Parse plot details if it's a JSON string
            if isinstance(plot_details, str):
                try:
                    plot_details = json.loads(plot_details)
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass
            
            # If we have Visualizations.plots, extract the first plot
            if "Visualizations.plots" in result:
                plots = result.get("Visualizations.plots", {})
                if plots:
                    # Get the first dataset
                    first_dataset = next(iter(plots))
                    dataset_plots = plots[first_dataset]
                    
                    # Get the first plot category that has plots
                    for category in ["data_exploration", "analysis", "model_performance"]:
                        if category in dataset_plots and dataset_plots[category]:
                            first_plot = dataset_plots[category][0]
                            plot_description = first_plot.get("description", plot_description)
                            if not plot_details:
                                plot_details = first_plot
                            break
            
            return {
                "plot_reference": plot_reference,
                "plot_description": plot_description,
                "plot_details": plot_details
            }
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to create {plot_type} visualization due to an error: {str(e)}"
            }