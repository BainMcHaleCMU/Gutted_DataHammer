"""
Modeling Agent

This module defines the ModelingAgent class for developing and evaluating predictive models.
"""

from typing import Any, Dict, List, Optional
import logging
import traceback

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import ModelingTaskAgent


class ModelingAgent(BaseAgent):
    """
    Agent responsible for developing and evaluating predictive or descriptive models.
    
    The Modeling Agent:
    - Selects appropriate modeling algorithms
    - Performs feature engineering/selection
    - Manages data splitting, model training, and evaluation
    - Requests visualizations for model evaluation
    """
    
    def __init__(self):
        """Initialize the Modeling Agent."""
        super().__init__(name="ModelingAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = ModelingTaskAgent()
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to model
                - target_variable: Name of the target variable
                - model_type: Optional type of model to build
                - goals: Optional list of modeling goals
                
        Returns:
            Dict containing:
                - models: Dict with model artifacts and metrics
                - visualization_requests: List of visualization requests
                - suggestions: List of suggested next steps
        """
        try:
            # Extract data reference from kwargs or environment
            data_reference = kwargs.get("data_reference")
            if not data_reference:
                # Try to find data in different locations in the environment
                if "cleaned_data" in environment:
                    data_reference = environment["cleaned_data"]
                elif "Cleaned Data" in environment and "processed_data" in environment["Cleaned Data"]:
                    # Just use all processed data
                    data_reference = None
                elif "loaded_data" in environment:
                    data_reference = environment["loaded_data"]
                elif "Raw Data" in environment:
                    data_reference = "Raw Data"
                    self.logger.warning("Using raw data for modeling. Results may be suboptimal.")
            
            # Extract other parameters
            target_variable = kwargs.get("target_variable")
            model_type = kwargs.get("model_type")
            goals = kwargs.get("goals", ["Build predictive models"])
            
            # Log the modeling request
            self.logger.info(f"Modeling request: data={data_reference}, target={target_variable}, model={model_type}")
            self.logger.info(f"Modeling goals: {goals}")
            
            # Validate environment
            if not environment:
                raise ValueError("Environment is empty or invalid")
            
            # Use the task agent to build models
            task_input = {
                "environment": environment,
                "goals": goals,
                "data_reference": data_reference,
                "target_variable": target_variable,
                "model_type": model_type
            }
            
            # Run the task agent
            self.logger.info("Running modeling task agent")
            result = self.task_agent.run(task_input)
            
            # Check for errors in the result
            if "error" in result:
                self.logger.error(f"Modeling task agent returned an error: {result['error']}")
                return {
                    "error": result["error"],
                    "models": {},
                    "visualization_requests": [],
                    "suggestions": ["Check data format and try again"]
                }
            
            # Extract the results
            trained_models = result.get("Models.trained_model", {})
            performance_metrics = result.get("Models.performance", {})
            
            # Create visualization requests based on models
            visualization_requests = self._generate_visualization_requests(trained_models, performance_metrics)
            
            # Generate suggestions based on results
            suggestions = self._generate_suggestions(trained_models, performance_metrics)
            
            # Combine model information
            model_results = self._combine_model_information(trained_models, performance_metrics)
            
            return {
                "models": model_results,
                "visualization_requests": visualization_requests,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"Error in ModelingAgent: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                "error": str(e),
                "models": {},
                "visualization_requests": [],
                "suggestions": ["Check data format and try again"]
            }
    
    def _generate_visualization_requests(
        self, 
        trained_models: Dict[str, Any], 
        performance_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate visualization requests based on trained models and their performance.
        
        Args:
            trained_models: Dictionary of trained models
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            List of visualization requests
        """
        visualization_requests = []
        
        # Process each dataset
        for dataset_name, models in trained_models.items():
            # Skip datasets with errors
            if isinstance(models, dict) and "error" in models:
                continue
                
            # Get the best model for this dataset
            best_model = None
            if dataset_name in performance_metrics:
                best_model = performance_metrics[dataset_name].get("best_model")
            
            # Add visualizations for the best model
            if best_model and best_model in models:
                model_info = models[best_model]
                model_type = model_info.get("type", "")
                
                # Add appropriate visualizations based on model type
                if "regression" in model_type or model_type in ["linear_regression", "arima", "prophet"]:
                    # Regression visualizations
                    visualization_requests.extend([
                        {"type": "actual_vs_predicted", "dataset": dataset_name, "model": best_model},
                        {"type": "residual_plot", "dataset": dataset_name, "model": best_model},
                        {"type": "feature_importance", "dataset": dataset_name, "model": best_model}
                    ])
                elif "classification" in model_type or model_type in ["logistic_regression", "random_forest"]:
                    # Classification visualizations
                    visualization_requests.extend([
                        {"type": "confusion_matrix", "dataset": dataset_name, "model": best_model},
                        {"type": "roc_curve", "dataset": dataset_name, "model": best_model},
                        {"type": "feature_importance", "dataset": dataset_name, "model": best_model}
                    ])
                elif "clustering" in model_type or model_type in ["kmeans", "hierarchical", "dbscan"]:
                    # Clustering visualizations
                    visualization_requests.extend([
                        {"type": "cluster_plot", "dataset": dataset_name, "model": best_model},
                        {"type": "silhouette_plot", "dataset": dataset_name, "model": best_model}
                    ])
                else:
                    # Generic visualizations
                    visualization_requests.append(
                        {"type": "model_performance", "dataset": dataset_name, "model": best_model}
                    )
        
        return visualization_requests
    
    def _generate_suggestions(
        self, 
        trained_models: Dict[str, Any], 
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Generate suggestions based on modeling results.
        
        Args:
            trained_models: Dictionary of trained models
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Check if we have any successful models
        has_successful_models = False
        has_errors = False
        
        for dataset_name, models in trained_models.items():
            if isinstance(models, dict) and "error" in models:
                has_errors = True
            else:
                has_successful_models = True
        
        # Add suggestions based on results
        if has_successful_models:
            suggestions.append("Run ReportingAgent to generate final report with model results")
            suggestions.append("Use VisualizationAgent to create visualizations of model performance")
            
            # Add suggestion for model deployment if appropriate
            suggestions.append("Consider deploying the best performing model for predictions")
        
        if has_errors:
            suggestions.append("Check data quality and format for datasets with errors")
            suggestions.append("Consider additional data cleaning steps before modeling")
        
        # Add general suggestions
        suggestions.append("Consider hyperparameter tuning to improve model performance")
        suggestions.append("Evaluate models on additional metrics relevant to your use case")
        
        return suggestions
    
    def _combine_model_information(
        self, 
        trained_models: Dict[str, Any], 
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine model information and performance metrics.
        
        Args:
            trained_models: Dictionary of trained models
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            Combined model information
        """
        model_results = {}
        
        # Process each dataset
        for dataset_name, models in trained_models.items():
            # Handle error cases
            if isinstance(models, dict) and "error" in models:
                model_results[dataset_name] = {"error": models["error"]}
                continue
            
            # Get performance metrics for this dataset
            dataset_metrics = performance_metrics.get(dataset_name, {})
            
            # Initialize results for this dataset
            model_results[dataset_name] = {}
            
            # Process each model
            for model_name, model_info in models.items():
                # Skip error entries
                if isinstance(model_info, dict) and "error" in model_info:
                    model_results[dataset_name][model_name] = {"error": model_info["error"]}
                    continue
                
                # Get metrics for this model
                model_metrics = dataset_metrics.get(model_name, {})
                
                # Combine information
                model_results[dataset_name][model_name] = {
                    "type": model_info.get("type", "Unknown"),
                    "features": model_info.get("features", []),
                    "target": model_info.get("target", ""),
                    "hyperparameters": model_info.get("hyperparameters", {}),
                    "performance": model_metrics.get("metrics", {}),
                    "cross_validation": model_metrics.get("cross_validation", {}),
                    "data_shape": model_info.get("data_shape", {}),
                    "timestamp": model_info.get("timestamp", "")
                }
            
            # Add best model information
            best_model = dataset_metrics.get("best_model")
            if best_model:
                model_results[dataset_name]["best_model"] = best_model
        
        return model_results