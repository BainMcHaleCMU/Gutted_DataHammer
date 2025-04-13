"""
Modeling Task Agent Module

This module defines the modeling task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import traceback

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class ModelingTaskAgent(BaseTaskAgent):
    """
    Task agent for building and evaluating predictive models.

    Responsibilities:
    - Feature selection
    - Model selection
    - Model training
    - Model evaluation
    - Hyperparameter tuning
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the ModelingTaskAgent."""
        super().__init__(name="ModelingAgent", llm=llm)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the modeling task.

        Args:
            input_data: Input data for the task containing:
                - environment: The shared environment state
                - goals: List of user-defined goals
                - data_reference: Optional reference to specific data to model
                - target_variable: Optional name of the target variable
                - model_type: Optional type of model to build

        Returns:
            Dict containing model and performance metrics
        """
        # Validate input data
        if not isinstance(input_data, dict):
            error_msg = f"Input data must be a dictionary, got {type(input_data)}"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg)

        # Extract input parameters
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        data_reference = input_data.get("data_reference")
        target_variable = input_data.get("target_variable")
        requested_model_type = input_data.get("model_type")

        self.logger.info(f"Starting modeling task with goals: {goals}")
        self.logger.info(f"Target variable: {target_variable}, Model type: {requested_model_type}")

        # Initialize results
        trained_models = {}
        performance_metrics = {}

        try:
            # Get cleaned data from the environment
            cleaned_data = environment.get("Cleaned Data", {})
            processed_data = cleaned_data.get("processed_data", {})

            # Validate processed data
            if not processed_data:
                error_msg = "No processed data found in environment"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)

            # Get analysis results
            analysis_results = environment.get("Analysis Results", {})
            findings = analysis_results.get("findings", {})

            # If data_reference is provided, filter to only that dataset
            if data_reference and data_reference in processed_data:
                datasets_to_process = {data_reference: processed_data[data_reference]}
            else:
                datasets_to_process = processed_data

            # Process each dataset
            for dataset_name, dataset_info in datasets_to_process.items():
                try:
                    # Skip datasets with errors
                    if dataset_info.get("type") == "error":
                        error_msg = f"Skipping dataset {dataset_name} due to processing error: {dataset_info.get('error', 'Unknown error')}"
                        self.logger.warning(error_msg)
                        trained_models[dataset_name] = {"error": error_msg}
                        performance_metrics[dataset_name] = {"error": error_msg}
                        continue

                    # Validate dataset structure
                    if not self._validate_dataset(dataset_info):
                        error_msg = f"Dataset {dataset_name} has invalid structure"
                        self.logger.error(error_msg)
                        trained_models[dataset_name] = {"error": error_msg}
                        performance_metrics[dataset_name] = {"error": error_msg}
                        continue

                    # Get dataset findings
                    dataset_findings = findings.get(dataset_name, {})

                    # Determine target variable
                    target = self._determine_target_variable(target_variable, dataset_info, dataset_findings)
                    if not target:
                        error_msg = f"Could not determine target variable for dataset {dataset_name}"
                        self.logger.error(error_msg)
                        trained_models[dataset_name] = {"error": error_msg}
                        performance_metrics[dataset_name] = {"error": error_msg}
                        continue

                    # Determine appropriate model types based on data characteristics and goals
                    model_types = self._determine_model_types(
                        dataset_info, 
                        dataset_findings, 
                        goals, 
                        requested_model_type
                    )

                    # Initialize results for this dataset
                    trained_models[dataset_name] = {}
                    performance_metrics[dataset_name] = {}

                    # Train models
                    for model_type in model_types:
                        try:
                            self.logger.info(f"Training {model_type} model for {dataset_name}")
                            
                            # Get features for this model
                            features = self._select_features(
                                dataset_info, 
                                dataset_findings, 
                                target, 
                                model_type
                            )
                            
                            # Get hyperparameters
                            hyperparameters = self._get_hyperparameters(model_type, dataset_info)
                            
                            # Train the model and get performance metrics
                            model_result, model_metrics = self._train_and_evaluate_model(
                                model_type, 
                                dataset_info, 
                                features, 
                                target, 
                                hyperparameters
                            )
                            
                            # Store results
                            trained_models[dataset_name][model_type] = model_result
                            performance_metrics[dataset_name][model_type] = model_metrics
                            
                        except Exception as model_error:
                            error_msg = f"Error training {model_type} model for {dataset_name}: {str(model_error)}"
                            self.logger.error(error_msg)
                            self.logger.debug(traceback.format_exc())
                            trained_models[dataset_name][model_type] = {"error": error_msg}
                            performance_metrics[dataset_name][model_type] = {"error": error_msg}

                    # Determine best model if we have any successful models
                    successful_models = {k: v for k, v in performance_metrics[dataset_name].items() 
                                        if not isinstance(v, dict) or "error" not in v}
                    
                    if successful_models:
                        best_model = self._determine_best_model(successful_models)
                        performance_metrics[dataset_name]["best_model"] = best_model
                    else:
                        performance_metrics[dataset_name]["best_model"] = None
                        self.logger.warning(f"No successful models trained for {dataset_name}")

                except Exception as dataset_error:
                    error_msg = f"Error processing dataset {dataset_name}: {str(dataset_error)}"
                    self.logger.error(error_msg)
                    self.logger.debug(traceback.format_exc())
                    trained_models[dataset_name] = {"error": error_msg}
                    performance_metrics[dataset_name] = {"error": error_msg}

            # Return the modeling results
            return {
                "Models.trained_model": trained_models,
                "Models.performance": performance_metrics,
            }
            
        except Exception as e:
            error_msg = f"Error in modeling task: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return self._create_error_response(error_msg)

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "Models.trained_model": {"error": error_msg},
            "Models.performance": {"error": error_msg},
            "error": error_msg
        }

    def _validate_dataset(self, dataset_info: Dict[str, Any]) -> bool:
        """
        Validate that the dataset has the required structure.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            True if valid, False otherwise
        """
        # Check if dataset has data
        if not dataset_info:
            return False
            
        # Check for required fields based on dataset type
        dataset_type = dataset_info.get("type")
        
        if dataset_type == "tabular":
            # For tabular data, check for columns and data
            return "columns" in dataset_info and "data" in dataset_info
        elif dataset_type == "time_series":
            # For time series, check for time column and data
            return "time_column" in dataset_info and "data" in dataset_info
        elif dataset_type == "text":
            # For text data, check for text field
            return "text_field" in dataset_info and "data" in dataset_info
        else:
            # Unknown dataset type
            return False

    def _determine_target_variable(
        self, 
        target_variable: Optional[str], 
        dataset_info: Dict[str, Any],
        dataset_findings: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine the target variable for modeling.
        
        Args:
            target_variable: User-specified target variable
            dataset_info: Dataset information
            dataset_findings: Findings from analysis
            
        Returns:
            Target variable name or None if it cannot be determined
        """
        # If target variable is explicitly provided, use it
        if target_variable:
            # Verify it exists in the dataset
            if dataset_info.get("type") == "tabular" and target_variable in dataset_info.get("columns", []):
                return target_variable
            else:
                self.logger.warning(f"Specified target variable '{target_variable}' not found in dataset")
                
        # Try to get target from dataset findings
        if "target_variable" in dataset_findings:
            return dataset_findings["target_variable"]
            
        # Try to get from dataset info
        if "target" in dataset_info:
            return dataset_info["target"]
            
        # Look for common target variable names
        common_targets = ["target", "label", "y", "class", "outcome", "result"]
        if dataset_info.get("type") == "tabular":
            columns = dataset_info.get("columns", [])
            for target in common_targets:
                if target in columns:
                    return target
                    
        # If we have key variables identified, use the last one as target
        key_variables = dataset_findings.get("key_variables", [])
        if key_variables:
            return key_variables[-1]
            
        return None

    def _determine_model_types(
        self, 
        dataset_info: Dict[str, Any],
        dataset_findings: Dict[str, Any],
        goals: List[str],
        requested_model_type: Optional[str]
    ) -> List[str]:
        """
        Determine appropriate model types based on data characteristics and goals.
        
        Args:
            dataset_info: Dataset information
            dataset_findings: Findings from analysis
            goals: User-defined goals
            requested_model_type: User-requested model type
            
        Returns:
            List of model types to try
        """
        # If model type is explicitly requested, use it
        if requested_model_type:
            return [requested_model_type]
            
        # Determine task type from goals or findings
        task_type = self._determine_task_type(goals, dataset_findings)
        
        # Get data characteristics
        data_type = dataset_info.get("type", "tabular")
        
        # Select models based on task and data type
        if task_type == "regression":
            if data_type == "time_series":
                return ["arima", "prophet", "linear_regression", "gradient_boosting"]
            else:
                return ["linear_regression", "random_forest", "gradient_boosting", "xgboost"]
        elif task_type == "classification":
            if dataset_findings.get("is_binary_classification", False):
                return ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
            else:
                return ["random_forest", "gradient_boosting", "xgboost", "multinomial_logistic"]
        elif task_type == "clustering":
            return ["kmeans", "hierarchical", "dbscan"]
        elif task_type == "time_series":
            return ["arima", "prophet", "lstm"]
        else:
            # Default to common models
            return ["linear_regression", "random_forest", "gradient_boosting"]

    def _determine_task_type(self, goals: List[str], dataset_findings: Dict[str, Any]) -> str:
        """
        Determine the type of modeling task from goals and findings.
        
        Args:
            goals: User-defined goals
            dataset_findings: Findings from analysis
            
        Returns:
            Task type (regression, classification, clustering, time_series)
        """
        # Check findings first
        if "task_type" in dataset_findings:
            return dataset_findings["task_type"]
            
        # Check target variable type
        if "target_type" in dataset_findings:
            target_type = dataset_findings["target_type"]
            if target_type == "continuous":
                return "regression"
            elif target_type in ["categorical", "binary"]:
                return "classification"
                
        # Look for keywords in goals
        goal_text = " ".join(goals).lower()
        if any(kw in goal_text for kw in ["regress", "predict value", "estimate", "forecast"]):
            return "regression"
        elif any(kw in goal_text for kw in ["classif", "categorize", "identify", "detect"]):
            return "classification"
        elif any(kw in goal_text for kw in ["cluster", "segment", "group"]):
            return "clustering"
        elif any(kw in goal_text for kw in ["time series", "temporal", "forecast"]):
            return "time_series"
            
        # Default to regression
        return "regression"

    def _select_features(
        self, 
        dataset_info: Dict[str, Any],
        dataset_findings: Dict[str, Any],
        target: str,
        model_type: str
    ) -> List[str]:
        """
        Select features for model training.
        
        Args:
            dataset_info: Dataset information
            dataset_findings: Findings from analysis
            target: Target variable
            model_type: Type of model
            
        Returns:
            List of feature names
        """
        # Get all available features
        if dataset_info.get("type") == "tabular":
            all_features = [col for col in dataset_info.get("columns", []) if col != target]
        else:
            all_features = dataset_info.get("features", [])
            
        # If no features available, return empty list
        if not all_features:
            return []
            
        # Use important features from findings if available
        important_features = dataset_findings.get("important_features", [])
        if important_features:
            return important_features
            
        # Use key variables if available
        key_variables = dataset_findings.get("key_variables", [])
        if key_variables:
            # Remove target from key variables if present
            return [var for var in key_variables if var != target]
            
        # Use all features as fallback
        return all_features

    def _get_hyperparameters(self, model_type: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get hyperparameters for a model type, potentially adjusted for dataset characteristics.
        
        Args:
            model_type: Type of model
            dataset_info: Dataset information
            
        Returns:
            Dictionary of hyperparameters
        """
        # Get dataset size to adjust hyperparameters
        data_size = len(dataset_info.get("data", [])) if "data" in dataset_info else 1000
        
        # Base hyperparameters by model type
        if model_type == "linear_regression":
            return {"fit_intercept": True, "normalize": False}
            
        elif model_type == "logistic_regression":
            return {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000 if data_size > 10000 else 500
            }
            
        elif model_type == "random_forest":
            # Adjust n_estimators based on dataset size
            n_estimators = min(100, max(10, data_size // 100))
            return {
                "n_estimators": n_estimators,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "bootstrap": True
            }
            
        elif model_type == "gradient_boosting":
            # Adjust learning rate based on dataset size
            learning_rate = 0.05 if data_size > 10000 else 0.1
            return {
                "n_estimators": 100,
                "learning_rate": learning_rate,
                "max_depth": 3,
                "subsample": 0.8,
                "min_samples_split": 2
            }
            
        elif model_type == "xgboost":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
            
        elif model_type == "kmeans":
            return {"n_clusters": 3, "init": "k-means++", "n_init": 10}
            
        elif model_type == "arima":
            return {"p": 1, "d": 1, "q": 1}
            
        elif model_type == "prophet":
            return {"seasonality_mode": "multiplicative", "yearly_seasonality": True}
            
        else:
            return {}

    def _train_and_evaluate_model(
        self, 
        model_type: str,
        dataset_info: Dict[str, Any],
        features: List[str],
        target: str,
        hyperparameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train and evaluate a model.
        
        Args:
            model_type: Type of model
            dataset_info: Dataset information
            features: List of feature names
            target: Target variable name
            hyperparameters: Model hyperparameters
            
        Returns:
            Tuple of (model_result, model_metrics)
        """
        # In a real implementation, this would train actual models
        # For now, we'll create structured metadata about the model
        
        # Create model metadata
        model_result = {
            "type": model_type,
            "features": features,
            "target": target,
            "hyperparameters": hyperparameters,
            "training_time": "N/A",  # Would be actual training time
            "timestamp": datetime.now().isoformat(),
            "data_shape": self._get_data_shape(dataset_info)
        }
        
        # Create performance metrics
        if "regression" in model_type or model_type in ["linear_regression", "arima", "prophet"]:
            metrics = {
                "metrics": {
                    "r2": None,  # Would be actual R2
                    "mse": None,  # Would be actual MSE
                    "mae": None,  # Would be actual MAE
                    "rmse": None  # Would be actual RMSE
                },
                "cross_validation": {
                    "method": "5-fold",
                    "r2_scores": None  # Would be actual CV scores
                }
            }
        else:
            metrics = {
                "metrics": {
                    "accuracy": None,  # Would be actual accuracy
                    "precision": None,  # Would be actual precision
                    "recall": None,  # Would be actual recall
                    "f1": None  # Would be actual F1
                },
                "cross_validation": {
                    "method": "5-fold",
                    "accuracy_scores": None  # Would be actual CV scores
                }
            }
        
        return model_result, metrics

    def _get_data_shape(self, dataset_info: Dict[str, Any]) -> Dict[str, int]:
        """Get the shape of the dataset."""
        if "data" in dataset_info:
            rows = len(dataset_info["data"])
            cols = len(dataset_info.get("columns", [])) if dataset_info.get("type") == "tabular" else 0
            return {"rows": rows, "columns": cols}
        return {"rows": 0, "columns": 0}

    def _determine_best_model(self, model_metrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Determine the best model based on performance metrics.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            Name of the best model
        """
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in model_metrics.items():
            # Skip models with errors
            if isinstance(metrics, dict) and "error" in metrics:
                continue
                
            # Get the appropriate metric based on model type
            if "regression" in model_name or model_name in ["linear_regression", "arima", "prophet"]:
                # For regression, use R2 (higher is better)
                score = metrics.get("metrics", {}).get("r2", -float('inf'))
            else:
                # For classification, use F1 (higher is better)
                score = metrics.get("metrics", {}).get("f1", -float('inf'))
                
            # Update best model if this one is better
            if score > best_score:
                best_score = score
                best_model = model_name
                
        return best_model
