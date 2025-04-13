"""
Visualization Task Agent Module

This module defines the visualization task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import json

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class VisualizationTaskAgent(BaseTaskAgent):
    """
    Task agent for creating data visualizations.

    Responsibilities:
    - Creating charts and graphs
    - Creating interactive visualizations
    - Creating dashboards
    - Visualizing model results
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the VisualizationTaskAgent."""
        super().__init__(name="VisualizationAgent", llm=llm)

    def _validate_data(self, data: Any) -> bool:
        """
        Validate that data exists and is in a usable format.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None:
            return False
            
        if isinstance(data, dict) and not data:
            return False
            
        if isinstance(data, list) and not data:
            return False
            
        return True
    
    def _get_column_names(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract column names from dataset info.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            List of column names
        """
        # Try different ways to get column names
        if "columns" in dataset_info:
            return dataset_info["columns"]
        elif "schema" in dataset_info:
            return list(dataset_info["schema"].keys())
        elif "sample" in dataset_info and isinstance(dataset_info["sample"], dict):
            return list(dataset_info["sample"].keys())
        elif "data" in dataset_info and isinstance(dataset_info["data"], dict):
            return list(dataset_info["data"].keys())
        
        # Default to empty list if no columns found
        return []
    
    def _get_numeric_columns(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract numeric column names from dataset info.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            List of numeric column names
        """
        numeric_columns = []
        
        # Try to get schema information
        schema = dataset_info.get("schema", {})
        if schema:
            for col, col_info in schema.items():
                col_type = col_info.get("type", "").lower()
                if any(num_type in col_type for num_type in ["int", "float", "double", "numeric", "number"]):
                    numeric_columns.append(col)
            
            if numeric_columns:
                return numeric_columns
        
        # If no schema or no numeric columns found, try to infer from sample data
        sample = dataset_info.get("sample", {})
        if sample:
            for col, value in sample.items():
                try:
                    float(value)  # Try to convert to float
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        return numeric_columns
    
    def _get_categorical_columns(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract categorical column names from dataset info.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            List of categorical column names
        """
        categorical_columns = []
        
        # Try to get schema information
        schema = dataset_info.get("schema", {})
        if schema:
            for col, col_info in schema.items():
                col_type = col_info.get("type", "").lower()
                if any(cat_type in col_type for cat_type in ["category", "string", "object", "bool", "enum"]):
                    categorical_columns.append(col)
            
            if categorical_columns:
                return categorical_columns
        
        # If no schema or no categorical columns found, try to infer from sample data
        sample = dataset_info.get("sample", {})
        if sample:
            for col, value in sample.items():
                if isinstance(value, (str, bool)) or (isinstance(value, int) and value in [0, 1]):
                    categorical_columns.append(col)
        
        return categorical_columns
    
    def _get_datetime_columns(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract datetime column names from dataset info.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            List of datetime column names
        """
        datetime_columns = []
        
        # Try to get schema information
        schema = dataset_info.get("schema", {})
        if schema:
            for col, col_info in schema.items():
                col_type = col_info.get("type", "").lower()
                if any(dt_type in col_type for dt_type in ["date", "time", "datetime", "timestamp"]):
                    datetime_columns.append(col)
            
            if datetime_columns:
                return datetime_columns
        
        # If no schema or no datetime columns found, try to infer from column names
        all_columns = self._get_column_names(dataset_info)
        for col in all_columns:
            col_lower = col.lower()
            if any(dt_term in col_lower for dt_term in ["date", "time", "year", "month", "day", "timestamp"]):
                datetime_columns.append(col)
        
        return datetime_columns
    
    def _create_data_exploration_visualizations(
        self, dataset_name: str, dataset_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create data exploration visualizations based on dataset info.
        
        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset information
            
        Returns:
            List of visualization specifications
        """
        visualizations = []
        
        # Get column information
        all_columns = self._get_column_names(dataset_info)
        numeric_columns = self._get_numeric_columns(dataset_info)
        categorical_columns = self._get_categorical_columns(dataset_info)
        datetime_columns = self._get_datetime_columns(dataset_info)
        
        # Create histograms for numeric columns
        for i, col in enumerate(numeric_columns[:3]):  # Limit to first 3 numeric columns
            visualizations.append({
                "type": "histogram",
                "title": f"Distribution of {col}",
                "x_axis": col,
                "y_axis": "frequency",
                "description": f"Histogram showing the distribution of values in {col}",
            })
        
        # Create box plots for numeric columns
        if len(numeric_columns) >= 2:
            visualizations.append({
                "type": "box_plot",
                "title": "Box Plot of Numeric Features",
                "features": numeric_columns[:5],  # Limit to first 5 numeric columns
                "description": "Box plot showing the distribution of numeric features",
            })
        
        # Create bar charts for categorical columns
        for i, col in enumerate(categorical_columns[:3]):  # Limit to first 3 categorical columns
            visualizations.append({
                "type": "bar_chart",
                "title": f"Category Distribution in {col}",
                "x_axis": col,
                "y_axis": "count",
                "description": f"Bar chart showing the distribution of categories in {col}",
            })
        
        # Create time series plots for datetime columns
        for i, date_col in enumerate(datetime_columns[:2]):  # Limit to first 2 datetime columns
            if numeric_columns:  # Need at least one numeric column for y-axis
                visualizations.append({
                    "type": "line_chart",
                    "title": f"Time Series of {numeric_columns[0]} over {date_col}",
                    "x_axis": date_col,
                    "y_axis": numeric_columns[0],
                    "description": f"Line chart showing {numeric_columns[0]} over time ({date_col})",
                })
        
        # If no visualizations could be created, add a placeholder
        if not visualizations:
            visualizations.append({
                "type": "text",
                "title": f"No Visualizations for {dataset_name}",
                "content": "Could not create visualizations due to insufficient or incompatible data",
                "description": "Placeholder for missing visualizations",
            })
        
        return visualizations
    
    def _create_analysis_visualizations(
        self, dataset_name: str, dataset_info: Dict[str, Any], dataset_insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create analysis visualizations based on dataset insights.
        
        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset information
            dataset_insights: Insights for the dataset
            
        Returns:
            List of visualization specifications
        """
        visualizations = []
        
        # Get column information
        numeric_columns = self._get_numeric_columns(dataset_info)
        datetime_columns = self._get_datetime_columns(dataset_info)
        
        # Create visualizations based on insights
        for insight in dataset_insights:
            insight_type = insight.get("type", "")
            
            if insight_type == "correlation" and len(numeric_columns) >= 2:
                # Get correlation variables from insight or use defaults
                x_col = insight.get("x_variable", numeric_columns[0])
                y_col = insight.get("y_variable", numeric_columns[1])
                
                # Ensure columns exist in the dataset
                if x_col not in numeric_columns:
                    x_col = numeric_columns[0]
                if y_col not in numeric_columns or y_col == x_col:
                    y_col = next((col for col in numeric_columns if col != x_col), numeric_columns[0])
                
                visualizations.append({
                    "type": "scatter_plot",
                    "title": f"Correlation between {x_col} and {y_col}",
                    "x_axis": x_col,
                    "y_axis": y_col,
                    "trend_line": True,
                    "description": insight.get("description", f"Scatter plot showing relationship between {x_col} and {y_col}"),
                })
                
            elif insight_type == "time_series" and datetime_columns and numeric_columns:
                # Get time series variables from insight or use defaults
                date_col = insight.get("time_variable", datetime_columns[0])
                value_col = insight.get("value_variable", numeric_columns[0])
                
                # Ensure columns exist in the dataset
                if date_col not in datetime_columns and datetime_columns:
                    date_col = datetime_columns[0]
                if value_col not in numeric_columns and numeric_columns:
                    value_col = numeric_columns[0]
                
                visualizations.append({
                    "type": "line_chart",
                    "title": f"Time Series Analysis of {value_col} over {date_col}",
                    "x_axis": date_col,
                    "y_axis": value_col,
                    "description": insight.get("description", f"Line chart showing {value_col} over time ({date_col})"),
                })
                
            elif insight_type == "distribution" and numeric_columns:
                # Get distribution variable from insight or use default
                value_col = insight.get("variable", numeric_columns[0])
                
                # Ensure column exists in the dataset
                if value_col not in numeric_columns and numeric_columns:
                    value_col = numeric_columns[0]
                
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution Analysis of {value_col}",
                    "x_axis": value_col,
                    "y_axis": "frequency",
                    "description": insight.get("description", f"Histogram showing the distribution of {value_col}"),
                })
        
        # If no visualizations could be created from insights, create a correlation matrix if possible
        if not visualizations and len(numeric_columns) >= 2:
            visualizations.append({
                "type": "heatmap",
                "title": "Correlation Matrix",
                "features": numeric_columns[:8],  # Limit to first 8 numeric columns
                "description": "Heatmap showing correlations between numeric features",
            })
        
        return visualizations
    
    def _create_model_performance_visualizations(
        self, 
        dataset_name: str, 
        dataset_performance: Dict[str, Any],
        trained_models: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create model performance visualizations.
        
        Args:
            dataset_name: Name of the dataset
            dataset_performance: Performance metrics for models
            trained_models: Trained model information
            
        Returns:
            List of visualization specifications
        """
        visualizations = []
        
        # Check if we have valid performance data
        if not self._validate_data(dataset_performance):
            return visualizations
        
        # Get best model if available
        best_model = dataset_performance.get("best_model")
        
        # Create model comparison visualization if we have multiple models
        models = [k for k in dataset_performance.keys() if k != "best_model"]
        if models:
            # Determine metrics based on model type
            metrics = []
            if best_model and "regression" in best_model:
                metrics = ["r2", "mse", "mae", "rmse"]
            else:
                metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            
            # Filter to metrics that are actually present in the data
            available_metrics = set()
            for model in models:
                model_metrics = dataset_performance.get(model, {})
                available_metrics.update(model_metrics.keys())
            
            metrics = [m for m in metrics if m in available_metrics]
            
            if metrics:  # Only create visualization if we have metrics
                visualizations.append({
                    "type": "bar_chart",
                    "title": "Model Performance Comparison",
                    "x_axis": "model",
                    "y_axis": "metric_value",
                    "models": models,
                    "metrics": metrics,
                    "description": (
                        f"Comparison of performance metrics across different models"
                        + (f", with {best_model} performing best" if best_model else "")
                    ),
                })
        
        # Create feature importance visualization for the best model if available
        if best_model and trained_models:
            dataset_models = trained_models.get(dataset_name, {})
            best_model_info = dataset_models.get(best_model, {})
            features = best_model_info.get("features", [])
            
            if features:
                visualizations.append({
                    "type": "bar_chart",
                    "title": f"Feature Importance for {best_model}",
                    "x_axis": "feature",
                    "y_axis": "importance",
                    "features": features,
                    "description": "Relative importance of features in the best performing model",
                })
        
        return visualizations

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the visualization task.

        Args:
            input_data: Input data for the task

        Returns:
            Dict containing visualizations
        """
        # Extract input data with proper validation
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Get specific plot parameters if provided
        plot_type = input_data.get("plot_type")
        plot_params = input_data.get("plot_params", {})
        data_reference = input_data.get("data_reference")
        
        # Log the start of visualization creation
        self.logger.info("Creating visualizations")
        
        try:
            # Get data from the environment with proper validation
            cleaned_data = environment.get("Cleaned Data", {})
            processed_data = cleaned_data.get("processed_data", {})
            
            # If no processed data is found, try to find data in other locations
            if not processed_data and data_reference:
                # Try to find the referenced data in the environment
                if data_reference in environment:
                    processed_data = {data_reference: environment[data_reference]}
                elif "." in data_reference:
                    # Handle nested references like "Cleaned Data.dataset1"
                    parts = data_reference.split(".")
                    current = environment
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            break
                    if current != environment:
                        processed_data = {"referenced_data": current}
            
            # If still no data, check other common locations
            if not processed_data:
                for key in ["Data", "loaded_data", "cleaned_data", "data"]:
                    if key in environment and environment[key]:
                        processed_data = {key: environment[key]}
                        break
            
            # Validate that we have data to work with
            if not self._validate_data(processed_data):
                self.logger.warning("No valid data found for visualization")
                return {
                    "Visualization.error": "No valid data found for visualization",
                    "Visualization.plot_path": None,
                    "Visualization.description": "Failed to create visualization due to missing data"
                }
            
            # Get analysis results and model information
            analysis_results = environment.get("Analysis Results", {})
            insights = analysis_results.get("insights", {})
            findings = analysis_results.get("findings", {})
            
            models = environment.get("Models", {})
            trained_models = models.get("trained_model", {})
            performance = models.get("performance", {})
            
            # Initialize results
            plots = {}
            dashboard = {
                "title": "Data Analysis Dashboard",
                "sections": [],
                "layout": "grid",
                "theme": "light",
            }
            
            # If a specific plot type is requested, create just that plot
            if plot_type and data_reference:
                # Create a single visualization based on the requested type
                self.logger.info(f"Creating specific {plot_type} visualization for {data_reference}")
                
                # Find the dataset info
                dataset_info = None
                if data_reference in processed_data:
                    dataset_info = processed_data[data_reference]
                else:
                    # Try to find the dataset in the processed data
                    for name, info in processed_data.items():
                        if name == data_reference or data_reference in name:
                            dataset_info = info
                            data_reference = name
                            break
                
                if not dataset_info:
                    self.logger.warning(f"Dataset {data_reference} not found")
                    return {
                        "Visualization.error": f"Dataset {data_reference} not found",
                        "Visualization.plot_path": None,
                        "Visualization.description": f"Failed to create {plot_type} visualization due to missing dataset"
                    }
                
                # Create the specific visualization
                visualization = None
                
                if plot_type == "histogram":
                    column = plot_params.get("column")
                    if not column:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        column = numeric_columns[0] if numeric_columns else None
                    
                    if column:
                        visualization = {
                            "type": "histogram",
                            "title": f"Distribution of {column}",
                            "x_axis": column,
                            "y_axis": "frequency",
                            "bins": plot_params.get("bins", 10),
                            "description": f"Histogram showing the distribution of values in {column}"
                        }
                
                elif plot_type == "scatter_plot":
                    x_column = plot_params.get("x_column")
                    y_column = plot_params.get("y_column")
                    
                    if not x_column or not y_column:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        if len(numeric_columns) >= 2:
                            x_column = x_column or numeric_columns[0]
                            y_column = y_column or numeric_columns[1]
                    
                    if x_column and y_column:
                        visualization = {
                            "type": "scatter_plot",
                            "title": f"Scatter Plot of {y_column} vs {x_column}",
                            "x_axis": x_column,
                            "y_axis": y_column,
                            "trend_line": plot_params.get("trend_line", True),
                            "description": f"Scatter plot showing relationship between {x_column} and {y_column}"
                        }
                
                elif plot_type == "bar_chart":
                    x_column = plot_params.get("x_column")
                    y_column = plot_params.get("y_column")
                    
                    if not x_column:
                        categorical_columns = self._get_categorical_columns(dataset_info)
                        x_column = categorical_columns[0] if categorical_columns else None
                    
                    if not y_column:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        y_column = numeric_columns[0] if numeric_columns else "count"
                    
                    if x_column:
                        visualization = {
                            "type": "bar_chart",
                            "title": f"Bar Chart of {y_column} by {x_column}",
                            "x_axis": x_column,
                            "y_axis": y_column,
                            "description": f"Bar chart showing {y_column} for each category in {x_column}"
                        }
                
                elif plot_type == "line_chart":
                    x_column = plot_params.get("x_column")
                    y_column = plot_params.get("y_column")
                    
                    if not x_column:
                        datetime_columns = self._get_datetime_columns(dataset_info)
                        x_column = datetime_columns[0] if datetime_columns else None
                    
                    if not y_column:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        y_column = numeric_columns[0] if numeric_columns else None
                    
                    if x_column and y_column:
                        visualization = {
                            "type": "line_chart",
                            "title": f"Line Chart of {y_column} over {x_column}",
                            "x_axis": x_column,
                            "y_axis": y_column,
                            "description": f"Line chart showing {y_column} over {x_column}"
                        }
                
                elif plot_type == "box_plot":
                    features = plot_params.get("features")
                    
                    if not features:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        features = numeric_columns[:5] if numeric_columns else None
                    
                    if features:
                        visualization = {
                            "type": "box_plot",
                            "title": "Box Plot of Numeric Features",
                            "features": features,
                            "description": "Box plot showing the distribution of numeric features"
                        }
                
                elif plot_type == "heatmap":
                    features = plot_params.get("features")
                    
                    if not features:
                        numeric_columns = self._get_numeric_columns(dataset_info)
                        features = numeric_columns[:8] if numeric_columns else None
                    
                    if features:
                        visualization = {
                            "type": "heatmap",
                            "title": "Correlation Matrix",
                            "features": features,
                            "description": "Heatmap showing correlations between numeric features"
                        }
                
                # Return the single visualization
                if visualization:
                    return {
                        "Visualization.plot_path": f"visualizations/{data_reference}_{plot_type}.png",
                        "Visualization.description": visualization["description"],
                        "Visualization.details": json.dumps(visualization)
                    }
                else:
                    return {
                        "Visualization.error": f"Could not create {plot_type} visualization with the provided parameters",
                        "Visualization.plot_path": None,
                        "Visualization.description": f"Failed to create {plot_type} visualization"
                    }
            
            # Process each dataset for comprehensive visualization
            for dataset_name, dataset_info in processed_data.items():
                try:
                    # Skip datasets with errors or invalid data
                    if not self._validate_data(dataset_info) or dataset_info.get("type") == "error":
                        self.logger.warning(
                            f"Skipping dataset {dataset_name} due to invalid data or error"
                        )
                        continue
                    
                    # Get dataset insights and findings
                    dataset_insights = insights.get(dataset_name, [])
                    dataset_findings = findings.get(dataset_name, {})
                    
                    # Initialize plots for this dataset
                    plots[dataset_name] = {
                        "data_exploration": [],
                        "analysis": [],
                        "model_performance": [],
                    }
                    
                    # Create data exploration visualizations
                    plots[dataset_name]["data_exploration"] = self._create_data_exploration_visualizations(
                        dataset_name, dataset_info
                    )
                    
                    # Create analysis visualizations
                    plots[dataset_name]["analysis"] = self._create_analysis_visualizations(
                        dataset_name, dataset_info, dataset_insights
                    )
                    
                    # Create model performance visualizations if applicable
                    if dataset_name in performance:
                        dataset_performance = performance.get(dataset_name, {})
                        plots[dataset_name]["model_performance"] = self._create_model_performance_visualizations(
                            dataset_name, dataset_performance, trained_models
                        )
                    
                    # Add dataset section to dashboard
                    dashboard_plots = []
                    
                    # Add data exploration plot if available
                    if plots[dataset_name]["data_exploration"]:
                        dashboard_plots.append({
                            "id": "distribution",
                            "title": "Data Distribution",
                            "plot_ref": f"{dataset_name}.data_exploration.0",
                        })
                    
                    # Add analysis plot if available
                    if plots[dataset_name]["analysis"]:
                        dashboard_plots.append({
                            "id": "correlation",
                            "title": "Feature Correlations",
                            "plot_ref": f"{dataset_name}.analysis.0",
                        })
                    
                    # Add model performance plot if available
                    if plots[dataset_name]["model_performance"]:
                        dashboard_plots.append({
                            "id": "model_comparison",
                            "title": "Model Comparison",
                            "plot_ref": f"{dataset_name}.model_performance.0",
                        })
                    
                    # Add section to dashboard if we have plots
                    if dashboard_plots:
                        dashboard["sections"].append({
                            "title": f"Analysis of {dataset_name}",
                            "plots": dashboard_plots,
                            "summary": dataset_findings.get("summary", ""),
                        })
                
                except Exception as e:
                    self.logger.error(
                        f"Error creating visualizations for dataset {dataset_name}: {str(e)}"
                    )
                    plots[dataset_name] = {"error": str(e)}
            
            # Add summary section to dashboard if we have findings
            key_findings = []
            for finding in findings.values():
                if isinstance(finding, dict) and "error" not in finding:
                    summary = finding.get("summary", "")
                    if summary:
                        key_findings.append(summary)
            
            if key_findings or dashboard["sections"]:
                dashboard["sections"].append({
                    "title": "Executive Summary",
                    "content": "This dashboard presents the results of our data analysis and modeling efforts.",
                    "key_findings": key_findings,
                })
            
            # Return the visualizations
            return {"Visualizations.plots": plots, "Visualizations.dashboard": dashboard}
            
        except Exception as e:
            self.logger.error(f"Error in visualization task agent: {str(e)}")
            return {
                "Visualization.error": str(e),
                "Visualization.plot_path": None,
                "Visualization.description": "Failed to create visualizations due to an error"
            }
