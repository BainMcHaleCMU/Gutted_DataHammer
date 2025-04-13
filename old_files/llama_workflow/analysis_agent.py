"""
Analysis Task Agent Module

This module defines the analysis task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import traceback
from collections import defaultdict

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class AnalysisTaskAgent(BaseTaskAgent):
    """
    Task agent for in-depth data analysis.

    Responsibilities:
    - Advanced statistical analysis
    - Hypothesis testing
    - Segmentation analysis
    - Time series analysis
    - Pattern discovery
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the AnalysisTaskAgent."""
        super().__init__(name="AnalysisAgent", llm=llm)

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the input data for the analysis task.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required keys
        if "environment" not in input_data:
            return False, "Missing 'environment' in input data"
            
        environment = input_data.get("environment", {})
        
        # Check for data to analyze
        if "Cleaned Data" not in environment and "Data Overview" not in environment:
            return False, "No data found in environment. Need either 'Cleaned Data' or 'Data Overview'"
            
        # If we have cleaned data, check for processed_data
        if "Cleaned Data" in environment:
            cleaned_data = environment["Cleaned Data"]
            if not isinstance(cleaned_data, dict):
                return False, "Invalid 'Cleaned Data' format, expected dictionary"
                
            if "processed_data" not in cleaned_data:
                return False, "No 'processed_data' found in 'Cleaned Data'"
                
            if not cleaned_data["processed_data"]:
                return False, "Empty 'processed_data' in 'Cleaned Data'"
                
        return True, ""

    def get_data_from_environment(self, environment: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Extract relevant data from the environment.
        
        Args:
            environment: The shared environment state
            
        Returns:
            Tuple of (processed_data, cleaning_steps, statistics)
        """
        # Get cleaned data from the environment
        cleaned_data = environment.get("Cleaned Data", {})
        processed_data = cleaned_data.get("processed_data", {})
        cleaning_steps = cleaned_data.get("cleaning_steps", {})

        # Get data overview information
        data_overview = environment.get("Data Overview", {})
        statistics = data_overview.get("statistics", {})
        
        return processed_data, cleaning_steps, statistics

    def analyze_numeric_data(self, dataset_name: str, numeric_columns: Dict[str, Any], dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze numeric columns and generate insights.
        
        Args:
            dataset_name: Name of the dataset
            numeric_columns: Dictionary of numeric columns and their statistics
            dataset_info: Information about the dataset
            
        Returns:
            List of insights about numeric data
        """
        insights = []
        
        if not numeric_columns:
            return insights
            
        # Add general numeric data insights
        insights.append({
            "type": "numeric_summary",
            "description": f"Dataset contains {len(numeric_columns)} numeric columns",
            "importance": "medium",
        })
        
        # Analyze each numeric column
        for col_name, col_stats in numeric_columns.items():
            # Skip if no statistics available
            if not col_stats:
                continue
                
            # Check for missing values
            missing_pct = col_stats.get("missing_percentage", 0)
            if missing_pct > 0:
                insights.append({
                    "type": "data_quality",
                    "description": f"Column '{col_name}' has {missing_pct:.1f}% missing values",
                    "importance": "high" if missing_pct > 20 else "medium",
                })
                
            # Check for outliers
            if "outliers" in col_stats and col_stats["outliers"]:
                insights.append({
                    "type": "outliers",
                    "description": f"Column '{col_name}' contains outliers",
                    "details": f"Outlier count: {col_stats.get('outlier_count', 'unknown')}",
                    "importance": "high",
                })
                
            # Check distribution
            if "skewness" in col_stats:
                skew = col_stats["skewness"]
                if abs(skew) > 1:
                    skew_type = "right" if skew > 0 else "left"
                    insights.append({
                        "type": "distribution",
                        "description": f"Column '{col_name}' shows a {skew_type}-skewed distribution",
                        "details": f"Skewness: {skew:.2f}",
                        "importance": "medium",
                    })
        
        # Analyze correlations if available
        if "correlations" in dataset_info:
            correlations = dataset_info["correlations"]
            strong_correlations = []
            
            for col1, corr_data in correlations.items():
                for col2, corr_value in corr_data.items():
                    if col1 != col2 and abs(corr_value) > 0.7:
                        corr_type = "positive" if corr_value > 0 else "negative"
                        strong_correlations.append((col1, col2, corr_value, corr_type))
            
            # Add insights for strong correlations
            for col1, col2, corr_value, corr_type in strong_correlations[:3]:  # Limit to top 3
                insights.append({
                    "type": "correlation",
                    "description": f"Strong {corr_type} correlation detected between '{col1}' and '{col2}'",
                    "details": f"Correlation coefficient: {corr_value:.2f}",
                    "importance": "high",
                })
                
        return insights

    def analyze_categorical_data(self, dataset_name: str, categorical_columns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze categorical columns and generate insights.
        
        Args:
            dataset_name: Name of the dataset
            categorical_columns: Dictionary of categorical columns and their statistics
            
        Returns:
            List of insights about categorical data
        """
        insights = []
        
        if not categorical_columns:
            return insights
            
        # Add general categorical data insights
        insights.append({
            "type": "categorical_summary",
            "description": f"Dataset contains {len(categorical_columns)} categorical columns",
            "importance": "medium",
        })
        
        # Analyze each categorical column
        for col_name, col_stats in categorical_columns.items():
            # Skip if no statistics available
            if not col_stats:
                continue
                
            # Check for missing values
            missing_pct = col_stats.get("missing_percentage", 0)
            if missing_pct > 0:
                insights.append({
                    "type": "data_quality",
                    "description": f"Column '{col_name}' has {missing_pct:.1f}% missing values",
                    "importance": "high" if missing_pct > 20 else "medium",
                })
                
            # Check for cardinality
            unique_count = col_stats.get("unique_count", 0)
            total_count = col_stats.get("total_count", 0)
            
            if unique_count and total_count:
                cardinality_ratio = unique_count / total_count if total_count > 0 else 0
                
                if cardinality_ratio > 0.9:
                    insights.append({
                        "type": "cardinality",
                        "description": f"Column '{col_name}' has high cardinality",
                        "details": f"{unique_count} unique values out of {total_count} total values",
                        "importance": "high",
                    })
                    
            # Check for imbalance
            if "value_counts" in col_stats:
                value_counts = col_stats["value_counts"]
                if value_counts and len(value_counts) > 1:
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    most_common_pct = most_common[1] / total_count * 100 if total_count > 0 else 0
                    
                    if most_common_pct > 75:
                        insights.append({
                            "type": "imbalance",
                            "description": f"Column '{col_name}' shows significant imbalance",
                            "details": f"{most_common_pct:.1f}% of values belong to category '{most_common[0]}'",
                            "importance": "high",
                        })
                        
        return insights

    def analyze_datetime_data(self, dataset_name: str, datetime_columns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze datetime columns and generate insights.
        
        Args:
            dataset_name: Name of the dataset
            datetime_columns: Dictionary of datetime columns and their statistics
            
        Returns:
            List of insights about datetime data
        """
        insights = []
        
        if not datetime_columns:
            return insights
            
        # Add general datetime data insights
        insights.append({
            "type": "datetime_summary",
            "description": f"Dataset contains {len(datetime_columns)} datetime columns",
            "importance": "medium",
        })
        
        # Analyze each datetime column
        for col_name, col_stats in datetime_columns.items():
            # Skip if no statistics available
            if not col_stats:
                continue
                
            # Check for missing values
            missing_pct = col_stats.get("missing_percentage", 0)
            if missing_pct > 0:
                insights.append({
                    "type": "data_quality",
                    "description": f"Column '{col_name}' has {missing_pct:.1f}% missing values",
                    "importance": "high" if missing_pct > 20 else "medium",
                })
                
            # Check for date range
            if "min_date" in col_stats and "max_date" in col_stats:
                min_date = col_stats["min_date"]
                max_date = col_stats["max_date"]
                
                insights.append({
                    "type": "date_range",
                    "description": f"Column '{col_name}' spans from {min_date} to {max_date}",
                    "importance": "medium",
                })
                
            # Check for seasonality if available
            if "seasonality" in col_stats:
                seasonality = col_stats["seasonality"]
                if seasonality:
                    insights.append({
                        "type": "time_series",
                        "description": f"Column '{col_name}' shows {seasonality} seasonality",
                        "importance": "high",
                    })
                    
        return insights

    def generate_findings(self, dataset_name: str, dataset_insights: List[Dict[str, Any]], 
                         numeric_columns: Dict[str, Any], categorical_columns: Dict[str, Any],
                         datetime_columns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate findings based on insights.
        
        Args:
            dataset_name: Name of the dataset
            dataset_insights: List of insights for the dataset
            numeric_columns: Dictionary of numeric columns and their statistics
            categorical_columns: Dictionary of categorical columns and their statistics
            datetime_columns: Dictionary of datetime columns and their statistics
            
        Returns:
            Dictionary of findings
        """
        # Count insights by type and importance
        insight_types = defaultdict(int)
        high_importance_count = 0
        
        for insight in dataset_insights:
            insight_type = insight.get("type", "unknown")
            insight_types[insight_type] += 1
            
            if insight.get("importance") == "high":
                high_importance_count += 1
                
        # Extract key variables
        key_variables = []
        
        # Add numeric columns with high importance insights
        for insight in dataset_insights:
            if insight.get("importance") == "high" and "column" in insight.get("description", ""):
                # Extract column name from description
                desc = insight.get("description", "")
                if "'" in desc:
                    col_name = desc.split("'")[1]
                    if col_name and col_name not in key_variables:
                        key_variables.append(col_name)
        
        # If no key variables found, use top numeric columns
        if not key_variables and numeric_columns:
            key_variables = list(numeric_columns.keys())[:3]
            
        # Identify potential issues
        potential_issues = []
        
        # Data quality issues
        data_quality_insights = [i for i in dataset_insights if i.get("type") == "data_quality"]
        if data_quality_insights:
            potential_issues.append("Missing values in multiple columns")
            
        # Imbalance issues
        imbalance_insights = [i for i in dataset_insights if i.get("type") == "imbalance"]
        if imbalance_insights:
            potential_issues.append("Imbalance in categorical variables")
            
        # Outlier issues
        outlier_insights = [i for i in dataset_insights if i.get("type") == "outliers"]
        if outlier_insights:
            potential_issues.append("Outliers present in numeric variables")
            
        # Distribution issues
        distribution_insights = [i for i in dataset_insights if i.get("type") == "distribution"]
        if distribution_insights:
            potential_issues.append("Skewed distributions in numeric variables")
            
        # Generate recommendations
        recommendations = []
        
        # Data quality recommendations
        if any(i.get("type") == "data_quality" for i in dataset_insights):
            recommendations.append("Address missing values through imputation or removal")
            
        # Outlier recommendations
        if any(i.get("type") == "outliers" for i in dataset_insights):
            recommendations.append("Consider treating outliers through capping or transformation")
            
        # Distribution recommendations
        if any(i.get("type") == "distribution" for i in dataset_insights):
            recommendations.append("Apply transformations to normalize skewed variables")
            
        # Imbalance recommendations
        if any(i.get("type") == "imbalance" for i in dataset_insights):
            recommendations.append("Consider resampling techniques for imbalanced categories")
            
        # Correlation recommendations
        if any(i.get("type") == "correlation" for i in dataset_insights):
            recommendations.append("Consider feature selection to address multicollinearity")
            
        # Add general recommendations
        recommendations.extend([
            "Normalize numeric features before modeling",
            "Consider feature engineering to create interaction terms",
        ])
        
        # Create findings dictionary
        findings = {
            "summary": f"Analysis of {dataset_name} revealed {len(dataset_insights)} insights ({high_importance_count} high importance)",
            "key_variables": key_variables[:5],  # Limit to top 5
            "potential_issues": potential_issues,
            "recommendations": recommendations[:7],  # Limit to top 7
        }
        
        return findings

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data analysis task.

        Args:
            input_data: Input data for the task

        Returns:
            Dict containing analysis results
        """
        self.logger.info("Starting data analysis task")
        
        # Validate input data
        is_valid, error_message = self.validate_input(input_data)
        if not is_valid:
            self.logger.error(f"Invalid input data: {error_message}")
            return {
                "Analysis Results.error": error_message,
                "Analysis Results.insights": {},
                "Analysis Results.findings": {},
            }
        
        environment = input_data.get("environment", {})
        goals = input_data.get("goals", [])
        
        # Extract data from environment
        processed_data, cleaning_steps, statistics = self.get_data_from_environment(environment)
        
        self.logger.info(f"Analyzing {len(processed_data)} datasets")

        # Initialize results
        insights = {}
        findings = {}
        errors = {}

        # Process each dataset
        for dataset_name, dataset_info in processed_data.items():
            try:
                # Skip datasets with errors
                if dataset_info.get("type") == "error":
                    self.logger.warning(
                        f"Skipping dataset {dataset_name} due to processing error"
                    )
                    errors[dataset_name] = f"Dataset has processing error: {dataset_info.get('error', 'Unknown error')}"
                    continue

                # Get dataset statistics
                dataset_stats = statistics.get(dataset_name, {})
                if not dataset_stats:
                    self.logger.warning(f"No statistics found for dataset {dataset_name}")
                    
                # Initialize insights for this dataset
                dataset_insights = []

                # Add general dataset insights
                rows = dataset_info.get('rows', 0)
                columns = dataset_info.get('columns', 0)
                dataset_insights.append({
                    "type": "general",
                    "description": f"Dataset contains {rows} rows and {columns} columns after cleaning",
                    "importance": "medium",
                })

                # Get column statistics
                numeric_columns = dataset_stats.get("numeric_columns", {})
                categorical_columns = dataset_stats.get("categorical_columns", {})
                datetime_columns = dataset_stats.get("datetime_columns", {})
                
                # Analyze numeric data
                numeric_insights = self.analyze_numeric_data(dataset_name, numeric_columns, dataset_info)
                dataset_insights.extend(numeric_insights)
                
                # Analyze categorical data
                categorical_insights = self.analyze_categorical_data(dataset_name, categorical_columns)
                dataset_insights.extend(categorical_insights)
                
                # Analyze datetime data
                datetime_insights = self.analyze_datetime_data(dataset_name, datetime_columns)
                dataset_insights.extend(datetime_insights)
                
                # Store insights for this dataset
                insights[dataset_name] = dataset_insights
                
                # Generate findings for this dataset
                findings[dataset_name] = self.generate_findings(
                    dataset_name, dataset_insights, numeric_columns, categorical_columns, datetime_columns
                )

            except Exception as e:
                self.logger.error(f"Error analyzing dataset {dataset_name}: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                errors[dataset_name] = str(e)
                insights[dataset_name] = [{
                    "type": "error",
                    "description": f"Error during analysis: {str(e)}",
                    "importance": "high",
                }]
                findings[dataset_name] = {"error": str(e)}

        # Prepare the final result
        result = {
            "Analysis Results.insights": insights,
            "Analysis Results.findings": findings,
        }
        
        # Add errors if any
        if errors:
            result["Analysis Results.errors"] = errors
            
        self.logger.info(f"Analysis task completed with {len(insights)} datasets analyzed")
        return result
