"""
Analysis Agent

This module defines the AnalysisAgent class for deriving deeper insights from data.
"""

from typing import Any, Dict, List, Optional
import logging
import traceback

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import AnalysisTaskAgent


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for deriving deeper insights from data relevant to the goals.
    
    The Analysis Agent:
    - Performs targeted statistical analyses
    - Conducts hypothesis testing
    - Investigates complex correlations
    - Analyzes outliers for insights
    - Performs segmentation if needed
    """
    
    def __init__(self):
        """Initialize the Analysis Agent."""
        super().__init__(name="AnalysisAgent")
        self.logger = logging.getLogger(__name__)
        self.task_agent = AnalysisTaskAgent()
    
    def generate_visualization_requests(self, insights: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Generate visualization requests based on insights.
        
        Args:
            insights: Dictionary of insights by dataset
            
        Returns:
            List of visualization requests
        """
        visualization_requests = []
        
        # Process each dataset's insights
        for dataset_name, dataset_insights in insights.items():
            # Find correlation insights for scatter plots
            correlation_insights = [i for i in dataset_insights if i.get("type") == "correlation"]
            for insight in correlation_insights:
                # Extract column names from description
                description = insight.get("description", "")
                if "between '" in description and "' and '" in description:
                    parts = description.split("between '")[1].split("' and '")
                    if len(parts) >= 2:
                        col1 = parts[0]
                        col2 = parts[1].split("'")[0]
                        
                        visualization_requests.append({
                            "type": "scatter",
                            "dataset": dataset_name,
                            "x": col1,
                            "y": col2,
                            "title": f"Correlation between {col1} and {col2}"
                        })
            
            # Find distribution insights for histograms
            distribution_insights = [i for i in dataset_insights if i.get("type") == "distribution"]
            for insight in distribution_insights:
                # Extract column name from description
                description = insight.get("description", "")
                if "Column '" in description and "'" in description:
                    col_name = description.split("'")[1]
                    
                    visualization_requests.append({
                        "type": "histogram",
                        "dataset": dataset_name,
                        "column": col_name,
                        "title": f"Distribution of {col_name}"
                    })
            
            # Find outlier insights for box plots
            outlier_insights = [i for i in dataset_insights if i.get("type") == "outliers"]
            for insight in outlier_insights:
                # Extract column name from description
                description = insight.get("description", "")
                if "Column '" in description and "'" in description:
                    col_name = description.split("'")[1]
                    
                    visualization_requests.append({
                        "type": "box_plot",
                        "dataset": dataset_name,
                        "column": col_name,
                        "title": f"Outliers in {col_name}"
                    })
            
            # Find imbalance insights for bar charts
            imbalance_insights = [i for i in dataset_insights if i.get("type") == "imbalance"]
            for insight in imbalance_insights:
                # Extract column name from description
                description = insight.get("description", "")
                if "Column '" in description and "'" in description:
                    col_name = description.split("'")[1]
                    
                    visualization_requests.append({
                        "type": "bar_chart",
                        "dataset": dataset_name,
                        "column": col_name,
                        "title": f"Category distribution in {col_name}"
                    })
        
        # Limit to a reasonable number of visualizations
        return visualization_requests[:10]
    
    def generate_suggestions(self, findings: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate suggestions for next steps based on findings.
        
        Args:
            findings: Dictionary of findings by dataset
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Collect all recommendations from findings
        all_recommendations = []
        for dataset_name, dataset_findings in findings.items():
            recommendations = dataset_findings.get("recommendations", [])
            all_recommendations.extend(recommendations)
        
        # Count frequency of each recommendation
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        sorted_recommendations = sorted(
            recommendation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Convert top recommendations to suggestions
        for rec, count in sorted_recommendations[:5]:
            suggestions.append(rec)
        
        # Add general next steps
        suggestions.extend([
            "Run ModelingAgent to build predictive models",
            "Use VisualizationAgent to create detailed visualizations",
            "Consider feature engineering based on the insights"
        ])
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:7]  # Limit to top 7
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - data_reference: Reference to the data to analyze
                - analysis_targets: Optional list of specific analysis targets
                
        Returns:
            Dict containing:
                - analysis_results: Dict with analysis findings
                - visualization_requests: List of visualization requests
                - suggestions: List of suggested next steps
        """
        # Extract data reference from kwargs or environment
        data_reference = kwargs.get("data_reference")
        if not data_reference:
            if "Cleaned Data" in environment:
                data_reference = "Cleaned Data"
            elif "cleaned_data" in environment:
                data_reference = "cleaned_data"
            elif "loaded_data" in environment:
                data_reference = "loaded_data"
            else:
                self.logger.warning("No data reference found in environment")
        
        analysis_targets = kwargs.get("analysis_targets", [])
        
        self.logger.info(f"Analyzing data: {data_reference}")
        
        # Use the task agent to analyze the data
        task_input = {
            "environment": environment,
            "goals": kwargs.get("goals", ["Derive deeper insights from data"]),
            "data_reference": data_reference,
            "analysis_targets": analysis_targets
        }
        
        try:
            # Run the task agent
            result = self.task_agent.run(task_input)
            
            # Extract the results
            insights = result.get("Analysis Results.insights", {})
            findings = result.get("Analysis Results.findings", {})
            errors = result.get("Analysis Results.errors", {})
            
            # Generate visualization requests based on insights
            visualization_requests = self.generate_visualization_requests(insights)
            
            # Generate suggestions based on findings
            suggestions = self.generate_suggestions(findings)
            
            # Prepare analysis results
            analysis_results = {}
            
            # Process each dataset's findings
            for dataset_name, dataset_findings in findings.items():
                # Skip datasets with errors
                if "error" in dataset_findings:
                    continue
                    
                # Extract key information
                summary = dataset_findings.get("summary", "")
                key_variables = dataset_findings.get("key_variables", [])
                potential_issues = dataset_findings.get("potential_issues", [])
                
                # Add to analysis results
                analysis_results[dataset_name] = {
                    "summary": summary,
                    "key_variables": key_variables,
                    "potential_issues": potential_issues
                }
            
            # Prepare the final result
            result = {
                "analysis_results": analysis_results,
                "insights": insights,
                "visualization_requests": visualization_requests,
                "suggestions": suggestions
            }
            
            # Add errors if any
            if errors:
                result["errors"] = errors
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "error": str(e),
                "analysis_results": {},
                "insights": {},
                "visualization_requests": [],
                "suggestions": ["Check data format and try again"]
            }