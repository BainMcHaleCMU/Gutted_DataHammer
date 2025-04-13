"""
Reporting Task Agent Module

This module defines the reporting task agent used in LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import json
import os
from datetime import datetime
import traceback

from llama_index.core.llms import LLM

from .base import BaseTaskAgent


class ReportingTaskAgent(BaseTaskAgent):
    """
    Task agent for generating reports and documentation.

    Responsibilities:
    - Creating summary reports
    - Documenting findings
    - Creating presentations
    - Generating recommendations
    """

    def __init__(self, llm: Optional[LLM] = None, output_dir: str = "./reports"):
        """
        Initialize the ReportingTaskAgent.
        
        Args:
            llm: Optional language model to use
            output_dir: Directory to save generated reports
        """
        super().__init__(name="ReportingAgent", llm=llm)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to create output directory: {str(e)}")

    def _validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and provide defaults for missing values.
        
        Args:
            input_data: Input data for the task
            
        Returns:
            Validated input data with defaults for missing values
        """
        # Ensure environment exists
        if "environment" not in input_data:
            self.logger.warning("No environment data provided, using empty environment")
            input_data["environment"] = {}
            
        # Ensure goals exist
        if "goals" not in input_data or not input_data["goals"]:
            self.logger.warning("No goals provided, using default goal")
            input_data["goals"] = ["Generate comprehensive data analysis report"]
            
        # Get report format
        report_format = input_data.get("report_format", "jupyter")
        if report_format not in ["jupyter", "markdown", "html", "pdf"]:
            self.logger.warning(f"Unsupported report format: {report_format}, defaulting to jupyter")
            report_format = "jupyter"
        input_data["report_format"] = report_format
        
        # Get sections
        if "sections" not in input_data or not isinstance(input_data["sections"], list):
            input_data["sections"] = []
            
        return input_data

    def _safe_get(self, data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """
        Safely get a value from nested dictionaries.
        
        Args:
            data: Dictionary to get value from
            keys: List of keys to traverse
            default: Default value if key doesn't exist
            
        Returns:
            Value at the specified keys or default if not found
        """
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def _generate_report_path(self, report_format: str) -> str:
        """
        Generate a unique file path for the report.
        
        Args:
            report_format: Format of the report
            
        Returns:
            File path for the report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".ipynb" if report_format == "jupyter" else f".{report_format}"
        filename = f"data_analysis_report_{timestamp}{extension}"
        return os.path.join(self.output_dir, filename)

    def _handle_insights(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process insights data and handle potential errors.
        
        Args:
            insights: Dictionary of insights by dataset
            
        Returns:
            List of Jupyter notebook cells for insights
        """
        cells = []
        
        # Add analysis section header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Data Analysis\n\nThis section presents the key insights from our analysis.",
        })
        
        # Process each dataset's insights
        if not insights:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": "*No insights available*",
            })
            return cells
            
        for dataset_name, dataset_insights in insights.items():
            insight_text = f"### Insights for {dataset_name}\n\n"
            
            if not dataset_insights:
                insight_text += "*No insights available for this dataset*"
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": insight_text,
                })
                continue
                
            if isinstance(dataset_insights, list):
                insight_bullets = []
                
                for insight in dataset_insights:
                    if not isinstance(insight, dict):
                        continue
                        
                    insight_type = insight.get('type', 'Insight')
                    description = insight.get('description', 'No description available')
                    details = insight.get('details', '')
                    
                    bullet = f"- **{insight_type}**: {description}"
                    if details:
                        bullet += f" ({details})"
                    
                    insight_bullets.append(bullet)
                
                if insight_bullets:
                    insight_text += "\n".join(insight_bullets)
                else:
                    insight_text += "*No valid insights available for this dataset*"
            else:
                insight_text += "*Insights data is not in the expected format*"
                
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": insight_text,
            })
            
        return cells

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reporting task.

        Args:
            input_data: Input data for the task
                - environment: The shared environment state
                - goals: List of analysis goals
                - report_format: Format for the report (jupyter, markdown, html, pdf)
                - sections: Optional list of sections to include

        Returns:
            Dict containing report and documentation
        """
        try:
            # Validate input data
            input_data = self._validate_input(input_data)
            
            # Extract validated data
            environment = input_data["environment"]
            goals = input_data["goals"]
            report_format = input_data["report_format"]
            sections = input_data["sections"]
            
            self.logger.info(f"Generating {report_format} report with {len(goals)} goals")

            # Get data from the environment with safe access
            data_overview = self._safe_get(environment, ["Data Overview"], {})
            summary = self._safe_get(data_overview, ["summary"], {})

            cleaned_data = self._safe_get(environment, ["Cleaned Data"], {})
            cleaning_steps = self._safe_get(cleaned_data, ["cleaning_steps"], {})

            analysis_results = self._safe_get(environment, ["Analysis Results"], {})
            insights = self._safe_get(analysis_results, ["insights"], {})
            findings = self._safe_get(analysis_results, ["findings"], {})

            models = self._safe_get(environment, ["Models"], {})
            trained_models = self._safe_get(models, ["trained_model"], {})
            performance = self._safe_get(models, ["performance"], {})

            visualizations = self._safe_get(environment, ["Visualizations"], {})
            plots = self._safe_get(visualizations, ["plots"], {})
            dashboard = self._safe_get(visualizations, ["dashboard"], {})

            # Create Jupyter notebook cells
            jupyter_cells = []

            # Add title and introduction
            jupyter_cells.extend([
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "# Data Analysis Report\n\n## Introduction\n\nThis notebook contains the results of our data analysis and modeling efforts.",
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Goals\n\n" + "\n".join([f"- {goal}" for goal in goals]),
                },
            ])

            # Add data overview section
            jupyter_cells.extend([
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Data Overview\n\nThis section provides an overview of the datasets used in the analysis.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display data overview\nimport pandas as pd\nimport json\n\n# Display summary of datasets\nprint('Dataset Summary:')\ntry:\n    summary_data = " + json.dumps(summary) + "\n    print(json.dumps(summary_data, indent=2))\nexcept Exception as e:\n    print(f\"Error displaying summary: {str(e)}\")",
                    "execution_count": None,
                    "outputs": [],
                },
            ])

            # Add data cleaning section
            jupyter_cells.extend([
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Data Cleaning\n\nThis section describes the data cleaning steps performed.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display cleaning steps\nimport pandas as pd\n\n# Display cleaning steps for each dataset\ntry:\n    cleaning_steps_data = " + json.dumps(cleaning_steps) + "\n    if not cleaning_steps_data:\n        print(\"No cleaning steps available\")\n    else:\n        for dataset_name, steps in cleaning_steps_data.items():\n            print(f'Cleaning steps for {dataset_name}:')\n            if not steps:\n                print(\"  No steps recorded\")\n                continue\n            for i, step in enumerate(steps):\n                if isinstance(step, dict):\n                    operation = step.get('operation', 'Unknown operation')\n                    description = step.get('description', 'No description')\n                    print(f\"  {i+1}. {operation}: {description}\")\n                else:\n                    print(f\"  {i+1}. {step}\")\n            print()\nexcept Exception as e:\n    print(f\"Error displaying cleaning steps: {str(e)}\")",
                    "execution_count": None,
                    "outputs": [],
                },
            ])

            # Add analysis section with insights
            jupyter_cells.extend(self._handle_insights(insights))

            # Add modeling section
            jupyter_cells.extend([
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Modeling Results\n\nThis section presents the results of our modeling efforts.",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": "# Code to display model performance\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Display performance metrics for each dataset and model\ntry:\n    performance_data = " + json.dumps(performance) + "\n    if not performance_data:\n        print(\"No model performance data available\")\n    else:\n        for dataset_name, dataset_perf in performance_data.items():\n            print(f'Model performance for {dataset_name}:')\n            if not isinstance(dataset_perf, dict):\n                print(\"  Invalid performance data format\")\n                continue\n                \n            if 'best_model' not in dataset_perf:\n                print(\"  No best model identified\")\n                continue\n                \n            best_model = dataset_perf['best_model']\n            print(f\"  Best model: {best_model}\")\n            \n            if best_model in dataset_perf and 'metrics' in dataset_perf[best_model]:\n                metrics = dataset_perf[best_model]['metrics']\n                print(f\"  Performance metrics: {metrics}\")\n                \n                # Only create visualization if we have numeric metrics\n                if isinstance(metrics, dict) and metrics:\n                    try:\n                        metric_names = list(metrics.keys())\n                        metric_values = list(metrics.values())\n                        \n                        # Check if values are numeric\n                        if all(isinstance(v, (int, float)) for v in metric_values):\n                            plt.figure(figsize=(10, 6))\n                            bars = plt.bar(metric_names, metric_values)\n                            plt.title(f'Performance Metrics for {best_model} on {dataset_name}')\n                            plt.ylabel('Value')\n                            plt.ylim(0, max(metric_values) * 1.2)\n                            \n                            # Add value labels on top of bars\n                            for bar in bars:\n                                height = bar.get_height()\n                                plt.text(bar.get_x() + bar.get_width()/2., height,\n                                        f'{height:.3f}', ha='center', va='bottom')\n                                        \n                            plt.show()\n                    except Exception as viz_error:\n                        print(f\"  Could not create visualization: {str(viz_error)}\")\n            else:\n                print(\"  No metrics available for the best model\")\n            print()\nexcept Exception as e:\n    print(f\"Error displaying model performance: {str(e)}\")",
                    "execution_count": None,
                    "outputs": [],
                },
            ])

            # Add visualization section
            jupyter_cells.extend([
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Visualizations\n\nThis section contains key visualizations from our analysis.",
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "### Dashboard Overview\n\n" + 
                    (f"The dashboard contains {len(dashboard.get('sections', []))} sections with various visualizations." 
                     if dashboard and 'sections' in dashboard else "No dashboard data available."),
                },
            ])

            # Add recommendations section
            recommendations = []
            if findings:
                for dataset_name, finding in findings.items():
                    if isinstance(finding, dict) and "recommendations" in finding:
                        for rec in finding.get("recommendations", []):
                            recommendations.append(f"- **{dataset_name}**: {rec}")

            recommendation_text = "## Recommendations\n\nBased on our analysis, we recommend the following actions:\n\n"
            if recommendations:
                recommendation_text += "\n".join(recommendations)
            else:
                recommendation_text += "*No specific recommendations available based on findings*\n\n"
                recommendation_text += "### General Recommendations:\n\n"
                recommendation_text += "- Ensure data quality through regular validation\n"
                recommendation_text += "- Consider collecting additional data points for more robust analysis\n"
                recommendation_text += "- Review analysis methodology for potential improvements"
                
            jupyter_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": recommendation_text,
            })

            # Create summary with current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            report_summary = {
                "title": "Data Analysis Report",
                "date": current_date,
                "datasets_analyzed": list(summary.keys()) if summary else [],
                "key_findings": [],
            }

            # Add key findings from each dataset
            if findings:
                for dataset_name, finding in findings.items():
                    if isinstance(finding, dict) and "summary" in finding:
                        report_summary["key_findings"].append({
                            "dataset": dataset_name,
                            "summary": finding.get("summary", ""),
                            "key_variables": finding.get("key_variables", []),
                        })

            # Create recommendations
            report_recommendations = {
                "data_quality": [
                    "Implement data validation checks to prevent missing values",
                    "Standardize data collection processes to ensure consistency",
                ],
                "analysis": [
                    "Conduct further analysis on correlations between key variables",
                    "Investigate seasonal patterns in time series data",
                ],
                "modeling": [
                    "Deploy the best performing model in a production environment",
                    "Regularly retrain models with new data to maintain accuracy",
                    "Consider ensemble methods to improve prediction performance",
                ],
                "business_actions": [
                    "Use insights to inform strategic decision making",
                    "Develop a data-driven approach to problem solving",
                    "Invest in data infrastructure to support ongoing analysis",
                ],
            }
            
            # Generate report path
            report_path = self._generate_report_path(report_format)
            
            # Create the final report object
            report = {
                "JupyterLogbook": {
                    "cells": jupyter_cells,
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {
                            "codemirror_mode": {"name": "ipython", "version": 3},
                            "file_extension": ".py",
                            "mimetype": "text/x-python",
                            "name": "python",
                            "nbconvert_exporter": "python",
                            "pygments_lexer": "ipython3",
                            "version": "3.8.10",
                        },
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                "Report.summary": report_summary,
                "Report.recommendations": report_recommendations,
                "Report.path": report_path,
                "Report.format": report_format,
                "Report.content": {
                    "title": "Data Analysis Report",
                    "date": current_date,
                    "sections": [
                        "Introduction", 
                        "Goals", 
                        "Data Overview", 
                        "Data Cleaning", 
                        "Data Analysis", 
                        "Modeling Results", 
                        "Visualizations", 
                        "Recommendations"
                    ] + sections,
                }
            }
            
            # Try to save the report to disk
            try:
                if report_format == "jupyter":
                    with open(report_path, 'w') as f:
                        json.dump(report["JupyterLogbook"], f, indent=2)
                    self.logger.info(f"Saved Jupyter notebook to {report_path}")
                else:
                    # For other formats, we'd need to convert the Jupyter notebook
                    # This is a placeholder for future implementation
                    self.logger.warning(f"Saving in {report_format} format not yet implemented")
            except Exception as save_error:
                self.logger.error(f"Failed to save report to {report_path}: {str(save_error)}")
                report["Report.save_error"] = str(save_error)
            
            return report
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            
            # Return error information
            return {
                "error": error_msg,
                "stack_trace": stack_trace,
                "Report.summary": {
                    "title": "Error Report",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "error": error_msg
                },
                "Report.content": {
                    "error": error_msg,
                    "input_data": str(input_data)[:1000] + "..." if len(str(input_data)) > 1000 else str(input_data)
                }
            }
