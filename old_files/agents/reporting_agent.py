"""
Reporting Agent

This module defines the ReportingAgent class for compiling final results into a report.
"""

from typing import Any, Dict, List, Optional
import logging
import os
import traceback

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import ReportingTaskAgent


class ReportingAgent(BaseAgent):
    """
    Agent responsible for compiling final results and process into a coherent report.
    
    The Reporting Agent:
    - Gathers key information from across the Environment
    - Synthesizes narrative explanations and summaries
    - Requests final summary visualizations
    - Produces the final output (polished JupyterLogbook or Markdown)
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize the Reporting Agent.
        
        Args:
            output_dir: Directory to save generated reports
        """
        super().__init__(name="ReportingAgent")
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Create the task agent with the same output directory
        self.task_agent = ReportingTaskAgent(output_dir=output_dir)
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - report_format: Optional format for the report (jupyter, markdown, html, pdf)
                - sections: Optional list of sections to include
                - goals: Optional list of analysis goals
                - output_dir: Optional directory to save the report
                
        Returns:
            Dict containing:
                - report: The final report path
                - report_content: The report content structure
                - visualization_requests: List of final visualization requests
                - error: Error message if an error occurred
        """
        # Extract parameters from kwargs with validation
        report_format = kwargs.get("report_format", "jupyter")
        if report_format not in ["jupyter", "markdown", "html", "pdf"]:
            self.logger.warning(f"Unsupported report format: {report_format}, defaulting to jupyter")
            report_format = "jupyter"
            
        sections = kwargs.get("sections", [])
        if not isinstance(sections, list):
            self.logger.warning("Sections parameter is not a list, using empty list")
            sections = []
            
        goals = kwargs.get("goals", ["Generate comprehensive data analysis report"])
        if not isinstance(goals, list) or not goals:
            self.logger.warning("Goals parameter is invalid, using default goal")
            goals = ["Generate comprehensive data analysis report"]
            
        # Check if output directory is specified in kwargs
        output_dir = kwargs.get("output_dir", self.output_dir)
        if output_dir != self.output_dir:
            # Create a new task agent with the specified output directory
            try:
                self.task_agent = ReportingTaskAgent(output_dir=output_dir)
                self.output_dir = output_dir
                self.logger.info(f"Using custom output directory: {output_dir}")
            except Exception as e:
                self.logger.error(f"Failed to use custom output directory: {str(e)}")
                # Continue with the default output directory
        
        self.logger.info(f"Generating report in {report_format} format with {len(goals)} goals")
        
        # Prepare the task input
        task_input = {
            "environment": environment,
            "goals": goals,
            "report_format": report_format,
            "sections": sections
        }
        
        try:
            # Run the task agent with comprehensive error handling
            result = self.task_agent.run(task_input)
            
            # Check for errors in the result
            if "error" in result:
                self.logger.error(f"Task agent reported an error: {result['error']}")
                return {
                    "error": result["error"],
                    "report": None,
                    "report_content": result.get("Report.content", {}),
                    "visualization_requests": []
                }
            
            # Extract the results with proper defaults
            report_path = result.get("Report.path", os.path.join(self.output_dir, "report.ipynb"))
            report_content = result.get("Report.content", {})
            report_summary = result.get("Report.summary", {})
            
            # Generate visualization requests based on the report content
            visualization_requests = [
                {"type": "summary_plot", "data": "all_results"}
            ]
            
            # Add dataset-specific visualization requests if available
            datasets = report_summary.get("datasets_analyzed", [])
            for dataset in datasets:
                visualization_requests.append({
                    "type": "dataset_summary",
                    "data": dataset,
                    "title": f"Summary of {dataset}"
                })
            
            return {
                "report": report_path,
                "report_content": report_content,
                "report_summary": report_summary,
                "visualization_requests": visualization_requests
            }
        except Exception as e:
            # Comprehensive error handling with stack trace
            error_msg = f"Error generating report: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            
            return {
                "error": error_msg,
                "stack_trace": stack_trace,
                "report": None,
                "report_content": {
                    "error": error_msg,
                    "input": {
                        "report_format": report_format,
                        "sections": sections,
                        "goals": goals
                    }
                },
                "visualization_requests": []
            }