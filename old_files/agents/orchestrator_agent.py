"""
Orchestrator Agent

This module defines the OrchestratorAgent class that coordinates the AI Agent Swarm.
"""

from typing import Any, Dict, List, Optional, Type
import logging
from datetime import datetime

from .base_agent import BaseAgent
from ..llama_workflow.workflow_manager import WorkflowManager


class OrchestratorAgent(BaseAgent):
    """
    Central coordinator for the AI Agent Swarm.

    The Orchestrator Agent is responsible for:
    - Initializing the Environment
    - Dynamically planning the workflow
    - Invoking specialized agents
    - Managing the Environment state
    - Updating the JupyterLogbook
    - Handling errors and coordinating corrective actions
    """

    def __init__(self):
        """Initialize the Orchestrator Agent."""
        super().__init__(name="OrchestratorAgent")
        self.available_agents = {}
        self.environment = self._initialize_environment()
        self.workflow_manager = None
        self.logger = logging.getLogger(__name__)

    def _initialize_environment(self) -> Dict[str, Any]:
        """
        Initialize the shared Environment state.

        Returns:
            Dict containing the initial Environment state
        """
        return {
            "Goals": [],
            "Data": {},
            "Data Overview": {},
            "Cleaned Data": {},
            "Analysis Results": {},
            "Models": {},
            "Visualizations": {},
            "JupyterLogbook": None,
            "Available Agents": {},
            "Execution State/Log": [],
            "Workflow": {"Current": None, "History": []},
        }

    def register_agent(
        self, agent_class: Type[BaseAgent], agent_name: Optional[str] = None
    ) -> None:
        """
        Register a specialized agent with the Orchestrator.

        Args:
            agent_class: The agent class to register
            agent_name: Optional custom name for the agent
        """
        agent_instance = agent_class()
        name = agent_name or agent_instance.name
        self.available_agents[name] = agent_instance

        # Update the Available Agents in the environment
        self.environment["Available Agents"][name] = {
            "name": name,
            "description": agent_instance.__class__.__doc__,
        }

    def run(self, environment: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the Orchestrator's primary functionality.

        This method implements the dynamic workflow planning and execution.

        Args:
            environment: Optional external environment state to use
            **kwargs: Additional arguments
                - goals: List of user-defined goals
                - data_sources: Dict of data source references
                - workflow_type: Optional type of workflow to execute

        Returns:
            Dict containing the final Environment state
        """
        # Use provided environment or the internal one
        env = environment if environment is not None else self.environment

        # Update goals and data if provided
        if "goals" in kwargs:
            env["Goals"] = kwargs["goals"]

        if "data_sources" in kwargs:
            env["Data"] = kwargs["data_sources"]

        workflow_type = kwargs.get("workflow_type", "auto")

        # Initialize JupyterLogbook if not already done
        if env["JupyterLogbook"] is None:
            env["JupyterLogbook"] = self._initialize_jupyter_logbook()

        # Log the initialization
        self._log_execution_state(
            "Orchestrator initialized", "System initialized with goals and data sources"
        )

        # Initialize workflow manager if not already done
        if self.workflow_manager is None:
            self.workflow_manager = WorkflowManager()
            self.workflow_manager.register_default_agents()

        # Create the workflow
        self._log_execution_state(
            "Planning workflow", f"Planning workflow based on goals: {env['Goals']}"
        )
        workflow = self.workflow_manager.create_workflow(env["Goals"], env)

        # Store the workflow execution log in the environment
        env["Workflow"]["Current"] = {
            "execution_log": self.workflow_manager.execution_log,
            "type": workflow_type,
        }

        # Log the workflow plan
        self._log_execution_state(
            "Workflow planned", f"Created workflow with LlamaIndex AgentWorkflow"
        )

        # Execute the workflow
        self._log_execution_state("Executing workflow", "Starting workflow execution")

        try:
            # Execute the workflow
            updated_env = self.workflow_manager.execute_workflow(workflow, env)

            # Update the environment with the workflow execution results
            env.update(updated_env)

            # Update workflow status in the environment
            env["Workflow"]["Current"][
                "execution_log"
            ] = self.workflow_manager.execution_log
            env["Workflow"]["Current"]["status"] = "completed"

            # Log the completion
            self._log_execution_state(
                "Workflow execution completed",
                "Workflow execution completed successfully",
            )
        except Exception as e:
            # Log the error
            self.logger.error(f"Error executing workflow: {str(e)}")
            self._log_execution_state("Workflow execution failed", f"Error: {str(e)}")

            # Update workflow status in the environment
            env["Workflow"]["Current"]["status"] = "failed"
            env["Workflow"]["Current"]["error"] = str(e)
            env["Workflow"]["Current"][
                "execution_log"
            ] = self.workflow_manager.execution_log

        # Archive the current workflow in history
        env["Workflow"]["History"].append(env["Workflow"]["Current"])

        return env

    def _initialize_jupyter_logbook(self) -> Any:
        """
        Initialize the JupyterLogbook.

        Returns:
            An initialized notebook object
        """
        # TODO: Implement notebook initialization using nbformat
        return {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

    def _log_execution_state(self, action: str, details: str) -> None:
        """
        Log an action in the Execution State/Log.

        Args:
            action: The action being logged
            details: Details about the action
        """
        log_entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.environment["Execution State/Log"].append(log_entry)
        self.logger.info(f"{action}: {details}")

    def invoke_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a specialized agent.

        Args:
            agent_name: Name of the agent to invoke
            **kwargs: Additional arguments to pass to the agent

        Returns:
            Dict containing the agent's results

        Raises:
            ValueError: If the agent is not registered
        """
        if agent_name not in self.available_agents:
            raise ValueError(f"Agent '{agent_name}' is not registered")

        agent = self.available_agents[agent_name]

        # Log the agent invocation
        self._log_execution_state(
            f"Invoking {agent_name}",
            f"Delegating task to {agent_name} with args: {kwargs}",
        )

        # Run the agent with the current environment
        result = agent.run(self.environment, **kwargs)

        # Log the agent completion
        self._log_execution_state(
            f"{agent_name} completed",
            f"Agent completed task: {kwargs.get('task', 'unknown')}",
        )

        return result

    def update_jupyter_logbook(
        self, markdown_content: str = None, code_content: str = None
    ) -> None:
        """
        Update the JupyterLogbook with new content.

        Args:
            markdown_content: Optional markdown content to add
            code_content: Optional code content to add
        """
        # TODO: Implement notebook update logic using nbformat
        pass
