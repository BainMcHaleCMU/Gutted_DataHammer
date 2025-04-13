"""
Workflow Manager Module

This module defines the WorkflowManager class that manages LlamaIndex agent workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Callable
import logging
from datetime import datetime
import asyncio

from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool


from .task_agents import (
    DataLoadingTaskAgent,
    ExplorationTaskAgent,
    CleaningTaskAgent,
    AnalysisTaskAgent,
    ModelingTaskAgent,
    VisualizationTaskAgent,
    ReportingTaskAgent,
    BaseTaskAgent,
)


class WorkflowManager:
    """
    Manages LlamaIndex agent workflows for data science tasks.

    The WorkflowManager is responsible for:
    - Creating and configuring task agents
    - Building dynamic workflows based on user goals
    - Executing workflows and tracking progress
    - Handling task outputs and environment updates
    """

    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the WorkflowManager.

        Args:
            llm: Optional language model to use for agents
        """
        self.llm = llm

        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, Any] = {}
        self.workflow: Optional[AgentWorkflow] = None
        self.execution_log: List[Dict[str, Any]] = []

    def register_agent(
        self, agent_class: Type[BaseTaskAgent], agent_name: Optional[str] = None
    ) -> None:
        """
        Register an agent with the workflow manager.

        Args:
            agent_class: The agent class to register
            agent_name: Optional custom name for the agent
        """
        agent_instance = agent_class(llm=self.llm)
        name = agent_name or agent_instance.name

        # Convert task agent to function agent with appropriate tools
        function_agent = self._create_function_agent(agent_instance, name)

        self.agents[name] = function_agent
        self.logger.info(f"Registered agent: {name}")

    def _create_function_agent(
        self, task_agent: BaseTaskAgent, name: str
    ) -> FunctionAgent:
        """
        Create a FunctionAgent from a task agent.

        Args:
            task_agent: The task agent to convert
            name: The name for the agent

        Returns:
            A FunctionAgent wrapping the task agent's functionality
        """
        # Create a tool that wraps the task agent's run method
        run_tool = FunctionTool.from_defaults(
            name=f"{name}_run",
            description=f"Execute the {name}'s primary functionality",
            fn=task_agent.run,
        )

        # Create and return a FunctionAgent with the tool
        return FunctionAgent.from_tools(
            tools=[run_tool],
            llm=self.llm,
            name=name,
            system_prompt=f"You are a {name} specialized in data science tasks.",
        )

    def register_default_agents(self) -> None:
        """Register all default agents."""
        self.register_agent(DataLoadingTaskAgent)
        self.register_agent(ExplorationTaskAgent)
        self.register_agent(CleaningTaskAgent)
        self.register_agent(AnalysisTaskAgent)
        self.register_agent(ModelingTaskAgent)
        self.register_agent(VisualizationTaskAgent)
        self.register_agent(ReportingTaskAgent)

    def create_workflow(
        self, goals: List[str], environment: Dict[str, Any]
    ) -> AgentWorkflow:
        """
        Create a workflow based on user goals and the current environment.

        Args:
            goals: List of user-defined goals
            environment: The shared environment state

        Returns:
            An AgentWorkflow configured with agents
        """
        if not self.agents:
            self.register_default_agents()

        # Determine the root agent - start with DataLoadingAgent
        root_agent_name = "DataLoadingAgent"
        if root_agent_name not in self.agents:
            available_agents = list(self.agents.keys())
            if not available_agents:
                raise ValueError(
                    "No agents registered. Call register_default_agents first."
                )
            root_agent_name = available_agents[0]

        # Create the workflow with all registered agents
        workflow = AgentWorkflow(
            agents=list(self.agents.values()),
            root_agent=self.agents[root_agent_name].name,
            initial_state={
                "environment": environment,
                "goals": goals,
                "current_step": "data_loading",
                "completed_steps": [],
                "results": {},
            },
        )

        self.workflow = workflow
        self.logger.info(f"Created workflow with {len(self.agents)} agents")
        return workflow

    async def execute_workflow_async(
        self, environment: Dict[str, Any], callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow asynchronously and update the environment.

        Args:
            environment: The shared environment state
            callbacks: Optional list of callback functions for handling events

        Returns:
            Updated environment after workflow execution
        """
        if not self.workflow:
            raise ValueError("No workflow created. Call create_workflow first.")

        # Execute the workflow
        self.logger.info("Starting workflow execution")
        self._log_execution("Workflow execution started", "Starting workflow execution")

        # Create a copy of the environment
        env = environment.copy()

        # Update initial state with environment
        self.workflow.initial_state["environment"] = env

        # Execute the workflow
        final_state = await self.workflow.arun(callbacks=callbacks)

        # Update the environment with the final state
        if "environment" in final_state:
            env = final_state["environment"]

        # Log completion
        self.logger.info("Workflow execution completed")
        self._log_execution(
            "Workflow execution completed", "Workflow execution completed successfully"
        )

        return env

    def execute_workflow(
        self, environment: Dict[str, Any], callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow and update the environment.

        Args:
            environment: The shared environment state
            callbacks: Optional list of callback functions for handling events

        Returns:
            Updated environment after workflow execution
        """
        return asyncio.run(self.execute_workflow_async(environment, callbacks))

    def _log_execution(self, action: str, details: str) -> None:
        """
        Log an execution action.

        Args:
            action: The action being logged
            details: Details about the action
        """
        log_entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.execution_log.append(log_entry)
