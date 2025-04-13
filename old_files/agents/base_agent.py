"""
Base Agent Class

This module defines the BaseAgent class that all specialized agents inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the AI Agent Swarm.
    
    All specialized agents inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        self.tools = []
    
    @abstractmethod
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments specific to the agent
            
        Returns:
            Dict containing results and any suggestions for next steps
        """
        pass
    
    def register_tool(self, tool: Any) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: The tool to register
        """
        self.tools.append(tool)
    
    def get_available_tools(self) -> List[Any]:
        """
        Get all tools available to this agent.
        
        Returns:
            List of tools
        """
        return self.tools