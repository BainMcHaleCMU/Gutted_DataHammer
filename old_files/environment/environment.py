"""
Environment Module

This module defines the Environment class that serves as the central, shared state repository.
"""

from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime


class Environment:
    """
    Central, shared state repository for the AI Agent Swarm.
    
    The Environment stores:
    - Goals
    - Data references
    - Data Overview
    - Cleaned Data
    - Analysis Results
    - Models
    - Visualizations
    - JupyterLogbook
    - Available Agents
    - Execution State/Log
    """
    
    def __init__(self, goals: List[str] = None, data_sources: Dict[str, Any] = None):
        """
        Initialize the Environment.
        
        Args:
            goals: Optional list of user-defined goals
            data_sources: Optional dict of data source references
        """
        self.state = {
            "Goals": goals or [],
            "Data": data_sources or {},
            "Data Overview": {},
            "Cleaned Data": {},
            "Analysis Results": {},
            "Models": {},
            "Visualizations": {},
            "JupyterLogbook": None,
            "Available Agents": {},
            "Execution State/Log": []
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current Environment state.
        
        Returns:
            Dict containing the current state
        """
        return self.state
    
    def update_state(self, key: str, value: Any) -> None:
        """
        Update a specific key in the Environment state.
        
        Args:
            key: The key to update
            value: The new value
        """
        if key not in self.state:
            raise ValueError(f"Invalid Environment key: {key}")
        
        self.state[key] = value
    
    def log_execution(self, action: str, details: str) -> None:
        """
        Log an action in the Execution State/Log.
        
        Args:
            action: The action being logged
            details: Details about the action
        """
        log_entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.state["Execution State/Log"].append(log_entry)
    
    def save_to_disk(self, path: str) -> None:
        """
        Save the Environment state to disk.
        
        Args:
            path: Path to save the state
        """
        # Create a serializable copy of the state
        serializable_state = self.state.copy()
        
        # Handle non-serializable objects
        if serializable_state["JupyterLogbook"] is not None:
            serializable_state["JupyterLogbook"] = "JupyterLogbook object (not serialized)"
        
        with open(path, 'w') as f:
            json.dump(serializable_state, f, indent=2, default=str)
    
    def load_from_disk(self, path: str) -> None:
        """
        Load the Environment state from disk.
        
        Args:
            path: Path to load the state from
        """
        with open(path, 'r') as f:
            loaded_state = json.load(f)
        
        # Update the current state with the loaded state
        for key, value in loaded_state.items():
            if key in self.state:
                self.state[key] = value