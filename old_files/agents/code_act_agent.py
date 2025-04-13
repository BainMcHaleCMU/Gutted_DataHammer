"""
Code Action Agent

This module defines the CodeActAgent class for securely executing Python code.
"""

from typing import Any, Dict, List, Optional
import subprocess
import tempfile
import os
import sys
import logging
import uuid
import json

from .base_agent import BaseAgent
from ..llama_workflow.task_agents import BaseTaskAgent


class CodeActAgent(BaseAgent):
    """
    Agent responsible for securely executing Python code snippets.
    
    The CodeAct Agent:
    - Receives Python code from other agents
    - Executes code in a secure, isolated environment
    - Captures stdout, stderr, return values, and artifact paths
    - Returns structured results to the requesting agent
    """
    
    def __init__(self):
        """Initialize the CodeAct Agent."""
        super().__init__(name="CodeActAgent")
        self.logger = logging.getLogger(__name__)
        self.execution_dir = os.path.join(tempfile.gettempdir(), "datahammer_executions")
        os.makedirs(self.execution_dir, exist_ok=True)
    
    def run(self, environment: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        Args:
            environment: The shared environment state
            **kwargs: Additional arguments
                - code: String containing Python code to execute
                - timeout: Optional timeout in seconds
                
        Returns:
            Dict containing:
                - stdout: Standard output from code execution
                - stderr: Standard error from code execution
                - return_value: Return value from code execution
                - artifact_paths: Paths to any artifacts created
        """
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30)
        
        if not code:
            return {
                "stdout": "",
                "stderr": "No code provided",
                "return_value": None,
                "artifact_paths": []
            }
        
        self.logger.info(f"Executing code with timeout {timeout}s")
        
        try:
            # Execute the code securely
            result = self._execute_code_securely(code, timeout)
            return result
        except Exception as e:
            self.logger.error(f"Error executing code: {str(e)}")
            return {
                "stdout": "",
                "stderr": f"Error executing code: {str(e)}",
                "return_value": None,
                "artifact_paths": []
            }
    
    def _execute_code_securely(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code in a secure environment.
        
        Args:
            code: String containing Python code to execute
            timeout: Timeout in seconds
            
        Returns:
            Dict containing execution results
            
        Note:
            This is a placeholder for the actual secure execution implementation.
            A real implementation would use Docker, RestrictedPython, or another
            sandboxing mechanism to ensure security.
        """
        # Create a unique execution ID
        execution_id = str(uuid.uuid4())
        execution_path = os.path.join(self.execution_dir, execution_id)
        os.makedirs(execution_path, exist_ok=True)
        
        # Create a Python file with the code
        code_file = os.path.join(execution_path, "code.py")
        with open(code_file, "w") as f:
            f.write(code)
        
        # Create a wrapper script that captures return values
        wrapper_code = f"""
import sys
import json
import traceback
import os

# Directory for artifacts
os.makedirs(os.path.join("{execution_path}", "artifacts"), exist_ok=True)

# Execute the code and capture the return value
try:
    # Add the execution path to sys.path
    sys.path.insert(0, "{execution_path}")
    
    # Execute the code
    exec_globals = {{"__name__": "__main__", "artifact_dir": os.path.join("{execution_path}", "artifacts")}}
    with open("{code_file}", "r") as f:
        code = f.read()
        exec(code, exec_globals)
    
    # Write the return value if available
    if "_return_value" in exec_globals:
        with open("{execution_path}/return_value.json", "w") as f:
            json.dump(exec_globals["_return_value"], f)
    
    # Write the artifact paths
    artifact_paths = []
    for root, dirs, files in os.walk(os.path.join("{execution_path}", "artifacts")):
        for file in files:
            artifact_paths.append(os.path.join(root, file))
    
    with open("{execution_path}/artifact_paths.json", "w") as f:
        json.dump(artifact_paths, f)
    
except Exception as e:
    # Write the error
    with open("{execution_path}/error.txt", "w") as f:
        f.write(traceback.format_exc())
    sys.stderr.write(traceback.format_exc())
    sys.exit(1)
"""
        
        wrapper_file = os.path.join(execution_path, "wrapper.py")
        with open(wrapper_file, "w") as f:
            f.write(wrapper_code)
        
        # Execute the wrapper script
        try:
            # For now, we'll use subprocess to execute the code
            # In a production environment, this should be replaced with a secure sandbox
            process = subprocess.Popen(
                [sys.executable, wrapper_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Check for return value
            return_value = None
            return_value_file = os.path.join(execution_path, "return_value.json")
            if os.path.exists(return_value_file):
                with open(return_value_file, "r") as f:
                    try:
                        return_value = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            # Check for artifact paths
            artifact_paths = []
            artifact_paths_file = os.path.join(execution_path, "artifact_paths.json")
            if os.path.exists(artifact_paths_file):
                with open(artifact_paths_file, "r") as f:
                    try:
                        artifact_paths = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_value": return_value,
                "artifact_paths": artifact_paths,
                "execution_path": execution_path
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "return_value": None,
                "artifact_paths": [],
                "execution_path": execution_path
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error executing code: {str(e)}",
                "return_value": None,
                "artifact_paths": [],
                "execution_path": execution_path
            }