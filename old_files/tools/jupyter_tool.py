"""
Jupyter Tool

This module defines the JupyterTool class for managing Jupyter notebooks.
"""

from typing import Any, Dict, List, Optional
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


class JupyterTool:
    """
    Tool for managing Jupyter notebooks.
    
    This tool is primarily used by the Orchestrator Agent to update
    the JupyterLogbook with markdown and code cells.
    """
    
    def __init__(self):
        """Initialize the Jupyter Tool."""
        pass
    
    def create_notebook(self) -> nbformat.NotebookNode:
        """
        Create a new notebook.
        
        Returns:
            A new notebook object
        """
        notebook = new_notebook()
        
        # Add a title cell
        title_cell = new_markdown_cell("# Data Science Project\n\n"
                                      "This notebook contains the data science process "
                                      "performed by the AI Agent Swarm.")
        notebook.cells.append(title_cell)
        
        return notebook
    
    def add_markdown_cell(self, notebook: nbformat.NotebookNode, content: str) -> nbformat.NotebookNode:
        """
        Add a markdown cell to a notebook.
        
        Args:
            notebook: The notebook to update
            content: The markdown content
            
        Returns:
            The updated notebook
        """
        cell = new_markdown_cell(content)
        notebook.cells.append(cell)
        return notebook
    
    def add_code_cell(self, notebook: nbformat.NotebookNode, code: str) -> nbformat.NotebookNode:
        """
        Add a code cell to a notebook.
        
        Args:
            notebook: The notebook to update
            code: The Python code
            
        Returns:
            The updated notebook
        """
        cell = new_code_cell(code)
        notebook.cells.append(cell)
        return notebook
    
    def save_notebook(self, notebook: nbformat.NotebookNode, path: str) -> None:
        """
        Save a notebook to disk.
        
        Args:
            notebook: The notebook to save
            path: Path to save the notebook
        """
        with open(path, 'w') as f:
            nbformat.write(notebook, f)
    
    def load_notebook(self, path: str) -> nbformat.NotebookNode:
        """
        Load a notebook from disk.
        
        Args:
            path: Path to load the notebook from
            
        Returns:
            The loaded notebook
        """
        return nbformat.read(path, as_version=4)