"""
Jupyter Logbook Module

This module defines utilities for managing the JupyterLogbook.
"""

from typing import Any, Dict, List, Optional
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


class JupyterLogbook:
    """
    Manager for the JupyterLogbook.
    
    The JupyterLogbook is a chronologically ordered sequence of markdown and code cells
    that provides a transparent, step-by-step, executable record of the data science process.
    """
    
    def __init__(self, notebook_path: Optional[str] = None):
        """
        Initialize the JupyterLogbook.
        
        Args:
            notebook_path: Optional path to an existing notebook file
        """
        if notebook_path:
            try:
                self.notebook = nbformat.read(notebook_path, as_version=4)
            except Exception as e:
                print(f"Error loading notebook: {e}")
                self.notebook = self._create_new_notebook()
        else:
            self.notebook = self._create_new_notebook()
    
    def _create_new_notebook(self) -> nbformat.NotebookNode:
        """
        Create a new notebook.
        
        Returns:
            A new notebook object
        """
        notebook = new_notebook()
        
        # Add a title cell
        title_cell = new_markdown_cell("# Data Science Project Logbook\n\n"
                                      "This notebook contains a chronological record of "
                                      "the data science process performed by the AI Agent Swarm.")
        notebook.cells.append(title_cell)
        
        return notebook
    
    def add_markdown_cell(self, content: str) -> None:
        """
        Add a markdown cell to the notebook.
        
        Args:
            content: The markdown content
        """
        cell = new_markdown_cell(content)
        self.notebook.cells.append(cell)
    
    def add_code_cell(self, code: str) -> None:
        """
        Add a code cell to the notebook.
        
        Args:
            code: The Python code
        """
        cell = new_code_cell(code)
        self.notebook.cells.append(cell)
    
    def save(self, path: str) -> None:
        """
        Save the notebook to disk.
        
        Args:
            path: Path to save the notebook
        """
        with open(path, 'w') as f:
            nbformat.write(self.notebook, f)
    
    def get_notebook(self) -> nbformat.NotebookNode:
        """
        Get the notebook object.
        
        Returns:
            The notebook object
        """
        return self.notebook