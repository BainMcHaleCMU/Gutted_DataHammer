"""
Visualization Tool

This module defines the VisualizationTool class for generating visualizations.
"""

from typing import Any, Dict, List, Optional, Union
import os
import logging
import json


class VisualizationTool:
    """
    Tool for generating visualizations.
    
    This tool is primarily used by the VisualizationAgent to generate
    plots and charts for data exploration and analysis.
    """
    
    def __init__(self):
        """Initialize the Visualization Tool."""
        self.logger = logging.getLogger(__name__)
        
        # Create visualizations directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
    
    def _validate_data_reference(self, data_reference: Any) -> bool:
        """
        Validate that the data reference exists and is usable.
        
        Args:
            data_reference: Reference to the data
            
        Returns:
            bool: True if data reference is valid, False otherwise
        """
        if data_reference is None:
            return False
            
        if isinstance(data_reference, str) and not data_reference.strip():
            return False
            
        return True
    
    def _validate_column(self, data_reference: Any, column: str) -> bool:
        """
        Validate that the column exists in the data.
        
        Args:
            data_reference: Reference to the data
            column: Column name to validate
            
        Returns:
            bool: True if column is valid, False otherwise
        """
        if not column or not isinstance(column, str):
            return False
            
        # In a real implementation, this would check if the column exists in the data
        # For now, we'll just return True
        return True
    
    def _get_output_path(self, plot_type: str, data_reference: str, **kwargs) -> str:
        """
        Generate a standardized output path for the plot.
        
        Args:
            plot_type: Type of plot
            data_reference: Reference to the data
            **kwargs: Additional parameters for the plot
            
        Returns:
            str: Path to save the plot
        """
        # Clean data reference for use in filename
        clean_ref = data_reference.replace("/", "_").replace(".", "_").replace(" ", "_")
        
        # Get additional identifiers from kwargs
        identifiers = []
        
        if "column" in kwargs:
            identifiers.append(kwargs["column"])
        if "x_column" in kwargs and "y_column" in kwargs:
            identifiers.append(f"{kwargs['x_column']}_{kwargs['y_column']}")
        elif "columns" in kwargs:
            if isinstance(kwargs["columns"], list):
                identifiers.append("_".join(kwargs["columns"][:2]))  # Use first two columns
            else:
                identifiers.append(kwargs["columns"])
        
        # Create filename
        if identifiers:
            filename = f"{clean_ref}_{plot_type}_{'_'.join(identifiers)}.png"
        else:
            filename = f"{clean_ref}_{plot_type}.png"
        
        return os.path.join("visualizations", filename)
    
    def generate_histogram(
        self, 
        data_reference: str, 
        column: str, 
        bins: int = 10,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a histogram.
        
        Args:
            data_reference: Reference to the data
            column: Column to plot
            bins: Number of bins
            title: Title for the plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            color: Color for the histogram
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if not self._validate_column(data_reference, column):
                raise ValueError(f"Invalid column: {column}")
                
            if not isinstance(bins, int) or bins <= 0:
                self.logger.warning(f"Invalid bins value: {bins}, using default of 10")
                bins = 10
            
            # Generate output path
            output_path = self._get_output_path(
                "histogram", 
                data_reference, 
                column=column, 
                bins=bins
            )
            
            # In a real implementation, this would use matplotlib or seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "histogram",
                "data_reference": data_reference,
                "column": column,
                "bins": bins,
                "title": title or f"Distribution of {column}",
                "x_label": x_label or column,
                "y_label": y_label or "Frequency",
                "color": color or "blue",
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": f"Histogram of {column} with {bins} bins",
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating histogram: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate histogram for {column}: {str(e)}"
            }
    
    def generate_scatter_plot(
        self, 
        data_reference: str, 
        x_column: str, 
        y_column: str,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        trend_line: bool = True,
        alpha: float = 0.7,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a scatter plot.
        
        Args:
            data_reference: Reference to the data
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Title for the plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            color: Color for the points
            trend_line: Whether to include a trend line
            alpha: Transparency of the points
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if not self._validate_column(data_reference, x_column):
                raise ValueError(f"Invalid x_column: {x_column}")
                
            if not self._validate_column(data_reference, y_column):
                raise ValueError(f"Invalid y_column: {y_column}")
                
            if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
                self.logger.warning(f"Invalid alpha value: {alpha}, using default of 0.7")
                alpha = 0.7
            
            # Generate output path
            output_path = self._get_output_path(
                "scatter", 
                data_reference, 
                x_column=x_column, 
                y_column=y_column
            )
            
            # In a real implementation, this would use matplotlib or seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "scatter_plot",
                "data_reference": data_reference,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"Scatter Plot of {y_column} vs {x_column}",
                "x_label": x_label or x_column,
                "y_label": y_label or y_column,
                "color": color or "blue",
                "trend_line": trend_line,
                "alpha": alpha,
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": f"Scatter plot of {y_column} vs {x_column}" + 
                                   (" with trend line" if trend_line else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scatter plot: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate scatter plot for {y_column} vs {x_column}: {str(e)}"
            }
    
    def generate_correlation_matrix(
        self, 
        data_reference: str,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        cmap: str = "coolwarm",
        annot: bool = True,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a correlation matrix.
        
        Args:
            data_reference: Reference to the data
            columns: List of columns to include in the matrix (None for all numeric columns)
            title: Title for the plot
            cmap: Colormap for the heatmap
            annot: Whether to annotate the heatmap with correlation values
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if columns is not None:
                if not isinstance(columns, list) or not columns:
                    raise ValueError("Columns must be a non-empty list")
                    
                for column in columns:
                    if not self._validate_column(data_reference, column):
                        raise ValueError(f"Invalid column: {column}")
            
            # Generate output path
            output_path = self._get_output_path(
                "correlation", 
                data_reference, 
                columns=columns
            )
            
            # In a real implementation, this would use seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "correlation_matrix",
                "data_reference": data_reference,
                "columns": columns,
                "title": title or "Correlation Matrix",
                "cmap": cmap,
                "annot": annot,
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": "Correlation matrix of numeric features" + 
                                   (f" ({', '.join(columns[:3])}...)" if columns else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating correlation matrix: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate correlation matrix: {str(e)}"
            }
    
    def generate_bar_chart(
        self, 
        data_reference: str, 
        x_column: str, 
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        horizontal: bool = False,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a bar chart.
        
        Args:
            data_reference: Reference to the data
            x_column: Column for categories (x-axis)
            y_column: Column for values (y-axis), if None, counts will be used
            title: Title for the plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            color: Color for the bars
            horizontal: Whether to create a horizontal bar chart
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if not self._validate_column(data_reference, x_column):
                raise ValueError(f"Invalid x_column: {x_column}")
                
            if y_column is not None and not self._validate_column(data_reference, y_column):
                raise ValueError(f"Invalid y_column: {y_column}")
            
            # Generate output path
            output_path = self._get_output_path(
                "bar", 
                data_reference, 
                x_column=x_column, 
                y_column=y_column
            )
            
            # In a real implementation, this would use matplotlib or seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "bar_chart",
                "data_reference": data_reference,
                "x_column": x_column,
                "y_column": y_column or "count",
                "title": title or f"Bar Chart of {y_column or 'Count'} by {x_column}",
                "x_label": x_label or x_column,
                "y_label": y_label or (y_column or "Count"),
                "color": color or "blue",
                "horizontal": horizontal,
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": f"Bar chart of {y_column or 'count'} by {x_column}" + 
                                   (" (horizontal)" if horizontal else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating bar chart: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate bar chart for {x_column}: {str(e)}"
            }
    
    def generate_line_chart(
        self, 
        data_reference: str, 
        x_column: str, 
        y_column: str,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color: Optional[str] = None,
        markers: bool = True,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a line chart.
        
        Args:
            data_reference: Reference to the data
            x_column: Column for x-axis (typically time)
            y_column: Column for y-axis
            title: Title for the plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            color: Color for the line
            markers: Whether to include markers on the line
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if not self._validate_column(data_reference, x_column):
                raise ValueError(f"Invalid x_column: {x_column}")
                
            if not self._validate_column(data_reference, y_column):
                raise ValueError(f"Invalid y_column: {y_column}")
            
            # Generate output path
            output_path = self._get_output_path(
                "line", 
                data_reference, 
                x_column=x_column, 
                y_column=y_column
            )
            
            # In a real implementation, this would use matplotlib or seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "line_chart",
                "data_reference": data_reference,
                "x_column": x_column,
                "y_column": y_column,
                "title": title or f"Line Chart of {y_column} over {x_column}",
                "x_label": x_label or x_column,
                "y_label": y_label or y_column,
                "color": color or "blue",
                "markers": markers,
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": f"Line chart of {y_column} over {x_column}" + 
                                   (" with markers" if markers else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating line chart: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate line chart for {y_column} over {x_column}: {str(e)}"
            }
    
    def generate_box_plot(
        self, 
        data_reference: str, 
        columns: List[str],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        palette: Optional[str] = None,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a box plot.
        
        Args:
            data_reference: Reference to the data
            columns: List of columns to include in the box plot
            title: Title for the plot
            x_label: Label for x-axis
            y_label: Label for y-axis
            palette: Color palette for the boxes
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if not isinstance(columns, list) or not columns:
                raise ValueError("Columns must be a non-empty list")
                
            for column in columns:
                if not self._validate_column(data_reference, column):
                    raise ValueError(f"Invalid column: {column}")
            
            # Generate output path
            output_path = self._get_output_path(
                "boxplot", 
                data_reference, 
                columns=columns
            )
            
            # In a real implementation, this would use seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "box_plot",
                "data_reference": data_reference,
                "columns": columns,
                "title": title or "Box Plot of Numeric Features",
                "x_label": x_label or "Features",
                "y_label": y_label or "Values",
                "palette": palette or "Set2",
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": f"Box plot of {', '.join(columns[:3])}" + 
                                   (f" and {len(columns) - 3} more" if len(columns) > 3 else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating box plot: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate box plot: {str(e)}"
            }
    
    def generate_heatmap(
        self, 
        data_reference: str,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        annot: bool = True,
        output_format: str = "png"
    ) -> Dict[str, Any]:
        """
        Generate a heatmap.
        
        Args:
            data_reference: Reference to the data
            columns: List of columns to include in the heatmap (None for all numeric columns)
            title: Title for the plot
            cmap: Colormap for the heatmap
            annot: Whether to annotate the heatmap with values
            output_format: Format for the output file
            
        Returns:
            Dict containing:
                - plot_reference: Path to the generated plot
                - plot_description: Description of the plot
                - plot_details: Additional details about the plot
                - error: Error message if visualization failed
        """
        try:
            # Validate inputs
            if not self._validate_data_reference(data_reference):
                raise ValueError("Invalid data reference")
                
            if columns is not None:
                if not isinstance(columns, list) or not columns:
                    raise ValueError("Columns must be a non-empty list")
                    
                for column in columns:
                    if not self._validate_column(data_reference, column):
                        raise ValueError(f"Invalid column: {column}")
            
            # Generate output path
            output_path = self._get_output_path(
                "heatmap", 
                data_reference, 
                columns=columns
            )
            
            # In a real implementation, this would use seaborn to generate the plot
            # For now, we'll just return the metadata
            
            # Create plot details
            plot_details = {
                "type": "heatmap",
                "data_reference": data_reference,
                "columns": columns,
                "title": title or "Heatmap",
                "cmap": cmap,
                "annot": annot,
                "output_format": output_format
            }
            
            return {
                "plot_reference": output_path,
                "plot_description": "Heatmap" + 
                                   (f" of {', '.join(columns[:3])}..." if columns else ""),
                "plot_details": plot_details
            }
            
        except Exception as e:
            self.logger.error(f"Error generating heatmap: {str(e)}")
            return {
                "error": str(e),
                "plot_reference": None,
                "plot_description": f"Failed to generate heatmap: {str(e)}"
            }