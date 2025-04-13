"""
Tests for the AnalysisTaskAgent class.
"""

import unittest
from unittest.mock import MagicMock, patch
import logging

from ..analysis_agent import AnalysisTaskAgent


class TestAnalysisTaskAgent(unittest.TestCase):
    """Test cases for the AnalysisTaskAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLM
        self.mock_llm = MagicMock()
        
        # Create an instance of AnalysisTaskAgent with the mock LLM
        self.agent = AnalysisTaskAgent(llm=self.mock_llm)
        
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_validate_input_valid(self):
        """Test validate_input with valid input."""
        # Create valid input data
        input_data = {
            "environment": {
                "Cleaned Data": {
                    "processed_data": {
                        "dataset1": {"rows": 100, "columns": 10}
                    }
                }
            }
        }
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertTrue(is_valid)
        self.assertEqual(error_message, "")
    
    def test_validate_input_missing_environment(self):
        """Test validate_input with missing environment."""
        # Create invalid input data
        input_data = {}
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "Missing 'environment' in input data")
    
    def test_validate_input_missing_data(self):
        """Test validate_input with missing data."""
        # Create invalid input data
        input_data = {
            "environment": {}
        }
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "No data found in environment. Need either 'Cleaned Data' or 'Data Overview'")
    
    def test_validate_input_invalid_cleaned_data(self):
        """Test validate_input with invalid cleaned data."""
        # Create invalid input data
        input_data = {
            "environment": {
                "Cleaned Data": "not a dictionary"
            }
        }
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "Invalid 'Cleaned Data' format, expected dictionary")
    
    def test_validate_input_missing_processed_data(self):
        """Test validate_input with missing processed_data."""
        # Create invalid input data
        input_data = {
            "environment": {
                "Cleaned Data": {}
            }
        }
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "No 'processed_data' found in 'Cleaned Data'")
    
    def test_validate_input_empty_processed_data(self):
        """Test validate_input with empty processed_data."""
        # Create invalid input data
        input_data = {
            "environment": {
                "Cleaned Data": {
                    "processed_data": {}
                }
            }
        }
        
        # Validate input
        is_valid, error_message = self.agent.validate_input(input_data)
        
        # Check results
        self.assertFalse(is_valid)
        self.assertEqual(error_message, "Empty 'processed_data' in 'Cleaned Data'")
    
    def test_get_data_from_environment(self):
        """Test get_data_from_environment."""
        # Create environment data
        environment = {
            "Cleaned Data": {
                "processed_data": {"dataset1": {"rows": 100}},
                "cleaning_steps": {"dataset1": ["step1", "step2"]}
            },
            "Data Overview": {
                "statistics": {"dataset1": {"numeric_columns": {"col1": {}}}}
            }
        }
        
        # Get data from environment
        processed_data, cleaning_steps, statistics = self.agent.get_data_from_environment(environment)
        
        # Check results
        self.assertEqual(processed_data, {"dataset1": {"rows": 100}})
        self.assertEqual(cleaning_steps, {"dataset1": ["step1", "step2"]})
        self.assertEqual(statistics, {"dataset1": {"numeric_columns": {"col1": {}}}})
    
    def test_analyze_numeric_data(self):
        """Test analyze_numeric_data."""
        # Create test data
        dataset_name = "test_dataset"
        numeric_columns = {
            "col1": {
                "missing_percentage": 15,
                "outliers": True,
                "outlier_count": 5,
                "skewness": 1.5
            },
            "col2": {
                "missing_percentage": 0,
                "outliers": False,
                "skewness": 0.2
            }
        }
        dataset_info = {
            "correlations": {
                "col1": {"col2": 0.8},
                "col2": {"col1": 0.8}
            }
        }
        
        # Analyze numeric data
        insights = self.agent.analyze_numeric_data(dataset_name, numeric_columns, dataset_info)
        
        # Check results
        self.assertEqual(len(insights), 5)  # 1 summary + 2 for col1 + 1 for correlation
        
        # Check summary insight
        self.assertEqual(insights[0]["type"], "numeric_summary")
        self.assertEqual(insights[0]["description"], "Dataset contains 2 numeric columns")
        
        # Check missing values insight
        missing_insights = [i for i in insights if "missing values" in i["description"]]
        self.assertEqual(len(missing_insights), 1)
        self.assertEqual(missing_insights[0]["description"], "Column 'col1' has 15.0% missing values")
        
        # Check outliers insight
        outlier_insights = [i for i in insights if i["type"] == "outliers"]
        self.assertEqual(len(outlier_insights), 1)
        self.assertEqual(outlier_insights[0]["description"], "Column 'col1' contains outliers")
        
        # Check distribution insight
        distribution_insights = [i for i in insights if i["type"] == "distribution"]
        self.assertEqual(len(distribution_insights), 1)
        self.assertEqual(distribution_insights[0]["description"], "Column 'col1' shows a right-skewed distribution")
        
        # Check correlation insight
        correlation_insights = [i for i in insights if i["type"] == "correlation"]
        self.assertEqual(len(correlation_insights), 1)
        self.assertTrue("Strong positive correlation detected between 'col1' and 'col2'" in correlation_insights[0]["description"])
    
    def test_analyze_categorical_data(self):
        """Test analyze_categorical_data."""
        # Create test data
        dataset_name = "test_dataset"
        categorical_columns = {
            "cat1": {
                "missing_percentage": 10,
                "unique_count": 5,
                "total_count": 100,
                "value_counts": {"A": 80, "B": 10, "C": 10}
            },
            "cat2": {
                "missing_percentage": 0,
                "unique_count": 50,
                "total_count": 100,
                "value_counts": {"X": 2, "Y": 2, "Z": 2}  # Many more values not shown
            }
        }
        
        # Analyze categorical data
        insights = self.agent.analyze_categorical_data(dataset_name, categorical_columns)
        
        # Check results
        self.assertEqual(len(insights), 4)  # 1 summary + 1 missing + 1 cardinality + 1 imbalance
        
        # Check summary insight
        self.assertEqual(insights[0]["type"], "categorical_summary")
        self.assertEqual(insights[0]["description"], "Dataset contains 2 categorical columns")
        
        # Check missing values insight
        missing_insights = [i for i in insights if "missing values" in i["description"]]
        self.assertEqual(len(missing_insights), 1)
        self.assertEqual(missing_insights[0]["description"], "Column 'cat1' has 10.0% missing values")
        
        # Check cardinality insight
        cardinality_insights = [i for i in insights if i["type"] == "cardinality"]
        self.assertEqual(len(cardinality_insights), 1)
        self.assertEqual(cardinality_insights[0]["description"], "Column 'cat2' has high cardinality")
        
        # Check imbalance insight
        imbalance_insights = [i for i in insights if i["type"] == "imbalance"]
        self.assertEqual(len(imbalance_insights), 1)
        self.assertEqual(imbalance_insights[0]["description"], "Column 'cat1' shows significant imbalance")
    
    def test_analyze_datetime_data(self):
        """Test analyze_datetime_data."""
        # Create test data
        dataset_name = "test_dataset"
        datetime_columns = {
            "date1": {
                "missing_percentage": 5,
                "min_date": "2020-01-01",
                "max_date": "2020-12-31",
                "seasonality": "quarterly"
            },
            "date2": {
                "missing_percentage": 0,
                "min_date": "2019-01-01",
                "max_date": "2021-12-31"
            }
        }
        
        # Analyze datetime data
        insights = self.agent.analyze_datetime_data(dataset_name, datetime_columns)
        
        # Check results
        self.assertEqual(len(insights), 5)  # 1 summary + 1 missing + 2 date ranges + 1 seasonality
        
        # Check summary insight
        self.assertEqual(insights[0]["type"], "datetime_summary")
        self.assertEqual(insights[0]["description"], "Dataset contains 2 datetime columns")
        
        # Check missing values insight
        missing_insights = [i for i in insights if "missing values" in i["description"]]
        self.assertEqual(len(missing_insights), 1)
        self.assertEqual(missing_insights[0]["description"], "Column 'date1' has 5.0% missing values")
        
        # Check date range insights
        date_range_insights = [i for i in insights if i["type"] == "date_range"]
        self.assertEqual(len(date_range_insights), 2)
        
        # Check seasonality insight
        seasonality_insights = [i for i in insights if i["type"] == "time_series"]
        self.assertEqual(len(seasonality_insights), 1)
        self.assertEqual(seasonality_insights[0]["description"], "Column 'date1' shows quarterly seasonality")
    
    def test_generate_findings(self):
        """Test generate_findings."""
        # Create test data
        dataset_name = "test_dataset"
        dataset_insights = [
            {
                "type": "general",
                "description": "Dataset contains 100 rows and 10 columns after cleaning",
                "importance": "medium"
            },
            {
                "type": "data_quality",
                "description": "Column 'col1' has 15.0% missing values",
                "importance": "medium"
            },
            {
                "type": "outliers",
                "description": "Column 'col2' contains outliers",
                "importance": "high"
            },
            {
                "type": "distribution",
                "description": "Column 'col3' shows a right-skewed distribution",
                "importance": "medium"
            },
            {
                "type": "correlation",
                "description": "Strong positive correlation detected between 'col1' and 'col2'",
                "importance": "high"
            },
            {
                "type": "imbalance",
                "description": "Column 'cat1' shows significant imbalance",
                "importance": "high"
            }
        ]
        numeric_columns = {"col1": {}, "col2": {}, "col3": {}}
        categorical_columns = {"cat1": {}}
        datetime_columns = {}
        
        # Generate findings
        findings = self.agent.generate_findings(
            dataset_name, dataset_insights, numeric_columns, categorical_columns, datetime_columns
        )
        
        # Check results
        self.assertIn("summary", findings)
        self.assertIn("key_variables", findings)
        self.assertIn("potential_issues", findings)
        self.assertIn("recommendations", findings)
        
        # Check key variables
        self.assertIn("col2", findings["key_variables"])
        
        # Check potential issues
        self.assertIn("Outliers present in numeric variables", findings["potential_issues"])
        self.assertIn("Skewed distributions in numeric variables", findings["potential_issues"])
        self.assertIn("Imbalance in categorical variables", findings["potential_issues"])
        
        # Check recommendations
        self.assertIn("Consider treating outliers through capping or transformation", findings["recommendations"])
        self.assertIn("Apply transformations to normalize skewed variables", findings["recommendations"])
        self.assertIn("Consider resampling techniques for imbalanced categories", findings["recommendations"])
        self.assertIn("Consider feature selection to address multicollinearity", findings["recommendations"])
    
    def test_run_with_valid_input(self):
        """Test run with valid input."""
        # Create valid input data
        input_data = {
            "environment": {
                "Cleaned Data": {
                    "processed_data": {
                        "dataset1": {
                            "rows": 100,
                            "columns": 10,
                            "type": "dataframe"
                        }
                    }
                },
                "Data Overview": {
                    "statistics": {
                        "dataset1": {
                            "numeric_columns": {
                                "col1": {"missing_percentage": 0, "outliers": False, "skewness": 0.1},
                                "col2": {"missing_percentage": 5, "outliers": True, "skewness": 1.2}
                            },
                            "categorical_columns": {
                                "cat1": {"missing_percentage": 0, "unique_count": 3, "total_count": 100, 
                                         "value_counts": {"A": 80, "B": 10, "C": 10}}
                            }
                        }
                    }
                }
            },
            "goals": ["Analyze data for insights"]
        }
        
        # Run the agent
        result = self.agent.run(input_data)
        
        # Check results
        self.assertIn("Analysis Results.insights", result)
        self.assertIn("Analysis Results.findings", result)
        self.assertIn("dataset1", result["Analysis Results.insights"])
        self.assertIn("dataset1", result["Analysis Results.findings"])
    
    def test_run_with_invalid_input(self):
        """Test run with invalid input."""
        # Create invalid input data
        input_data = {
            "environment": {}
        }
        
        # Run the agent
        result = self.agent.run(input_data)
        
        # Check results
        self.assertIn("Analysis Results.error", result)
        self.assertEqual(
            result["Analysis Results.error"],
            "No data found in environment. Need either 'Cleaned Data' or 'Data Overview'"
        )
    
    def test_run_with_error_in_dataset(self):
        """Test run with error in dataset processing."""
        # Create input data with an error dataset
        input_data = {
            "environment": {
                "Cleaned Data": {
                    "processed_data": {
                        "dataset1": {
                            "type": "error",
                            "error": "Failed to load dataset"
                        },
                        "dataset2": {
                            "rows": 100,
                            "columns": 10
                        }
                    }
                },
                "Data Overview": {
                    "statistics": {
                        "dataset2": {
                            "numeric_columns": {"col1": {}}
                        }
                    }
                }
            }
        }
        
        # Run the agent
        result = self.agent.run(input_data)
        
        # Check results
        self.assertIn("Analysis Results.insights", result)
        self.assertIn("Analysis Results.findings", result)
        self.assertNotIn("dataset1", result["Analysis Results.insights"])
        self.assertIn("dataset2", result["Analysis Results.insights"])


if __name__ == "__main__":
    unittest.main()