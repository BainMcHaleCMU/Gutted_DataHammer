import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import UserInputForm from './UserInputForm';

describe('UserInputForm', () => {
  const mockOnSubmit = jest.fn();
  
  beforeEach(() => {
    mockOnSubmit.mockClear();
  });
  
  test('renders the form with all elements', () => {
    render(<UserInputForm onSubmit={mockOnSubmit} />);
    
    // Check for heading
    expect(screen.getByText('Data Processing Instructions')).toBeInTheDocument();
    
    // Check for form elements
    expect(screen.getByLabelText(/Data Type:/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Instructions:/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Process Data/i })).toBeInTheDocument();
    
    // Check for data type options
    expect(screen.getByRole('option', { name: 'CSV' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'JSON' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'XML' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Plain Text' })).toBeInTheDocument();
  });
  
  test('shows error when submitting empty instructions', async () => {
    render(<UserInputForm onSubmit={mockOnSubmit} />);
    
    // Submit form without entering instructions
    fireEvent.click(screen.getByRole('button', { name: /Process Data/i }));
    
    // Check for error message
    expect(screen.getByText(/Please enter instructions/i)).toBeInTheDocument();
    
    // Verify onSubmit was not called
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });
  
  test('calls onSubmit with correct data when form is submitted', async () => {
    render(<UserInputForm onSubmit={mockOnSubmit} />);
    
    // Enter instructions
    fireEvent.change(screen.getByLabelText(/Instructions:/i), {
      target: { value: 'Filter rows where age > 30' },
    });
    
    // Select data type
    fireEvent.change(screen.getByLabelText(/Data Type:/i), {
      target: { value: 'json' },
    });
    
    // Submit form
    fireEvent.click(screen.getByRole('button', { name: /Process Data/i }));
    
    // Verify onSubmit was called with correct arguments
    expect(mockOnSubmit).toHaveBeenCalledWith('Filter rows where age > 30', 'json');
  });
  
  test('disables submit button while processing', async () => {
    // Mock implementation that doesn't resolve immediately
    const slowMockOnSubmit = jest.fn(() => {
      return new Promise(resolve => {
        setTimeout(resolve, 100);
      });
    });
    
    render(<UserInputForm onSubmit={slowMockOnSubmit} />);
    
    // Enter instructions
    fireEvent.change(screen.getByLabelText(/Instructions:/i), {
      target: { value: 'Calculate average of salary column' },
    });
    
    // Submit form
    fireEvent.click(screen.getByRole('button', { name: /Process Data/i }));
    
    // Button should be disabled and show "Processing..."
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent('Processing...');
    
    // Wait for processing to complete
    await waitFor(() => {
      expect(button).not.toBeDisabled();
      expect(button).toHaveTextContent('Process Data');
    });
  });
});