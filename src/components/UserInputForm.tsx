import React, { useState } from 'react';
import './UserInputForm.css';

interface UserInputFormProps {
  onSubmit: (instructions: string, dataType: string) => void;
}

const UserInputForm: React.FC<UserInputFormProps> = ({ onSubmit }) => {
  const [instructions, setInstructions] = useState('');
  const [dataType, setDataType] = useState('csv');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!instructions.trim()) {
      setError('Please enter instructions for data processing');
      return;
    }
    
    setError('');
    setIsProcessing(true);
    
    try {
      // Call the onSubmit prop with the form data
      onSubmit(instructions, dataType);
      
      // Reset form after successful submission
      setInstructions('');
    } catch (err) {
      setError('An error occurred while processing your request');
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="user-input-form-container">
      <h2>Data Processing Instructions</h2>
      <p className="form-description">
        Describe what you want the system to do with your data
      </p>
      
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="dataType">Data Type:</label>
          <select
            id="dataType"
            value={dataType}
            onChange={(e) => setDataType(e.target.value)}
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
            <option value="xml">XML</option>
            <option value="text">Plain Text</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="instructions">Instructions:</label>
          <textarea
            id="instructions"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            placeholder="Example: Filter rows where the 'age' column is greater than 30, then calculate the average of the 'salary' column"
            rows={5}
            required
          />
        </div>
        
        <button 
          type="submit" 
          className="submit-button"
          disabled={isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Process Data'}
        </button>
      </form>
    </div>
  );
};

export default UserInputForm;