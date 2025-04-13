/**
 * Service for data processing
 * Connects to the backend API for data analysis
 */

import { analyzeFile } from './api';

export interface ProcessingResult {
  success: boolean;
  message: string;
  data?: any;
  error?: string;
}

// Global file storage - this is a temporary solution
// In a production app, we would use a more robust state management solution
let currentFile: File | null = null;

/**
 * Set the current file for processing
 * @param file The file to be processed
 */
export const setCurrentFile = (file: File | null) => {
  currentFile = file;
};

/**
 * Get the current file
 * @returns The current file or null if none is set
 */
export const getCurrentFile = (): File | null => {
  return currentFile;
};

/**
 * Process data based on user instructions
 * @param instructions User-provided instructions for data processing
 * @param dataType Type of data to process (csv, json, xml, text)
 * @returns Promise with processing result
 */
export const processData = async (
  instructions: string,
  dataType: string
): Promise<ProcessingResult> => {
  // Log the processing request
  console.log(`Processing ${dataType} data with instructions: ${instructions}`);
  
  // Check if we have a file to process
  if (!currentFile) {
    return {
      success: false,
      message: 'No file available for processing',
      error: 'Please upload a file first',
    };
  }
  
  try {
    // Call the API to analyze the file with user instructions
    const result = await analyzeFile(currentFile, instructions);
    return result;
  } catch (error) {
    console.error('Error in processData:', error);
    return {
      success: false,
      message: 'Failed to process data',
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
};

/**
 * Validate user instructions
 * @param instructions User-provided instructions
 * @returns Object with validation result
 */
export const validateInstructions = (instructions: string): { 
  valid: boolean; 
  error?: string;
} => {
  if (!instructions || instructions.trim() === '') {
    return {
      valid: false,
      error: 'Instructions cannot be empty',
    };
  }
  
  if (instructions.length < 10) {
    return {
      valid: false,
      error: 'Instructions must be at least 10 characters long',
    };
  }
  
  return { valid: true };
};