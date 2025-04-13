import axios from 'axios';

// Define the base URL for the API
const API_BASE_URL = 'http://localhost:8000';

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload and analyze a file with user instructions
 * @param file The file to analyze
 * @param userInstructions User-provided instructions for analysis
 * @returns Promise with the analysis result
 */
export const analyzeFile = async (file: File, userInstructions: string) => {
  // Create a FormData object to send the file
  const formData = new FormData();
  formData.append('file', file);
  formData.append('user_instructions', userInstructions);

  try {
    const response = await api.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // console.log(response)
    
    // // Extract markdown content between code fences (```)
    // let markdownContent = '';
    // const responseText = typeof response.data === 'string' ? response.data : JSON.stringify(response.data);
    // const markdownRegex = /```([\s\S]*?)```/g;
    // const matches = responseText.match(markdownRegex);

    // if (matches && matches.length > 0) {
    //   // Join all markdown blocks found
    //   markdownContent = matches.map(match => match.replace(/```/g, '').trim()).join('\n\n');
    //   // response.data = { 
    //   //   ...response.data,
    //   //   extractedMarkdown: markdownContent 
    //   // };
    // } else {
    //   markdownContent = responseText;
    // }
    return {
      success: true,
      message: 'Analysis completed successfully',
      data: response,
    };
  } catch (error) {
    console.error('Error analyzing file:', error);
    return {
      success: false,
      message: 'Failed to analyze file',
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
};