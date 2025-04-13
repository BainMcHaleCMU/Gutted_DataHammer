'use client';

import dynamic from 'next/dynamic';
import { Box, VStack, Heading, Button, Text, Textarea, useToast } from '@chakra-ui/react';
import { useState } from 'react';
import { processData, setCurrentFile } from '@/lib/dataProcessingService';

// Use dynamic import to avoid hydration issues with client components
const FileUpload = dynamic(() => import('@/components/FileUpload'), {
  ssr: false,
});

export default function HomeClient() {
  const toast = useToast();
  const [insights, setInsights] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [dataType, setDataType] = useState('csv');

  const handleProcessingInstructions = async () => {
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please upload a file first',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!insights.trim()) {
      toast({
        title: 'No instructions provided',
        description: 'Please enter your data analysis goals',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const result = await processData(insights, dataType);
      
      if (result.success) {
        setAnalysisResult(result.data || {});
        toast({
          title: 'Success',
          description: result.message,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Error',
          description: result.error || 'An error occurred',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to process data',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Updated to receive dataType from FileUpload component and store file in service
  const handleFileSelected = (selectedFile: File, inferredDataType: string) => {
    setFile(selectedFile);
    setDataType(inferredDataType);
    // Store the file in the service for later use
    setCurrentFile(selectedFile);
  };

  // Add handler for file deletion
  const handleFileDelete = () => {
    setFile(null);
    setDataType('csv');
    setAnalysisResult(null);
    // Also clear the file in the service
    setCurrentFile(null);
  };

  return (
    <Box as="main" minH="100vh" py={8} px={4}>
      <VStack spacing={8} align="stretch" maxW="800px" mx="auto">
        <Heading as="h1" size="xl" textAlign="center">
          DataHammer
        </Heading>
        
        {/* Insights Goals Text Area */}
        <Box w="full">
          <Text mb={2} fontWeight="bold">What insights do you want from your data?</Text>
          <Textarea
            value={insights}
            onChange={(e) => setInsights(e.target.value)}
            placeholder="Example: Predict sales for the next quarter. Help me understand what inventory to stock up on."
            size="lg"
            rows={5}
            resize="vertical"
          />
        </Box>
        
        {/* File Upload Component - Now handles file display and deletion internally */}
        <Box w="full">
          <FileUpload 
            onFileSelect={handleFileSelected} 
            onFileDelete={handleFileDelete}
            selectedFile={file}
          />
        </Box>
        
        {/* Analyze Button */}
        <Button
          colorScheme="blue"
          size="lg"
          onClick={handleProcessingInstructions}
          isLoading={isProcessing}
          loadingText="Analyzing..."
          isDisabled={!file || !insights.trim()}
        >
          Analyze Data
        </Button>
        
        {/* Analysis Results */}
        {analysisResult && (
          <Box w="full" p={6} borderWidth={1} borderRadius="md" bg="white">
            <Heading as="h2" size="md" mb={4}>Analysis Results</Heading>
            <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
          </Box>
        )}
      </VStack>
    </Box>
  );
}