'use client';

import dynamic from 'next/dynamic';
import { 
  Box, VStack, Heading, Button, Text, Textarea, useToast,
  Container, useColorModeValue, Icon, Flex, Badge
} from '@chakra-ui/react';
import { useState } from 'react';
import { processData, setCurrentFile } from '@/lib/dataProcessingService';
import { FaLightbulb, FaFileAlt, FaChartBar } from 'react-icons/fa';

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

  const handleFileSelected = (selectedFile: File, inferredDataType: string) => {
    setFile(selectedFile);
    setDataType(inferredDataType);
    setCurrentFile(selectedFile);
  };

  const handleFileDelete = () => {
    setFile(null);
    setDataType('csv');
    setAnalysisResult(null);
    setCurrentFile(null);
  };

  const summary_string = analysisResult?.data?.final_report?.response?.blocks[0]?.text || '';

  return (
    <Box 
      as="main" 
      minH="100vh" 
      py={12} 
      px={4} 
      bg={useColorModeValue('gray.50', 'gray.900')}
    >
      <Container maxW="900px">
        <VStack spacing={8} align="stretch">
          <Box textAlign="center" py={4}>
            <Heading 
              as="h1" 
              size="2xl" 
              bgGradient="linear(to-r, blue.400, teal.400)" 
              bgClip="text"
              letterSpacing="tight"
            >
              DataHammer
            </Heading>
            <Text color={useColorModeValue('gray.600', 'gray.400')} mt={2}>
              Transform your raw data into actionable insights
            </Text>
          </Box>
          
          {/* Insights Goals Text Area */}
          <Box 
            w="full" 
            bg={useColorModeValue('white', 'gray.800')} 
            p={6} 
            borderRadius="lg" 
            boxShadow="md"
            borderWidth="1px"
            borderColor={useColorModeValue('gray.200', 'gray.700')}
            transition="all 0.3s"
            _hover={{ boxShadow: "lg" }}
          >
            <Flex mb={4} align="center">
              <Icon as={FaLightbulb} mr={3} color="yellow.400" boxSize={5} />
              <Text fontWeight="bold" fontSize="lg">What insights do you want from your data?</Text>
            </Flex>
            <Textarea
              value={insights}
              onChange={(e) => setInsights(e.target.value)}
              placeholder="Example: Predict sales for the next quarter. Help me understand what inventory to stock up on."
              size="md"
              rows={4}
              resize="vertical"
              borderRadius="md"
              focusBorderColor="blue.400"
              _hover={{ borderColor: 'blue.300' }}
              bg={useColorModeValue('white', 'gray.700')}
            />
          </Box>
          
          {/* File Upload Component */}
          <Box 
            w="full" 
            bg={useColorModeValue('white', 'gray.800')} 
            p={6} 
            borderRadius="lg" 
            boxShadow="md"
            borderWidth="1px"
            borderColor={useColorModeValue('gray.200', 'gray.700')}
            transition="all 0.3s"
            _hover={{ boxShadow: "lg" }}
          >
            <Flex mb={4} align="center">
              <Icon as={FaFileAlt} mr={3} color="blue.400" boxSize={5} />
              <Text fontWeight="bold" fontSize="lg">Upload Your Data File</Text>
            </Flex>
            <FileUpload 
              onFileSelect={handleFileSelected} 
              onFileDelete={handleFileDelete}
              selectedFile={file}
            />
          </Box>
          
          {/* Analyze Button */}
          <Flex justifyContent="center" mt={2}>
            <Button
              colorScheme="blue"
              size="lg"
              onClick={handleProcessingInstructions}
              isLoading={isProcessing}
              loadingText="Analyzing..."
              isDisabled={!file || !insights.trim()}
              borderRadius="full"
              px={10}
              py={6}
              _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
              transition="all 0.2s"
              fontSize="md"
              rightIcon={<FaChartBar />}
            >
              Analyze Data
            </Button>
          </Flex>
          
          {/* Analysis Results */}
          {analysisResult && (
            <Box 
              w="full" 
              p={6} 
              borderRadius="lg" 
              bg={useColorModeValue('white', 'gray.800')}
              boxShadow="lg"
              borderWidth="1px"
              borderColor={useColorModeValue('blue.100', 'blue.700')}
              mt={4}
            >
              <Flex align="center" mb={4}>
                <Icon as={FaChartBar} mr={3} color="green.400" boxSize={5} />
                <Heading as="h2" size="md">Analysis Results</Heading>
                <Badge ml={3} colorScheme="green" fontSize="sm">Complete</Badge>
              </Flex>
              <Box 
                maxH="500px" 
                overflowY="auto" 
                p={4}
                borderRadius="md"
                bg={useColorModeValue('gray.50', 'gray.700')}
              >
                <Text whiteSpace="pre-wrap" fontFamily="system-ui">{summary_string}</Text>
              </Box>
            </Box>
          )}
        </VStack>
      </Container>
    </Box>
  );
}