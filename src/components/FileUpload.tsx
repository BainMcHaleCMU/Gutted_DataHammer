'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Box,
  Center,
  Icon,
  Text,
  useToast,
  Button,
  Flex,
  CloseButton
} from '@chakra-ui/react'

// Update props to include onFileDelete
interface FileUploadProps {
  onFileSelect: (file: File, dataType: string) => void;
  onFileDelete?: () => void;
  selectedFile: File | null;
}

// Helper function to infer data type from file extension
const inferDataType = (filename: string): string => {
  const extension = filename.split('.').pop()?.toLowerCase() || '';
  
  switch (extension) {
    case 'csv':
      return 'csv';
    case 'json':
      return 'json';
    case 'xml':
      return 'xml';
    case 'txt':
      return 'text';
    case 'xls':
    case 'xlsx':
      return 'csv'; // Spreadsheets typically convert to CSV for processing
    default:
      return 'csv'; // Default to CSV if unknown
  }
};

export default function FileUpload({ onFileSelect, onFileDelete, selectedFile }: FileUploadProps) {
  const toast = useToast()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      
      // Infer the data type from the file extension
      const dataType = inferDataType(selectedFile.name);
      
      // Pass both file and inferred data type to parent component
      onFileSelect(selectedFile, dataType);
      
      toast({
        title: 'File uploaded',
        description: `${selectedFile.name} is ready for analysis`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    }
  }, [toast, onFileSelect])
  
  const onDropRejected = useCallback(() => {
    toast({
      title: 'File not supported',
      description: 'Please upload a file with a supported format (.csv, .xls, .xlsx, .json, .xml, .txt)',
      status: 'error',
      duration: 4000,
      isClosable: true,
    })
  }, [toast])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'text/xml': ['.xml'],
      'application/xml': ['.xml'],
      'text/plain': ['.txt']
    },
    maxFiles: 1
  })

  // If we have a file, display it with delete option
  if (selectedFile) {
    return (
      <Box
        w="full"
        p={4}
        borderWidth={1}
        borderRadius="md"
        borderColor="gray.300"
        bg="gray.50"
      >
        <Flex justify="space-between" align="center">
          <Box>
            <Text fontWeight="bold">Selected file:</Text>
            <Text>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)</Text>
            <Text fontSize="sm" color="gray.500">
              Type: {inferDataType(selectedFile.name).toUpperCase()}
            </Text>
          </Box>
          <CloseButton 
            onClick={() => {
              if (onFileDelete) {
                onFileDelete();
                toast({
                  title: 'File removed',
                  description: 'You can now upload another file',
                  status: 'info',
                  duration: 3000,
                  isClosable: true,
                });
              }
            }} 
            size="md"
            colorScheme="red"
            aria-label="Remove file"
          />
        </Flex>
      </Box>
    )
  }

  return (
    <Box
      {...getRootProps()}
      w="full"
      p={10}
      borderWidth={2}
      borderRadius="md"
      borderStyle="dashed"
      borderColor={isDragActive ? "blue.400" : "gray.300"}
      bg={isDragActive ? "blue.50" : "gray.50"}
      cursor="pointer"
      transition="all 0.2s"
      _hover={{ borderColor: "blue.300", bg: "blue.50" }}
    >
      <input {...getInputProps()} />
      <Center flexDir="column">
        <Icon boxSize={12} color="gray.400" />
        {isDragActive ? (
          <Text mt={4} textAlign="center">Drop the file here...</Text>
        ) : (
          <Text mt={4} textAlign="center">
            Drag and drop a file here, or click to select a file
          </Text>
        )}
        <Text fontSize="sm" color="gray.500" mt={2}>
          Supported formats: .csv, .xls, .xlsx, .json, .xml, .txt
        </Text>
      </Center>
    </Box>
  )
}