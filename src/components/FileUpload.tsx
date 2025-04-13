'use client'

import { useState, useRef } from 'react';
import { 
  Box, Button, Text, Flex, Icon, 
  useColorModeValue, VStack, HStack,
  Progress
} from '@chakra-ui/react';
import { FaUpload, FaFile, FaTrash } from 'react-icons/fa';

interface FileUploadProps {
  onFileSelect: (file: File, dataType: string) => void;
  onFileDelete: () => void;
  selectedFile: File | null;
}

export default function FileUpload({ onFileSelect, onFileDelete, selectedFile }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const fileType = file.name.split('.').pop()?.toLowerCase() || 'csv';
      onFileSelect(file, fileType);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      const fileType = file.name.split('.').pop()?.toLowerCase() || 'csv';
      onFileSelect(file, fileType);
    }
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'csv':
        return '📊';
      case 'json':
        return '📋';
      case 'xlsx':
      case 'xls':
        return '📑';
      default:
        return '📄';
    }
  };

  return (
    <Box width="100%">
      {!selectedFile ? (
        <Box
          border="2px dashed"
          borderColor={isDragging ? "blue.400" : useColorModeValue("gray.300", "gray.600")}
          borderRadius="lg"
          p={6}
          textAlign="center"
          bg={isDragging ? useColorModeValue("blue.50", "blue.900") : "transparent"}
          transition="all 0.2s"
          _hover={{ borderColor: "blue.400", bg: useColorModeValue("blue.50", "blue.900") }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv,.json,.xlsx,.xls"
            style={{ display: 'none' }}
            data-testid="file-input"
          />
          <VStack spacing={4}>
            <Icon as={FaUpload} w={10} h={10} color={useColorModeValue("blue.500", "blue.300")} />
            <Text fontSize="lg" fontWeight="medium">
              Drag and drop your file here, or
            </Text>
            <Button
              onClick={() => fileInputRef.current?.click()}
              colorScheme="blue"
              variant="outline"
              size="md"
              _hover={{
                transform: "translateY(-1px)",
                shadow: "sm",
              }}
            >
              Browse Files
            </Button>
            <Text fontSize="sm" color={useColorModeValue("gray.500", "gray.400")}>
              Supports CSV, JSON, XLSX files
            </Text>
          </VStack>
        </Box>
      ) : (
        <Box
          borderRadius="lg"
          p={4}
          bg={useColorModeValue("blue.50", "gray.700")}
          borderWidth="1px"
          borderColor={useColorModeValue("blue.200", "blue.600")}
        >
          <Flex justify="space-between" align="center">
            <HStack spacing={3}>
              <Box fontSize="2xl" lineHeight="1">
                {getFileIcon(selectedFile.name)}
              </Box>
              <VStack align="start" spacing={0}>
                <Text fontWeight="medium" isTruncated maxW="250px">
                  {selectedFile.name}
                </Text>
                <Text fontSize="sm" color={useColorModeValue("gray.600", "gray.400")}>
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </Text>
              </VStack>
            </HStack>
            <Button
              onClick={onFileDelete}
              colorScheme="red"
              variant="ghost"
              size="sm"
              leftIcon={<FaTrash />}
            >
              Remove
            </Button>
          </Flex>
          <Progress size="xs" colorScheme="blue" value={100} mt={3} />
        </Box>
      )}
    </Box>
  );
}