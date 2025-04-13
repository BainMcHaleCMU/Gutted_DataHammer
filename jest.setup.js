// Import jest-dom matchers
import '@testing-library/jest-dom';

// Mock next/dynamic to avoid issues with dynamic imports in tests
jest.mock('next/dynamic', () => ({
  __esModule: true,
  default: (...args) => {
    const dynamicModule = jest.requireActual('next/dynamic');
    const mockDynamic = dynamicModule.default;
    const [importFunc, options] = args;
    const component = importFunc();
    return mockDynamic(() => component, {
      ...options,
      loading: () => null,
    });
  },
}));

// Mock useToast from Chakra UI
jest.mock('@chakra-ui/react', () => {
  const originalModule = jest.requireActual('@chakra-ui/react');
  return {
    __esModule: true,
    ...originalModule,
    useToast: () => jest.fn(),
  };
});