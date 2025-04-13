/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Add basePath for GitHub Pages deployment only in production
  basePath: process.env.NODE_ENV === 'production' ? '/DataHammer' : '',
};

module.exports = nextConfig;