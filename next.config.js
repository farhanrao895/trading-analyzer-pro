/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  // Allow large images in analysis responses
  experimental: {
    largePageDataBytes: 128 * 1024 * 1024, // 128MB
  },
}

module.exports = nextConfig
