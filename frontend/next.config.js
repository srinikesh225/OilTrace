/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Backend base URL. Overridable at build time; defaults to the local API.
  env: {
    NEXT_PUBLIC_API_BASE:
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000",
  },
};

module.exports = nextConfig;
