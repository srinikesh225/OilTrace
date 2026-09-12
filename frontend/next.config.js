/** @type {import('next').NextConfig} */

// Frontend and backend are deployed on separate domains. The browser calls the
// backend directly at NEXT_PUBLIC_API_URL (a NEXT_PUBLIC_* var, so it is
// inlined into the client bundle at build time). CORS on the backend allows the
// frontend origin. No rewrites/proxy: this app talks to the backend cross-origin.
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
