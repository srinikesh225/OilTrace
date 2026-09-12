/** @type {import('next').NextConfig} */

// The browser always calls the same-origin path /api/*. Next rewrites those to
// the backend at request time (server-side), so:
//   - the deployed link is just the frontend URL (no cross-origin, no CORS),
//   - no backend URL is baked into the client bundle at build time,
//   - locally it targets the dev backend on :8000.
// Set BACKEND_ORIGIN in the deploy environment (a bare host is fine; https is
// assumed when no scheme is given).
function backendOrigin() {
  const raw = process.env.BACKEND_ORIGIN || "http://localhost:8000";
  return /^https?:\/\//.test(raw) ? raw : `https://${raw}`;
}

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backendOrigin()}/:path*` },
    ];
  },
};

module.exports = nextConfig;
