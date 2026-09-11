import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta
          name="description"
          content="OILTRACE — trace a satellite-observed oil slick back to the vessel that released it."
        />
        <link
          rel="icon"
          href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='5' fill='%230b1f2a'/%3E%3Cpath d='M6 21c3-6 6 3 10-3s6 4 10-2' stroke='%23e0a94a' stroke-width='2.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E"
        />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
