import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Trading Analyzer Pro - AI-Powered Technical Analysis',
  description: 'Advanced professional-grade chart analysis with Gemini AI, Binance integration, and comprehensive technical indicators for spot trading.',
  keywords: 'trading, technical analysis, crypto, bitcoin, AI, Gemini, chart analysis, indicators',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" 
          rel="stylesheet" 
        />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📊</text></svg>" />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}
