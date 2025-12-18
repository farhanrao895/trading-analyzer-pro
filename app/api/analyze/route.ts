import { NextRequest, NextResponse } from 'next/server'

// Get backend URL from environment or use default
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'
const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'  // Fallback

const INTERVAL_MAP: Record<string, string> = {
  '1m': '1', '5m': '5', '15m': '15', '30m': '30',
  '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const symbol = (formData.get('symbol') as string) || 'BTCUSDT'
    const timeframe = (formData.get('timeframe') as string) || '4h'

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    // Fetch Bybit data from Vercel (not blocked) before sending to backend
    const bybitInterval = INTERVAL_MAP[timeframe.toLowerCase()] || timeframe
    const [priceRes, klinesRes, depthRes] = await Promise.all([
      fetch(`${BYBIT_BASE_URL}/tickers?category=spot&symbol=${symbol}`).catch(() => null),
      fetch(`${BYBIT_BASE_URL}/kline?category=spot&symbol=${symbol}&interval=${bybitInterval}&limit=200`).catch(() => null),
      fetch(`${BYBIT_BASE_URL}/orderbook?category=spot&symbol=${symbol}&limit=20`).catch(() => null)
    ])

    let priceData = null
    let klinesData = null
    let depthData = null

    // Parse Bybit responses
    if (priceRes?.ok) {
      const bybitPrice = await priceRes.json()
      if (bybitPrice.retCode === 0) priceData = bybitPrice.result
    }
    if (klinesRes?.ok) {
      const bybitKlines = await klinesRes.json()
      if (bybitKlines.retCode === 0) klinesData = bybitKlines.result
    }
    if (depthRes?.ok) {
      const bybitDepth = await depthRes.json()
      if (bybitDepth.retCode === 0) depthData = bybitDepth.result
    }

    // Fallback to Binance if Bybit fails
    if (!priceData || !klinesData || !depthData) {
      const [binancePriceRes, binanceKlinesRes, binanceDepthRes] = await Promise.all([
        !priceData ? fetch(`${BINANCE_BASE_URL}/ticker/24hr?symbol=${symbol}`).catch(() => null) : null,
        !klinesData ? fetch(`${BINANCE_BASE_URL}/klines?symbol=${symbol}&interval=${timeframe}&limit=500`).catch(() => null) : null,
        !depthData ? fetch(`${BINANCE_BASE_URL}/depth?symbol=${symbol}&limit=20`).catch(() => null) : null
      ])
      
      if (!priceData && binancePriceRes?.ok) priceData = await binancePriceRes.json()
      if (!klinesData && binanceKlinesRes?.ok) klinesData = await binanceKlinesRes.json()
      if (!depthData && binanceDepthRes?.ok) depthData = await binanceDepthRes.json()
    }

    // Forward to Python backend with Binance data
    const analyzeFormData = new FormData()
    analyzeFormData.append('file', file)
    analyzeFormData.append('symbol', symbol)
    analyzeFormData.append('timeframe', timeframe)
    if (priceData) analyzeFormData.append('price_data', JSON.stringify(priceData))
    if (klinesData) analyzeFormData.append('klines_data', JSON.stringify(klinesData))
    if (depthData) analyzeFormData.append('depth_data', JSON.stringify(depthData))

    const response = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: analyzeFormData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Backend error' }))
      return NextResponse.json(error, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error: any) {
    console.error('Analyze error:', error)
    return NextResponse.json(
      { error: error.message || 'Analysis failed. Make sure backend is running.' },
      { status: 500 }
    )
  }
}
