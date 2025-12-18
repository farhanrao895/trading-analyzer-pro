import { NextRequest, NextResponse } from 'next/server'

// Get backend URL from environment or use default
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

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

    // Fetch Binance data from Vercel (not blocked) before sending to backend
    const [priceRes, klinesRes, depthRes] = await Promise.all([
      fetch(`${BINANCE_BASE_URL}/ticker/24hr?symbol=${symbol}`).catch(() => null),
      fetch(`${BINANCE_BASE_URL}/klines?symbol=${symbol}&interval=${timeframe}&limit=500`).catch(() => null),
      fetch(`${BINANCE_BASE_URL}/depth?symbol=${symbol}&limit=20`).catch(() => null)
    ])

    const priceData = priceRes?.ok ? await priceRes.json() : null
    const klinesData = klinesRes?.ok ? await klinesRes.json() : null
    const depthData = depthRes?.ok ? await depthRes.json() : null

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
