import { NextResponse } from 'next/server'

const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

export async function GET(
  request: Request,
  { params }: { params: { symbol: string; interval: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    const interval = params.interval
    const limit = new URL(request.url).searchParams.get('limit') || '500'
    
    const res = await fetch(
      `${BINANCE_BASE_URL}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    )
    
    if (!res.ok) {
      return NextResponse.json(
        { error: `Failed to fetch klines for ${symbol}` },
        { status: 404 }
      )
    }
    
    const data = await res.json()
    const klines = data.map((k: any[]) => ({
      open_time: k[0],
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5]),
      close_time: k[6],
      quote_volume: parseFloat(k[7]),
      trades: k[8]
    }))
    
    return NextResponse.json({
      symbol,
      interval,
      klines
    })
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch klines' },
      { status: 500 }
    )
  }
}

