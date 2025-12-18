import { NextResponse } from 'next/server'

const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'  // Fallback

const INTERVAL_MAP: Record<string, string> = {
  '1m': '1', '5m': '5', '15m': '15', '30m': '30',
  '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
}

export async function GET(
  request: Request,
  { params }: { params: { symbol: string; interval: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    const interval = params.interval
    const limit = Math.min(parseInt(new URL(request.url).searchParams.get('limit') || '200'), 200)
    
    // Try Bybit first
    const bybitInterval = INTERVAL_MAP[interval.toLowerCase()] || interval
    const bybitRes = await fetch(
      `${BYBIT_BASE_URL}/kline?category=spot&symbol=${symbol}&interval=${bybitInterval}&limit=${limit}`
    )
    
    if (bybitRes.ok) {
      const bybitData = await bybitRes.json()
      if (bybitData.retCode === 0 && bybitData.result?.list) {
        const klines = bybitData.result.list
          .reverse()  // Bybit returns newest first, reverse for chronological
          .map((k: any[]) => ({
            open_time: parseInt(k[0]),
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
            close_time: parseInt(k[0]) + (parseInt(bybitInterval) * 60000 || 86400000),
            quote_volume: parseFloat(k[6] || k[5]) * parseFloat(k[4]),
            trades: 0
          }))
        
        return NextResponse.json({ symbol, interval, klines })
      }
    }
    
    // Fallback to Binance
    const res = await fetch(
      `${BINANCE_BASE_URL}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    )
    
    if (res.ok) {
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
      
      return NextResponse.json({ symbol, interval, klines })
    }
    
    return NextResponse.json(
      { error: `Failed to fetch klines for ${symbol}` },
      { status: 404 }
    )
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch klines' },
      { status: 500 }
    )
  }
}

