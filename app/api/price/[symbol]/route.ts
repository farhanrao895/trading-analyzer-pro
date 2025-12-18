import { NextResponse } from 'next/server'

const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

export async function GET(
  request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    const res = await fetch(`${BINANCE_BASE_URL}/ticker/24hr?symbol=${symbol}`)
    
    if (!res.ok) {
      return NextResponse.json(
        { error: `Symbol ${symbol} not found` },
        { status: 404 }
      )
    }
    
    const data = await res.json()
    
    return NextResponse.json({
      symbol: data.symbol,
      current_price: parseFloat(data.lastPrice),
      price_change_24h: parseFloat(data.priceChange),
      price_change_pct: parseFloat(data.priceChangePercent),
      high_24h: parseFloat(data.highPrice),
      low_24h: parseFloat(data.lowPrice),
      volume_24h: parseFloat(data.volume),
      quote_volume: parseFloat(data.quoteVolume),
      open_price: parseFloat(data.openPrice),
      weighted_avg_price: parseFloat(data.weightedAvgPrice)
    })
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch price' },
      { status: 500 }
    )
  }
}

