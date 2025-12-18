import { NextResponse } from 'next/server'

const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

export async function GET(
  request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    const limit = new URL(request.url).searchParams.get('limit') || '20'
    
    const res = await fetch(
      `${BINANCE_BASE_URL}/depth?symbol=${symbol}&limit=${limit}`
    )
    
    if (!res.ok) {
      return NextResponse.json(
        { error: `Failed to fetch depth for ${symbol}` },
        { status: 404 }
      )
    }
    
    const data = await res.json()
    
    const bids = data.bids?.map((b: any[]) => ({
      price: parseFloat(b[0]),
      quantity: parseFloat(b[1])
    })) || []
    
    const asks = data.asks?.map((a: any[]) => ({
      price: parseFloat(a[0]),
      quantity: parseFloat(a[1])
    })) || []
    
    const largest_bid = bids.length > 0 
      ? bids.reduce((max, b) => b.quantity > max.quantity ? b : max, bids[0])
      : { price: 0, quantity: 0 }
    
    const largest_ask = asks.length > 0
      ? asks.reduce((max, a) => a.quantity > max.quantity ? a : max, asks[0])
      : { price: 0, quantity: 0 }
    
    return NextResponse.json({
      symbol,
      bids,
      asks,
      largest_bid_wall: largest_bid,
      largest_ask_wall: largest_ask,
      bid_depth: bids.reduce((sum, b) => sum + b.quantity, 0),
      ask_depth: asks.reduce((sum, a) => sum + a.quantity, 0)
    })
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch depth' },
      { status: 500 }
    )
  }
}

