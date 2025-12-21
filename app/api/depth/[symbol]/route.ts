import { NextResponse } from 'next/server'

const OKX_BASE_URL = 'https://www.okx.com/api/v5/market'
const KRAKEN_BASE_URL = 'https://api.kraken.com/0/public'
const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

const SYMBOL_TO_KRAKEN: Record<string, string> = {
  'BTCUSDT': 'XXBTZUSD', 'ETHUSDT': 'XETHZUSD', 'SOLUSDT': 'SOLUSD',
  'XRPUSDT': 'XXRPZUSD', 'ADAUSDT': 'ADAUSD', 'DOGEUSDT': 'XDGUSD',
  'DOTUSDT': 'DOTUSD', 'LINKUSDT': 'LINKUSD', 'LTCUSDT': 'XLTCZUSD'
}

type DepthEntry = { price: number; quantity: number }

function formatDepthResponse(symbol: string, bids: DepthEntry[], asks: DepthEntry[], source: string) {
  const largest_bid = bids.length > 0 
    ? bids.reduce((max, b) => b.quantity > max.quantity ? b : max, bids[0])
    : { price: 0, quantity: 0 }
  
  const largest_ask = asks.length > 0
    ? asks.reduce((max, a) => a.quantity > max.quantity ? a : max, asks[0])
    : { price: 0, quantity: 0 }
  
  return {
    symbol,
    bids,
    asks,
    largest_bid_wall: largest_bid,
    largest_ask_wall: largest_ask,
    bid_depth: bids.reduce((sum, b) => sum + b.quantity, 0),
    ask_depth: asks.reduce((sum, a) => sum + a.quantity, 0),
    source
  }
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ symbol: string }> | { symbol: string } }
) {
  try {
    // Handle both Next.js 14 and 15 (params might be Promise or object)
    const resolvedParams = params instanceof Promise ? await params : params
    const symbol = resolvedParams.symbol.toUpperCase()
    const limit = Math.min(parseInt(new URL(request.url).searchParams.get('limit') || '20'), 50)
    
    // 1. Try OKX first (not blocked, good order book data)
    console.log(`Fetching depth for ${symbol} from OKX...`)
    const okxSymbol = `${symbol.replace('USDT', '')}-USDT`
    const okxRes = await fetch(
      `${OKX_BASE_URL}/books?instId=${okxSymbol}&sz=${limit}`
    ).catch(() => null)
    
    if (okxRes?.ok) {
      const okxData = await okxRes.json()
      if (okxData.code === '0' && okxData.data?.length > 0) {
        const book = okxData.data[0]
        const bids = (book.bids || []).map((b: any[]) => ({
          price: parseFloat(b[0]),
          quantity: parseFloat(b[1])
        }))
        const asks = (book.asks || []).map((a: any[]) => ({
          price: parseFloat(a[0]),
          quantity: parseFloat(a[1])
        }))
        if (bids.length > 0 || asks.length > 0) {
          console.log(`Got depth from OKX: ${bids.length} bids, ${asks.length} asks`)
          return NextResponse.json(formatDepthResponse(symbol, bids, asks, 'okx'))
        }
      }
    }
    
    // 2. Try Kraken
    const krakenPair = SYMBOL_TO_KRAKEN[symbol]
    if (krakenPair) {
      console.log(`OKX failed, trying Kraken for ${symbol}...`)
      const krakenRes = await fetch(
        `${KRAKEN_BASE_URL}/Depth?pair=${krakenPair}&count=${limit}`
      ).catch(() => null)
      
      if (krakenRes?.ok) {
        const krakenData = await krakenRes.json()
        if (!krakenData.error?.length && krakenData.result?.[krakenPair]) {
          const book = krakenData.result[krakenPair]
          const bids = (book.bids || []).map((b: any[]) => ({
            price: parseFloat(b[0]),
            quantity: parseFloat(b[1])
          }))
          const asks = (book.asks || []).map((a: any[]) => ({
            price: parseFloat(a[0]),
            quantity: parseFloat(a[1])
          }))
          if (bids.length > 0 || asks.length > 0) {
            console.log(`Got depth from Kraken: ${bids.length} bids, ${asks.length} asks`)
            return NextResponse.json(formatDepthResponse(symbol, bids, asks, 'kraken'))
          }
        }
      }
    }
    
    // 3. Try Bybit
    console.log(`Kraken failed, trying Bybit for ${symbol}...`)
    const bybitRes = await fetch(
      `${BYBIT_BASE_URL}/orderbook?category=spot&symbol=${symbol}&limit=${limit}`
    ).catch(() => null)
    
    if (bybitRes?.ok) {
      const bybitData = await bybitRes.json()
      if (bybitData.retCode === 0 && bybitData.result) {
        const bids = (bybitData.result.b || []).map((b: any[]) => ({
          price: parseFloat(b[0]),
          quantity: parseFloat(b[1])
        }))
        const asks = (bybitData.result.a || []).map((a: any[]) => ({
          price: parseFloat(a[0]),
          quantity: parseFloat(a[1])
        }))
        if (bids.length > 0 || asks.length > 0) {
          console.log(`Got depth from Bybit: ${bids.length} bids, ${asks.length} asks`)
          return NextResponse.json(formatDepthResponse(symbol, bids, asks, 'bybit'))
        }
      }
    }
    
    // 4. Try Binance (last resort)
    console.log(`Bybit failed, trying Binance for ${symbol} (last resort)...`)
    const res = await fetch(
      `${BINANCE_BASE_URL}/depth?symbol=${symbol}&limit=${limit}`
    ).catch(() => null)
    
    if (res?.ok) {
      const data = await res.json()
      const bids = (data.bids || []).map((b: any[]) => ({
        price: parseFloat(b[0]),
        quantity: parseFloat(b[1])
      }))
      const asks = (data.asks || []).map((a: any[]) => ({
        price: parseFloat(a[0]),
        quantity: parseFloat(a[1])
      }))
      console.log(`Got depth from Binance: ${bids.length} bids, ${asks.length} asks`)
      return NextResponse.json(formatDepthResponse(symbol, bids, asks, 'binance'))
    }
    
    // Return empty depth (non-critical data)
    console.log(`All depth sources failed for ${symbol}, returning empty`)
    return NextResponse.json(formatDepthResponse(symbol, [], [], 'none'))
  } catch (error: any) {
    console.error('Depth fetch error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch depth' },
      { status: 500 }
    )
  }
}

