import { NextResponse } from 'next/server'

const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'
const POPULAR_PAIRS = [
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
  'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
  'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'NEARUSDT',
  'APTUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'SEIUSDT',
  'PEPEUSDT', 'SHIBUSDT', 'WIFUSDT', 'BONKUSDT', 'INJUSDT'
]

export async function GET() {
  try {
    const res = await fetch(`${BINANCE_BASE_URL}/exchangeInfo`, {
      next: { revalidate: 3600 } // Cache for 1 hour
    })
    const data = await res.json()
    
    const symbols = data.symbols
      ?.filter((s: any) => s.quoteAsset === 'USDT' && s.status === 'TRADING')
      .map((s: any) => s.symbol)
      .sort() || POPULAR_PAIRS
    
    return NextResponse.json({
      symbols,
      popular: POPULAR_PAIRS
    })
  } catch (error) {
    return NextResponse.json({
      symbols: POPULAR_PAIRS,
      popular: POPULAR_PAIRS
    })
  }
}

