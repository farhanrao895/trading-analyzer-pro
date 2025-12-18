import { NextResponse } from 'next/server'

const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'  // Fallback

export async function GET(
  request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    
    // Try Bybit first (not blocked)
    const bybitRes = await fetch(`${BYBIT_BASE_URL}/tickers?category=spot&symbol=${symbol}`)
    if (bybitRes.ok) {
      const bybitData = await bybitRes.json()
      if (bybitData.retCode === 0 && bybitData.result?.list?.length > 0) {
        const ticker = bybitData.result.list[0]
        const prevPrice = parseFloat(ticker.prevPrice24h || ticker.lastPrice)
        const currentPrice = parseFloat(ticker.lastPrice)
        
        return NextResponse.json({
          symbol: ticker.symbol,
          current_price: currentPrice,
          price_change_24h: currentPrice - prevPrice,
          price_change_pct: parseFloat(ticker.price24hPcnt || 0) * 100,
          high_24h: parseFloat(ticker.highPrice24h || ticker.lastPrice),
          low_24h: parseFloat(ticker.lowPrice24h || ticker.lastPrice),
          volume_24h: parseFloat(ticker.volume24h || 0),
          quote_volume: parseFloat(ticker.turnover24h || 0),
          open_price: prevPrice,
          weighted_avg_price: currentPrice
        })
      }
    }
    
    // Fallback to Binance
    const res = await fetch(`${BINANCE_BASE_URL}/ticker/24hr?symbol=${symbol}`)
    if (res.ok) {
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
    }
    
    return NextResponse.json(
      { error: `Symbol ${symbol} not found` },
      { status: 404 }
    )
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch price' },
      { status: 500 }
    )
  }
}

