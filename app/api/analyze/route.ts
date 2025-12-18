import { NextRequest, NextResponse } from 'next/server'

// Get backend URL from environment or use default
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'
const COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
const OKX_BASE_URL = 'https://www.okx.com/api/v5/market'
const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

// Symbol mapping: Trading pair -> CoinGecko ID
const SYMBOL_TO_COINGECKO: Record<string, string> = {
  'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum', 'BNBUSDT': 'binancecoin',
  'SOLUSDT': 'solana', 'XRPUSDT': 'ripple', 'ADAUSDT': 'cardano',
  'DOGEUSDT': 'dogecoin', 'AVAXUSDT': 'avalanche-2', 'DOTUSDT': 'polkadot',
  'MATICUSDT': 'matic-network', 'LINKUSDT': 'chainlink', 'ATOMUSDT': 'cosmos',
  'LTCUSDT': 'litecoin', 'UNIUSDT': 'uniswap', 'NEARUSDT': 'near',
  'APTUSDT': 'aptos', 'OPUSDT': 'optimism', 'ARBUSDT': 'arbitrum',
  'SUIUSDT': 'sui', 'SEIUSDT': 'sei-network', 'PEPEUSDT': 'pepe',
  'SHIBUSDT': 'shiba-inu', 'WIFUSDT': 'dogwifcoin', 'BONKUSDT': 'bonk',
  'INJUSDT': 'injective-protocol'
}

const OKX_INTERVAL_MAP: Record<string, string> = {
  '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
  '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
}

const INTERVAL_TO_DAYS: Record<string, number> = {
  '1m': 1, '5m': 1, '15m': 1, '30m': 1,
  '1h': 1, '4h': 1, '1d': 1, '1w': 7
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const symbol = (formData.get('symbol') as string)?.toUpperCase() || 'BTCUSDT'
    const timeframe = (formData.get('timeframe') as string)?.toLowerCase() || '4h'

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    let priceData = null
    let klinesData = null
    let depthData = null

    // 1. Try CoinGecko for price (primary - not blocked)
    console.log(`Fetching price for analysis from CoinGecko...`)
    const coinId = SYMBOL_TO_COINGECKO[symbol] || symbol.replace('USDT', '').toLowerCase()
    const cgPriceRes = await fetch(
      `${COINGECKO_BASE_URL}/coins/${coinId}?localization=false&tickers=false&community_data=false&developer_data=false`
    ).catch(() => null)
    
    if (cgPriceRes?.ok) {
      const data = await cgPriceRes.json()
      if (data.market_data) {
        const md = data.market_data
        priceData = {
          symbol,
          current_price: md.current_price?.usd || 0,
          price_change_24h: md.price_change_24h || 0,
          price_change_pct: md.price_change_percentage_24h || 0,
          high_24h: md.high_24h?.usd || 0,
          low_24h: md.low_24h?.usd || 0,
          volume_24h: md.total_volume?.usd || 0,
          quote_volume: md.total_volume?.usd || 0
        }
        console.log(`Got price from CoinGecko: ${priceData.current_price}`)
      }
    }
    
    // Fallback to OKX for price
    if (!priceData) {
      console.log(`CoinGecko failed, trying OKX for price...`)
      const okxSymbol = `${symbol.replace('USDT', '')}-USDT`
      const okxRes = await fetch(`${OKX_BASE_URL}/ticker?instId=${okxSymbol}`).catch(() => null)
      if (okxRes?.ok) {
        const okxData = await okxRes.json()
        if (okxData.code === '0' && okxData.data?.length > 0) {
          const ticker = okxData.data[0]
          const last = parseFloat(ticker.last || 0)
          const open24h = parseFloat(ticker.open24h || last)
          priceData = {
            symbol,
            current_price: last,
            price_change_24h: last - open24h,
            price_change_pct: open24h ? ((last - open24h) / open24h * 100) : 0,
            high_24h: parseFloat(ticker.high24h || last),
            low_24h: parseFloat(ticker.low24h || last),
            volume_24h: parseFloat(ticker.vol24h || 0),
            quote_volume: parseFloat(ticker.volCcy24h || 0)
          }
          console.log(`Got price from OKX: ${priceData.current_price}`)
        }
      }
    }
    
    // 2. Try OKX for klines (better kline data)
    console.log(`Fetching klines for analysis from OKX...`)
    const okxSymbol = `${symbol.replace('USDT', '')}-USDT`
    const okxInterval = OKX_INTERVAL_MAP[timeframe] || '1H'
    const okxKlinesRes = await fetch(
      `${OKX_BASE_URL}/candles?instId=${okxSymbol}&bar=${okxInterval}&limit=100`
    ).catch(() => null)
    
    if (okxKlinesRes?.ok) {
      const okxData = await okxKlinesRes.json()
      if (okxData.code === '0' && okxData.data?.length > 0) {
        klinesData = okxData.data.reverse().map((k: any[]) => ({
          open_time: parseInt(k[0]),
          open: parseFloat(k[1]),
          high: parseFloat(k[2]),
          low: parseFloat(k[3]),
          close: parseFloat(k[4]),
          volume: parseFloat(k[5])
        }))
        console.log(`Got ${klinesData.length} klines from OKX`)
      }
    }
    
    // Fallback to CoinGecko OHLC for klines
    if (!klinesData) {
      console.log(`OKX klines failed, trying CoinGecko OHLC...`)
      const days = INTERVAL_TO_DAYS[timeframe] || 1
      const cgOhlcRes = await fetch(
        `${COINGECKO_BASE_URL}/coins/${coinId}/ohlc?vs_currency=usd&days=${days}`
      ).catch(() => null)
      
      if (cgOhlcRes?.ok) {
        const cgData = await cgOhlcRes.json()
        if (Array.isArray(cgData) && cgData.length > 0) {
          klinesData = cgData.map((candle: number[]) => ({
            open_time: candle[0],
            open: candle[1],
            high: candle[2],
            low: candle[3],
            close: candle[4],
            volume: 0
          }))
          console.log(`Got ${klinesData.length} klines from CoinGecko`)
        }
      }
    }
    
    // 3. Try OKX for depth (order book)
    console.log(`Fetching depth for analysis from OKX...`)
    const okxDepthRes = await fetch(
      `${OKX_BASE_URL}/books?instId=${okxSymbol}&sz=20`
    ).catch(() => null)
    
    if (okxDepthRes?.ok) {
      const okxData = await okxDepthRes.json()
      if (okxData.code === '0' && okxData.data?.length > 0) {
        depthData = okxData.data[0]
        console.log(`Got depth from OKX`)
      }
    }
    
    // Fallback to Bybit for any missing data
    if (!priceData || !klinesData || !depthData) {
      console.log(`Falling back to Bybit for missing data...`)
      const bybitInterval = { '1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '4h': '240', '1d': 'D', '1w': 'W' }[timeframe] || '240'
      const [bybitPriceRes, bybitKlinesRes, bybitDepthRes] = await Promise.all([
        !priceData ? fetch(`${BYBIT_BASE_URL}/tickers?category=spot&symbol=${symbol}`).catch(() => null) : null,
        !klinesData ? fetch(`${BYBIT_BASE_URL}/kline?category=spot&symbol=${symbol}&interval=${bybitInterval}&limit=200`).catch(() => null) : null,
        !depthData ? fetch(`${BYBIT_BASE_URL}/orderbook?category=spot&symbol=${symbol}&limit=20`).catch(() => null) : null
      ])
      
      if (!priceData && bybitPriceRes?.ok) {
        const data = await bybitPriceRes.json()
        if (data.retCode === 0 && data.result?.list?.length > 0) {
          const ticker = data.result.list[0]
          priceData = {
            symbol,
            current_price: parseFloat(ticker.lastPrice),
            price_change_24h: parseFloat(ticker.lastPrice) - parseFloat(ticker.prevPrice24h || ticker.lastPrice),
            price_change_pct: parseFloat(ticker.price24hPcnt || 0) * 100,
            high_24h: parseFloat(ticker.highPrice24h || ticker.lastPrice),
            low_24h: parseFloat(ticker.lowPrice24h || ticker.lastPrice),
            volume_24h: parseFloat(ticker.volume24h || 0),
            quote_volume: parseFloat(ticker.turnover24h || 0)
          }
          console.log(`Got price from Bybit: ${priceData.current_price}`)
        }
      }
      if (!klinesData && bybitKlinesRes?.ok) {
        const data = await bybitKlinesRes.json()
        if (data.retCode === 0 && data.result?.list?.length > 0) {
          klinesData = data.result.list.reverse().map((k: any[]) => ({
            open_time: parseInt(k[0]),
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5])
          }))
          console.log(`Got ${klinesData.length} klines from Bybit`)
        }
      }
      if (!depthData && bybitDepthRes?.ok) {
        const data = await bybitDepthRes.json()
        if (data.retCode === 0) depthData = data.result
        console.log(`Got depth from Bybit`)
      }
    }

    // Forward to Python backend with data
    const analyzeFormData = new FormData()
    analyzeFormData.append('file', file)
    analyzeFormData.append('symbol', symbol)
    analyzeFormData.append('timeframe', timeframe)
    if (priceData) analyzeFormData.append('price_data_json', JSON.stringify(priceData))
    if (klinesData) analyzeFormData.append('klines_data_json', JSON.stringify(klinesData))
    if (depthData) analyzeFormData.append('depth_data_json', JSON.stringify(depthData))

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
