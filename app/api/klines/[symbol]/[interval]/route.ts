import { NextResponse } from 'next/server'

const COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
const OKX_BASE_URL = 'https://www.okx.com/api/v5/market'
const BYBIT_BASE_URL = 'https://api.bybit.com/v5/market'
const BINANCE_BASE_URL = 'https://api.binance.com/api/v3'

// Interval mappings for different exchanges
const BYBIT_INTERVAL_MAP: Record<string, string> = {
  '1m': '1', '5m': '5', '15m': '15', '30m': '30',
  '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
}

const OKX_INTERVAL_MAP: Record<string, string> = {
  '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
  '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
}

// Map interval to CoinGecko days parameter
const INTERVAL_TO_DAYS: Record<string, number> = {
  '1m': 1, '5m': 1, '15m': 1, '30m': 1,
  '1h': 1, '4h': 1, '1d': 1, '1w': 7
}

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

export async function GET(
  request: Request,
  { params }: { params: Promise<{ symbol: string; interval: string }> | { symbol: string; interval: string } }
) {
  try {
    // Handle both Next.js 14 and 15 (params might be Promise or object)
    const resolvedParams = params instanceof Promise ? await params : params
    const symbol = resolvedParams.symbol.toUpperCase()
    const interval = resolvedParams.interval.toLowerCase()
    const limit = Math.min(parseInt(new URL(request.url).searchParams.get('limit') || '100'), 200)
    
    // 1. Try OKX first (has good kline data, not blocked)
    console.log(`Fetching klines for ${symbol} ${interval} from OKX...`)
    const okxSymbol = `${symbol.replace('USDT', '')}-USDT`
    const okxInterval = OKX_INTERVAL_MAP[interval] || '1H'
    const okxRes = await fetch(
      `${OKX_BASE_URL}/candles?instId=${okxSymbol}&bar=${okxInterval}&limit=${limit}`
    ).catch(() => null)
    
    if (okxRes?.ok) {
      const okxData = await okxRes.json()
      if (okxData.code === '0' && okxData.data?.length > 0) {
        const klines = okxData.data.reverse().map((k: any[]) => ({
          open_time: parseInt(k[0]),
          open: parseFloat(k[1]),
          high: parseFloat(k[2]),
          low: parseFloat(k[3]),
          close: parseFloat(k[4]),
          volume: parseFloat(k[5]),
          close_time: parseInt(k[0]) + 3600000,
          quote_volume: parseFloat(k[6] || 0),
          trades: 0
        }))
        console.log(`Got ${klines.length} klines from OKX`)
        return NextResponse.json({ symbol, interval, klines, source: 'okx' })
      }
    }
    
    // 2. Try CoinGecko OHLC (limited intervals)
    console.log(`OKX failed, trying CoinGecko for ${symbol}...`)
    const coinId = SYMBOL_TO_COINGECKO[symbol] || symbol.replace('USDT', '').toLowerCase()
    const days = INTERVAL_TO_DAYS[interval] || 1
    const cgRes = await fetch(
      `${COINGECKO_BASE_URL}/coins/${coinId}/ohlc?vs_currency=usd&days=${days}`
    ).catch(() => null)
    
    if (cgRes?.ok) {
      const cgData = await cgRes.json()
      if (Array.isArray(cgData) && cgData.length > 0) {
        const klines = cgData.map((candle: number[]) => ({
          open_time: candle[0],
          open: candle[1],
          high: candle[2],
          low: candle[3],
          close: candle[4],
          volume: 0,  // CoinGecko OHLC doesn't include volume
          close_time: candle[0] + 3600000,
          quote_volume: 0,
          trades: 0
        }))
        console.log(`Got ${klines.length} klines from CoinGecko`)
        return NextResponse.json({ symbol, interval, klines, source: 'coingecko' })
      }
    }
    
    // 3. Try Bybit
    console.log(`CoinGecko failed, trying Bybit for ${symbol}...`)
    const bybitInterval = BYBIT_INTERVAL_MAP[interval] || '60'
    const bybitRes = await fetch(
      `${BYBIT_BASE_URL}/kline?category=spot&symbol=${symbol}&interval=${bybitInterval}&limit=${limit}`
    ).catch(() => null)
    
    if (bybitRes?.ok) {
      const bybitData = await bybitRes.json()
      if (bybitData.retCode === 0 && bybitData.result?.list?.length > 0) {
        const klines = bybitData.result.list.reverse().map((k: any[]) => ({
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
        console.log(`Got ${klines.length} klines from Bybit`)
        return NextResponse.json({ symbol, interval, klines, source: 'bybit' })
      }
    }
    
    // 4. Try Binance (last resort)
    console.log(`Bybit failed, trying Binance for ${symbol} (last resort)...`)
    const res = await fetch(
      `${BINANCE_BASE_URL}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    ).catch(() => null)
    
    if (res?.ok) {
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
      console.log(`Got ${klines.length} klines from Binance`)
      return NextResponse.json({ symbol, interval, klines, source: 'binance' })
    }
    
    return NextResponse.json(
      { error: `Could not fetch klines for ${symbol} from any source` },
      { status: 404 }
    )
  } catch (error: any) {
    console.error('Klines fetch error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch klines' },
      { status: 500 }
    )
  }
}

