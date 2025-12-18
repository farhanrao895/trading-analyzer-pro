import { NextResponse } from 'next/server'

const COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
const OKX_BASE_URL = 'https://www.okx.com/api/v5/market'
const KRAKEN_BASE_URL = 'https://api.kraken.com/0/public'
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
  'INJUSDT': 'injective-protocol', 'TRXUSDT': 'tron', 'TONUSDT': 'the-open-network'
}

const SYMBOL_TO_KRAKEN: Record<string, string> = {
  'BTCUSDT': 'XXBTZUSD', 'ETHUSDT': 'XETHZUSD', 'SOLUSDT': 'SOLUSD',
  'XRPUSDT': 'XXRPZUSD', 'ADAUSDT': 'ADAUSD', 'DOGEUSDT': 'XDGUSD',
  'DOTUSDT': 'DOTUSD', 'LINKUSDT': 'LINKUSD', 'LTCUSDT': 'XLTCZUSD'
}

export async function GET(
  request: Request,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase()
    const coinId = SYMBOL_TO_COINGECKO[symbol] || symbol.replace('USDT', '').toLowerCase()
    
    // 1. Try CoinGecko first (primary - not blocked)
    console.log(`Fetching price for ${symbol} from CoinGecko...`)
    const cgRes = await fetch(
      `${COINGECKO_BASE_URL}/coins/${coinId}?localization=false&tickers=false&community_data=false&developer_data=false`
    ).catch(() => null)
    
    if (cgRes?.ok) {
      const data = await cgRes.json()
      if (data.market_data) {
        const md = data.market_data
        console.log(`Got price from CoinGecko: ${md.current_price?.usd}`)
        return NextResponse.json({
          symbol,
          current_price: md.current_price?.usd || 0,
          price_change_24h: md.price_change_24h || 0,
          price_change_pct: md.price_change_percentage_24h || 0,
          high_24h: md.high_24h?.usd || 0,
          low_24h: md.low_24h?.usd || 0,
          volume_24h: md.total_volume?.usd || 0,
          quote_volume: md.total_volume?.usd || 0,
          open_price: (md.current_price?.usd || 0) - (md.price_change_24h || 0),
          weighted_avg_price: md.current_price?.usd || 0,
          source: 'coingecko'
        })
      }
    }
    
    // 2. Try OKX (fallback 1)
    console.log(`CoinGecko failed, trying OKX for ${symbol}...`)
    const okxSymbol = `${symbol.replace('USDT', '')}-USDT`
    const okxRes = await fetch(`${OKX_BASE_URL}/ticker?instId=${okxSymbol}`).catch(() => null)
    
    if (okxRes?.ok) {
      const okxData = await okxRes.json()
      if (okxData.code === '0' && okxData.data?.length > 0) {
        const ticker = okxData.data[0]
        const last = parseFloat(ticker.last || 0)
        const open24h = parseFloat(ticker.open24h || last)
        console.log(`Got price from OKX: ${last}`)
        return NextResponse.json({
          symbol,
          current_price: last,
          price_change_24h: last - open24h,
          price_change_pct: open24h ? ((last - open24h) / open24h * 100) : 0,
          high_24h: parseFloat(ticker.high24h || last),
          low_24h: parseFloat(ticker.low24h || last),
          volume_24h: parseFloat(ticker.vol24h || 0),
          quote_volume: parseFloat(ticker.volCcy24h || 0),
          open_price: open24h,
          weighted_avg_price: last,
          source: 'okx'
        })
      }
    }
    
    // 3. Try Kraken (fallback 2)
    const krakenPair = SYMBOL_TO_KRAKEN[symbol]
    if (krakenPair) {
      console.log(`OKX failed, trying Kraken for ${symbol}...`)
      const krakenRes = await fetch(`${KRAKEN_BASE_URL}/Ticker?pair=${krakenPair}`).catch(() => null)
      
      if (krakenRes?.ok) {
        const krakenData = await krakenRes.json()
        if (!krakenData.error?.length && krakenData.result?.[krakenPair]) {
          const ticker = krakenData.result[krakenPair]
          const last = parseFloat(ticker.c?.[0] || 0)
          const open = parseFloat(ticker.o || last)
          console.log(`Got price from Kraken: ${last}`)
          return NextResponse.json({
            symbol,
            current_price: last,
            price_change_24h: last - open,
            price_change_pct: open ? ((last - open) / open * 100) : 0,
            high_24h: parseFloat(ticker.h?.[1] || last),
            low_24h: parseFloat(ticker.l?.[1] || last),
            volume_24h: parseFloat(ticker.v?.[1] || 0),
            quote_volume: 0,
            open_price: open,
            weighted_avg_price: last,
            source: 'kraken'
          })
        }
      }
    }
    
    // 4. Try Bybit (fallback 3)
    console.log(`Kraken failed, trying Bybit for ${symbol}...`)
    const bybitRes = await fetch(`${BYBIT_BASE_URL}/tickers?category=spot&symbol=${symbol}`).catch(() => null)
    
    if (bybitRes?.ok) {
      const bybitData = await bybitRes.json()
      if (bybitData.retCode === 0 && bybitData.result?.list?.length > 0) {
        const ticker = bybitData.result.list[0]
        const currentPrice = parseFloat(ticker.lastPrice)
        const prevPrice = parseFloat(ticker.prevPrice24h || ticker.lastPrice)
        console.log(`Got price from Bybit: ${currentPrice}`)
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
          weighted_avg_price: currentPrice,
          source: 'bybit'
        })
      }
    }
    
    // 5. Try Binance (last resort)
    console.log(`Bybit failed, trying Binance for ${symbol} (last resort)...`)
    const res = await fetch(`${BINANCE_BASE_URL}/ticker/24hr?symbol=${symbol}`).catch(() => null)
    
    if (res?.ok) {
      const data = await res.json()
      console.log(`Got price from Binance: ${data.lastPrice}`)
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
        weighted_avg_price: parseFloat(data.weightedAvgPrice),
        source: 'binance'
      })
    }
    
    return NextResponse.json(
      { error: `Could not fetch price for ${symbol} from any source` },
      { status: 404 }
    )
  } catch (error: any) {
    console.error('Price fetch error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch price' },
      { status: 500 }
    )
  }
}

