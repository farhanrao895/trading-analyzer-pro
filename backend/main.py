"""
Trading Analyzer Pro - Advanced Professional Technical Analysis Assistant
==========================================================================
Implements: Binance API + Indicator Engine + Gemini AI + Chart Annotations
"""

import os
import base64
import io
import json
import re
import math
import httpx
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

load_dotenv()

# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(title="Trading Analyzer Pro API", version="2.0.0")

# CORS - Allow all origins for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel and any other origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY not set. AI analysis will fail.")
else:
    genai.configure(api_key=gemini_api_key)

# Try different Gemini models (prioritize gemini-2.5-flash)
model = None
for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash-latest', 'gemini-1.5-flash']:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Using Gemini model: {model_name}")
        break
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")

# ============================================================
# CONSTANTS
# ============================================================

# API URLs - CoinGecko primary (not blocked), fallbacks for redundancy
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
OKX_BASE_URL = "https://www.okx.com/api/v5/market"
KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
BYBIT_BASE_URL = "https://api.bybit.com/v5/market"
BINANCE_BASE_URL = "https://api.binance.com/api/v3"  # Last resort fallback

POPULAR_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "ATOMUSDT", "LTCUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "OPUSDT", "ARBUSDT", "SUIUSDT", "SEIUSDT",
    "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "BONKUSDT", "INJUSDT"
]

# Symbol mapping: Trading pair -> CoinGecko ID
SYMBOL_TO_COINGECKO = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "BNBUSDT": "binancecoin",
    "SOLUSDT": "solana", "XRPUSDT": "ripple", "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin", "AVAXUSDT": "avalanche-2", "DOTUSDT": "polkadot",
    "MATICUSDT": "matic-network", "LINKUSDT": "chainlink", "ATOMUSDT": "cosmos",
    "LTCUSDT": "litecoin", "UNIUSDT": "uniswap", "NEARUSDT": "near",
    "APTUSDT": "aptos", "OPUSDT": "optimism", "ARBUSDT": "arbitrum",
    "SUIUSDT": "sui", "SEIUSDT": "sei-network", "PEPEUSDT": "pepe",
    "SHIBUSDT": "shiba-inu", "WIFUSDT": "dogwifcoin", "BONKUSDT": "bonk",
    "INJUSDT": "injective-protocol", "TRXUSDT": "tron", "TONUSDT": "the-open-network",
    "AAVEUSDT": "aave", "MKRUSDT": "maker", "RNDRUSDT": "render-token"
}

# Symbol mapping for OKX: BTCUSDT -> BTC-USDT
def symbol_to_okx(symbol: str) -> str:
    """Convert BTCUSDT -> BTC-USDT for OKX"""
    s = symbol.upper()
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    return s

# Symbol mapping for Kraken: BTCUSDT -> XXBTZUSD
SYMBOL_TO_KRAKEN = {
    "BTCUSDT": "XXBTZUSD", "ETHUSDT": "XETHZUSD", "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XXRPZUSD", "ADAUSDT": "ADAUSD", "DOGEUSDT": "XDGUSD",
    "DOTUSDT": "DOTUSD", "LINKUSDT": "LINKUSD", "LTCUSDT": "XLTCZUSD",
    "UNIUSDT": "UNIUSD", "ATOMUSDT": "ATOMUSD", "AVAXUSDT": "AVAXUSD"
}

TIMEFRAMES = {
    "1m": "1 Minute", "5m": "5 Minutes", "15m": "15 Minutes",
    "30m": "30 Minutes", "1h": "1 Hour", "4h": "4 Hours",
    "1d": "1 Day", "1w": "1 Week"
}

# Map intervals to CoinGecko days parameter
INTERVAL_TO_DAYS = {
    "1m": 1, "5m": 1, "15m": 1, "30m": 1,
    "1h": 1, "4h": 1, "1d": 1, "1w": 7
}

# Map intervals to Bybit format
INTERVAL_MAP = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1d": "D", "1w": "W"
}

# ============================================================
# PART 1: MULTI-SOURCE API INTEGRATION
# ============================================================

async def fetch_coingecko(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch from CoinGecko API (primary - not blocked)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"{COINGECKO_BASE_URL}/{endpoint}"
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            print(f"CoinGecko API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"CoinGecko fetch error: {e}")
    return None

async def fetch_coingecko_price(symbol: str) -> Optional[Dict]:
    """Get price data from CoinGecko for a symbol"""
    coin_id = SYMBOL_TO_COINGECKO.get(symbol.upper())
    if not coin_id:
        # Try to extract base from symbol (e.g., BTCUSDT -> btc)
        base = symbol.upper().replace("USDT", "").replace("USD", "").lower()
        coin_id = base
    
    data = await fetch_coingecko("simple/price", {
        "ids": coin_id,
        "vs_currencies": "usd",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_last_updated_at": "true"
    })
    
    if data and coin_id in data:
        coin_data = data[coin_id]
        return {
            "symbol": symbol.upper(),
            "current_price": coin_data.get("usd", 0),
            "price_change_pct": coin_data.get("usd_24h_change", 0),
            "volume_24h": coin_data.get("usd_24h_vol", 0),
        }
    return None

async def fetch_coingecko_market(symbol: str) -> Optional[Dict]:
    """Get detailed market data from CoinGecko"""
    coin_id = SYMBOL_TO_COINGECKO.get(symbol.upper())
    if not coin_id:
        base = symbol.upper().replace("USDT", "").replace("USD", "").lower()
        coin_id = base
    
    data = await fetch_coingecko(f"coins/{coin_id}", {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false"
    })
    
    if data and "market_data" in data:
        md = data["market_data"]
        return {
            "symbol": symbol.upper(),
            "current_price": md.get("current_price", {}).get("usd", 0),
            "price_change_24h": md.get("price_change_24h", 0),
            "price_change_pct": md.get("price_change_percentage_24h", 0),
            "high_24h": md.get("high_24h", {}).get("usd", 0),
            "low_24h": md.get("low_24h", {}).get("usd", 0),
            "volume_24h": md.get("total_volume", {}).get("usd", 0),
            "market_cap": md.get("market_cap", {}).get("usd", 0)
        }
    return None

async def fetch_coingecko_ohlc(symbol: str, days: int = 1) -> Optional[List]:
    """Get OHLC data from CoinGecko"""
    coin_id = SYMBOL_TO_COINGECKO.get(symbol.upper())
    if not coin_id:
        base = symbol.upper().replace("USDT", "").replace("USD", "").lower()
        coin_id = base
    
    data = await fetch_coingecko(f"coins/{coin_id}/ohlc", {
        "vs_currency": "usd",
        "days": str(days)
    })
    
    if data and isinstance(data, list):
        # CoinGecko returns: [[timestamp, open, high, low, close], ...]
        klines = []
        for candle in data:
            if len(candle) >= 5:
                klines.append({
                    "open_time": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": 0,  # CoinGecko OHLC doesn't include volume
                    "close_time": candle[0] + 3600000  # Estimate
                })
        return klines
    return None

async def fetch_okx(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch from OKX API (fallback 1)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"{OKX_BASE_URL}/{endpoint}"
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "0":  # OKX success code
                    return data.get("data")
            print(f"OKX API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"OKX fetch error: {e}")
    return None

async def fetch_okx_price(symbol: str) -> Optional[Dict]:
    """Get price from OKX"""
    okx_symbol = symbol_to_okx(symbol)
    data = await fetch_okx("ticker", {"instId": okx_symbol})
    if data and len(data) > 0:
        ticker = data[0]
        last = float(ticker.get("last", 0))
        open_24h = float(ticker.get("open24h", last))
        return {
            "symbol": symbol.upper(),
            "current_price": last,
            "price_change_24h": last - open_24h,
            "price_change_pct": ((last - open_24h) / open_24h * 100) if open_24h else 0,
            "high_24h": float(ticker.get("high24h", last)),
            "low_24h": float(ticker.get("low24h", last)),
            "volume_24h": float(ticker.get("vol24h", 0)),
            "quote_volume": float(ticker.get("volCcy24h", 0))
        }
    return None

async def fetch_okx_klines(symbol: str, interval: str, limit: int = 100) -> Optional[List]:
    """Get klines from OKX"""
    okx_symbol = symbol_to_okx(symbol)
    # OKX interval mapping
    okx_interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"
    }
    okx_bar = okx_interval_map.get(interval.lower(), "1H")
    
    data = await fetch_okx("candles", {
        "instId": okx_symbol,
        "bar": okx_bar,
        "limit": str(limit)
    })
    
    if data:
        klines = []
        for k in reversed(data):  # OKX returns newest first
            klines.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[0]) + 3600000,
                "quote_volume": float(k[6]) if len(k) > 6 else 0
            })
        return klines
    return None

async def fetch_kraken(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch from Kraken API (fallback 2)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"{KRAKEN_BASE_URL}/{endpoint}"
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("error"):
                    return data.get("result")
            print(f"Kraken API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"Kraken fetch error: {e}")
    return None

async def fetch_kraken_price(symbol: str) -> Optional[Dict]:
    """Get price from Kraken"""
    kraken_pair = SYMBOL_TO_KRAKEN.get(symbol.upper())
    if not kraken_pair:
        return None
    
    data = await fetch_kraken("Ticker", {"pair": kraken_pair})
    if data and kraken_pair in data:
        ticker = data[kraken_pair]
        last = float(ticker.get("c", [0])[0])
        open_24h = float(ticker.get("o", last))
        return {
            "symbol": symbol.upper(),
            "current_price": last,
            "price_change_24h": last - open_24h,
            "price_change_pct": ((last - open_24h) / open_24h * 100) if open_24h else 0,
            "high_24h": float(ticker.get("h", [0, last])[1]),
            "low_24h": float(ticker.get("l", [0, last])[1]),
            "volume_24h": float(ticker.get("v", [0, 0])[1]),
            "quote_volume": 0
        }
    return None

async def fetch_bybit(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch from Bybit API (fallback 3)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"{BYBIT_BASE_URL}/{endpoint}"
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("retCode") == 0:
                    return data.get("result")
            print(f"Bybit API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"Bybit fetch error: {e}")
    return None

async def fetch_binance(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch from Binance API (last resort fallback)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"{BINANCE_BASE_URL}/{endpoint}"
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            print(f"Binance API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"Binance fetch error: {e}")
    return None


@app.get("/api/symbols")
async def get_symbols():
    """GET /api/symbols - Fetch all USDT trading pairs"""
    data = await fetch_binance("exchangeInfo")
    if data:
        symbols = [
            s["symbol"] for s in data.get("symbols", [])
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        ]
        return {"symbols": sorted(symbols), "popular": POPULAR_PAIRS}
    return {"symbols": POPULAR_PAIRS, "popular": POPULAR_PAIRS}


@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    """GET /api/price/{symbol} - Real-time price with multi-source fallback"""
    symbol = symbol.upper()
    
    # 1. Try CoinGecko (primary - not blocked)
    print(f"Fetching price for {symbol} from CoinGecko...")
    data = await fetch_coingecko_market(symbol)
    if data:
        print(f"Got price from CoinGecko: {data.get('current_price')}")
        return {
            "symbol": symbol,
            "current_price": data.get("current_price", 0),
            "price_change_24h": data.get("price_change_24h", 0),
            "price_change_pct": data.get("price_change_pct", 0),
            "high_24h": data.get("high_24h", 0),
            "low_24h": data.get("low_24h", 0),
            "volume_24h": data.get("volume_24h", 0),
            "quote_volume": data.get("volume_24h", 0),
            "open_price": data.get("current_price", 0) - data.get("price_change_24h", 0),
            "weighted_avg_price": data.get("current_price", 0),
            "source": "coingecko"
        }
    
    # 2. Try OKX (fallback 1)
    print(f"CoinGecko failed, trying OKX for {symbol}...")
    data = await fetch_okx_price(symbol)
    if data:
        print(f"Got price from OKX: {data.get('current_price')}")
        data["source"] = "okx"
        return data
    
    # 3. Try Kraken (fallback 2)
    print(f"OKX failed, trying Kraken for {symbol}...")
    data = await fetch_kraken_price(symbol)
    if data:
        print(f"Got price from Kraken: {data.get('current_price')}")
        data["source"] = "kraken"
        return data
    
    # 4. Try Bybit (fallback 3)
    print(f"Kraken failed, trying Bybit for {symbol}...")
    bybit_data = await fetch_bybit("tickers", {"category": "spot", "symbol": symbol})
    if bybit_data and bybit_data.get("list"):
        ticker = bybit_data["list"][0]
        print(f"Got price from Bybit: {ticker.get('lastPrice')}")
        return {
            "symbol": symbol,
            "current_price": float(ticker.get("lastPrice", 0)),
            "price_change_24h": float(ticker.get("lastPrice", 0)) - float(ticker.get("prevPrice24h", ticker.get("lastPrice", 0))),
            "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
            "high_24h": float(ticker.get("highPrice24h", 0)),
            "low_24h": float(ticker.get("lowPrice24h", 0)),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "quote_volume": float(ticker.get("turnover24h", 0)),
            "open_price": float(ticker.get("prevPrice24h", 0)),
            "weighted_avg_price": float(ticker.get("lastPrice", 0)),
            "source": "bybit"
        }
    
    # 5. Try Binance (last resort)
    print(f"Bybit failed, trying Binance for {symbol} (last resort)...")
    data = await fetch_binance("ticker/24hr", {"symbol": symbol})
    if data:
        print(f"Got price from Binance: {data.get('lastPrice')}")
        return {
            "symbol": data["symbol"],
            "current_price": float(data["lastPrice"]),
            "price_change_24h": float(data["priceChange"]),
            "price_change_pct": float(data["priceChangePercent"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
            "volume_24h": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),
            "open_price": float(data["openPrice"]),
            "weighted_avg_price": float(data["weightedAvgPrice"]),
            "source": "binance"
        }
    
    raise HTTPException(status_code=404, detail=f"Could not fetch price for {symbol} from any source")


@app.get("/api/klines/{symbol}/{interval}")
async def get_klines(symbol: str, interval: str, limit: int = Query(default=500, le=1000)):
    """GET /api/klines/{symbol}/{interval} - Historical candlestick data with multi-source fallback"""
    symbol = symbol.upper()
    
    # 1. Try OKX first (has better kline data than CoinGecko)
    print(f"Fetching klines for {symbol} {interval} from OKX...")
    klines = await fetch_okx_klines(symbol, interval, min(limit, 500))
    if klines and len(klines) > 0:
        # Ensure chronological order (oldest to newest)
        klines.sort(key=lambda x: x["open_time"])
        print(f"Got {len(klines)} klines from OKX")
        return {"symbol": symbol, "interval": interval, "klines": klines, "source": "okx"}
    
    # 2. Try CoinGecko OHLC (limited intervals)
    print(f"OKX failed, trying CoinGecko OHLC for {symbol}...")
    days = INTERVAL_TO_DAYS.get(interval.lower(), 1)
    cg_klines = await fetch_coingecko_ohlc(symbol, days)
    if cg_klines and len(cg_klines) > 0:
        # Ensure chronological order (oldest to newest)
        cg_klines.sort(key=lambda x: x["open_time"])
        print(f"Got {len(cg_klines)} klines from CoinGecko")
        return {"symbol": symbol, "interval": interval, "klines": cg_klines, "source": "coingecko"}
    
    # 3. Try Bybit
    print(f"CoinGecko failed, trying Bybit for {symbol}...")
    bybit_interval = INTERVAL_MAP.get(interval.lower(), interval)
    data = await fetch_bybit("kline", {
        "category": "spot",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": min(limit, 500)
    })
    if data and data.get("list"):
        klines = []
        for k in reversed(data["list"]):
            klines.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[0]) + (int(bybit_interval) * 60000 if bybit_interval.isdigit() else 86400000),
                "quote_volume": float(k[6]) if len(k) > 6 else float(k[5]) * float(k[4]),
                "trades": 0
            })
        # Ensure chronological order (oldest to newest)
        klines.sort(key=lambda x: x["open_time"])
        print(f"Got {len(klines)} klines from Bybit")
        return {"symbol": symbol, "interval": interval, "klines": klines, "source": "bybit"}
    
    # 4. Try Binance (last resort)
    print(f"Bybit failed, trying Binance for {symbol} (last resort)...")
    data = await fetch_binance("klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)  # Binance allows up to 1000
    })
    if data:
        klines = []
        for k in data:
            klines.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
                "quote_volume": float(k[7]),
                "trades": k[8]
            })
        # Binance returns in chronological order, but ensure it's sorted
        klines.sort(key=lambda x: x["open_time"])
        print(f"Got {len(klines)} klines from Binance")
        return {"symbol": symbol, "interval": interval, "klines": klines, "source": "binance"}
    
    raise HTTPException(status_code=404, detail=f"Could not fetch klines for {symbol} from any source")


@app.get("/api/depth/{symbol}")
async def get_depth(symbol: str, limit: int = Query(default=20, le=100)):
    """GET /api/depth/{symbol} - Order book depth data with multi-source fallback"""
    symbol = symbol.upper()
    
    def format_depth_response(bids, asks, source):
        largest_bid = max(bids, key=lambda x: x["quantity"]) if bids else {"price": 0, "quantity": 0}
        largest_ask = max(asks, key=lambda x: x["quantity"]) if asks else {"price": 0, "quantity": 0}
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "largest_bid_wall": largest_bid,
            "largest_ask_wall": largest_ask,
            "bid_depth": sum(b["quantity"] for b in bids),
            "ask_depth": sum(a["quantity"] for a in asks),
            "source": source
        }
    
    # 1. Try OKX first (CoinGecko doesn't have order book)
    print(f"Fetching depth for {symbol} from OKX...")
    okx_symbol = symbol_to_okx(symbol)
    data = await fetch_okx("books", {"instId": okx_symbol, "sz": str(limit)})
    if data and len(data) > 0:
        book = data[0]
        bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in book.get("bids", [])]
        asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in book.get("asks", [])]
        if bids or asks:
            print(f"Got depth from OKX: {len(bids)} bids, {len(asks)} asks")
            return format_depth_response(bids, asks, "okx")
    
    # 2. Try Kraken
    print(f"OKX failed, trying Kraken for {symbol}...")
    kraken_pair = SYMBOL_TO_KRAKEN.get(symbol)
    if kraken_pair:
        data = await fetch_kraken("Depth", {"pair": kraken_pair, "count": str(limit)})
        if data and kraken_pair in data:
            book = data[kraken_pair]
            bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in book.get("bids", [])]
            asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in book.get("asks", [])]
            if bids or asks:
                print(f"Got depth from Kraken: {len(bids)} bids, {len(asks)} asks")
                return format_depth_response(bids, asks, "kraken")
    
    # 3. Try Bybit
    print(f"Kraken failed, trying Bybit for {symbol}...")
    data = await fetch_bybit("orderbook", {
        "category": "spot",
        "symbol": symbol,
        "limit": min(limit, 50)
    })
    if data:
        bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in data.get("b", [])]
        asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in data.get("a", [])]
        if bids or asks:
            print(f"Got depth from Bybit: {len(bids)} bids, {len(asks)} asks")
            return format_depth_response(bids, asks, "bybit")
    
    # 4. Try Binance (last resort)
    print(f"Bybit failed, trying Binance for {symbol} (last resort)...")
    data = await fetch_binance("depth", {"symbol": symbol, "limit": limit})
    if data:
        bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in data.get("bids", [])]
        asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in data.get("asks", [])]
        if bids or asks:
            print(f"Got depth from Binance: {len(bids)} bids, {len(asks)} asks")
            return format_depth_response(bids, asks, "binance")
    
    # Return empty depth if all fail (non-critical)
    print(f"All depth sources failed for {symbol}, returning empty")
    return format_depth_response([], [], "none")


@app.get("/api/timeframes")
async def get_timeframes():
    """GET /api/timeframes - Available timeframes"""
    return {"timeframes": TIMEFRAMES}


# ============================================================
# PART 2: INDICATOR CALCULATION ENGINE
# ============================================================

class IndicatorEngine:
    """Technical indicator calculation engine"""
    
    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> Dict:
        """RSI (Relative Strength Index) - TradingView Compatible
        
        Uses Wilder's RMA with α = 1/period (NOT standard EMA α = 2/(period+1))
        First RMA value = SMA of first `period` values (seed)
        Subsequent: RMA = (previous_RMA × (period - 1) + current) / period
        """
        if len(closes) < period + 1:
            return {"value": 50.0, "signal": "neutral", "period": period}
        
        # Calculate price changes
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Separate gains and losses
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        # First average = SMA of first `period` values (TradingView seed method)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Wilder's smoothing (RMA) for subsequent values
        # Formula: RMA = (prev_RMA * (period - 1) + current) / period
        # This is equivalent to: RMA = prev_RMA + (current - prev_RMA) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Handle edge cases exactly like TradingView
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0  # No movement = neutral
        elif avg_gain == 0:
            rsi = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        rsi = round(rsi, 2)
        
        # Signal determination
        if rsi < 30:
            signal = "oversold"
        elif rsi > 70:
            signal = "overbought"
        else:
            signal = "neutral"
        
        return {"value": rsi, "signal": signal, "period": period}
    
    @staticmethod
    def calculate_ema(closes: List[float], period: int) -> float:
        """EMA (Exponential Moving Average) - TradingView compatible
        
        Formula matches TradingView's ema() function:
        - Multiplier = 2 / (period + 1)
        - Initial value = SMA of first period values
        - Subsequent: EMA = (Price × Multiplier) + (Previous EMA × (1 - Multiplier))
        """
        if len(closes) < period:
            return closes[-1] if closes else 0
        
        # TradingView uses: multiplier = 2 / (period + 1)
        multiplier = 2.0 / (period + 1.0)
        
        # Start with SMA of first period values (TradingView standard)
        ema = sum(closes[:period]) / float(period)
        
        # Calculate EMA for rest of data using TradingView's exact formula
        for price in closes[period:]:
            # Equivalent formulas:
            # ema = (price * multiplier) + (ema * (1.0 - multiplier))
            # This is the same as: ema = (price - ema) * multiplier + ema
            ema = (price * multiplier) + (ema * (1.0 - multiplier))
        
        return round(ema, 4)
    
    @staticmethod
    def calculate_ema_series(closes: List[float], period: int) -> List[float]:
        """Calculate full EMA series (optimized for MACD) - TradingView compatible
        
        Returns the complete EMA series for all data points after initial period.
        Uses TradingView's exact EMA formula for consistency.
        """
        if len(closes) < period:
            return [closes[-1]] if closes else [0]
        
        # TradingView uses: multiplier = 2 / (period + 1)
        multiplier = 2.0 / (period + 1.0)
        
        # Start with SMA of first period values (TradingView standard)
        ema = sum(closes[:period]) / float(period)
        ema_series = [ema]
        
        # Calculate EMA for rest of data using TradingView's exact formula
        for price in closes[period:]:
            ema = (price * multiplier) + (ema * (1.0 - multiplier))
            ema_series.append(ema)
        
        return ema_series
    
    @staticmethod
    def calculate_sma(closes: List[float], period: int) -> float:
        """SMA (Simple Moving Average)"""
        if len(closes) < period:
            return closes[-1] if closes else 0
        return round(sum(closes[-period:]) / period, 4)
    
    @staticmethod
    def calculate_macd(closes: List[float]) -> Dict:
        """MACD (12, 26, 9) - Optimized calculation
        
        Builds full EMA series once instead of recalculating for each period.
        This significantly improves performance.
        """
        if len(closes) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        # Calculate full EMA series for 12 and 26 periods
        ema12_series = IndicatorEngine.calculate_ema_series(closes, 12)
        ema26_series = IndicatorEngine.calculate_ema_series(closes, 26)
        
        # Build MACD line series from the difference
        # Align series: EMA26 starts at index 0 (corresponds to closes[25])
        # EMA12 starts at index 14 to align with EMA26 start
        offset = 26 - 12  # 14 periods
        macd_series = []
        
        for i in range(len(ema26_series)):
            ema12_idx = i + offset
            if ema12_idx < len(ema12_series):
                macd_val = ema12_series[ema12_idx] - ema26_series[i]
                macd_series.append(macd_val)
        
        if not macd_series:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        macd_line = macd_series[-1]
        
        # Apply 9-period EMA to MACD line series for signal line
        if len(macd_series) >= 9:
            signal_series = IndicatorEngine.calculate_ema_series(macd_series, 9)
            signal_line = signal_series[-1]
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        if histogram > 0 and macd_line > signal_line:
            trend = "bullish"
        elif histogram < 0 and macd_line < signal_line:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "macd": round(macd_line, 6),
            "signal": round(signal_line, 6),
            "histogram": round(histogram, 6),
            "trend": trend
        }
    
    @staticmethod
    def find_peaks_and_troughs(values: List[float], lookback: int = 5) -> tuple:
        """Find peaks (local maxima) and troughs (local minima) using Williams Fractals approach
        
        A valid fractal requires `lookback` bars on each side with lower highs (for peaks)
        or higher lows (for troughs). This matches TradingView's fractal detection.
        
        Args:
            values: List of price or indicator values
            lookback: Number of bars on each side to confirm (default 5, minimum 2)
        
        Returns:
            Tuple of (peaks, troughs) where each is a list of (index, value) tuples
        """
        lookback = max(2, lookback)  # Minimum 2 bars each side
        peaks = []
        troughs = []
        
        for i in range(lookback, len(values) - lookback):
            # Check for peak (fractal high)
            is_peak = True
            for j in range(1, lookback + 1):
                if values[i - j] >= values[i] or values[i + j] >= values[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append((i, values[i]))
            
            # Check for trough (fractal low)
            is_trough = True
            for j in range(1, lookback + 1):
                if values[i - j] <= values[i] or values[i + j] <= values[i]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append((i, values[i]))
        
        return peaks, troughs
    
    @staticmethod
    def detect_divergence(prices: List[float], indicator_values: List[float], lookback: int = 50) -> Dict:
        """Detect bullish/bearish divergences - Professional Implementation
        
        Uses Williams Fractals for swing detection with proper filtering:
        - Minimum 5 bars between swings
        - Maximum 60 bars between swings  
        - Minimum indicator difference of 3 points (for RSI 0-100 scale)
        - Line-of-sight validation
        
        Args:
            prices: List of closing prices
            indicator_values: List of indicator values (RSI, MACD histogram, etc.)
            lookback: Number of recent bars to analyze (default 50)
        
        Returns:
            Dict with divergence type, signal, and strength
        """
        if len(prices) < lookback or len(indicator_values) < lookback:
            return {"type": "none", "signal": "neutral", "strength": "none"}
        
        # Ensure same length
        min_len = min(len(prices), len(indicator_values))
        recent_prices = prices[-min(lookback, min_len):]
        recent_indicators = indicator_values[-min(lookback, min_len):]
        
        # Find peaks and troughs with proper lookback (5 bars each side)
        price_peaks, price_troughs = IndicatorEngine.find_peaks_and_troughs(recent_prices, lookback=5)
        ind_peaks, ind_troughs = IndicatorEngine.find_peaks_and_troughs(recent_indicators, lookback=5)
        
        divergence_type = "none"
        signal = "neutral"
        strength = "none"
        
        # Filtering parameters
        MIN_BARS_BETWEEN = 5
        MAX_BARS_BETWEEN = 60
        MIN_INDICATOR_DIFF = 3  # For RSI (0-100 scale)
        
        # Check for BULLISH DIVERGENCE (price lower low, indicator higher low)
        if len(price_troughs) >= 2 and len(ind_troughs) >= 2:
            # Get last two troughs
            pt1_idx, pt1_val = price_troughs[-2]
            pt2_idx, pt2_val = price_troughs[-1]
            it1_idx, it1_val = ind_troughs[-2]
            it2_idx, it2_val = ind_troughs[-1]
            
            bars_between = pt2_idx - pt1_idx
            
            # Apply filters
            if MIN_BARS_BETWEEN <= bars_between <= MAX_BARS_BETWEEN:
                indicator_diff = abs(it2_val - it1_val)
                
                if indicator_diff >= MIN_INDICATOR_DIFF:
                    # Regular bullish: Price lower low, indicator higher low
                    if pt2_val < pt1_val and it2_val > it1_val:
                        # Line-of-sight check: No intermediate trough lower than both
                        intermediate_valid = True
                        for idx, val in price_troughs:
                            if pt1_idx < idx < pt2_idx and val < min(pt1_val, pt2_val):
                                intermediate_valid = False
                                break
                        
                        if intermediate_valid:
                            divergence_type = "bullish_regular"
                            signal = "bullish"
                            # Strength based on indicator difference
                            strength = "strong" if indicator_diff >= 10 else "moderate" if indicator_diff >= 5 else "weak"
                    
                    # Hidden bullish: Price higher low, indicator lower low
                    elif pt2_val > pt1_val and it2_val < it1_val:
                        if divergence_type == "none":
                            divergence_type = "bullish_hidden"
                            signal = "bullish"
                            strength = "moderate"
        
        # Check for BEARISH DIVERGENCE (price higher high, indicator lower high)
        if len(price_peaks) >= 2 and len(ind_peaks) >= 2 and divergence_type == "none":
            # Get last two peaks
            pp1_idx, pp1_val = price_peaks[-2]
            pp2_idx, pp2_val = price_peaks[-1]
            ip1_idx, ip1_val = ind_peaks[-2]
            ip2_idx, ip2_val = ind_peaks[-1]
            
            bars_between = pp2_idx - pp1_idx
            
            # Apply filters
            if MIN_BARS_BETWEEN <= bars_between <= MAX_BARS_BETWEEN:
                indicator_diff = abs(ip2_val - ip1_val)
                
                if indicator_diff >= MIN_INDICATOR_DIFF:
                    # Regular bearish: Price higher high, indicator lower high
                    if pp2_val > pp1_val and ip2_val < ip1_val:
                        # Line-of-sight check
                        intermediate_valid = True
                        for idx, val in price_peaks:
                            if pp1_idx < idx < pp2_idx and val > max(pp1_val, pp2_val):
                                intermediate_valid = False
                                break
                        
                        if intermediate_valid:
                            divergence_type = "bearish_regular"
                            signal = "bearish"
                            strength = "strong" if indicator_diff >= 10 else "moderate" if indicator_diff >= 5 else "weak"
                    
                    # Hidden bearish: Price lower high, indicator higher high
                    elif pp2_val < pp1_val and ip2_val > ip1_val:
                        divergence_type = "bearish_hidden"
                        signal = "bearish"
                        strength = "moderate"
        
        return {
            "type": divergence_type,
            "signal": signal,
            "strength": strength
        }
    
    @staticmethod
    def calculate_bollinger(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Bollinger Bands - TradingView Compatible
        
        Uses POPULATION standard deviation (divides by N, not N-1)
        This is confirmed by John Bollinger himself.
        """
        if len(closes) < period:
            price = closes[-1] if closes else 0
            return {"upper": price, "middle": price, "lower": price, "bandwidth": 0, "position": "middle"}
        
        # Calculate SMA (middle band)
        sma = sum(closes[-period:]) / float(period)
        
        # Calculate POPULATION standard deviation (TradingView uses biased=true by default)
        # Population std dev divides by N, not N-1
        variance = sum((p - sma) ** 2 for p in closes[-period:]) / float(period)
        std = math.sqrt(variance)
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        bandwidth = ((upper - lower) / sma) * 100 if sma > 0 else 0
        
        current_price = closes[-1]
        if current_price <= lower:
            position = "lower_band"
        elif current_price >= upper:
            position = "upper_band"
        else:
            position = "middle"
        
        return {
            "upper": round(upper, 4),
            "middle": round(sma, 4),
            "lower": round(lower, 4),
            "bandwidth": round(bandwidth, 2),
            "position": position
        }
    
    @staticmethod
    def calculate_atr(klines: List[Dict], period: int = 14) -> float:
        """ATR (Average True Range) using Wilder's smoothing method
        
        Initial ATR = Simple Average of first 14 True Ranges
        Subsequent ATR = ((Previous ATR × 13) + Current True Range) / 14
        
        This matches RSI methodology and is more responsive to volatility changes.
        """
        if len(klines) < period + 1:
            return 0
        
        # Calculate all True Ranges
        trs = []
        for i in range(1, len(klines)):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_close = klines[i-1]["close"]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)
        
        if len(trs) < period:
            return round(sum(trs) / len(trs), 4) if trs else 0
        
        # Initial ATR = Simple Average of first 14 True Ranges
        atr = sum(trs[:period]) / period
        
        # Apply Wilder's smoothing for subsequent values
        for i in range(period, len(trs)):
            atr = ((atr * (period - 1)) + trs[i]) / period
        
        return round(atr, 4)
    
    @staticmethod
    def find_support_resistance(klines: List[Dict], num_levels: int = 3) -> Dict:
        """Find support and resistance levels from price action"""
        if len(klines) < 10:
            return {"support": [], "resistance": []}
        
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        closes = [k["close"] for k in klines]
        current_price = closes[-1]
        
        # Find pivot points (swing highs and lows)
        pivot_highs = []
        pivot_lows = []
        
        lookback = 5
        for i in range(lookback, len(klines) - lookback):
            # Check for swing high
            is_swing_high = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                           all(highs[i] > highs[i+j] for j in range(1, lookback+1))
            if is_swing_high:
                pivot_highs.append({"price": highs[i], "index": i})
            
            # Check for swing low
            is_swing_low = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                          all(lows[i] < lows[i+j] for j in range(1, lookback+1))
            if is_swing_low:
                pivot_lows.append({"price": lows[i], "index": i})
        
        # Cluster similar price levels
        def cluster_levels(levels: List[Dict], threshold_pct: float = 0.5) -> List[Dict]:
            if not levels:
                return []
            
            sorted_levels = sorted(levels, key=lambda x: x["price"])
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level["price"] - current_cluster[-1]["price"]) / current_cluster[-1]["price"] < threshold_pct / 100:
                    current_cluster.append(level)
                else:
                    avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                    clusters.append({"price": round(avg_price, 4), "touches": len(current_cluster)})
                    current_cluster = [level]
            
            if current_cluster:
                avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                clusters.append({"price": round(avg_price, 4), "touches": len(current_cluster)})
            
            return clusters
        
        support_levels = cluster_levels(pivot_lows)
        resistance_levels = cluster_levels(pivot_highs)
        
        # Filter to levels near current price
        support = [s for s in support_levels if s["price"] < current_price]
        resistance = [r for r in resistance_levels if r["price"] > current_price]
        
        # Sort and limit
        support = sorted(support, key=lambda x: x["price"], reverse=True)[:num_levels]
        resistance = sorted(resistance, key=lambda x: x["price"])[:num_levels]
        
        # Add strength rating
        for s in support:
            s["strength"] = "strong" if s["touches"] >= 3 else "moderate" if s["touches"] >= 2 else "weak"
        for r in resistance:
            r["strength"] = "strong" if r["touches"] >= 3 else "moderate" if r["touches"] >= 2 else "weak"
        
        return {"support": support, "resistance": resistance}
    
    @staticmethod
    def calculate_fibonacci(klines: List[Dict], lookback: int = 50) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if len(klines) < lookback:
            lookback = len(klines)
        
        recent_klines = klines[-lookback:]
        swing_high = max(k["high"] for k in recent_klines)
        swing_low = min(k["low"] for k in recent_klines)
        
        diff = swing_high - swing_low
        
        levels = {
            "0.0": round(swing_high, 4),
            "0.236": round(swing_high - diff * 0.236, 4),
            "0.382": round(swing_high - diff * 0.382, 4),
            "0.5": round(swing_high - diff * 0.5, 4),
            "0.618": round(swing_high - diff * 0.618, 4),
            "0.786": round(swing_high - diff * 0.786, 4),
            "1.0": round(swing_low, 4),
            # Extensions
            "1.272": round(swing_low - diff * 0.272, 4),
            "1.618": round(swing_low - diff * 0.618, 4),
        }
        
        return {
            "levels": levels,
            "swing_high": swing_high,
            "swing_low": swing_low
        }
    
    @staticmethod
    def analyze_volume(klines: List[Dict], period: int = 20) -> Dict:
        """Analyze volume trends"""
        if len(klines) < period:
            return {"current": 0, "average": 0, "ratio": 1.0, "trend": "neutral"}
        
        volumes = [k["volume"] for k in klines]
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-period:]) / period
        
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check if volume is increasing on up moves or down moves
        recent_closes = [k["close"] for k in klines[-10:]]
        recent_volumes = volumes[-10:]
        
        up_volume = sum(v for i, v in enumerate(recent_volumes[1:], 1) 
                       if recent_closes[i] > recent_closes[i-1])
        down_volume = sum(v for i, v in enumerate(recent_volumes[1:], 1) 
                         if recent_closes[i] < recent_closes[i-1])
        
        if up_volume > down_volume * 1.2:
            trend = "bullish"
        elif down_volume > up_volume * 1.2:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "current": round(current_volume, 2),
            "average": round(avg_volume, 2),
            "ratio": round(ratio, 2),
            "trend": trend,
            "up_volume": round(up_volume, 2),
            "down_volume": round(down_volume, 2)
        }
    
    @staticmethod
    def calculate_obv(klines: List[Dict]) -> Dict:
        """On-Balance Volume (OBV) - TradingView Compatible
        
        Rules:
        - If close > prev_close: OBV = prev_OBV + volume
        - If close < prev_close: OBV = prev_OBV - volume  
        - If close = prev_close: OBV = prev_OBV (unchanged)
        
        Starting value: First bar's volume (positive or negative based on implicit direction)
        TradingView uses: cumsum(sign(change(close)) * volume)
        """
        if len(klines) < 2:
            return {"obv": 0, "trend": "neutral", "divergence": "none", "signal": "neutral"}
        
        closes = [k["close"] for k in klines]
        volumes = [k.get("volume", 0) for k in klines]
        
        # TradingView OBV calculation
        # First value: We need at least 2 bars to determine direction
        # Start OBV at 0, then accumulate based on price direction
        obv_series = [0]  # First bar has no previous reference, start at 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv_series.append(obv_series[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv_series.append(obv_series[-1] - volumes[i])
            else:
                obv_series.append(obv_series[-1])  # Unchanged when price unchanged
        
        current_obv = obv_series[-1]
        
        # Calculate OBV trend (using last 10 periods)
        lookback = min(10, len(obv_series) - 1)
        if lookback >= 2:
            obv_start = obv_series[-lookback]
            obv_end = obv_series[-1]
            
            # Avoid division by zero
            if obv_start != 0:
                obv_change_pct = ((obv_end - obv_start) / abs(obv_start)) * 100
            else:
                obv_change_pct = 100 if obv_end > 0 else (-100 if obv_end < 0 else 0)
            
            if obv_change_pct > 5:
                obv_trend = "increasing"
            elif obv_change_pct < -5:
                obv_trend = "decreasing"
            else:
                obv_trend = "neutral"
        else:
            obv_trend = "neutral"
            obv_change_pct = 0
        
        # Check for OBV vs Price divergence
        divergence = "none"
        signal = "neutral"
        
        if lookback >= 5:
            price_start = closes[-lookback]
            price_end = closes[-1]
            
            if price_start != 0:
                price_change_pct = ((price_end - price_start) / price_start) * 100
            else:
                price_change_pct = 0
            
            # Bullish divergence: Price down, OBV up
            if price_change_pct < -2 and obv_change_pct > 3:
                divergence = "bullish"
                signal = "bullish"
            # Bearish divergence: Price up, OBV down
            elif price_change_pct > 2 and obv_change_pct < -3:
                divergence = "bearish"
                signal = "bearish"
            # Confirmation: Both moving same direction
            elif price_change_pct > 2 and obv_change_pct > 3:
                signal = "bullish"
            elif price_change_pct < -2 and obv_change_pct < -3:
                signal = "bearish"
        
        return {
            "obv": round(current_obv, 2),
            "trend": obv_trend,
            "divergence": divergence,
            "signal": signal
        }

    @staticmethod
    def calculate_stoch_rsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14, 
                            smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """Stochastic RSI - TradingView Compatible
        
        Formula:
        1. Calculate RSI
        2. Apply Stochastic formula to RSI values
        3. Smooth with SMA for %K and %D lines
        
        Default parameters: 14, 14, 3, 3 (matching TradingView)
        First valid value appears at bar: rsi_period + stoch_period - 1 = 27
        """
        if len(closes) < rsi_period + stoch_period:
            return {"k": 50.0, "d": 50.0, "signal": "neutral"}
        
        # Step 1: Calculate RSI series
        rsi_values = []
        for i in range(rsi_period, len(closes)):
            subset = closes[:i+1]
            rsi_data = IndicatorEngine.calculate_rsi(subset, rsi_period)
            rsi_values.append(rsi_data["value"])
        
        if len(rsi_values) < stoch_period:
            return {"k": 50.0, "d": 50.0, "signal": "neutral"}
        
        # Step 2: Calculate Stochastic of RSI
        stoch_rsi_values = []
        for i in range(stoch_period - 1, len(rsi_values)):
            window = rsi_values[i - stoch_period + 1:i + 1]
            highest_rsi = max(window)
            lowest_rsi = min(window)
            
            if highest_rsi == lowest_rsi:
                stoch_rsi = 50.0  # Neutral when no range
            else:
                stoch_rsi = ((rsi_values[i] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100
            
            stoch_rsi_values.append(stoch_rsi)
        
        if len(stoch_rsi_values) < smooth_k:
            return {"k": 50.0, "d": 50.0, "signal": "neutral"}
        
        # Step 3: Smooth %K with SMA
        k_values = []
        for i in range(smooth_k - 1, len(stoch_rsi_values)):
            k_sma = sum(stoch_rsi_values[i - smooth_k + 1:i + 1]) / smooth_k
            k_values.append(k_sma)
        
        if len(k_values) < smooth_d:
            return {"k": k_values[-1] if k_values else 50.0, "d": 50.0, "signal": "neutral"}
        
        # Step 4: Smooth %D with SMA of %K
        d_value = sum(k_values[-smooth_d:]) / smooth_d
        k_value = k_values[-1]
        
        # Determine signal
        if k_value < 20 and d_value < 20:
            signal = "oversold"
        elif k_value > 80 and d_value > 80:
            signal = "overbought"
        elif k_value > d_value and k_value < 50:
            signal = "bullish_cross"
        elif k_value < d_value and k_value > 50:
            signal = "bearish_cross"
        else:
            signal = "neutral"
        
        return {
            "k": round(k_value, 2),
            "d": round(d_value, 2),
            "signal": signal
        }

    @staticmethod
    def calculate_adx(klines: List[Dict], period: int = 14) -> Dict:
        """ADX (Average Directional Index) - TradingView Compatible
        
        Uses Wilder's smoothing (RMA) for all components.
        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend
        - ADX < 20: Weak/no trend (ranging market)
        
        Also returns +DI and -DI for trend direction.
        """
        if len(klines) < period + 1:
            return {"adx": 0, "plus_di": 0, "minus_di": 0, "trend_strength": "weak"}
        
        # Calculate +DM, -DM, and TR
        plus_dm_list = []
        minus_dm_list = []
        tr_list = []
        
        for i in range(1, len(klines)):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_high = klines[i-1]["high"]
            prev_low = klines[i-1]["low"]
            prev_close = klines[i-1]["close"]
            
            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low
            
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            
            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return {"adx": 0, "plus_di": 0, "minus_di": 0, "trend_strength": "weak"}
        
        # Wilder's smoothing for ATR, +DM, -DM (seed with SMA)
        atr = sum(tr_list[:period]) / period
        smooth_plus_dm = sum(plus_dm_list[:period]) / period
        smooth_minus_dm = sum(minus_dm_list[:period]) / period
        
        dx_values = []
        
        for i in range(period, len(tr_list)):
            # Wilder's smoothing
            atr = (atr * (period - 1) + tr_list[i]) / period
            smooth_plus_dm = (smooth_plus_dm * (period - 1) + plus_dm_list[i]) / period
            smooth_minus_dm = (smooth_minus_dm * (period - 1) + minus_dm_list[i]) / period
            
            # Calculate +DI and -DI
            plus_di = (smooth_plus_dm / atr * 100) if atr > 0 else 0
            minus_di = (smooth_minus_dm / atr * 100) if atr > 0 else 0
            
            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
            else:
                dx = 0
            dx_values.append(dx)
        
        if len(dx_values) < period:
            return {"adx": 0, "plus_di": 0, "minus_di": 0, "trend_strength": "weak"}
        
        # ADX = Wilder's smoothed DX
        adx = sum(dx_values[:period]) / period
        for i in range(period, len(dx_values)):
            adx = (adx * (period - 1) + dx_values[i]) / period
        
        # Final +DI and -DI
        final_plus_di = (smooth_plus_dm / atr * 100) if atr > 0 else 0
        final_minus_di = (smooth_minus_dm / atr * 100) if atr > 0 else 0
        
        # Determine trend strength
        if adx >= 25:
            trend_strength = "strong"
        elif adx >= 20:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"
        
        return {
            "adx": round(adx, 2),
            "plus_di": round(final_plus_di, 2),
            "minus_di": round(final_minus_di, 2),
            "trend_strength": trend_strength,
            "trend_direction": "bullish" if final_plus_di > final_minus_di else "bearish"
        }


def calculate_all_indicators(klines: List[Dict]) -> Dict:
    """Calculate all indicators from kline data
    
    Ensures klines are sorted chronologically (oldest to newest) for accurate calculations.
    """
    if not klines:
        return {}
    
    # Ensure klines are sorted chronologically (oldest to newest)
    # This is critical for accurate EMA, RSI, MACD, and other indicator calculations
    klines = sorted(klines, key=lambda x: x["open_time"])
    
    closes = [k["close"] for k in klines]
    current_price = closes[-1]
    
    engine = IndicatorEngine()
    
    # RSI
    rsi = engine.calculate_rsi(closes, 14)
    
    # Stochastic RSI (NEW)
    stoch_rsi = engine.calculate_stoch_rsi(closes, 14, 14, 3, 3)
    
    # ADX (NEW)
    adx = engine.calculate_adx(klines, 14)
    
    # EMAs
    ema20 = engine.calculate_ema(closes, 20)
    ema50 = engine.calculate_ema(closes, 50)
    ema200 = engine.calculate_ema(closes, 200) if len(closes) >= 200 else engine.calculate_ema(closes, min(len(closes), 100))
    
    # EMA alignment
    if ema20 > ema50 > ema200:
        ema_alignment = "bullish"
    elif ema20 < ema50 < ema200:
        ema_alignment = "bearish"
    else:
        ema_alignment = "mixed"
    
    # Price vs EMAs
    above_count = sum([1 for ema in [ema20, ema50, ema200] if current_price > ema])
    if above_count == 3:
        price_vs_ema = "above_all"
    elif above_count == 0:
        price_vs_ema = "below_all"
    else:
        price_vs_ema = "mixed"
    
    # MACD
    macd = engine.calculate_macd(closes)
    
    # Calculate RSI values for divergence detection
    rsi_values = []
    for i in range(14, len(closes)):
        rsi_data = engine.calculate_rsi(closes[:i+1], 14)
        rsi_values.append(rsi_data["value"])
    
    # Calculate MACD histogram values for divergence detection
    macd_histogram_values = []
    for i in range(26, len(closes)):
        macd_data = engine.calculate_macd(closes[:i+1])
        macd_histogram_values.append(macd_data["histogram"])
    
    # Detect divergences (using default lookback=50 from detect_divergence)
    rsi_divergence = engine.detect_divergence(closes, rsi_values, lookback=50) if len(rsi_values) >= 50 else {"type": "none", "signal": "neutral", "strength": "none"}
    macd_divergence = engine.detect_divergence(closes, macd_histogram_values, lookback=50) if len(macd_histogram_values) >= 50 else {"type": "none", "signal": "neutral", "strength": "none"}
    
    # Combine divergences (prioritize RSI if both exist)
    divergence = rsi_divergence if rsi_divergence["type"] != "none" else macd_divergence
    if divergence["type"] == "none":
        divergence = {"type": "none", "signal": "neutral", "strength": "none", "indicator": "none"}
    else:
        divergence["indicator"] = "RSI" if rsi_divergence["type"] != "none" else "MACD"
    
    # Bollinger Bands
    bollinger = engine.calculate_bollinger(closes, 20, 2.0)
    
    # ATR
    atr = engine.calculate_atr(klines, 14)
    
    # Support/Resistance
    sr_levels = engine.find_support_resistance(klines, 3)
    
    # Fibonacci
    fib = engine.calculate_fibonacci(klines, 50)
    
    # Volume
    volume = engine.analyze_volume(klines, 20)
    
    # On-Balance Volume (OBV)
    obv = engine.calculate_obv(klines)
    
    # Determine overall trend
    bullish_signals = 0
    bearish_signals = 0
    
    if rsi["signal"] == "oversold": bullish_signals += 1
    elif rsi["signal"] == "overbought": bearish_signals += 1
    
    if macd["trend"] == "bullish": bullish_signals += 1
    elif macd["trend"] == "bearish": bearish_signals += 1
    
    if ema_alignment == "bullish": bullish_signals += 1
    elif ema_alignment == "bearish": bearish_signals += 1
    
    if price_vs_ema == "above_all": bullish_signals += 1
    elif price_vs_ema == "below_all": bearish_signals += 1
    
    if volume["trend"] == "bullish": bullish_signals += 1
    elif volume["trend"] == "bearish": bearish_signals += 1
    
    if bullish_signals > bearish_signals + 1:
        trend = "bullish"
    elif bearish_signals > bullish_signals + 1:
        trend = "bearish"
    else:
        trend = "neutral"
    
    return {
        "current_price": current_price,
        "rsi": rsi,
        "stoch_rsi": stoch_rsi,
        "adx": adx,
        "ema": {
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "alignment": ema_alignment,
            "price_vs_ema": price_vs_ema
        },
        "macd": macd,
        "divergence": divergence,
        "bollinger": bollinger,
        "atr": atr,
        "support_resistance": sr_levels,
        "fibonacci": fib,
        "volume": volume,
        "obv": obv,
        "trend": trend
    }


def calculate_long_trade_setup(
    current_price: float,
    supports: List[Dict],
    resistances: List[Dict],
    atr_value: float,
    confluence_score: float = 50,
    higher_tf_trend: str = "neutral"
) -> Dict:
    """Calculate LONG trade setup with ATR-based stop loss and R-multiple take profits
    
    Professional risk management:
    - Entry: Nearest support level (or current price if no support)
    - Stop Loss: Support - (ATR × 1.5) with minimum 0.5% buffer, always below support
    - Take Profits: R-multiples (1.5R, 2.5R, 4R) where R = Entry - SL
    
    Args:
        current_price: Current market price
        supports: List of support levels [{"price": float, "strength": str}, ...]
        resistances: List of resistance levels [{"price": float, "strength": str}, ...]
        atr_value: Current ATR(14) value
        confluence_score: Confluence score (0-100)
        higher_tf_trend: Higher timeframe trend ("bullish", "bearish", "neutral")
    
    Returns:
        Dict with entry, stop_loss, tp1, tp2, tp3, risk_per_trade, risk_reward_ratio, bias
    """
    
    # Determine bias: Only LONG or NEUTRAL for spot trading
    # Only recommend LONG when confluence >= 60 AND higher TF is bullish or neutral
    if confluence_score >= 60 and higher_tf_trend != "bearish":
        bias = "long"
        confidence = "high" if confluence_score >= 75 else "medium"
    else:
        bias = "neutral"
        confidence = "low"
    
    # Entry: Nearest support level (or current price if no support)
    if supports and len(supports) > 0:
        nearest_support = supports[0]["price"]
        entry_price = nearest_support
        entry_reasoning = f"Entry at nearest support level ${nearest_support:,.2f}"
    else:
        entry_price = current_price
        entry_reasoning = "Entry at current price (no clear support identified)"
    
    # Stop Loss: Must be below support with ATR buffer
    if atr_value > 0:
        # Standard ATR-based SL: Support - (ATR × 1.5)
        atr_buffer = atr_value * 1.5
        min_buffer = entry_price * 0.005  # Minimum 0.5% buffer
        buffer = max(atr_buffer, min_buffer)
        
        if supports and len(supports) > 0:
            # SL below support with buffer
            sl_price = supports[0]["price"] - buffer
            sl_reasoning = f"Stop loss below support (${supports[0]['price']:,.2f}) with ATR buffer"
        else:
            # No support - use ATR below entry
            sl_price = entry_price - buffer
            sl_reasoning = f"Stop loss below entry with ATR×1.5 buffer (${buffer:,.2f})"
        
        # Ensure SL is always below entry
        if sl_price >= entry_price:
            sl_price = entry_price - buffer
            sl_reasoning = "Stop loss adjusted to maintain risk (below entry with ATR buffer)"
    else:
        # Fallback: 3% below entry if no ATR
        sl_price = entry_price * 0.97
        sl_reasoning = "Stop loss 3% below entry (fallback - no ATR available)"
    
    # Calculate Risk (R)
    risk_per_trade = entry_price - sl_price
    risk_pct = (risk_per_trade / entry_price) * 100 if entry_price > 0 else 0
    
    # Take Profits using R-multiples
    tp1_price = entry_price + (risk_per_trade * 1.5)
    tp2_price = entry_price + (risk_per_trade * 2.5)
    tp3_price = entry_price + (risk_per_trade * 4.0)
    
    tp1_reasoning = "TP1: 1.5R target"
    tp2_reasoning = "TP2: 2.5R target"
    tp3_reasoning = "TP3: 4R extended target"
    
    # Adjust TPs if resistance levels are close (within 1% tolerance)
    if resistances:
        for i, res in enumerate(resistances[:3]):
            res_price = res["price"]
            
            # Check TP1
            if abs(res_price - tp1_price) / tp1_price < 0.01 and res_price > entry_price:
                tp1_price = res_price
                tp1_reasoning = f"TP1: Adjusted to resistance at ${res_price:,.2f}"
            
            # Check TP2
            elif abs(res_price - tp2_price) / tp2_price < 0.01 and res_price > tp1_price:
                tp2_price = res_price
                tp2_reasoning = f"TP2: Adjusted to resistance at ${res_price:,.2f}"
            
            # Check TP3
            elif abs(res_price - tp3_price) / tp3_price < 0.01 and res_price > tp2_price:
                tp3_price = res_price
                tp3_reasoning = f"TP3: Adjusted to resistance at ${res_price:,.2f}"
    
    # Validate: TPs must be in ascending order and above entry
    tp1_price = max(tp1_price, entry_price * 1.01)  # At least 1% above entry
    tp2_price = max(tp2_price, tp1_price * 1.01)    # Above TP1
    tp3_price = max(tp3_price, tp2_price * 1.01)    # Above TP2
    
    # Calculate risk-reward ratios
    rr1 = round((tp1_price - entry_price) / risk_per_trade, 1) if risk_per_trade > 0 else 1.5
    rr2 = round((tp2_price - entry_price) / risk_per_trade, 1) if risk_per_trade > 0 else 2.5
    rr3 = round((tp3_price - entry_price) / risk_per_trade, 1) if risk_per_trade > 0 else 4.0
    
    return {
        "bias": bias,
        "confidence": confidence,
        "entry": {
            "price": round(entry_price, 4),
            "reasoning": entry_reasoning
        },
        "stop_loss": {
            "price": round(sl_price, 4),
            "reasoning": sl_reasoning
        },
        "tp1": {
            "price": round(tp1_price, 4),
            "risk_reward": f"1:{rr1}",
            "reasoning": tp1_reasoning
        },
        "tp2": {
            "price": round(tp2_price, 4),
            "risk_reward": f"1:{rr2}",
            "reasoning": tp2_reasoning
        },
        "tp3": {
            "price": round(tp3_price, 4),
            "risk_reward": f"1:{rr3}",
            "reasoning": tp3_reasoning
        },
        "risk_per_trade": round(risk_per_trade, 4),
        "risk_pct": round(risk_pct, 2),
        "higher_tf_trend": higher_tf_trend
    }


async def check_higher_timeframe_trend(symbol: str, base_timeframe: str) -> Dict:
    """Check higher timeframe trend for 4h and below timeframes
    
    For timeframes 4h and below: Fetch 1D klines and check EMA alignment.
    Only allow LONG trades when higher timeframe trend is bullish or neutral.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        base_timeframe: Current chart timeframe
    
    Returns:
        Dict with trend status, alignment, and whether to allow longs
    """
    
    # Define which timeframes need higher TF confirmation
    needs_confirmation = ["1m", "5m", "15m", "30m", "1h", "4h"]
    
    if base_timeframe.lower() not in needs_confirmation:
        # Weekly/daily charts don't need higher TF confirmation
        return {
            "trend": "neutral",
            "alignment": "not_checked",
            "allow_longs": True,
            "higher_timeframe": "none"
        }
    
    try:
        # Fetch daily klines for higher timeframe analysis
        daily_klines = await fetch_okx_klines(symbol, "1d", 200)
        
        if not daily_klines or len(daily_klines) < 50:
            # Try other sources
            daily_klines = await fetch_coingecko_ohlc(symbol, 30)  # 30 days
        
        if not daily_klines or len(daily_klines) < 20:
            return {
                "trend": "neutral",
                "alignment": "insufficient_data",
                "allow_longs": True,
                "higher_timeframe": "1d"
            }
        
        # Calculate EMAs on daily timeframe
        closes = [k["close"] for k in daily_klines]
        engine = IndicatorEngine()
        
        ema20 = engine.calculate_ema(closes, 20)
        ema50 = engine.calculate_ema(closes, 50) if len(closes) >= 50 else ema20
        ema200 = engine.calculate_ema(closes, min(len(closes), 200)) if len(closes) >= 100 else ema50
        
        current_price = closes[-1]
        
        # Determine trend alignment
        if ema20 > ema50 > ema200:
            trend = "bullish"
            alignment = "ema20 > ema50 > ema200"
            allow_longs = True
        elif ema20 < ema50 < ema200:
            trend = "bearish"
            alignment = "ema20 < ema50 < ema200"
            allow_longs = False  # Don't take longs against higher TF trend
        else:
            trend = "neutral"
            alignment = "mixed"
            allow_longs = True  # Neutral is okay for longs
        
        # Additional check: price vs EMAs
        if current_price < ema200:
            # Price below long-term trend - be cautious
            if trend != "bearish":
                trend = "cautious"
                allow_longs = True  # Still allow but mark as cautious
        
        return {
            "trend": trend,
            "alignment": alignment,
            "allow_longs": allow_longs,
            "higher_timeframe": "1d",
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "price_vs_ema200": "above" if current_price > ema200 else "below"
        }
        
    except Exception as e:
        print(f"Error checking higher timeframe trend: {e}")
        return {
            "trend": "neutral",
            "alignment": "error",
            "allow_longs": True,
            "higher_timeframe": "1d",
            "error": str(e)
        }


@app.get("/api/indicators/{symbol}/{interval}")
async def get_indicators(symbol: str, interval: str):
    """GET /api/indicators - Pre-calculated indicators
    
    Uses 500 candles for accurate EMA200 and other indicator calculations.
    """
    klines_data = await get_klines(symbol, interval, 500)
    klines = klines_data["klines"]
    
    # Ensure klines are sorted chronologically (oldest to newest)
    klines.sort(key=lambda x: x["open_time"])
    
    indicators = calculate_all_indicators(klines)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "indicators": indicators
    }


# ============================================================
# PART 3: PRICE-TO-PIXEL MAPPING
# ============================================================

def extract_price_scale(klines: List[Dict], buffer_pct: float = 0.05) -> Dict:
    """Extract price scale from kline data"""
    if not klines:
        return {"max_price": 100, "min_price": 0}
    
    highs = [k["high"] for k in klines[-100:]]
    lows = [k["low"] for k in klines[-100:]]
    
    max_price = max(highs)
    min_price = min(lows)
    
    buffer = (max_price - min_price) * buffer_pct
    max_price += buffer
    min_price -= buffer
    
    return {
        "max_price": round(max_price, 4),
        "min_price": round(min_price, 4)
    }


def price_to_y(price: float, max_price: float, min_price: float, img_height: int) -> int:
    """Convert price to Y pixel coordinate"""
    chart_top_y = 70
    chart_bottom_y = img_height - 80
    chart_height = chart_bottom_y - chart_top_y
    
    if max_price == min_price:
        return chart_top_y + chart_height // 2
    
    y = chart_top_y + ((max_price - price) / (max_price - min_price)) * chart_height
    return int(max(chart_top_y, min(chart_bottom_y, y)))


# ============================================================
# PART 4: GEMINI AI ANALYSIS
# ============================================================

def build_analysis_prompt(
    symbol: str,
    timeframe: str,
    price_data: Dict,
    indicators: Dict,
    depth_data: Dict,
    higher_tf_trend: Dict = None
) -> str:
    """Build Chain-of-Thought analysis prompt - Simplified for pattern recognition only
    
    Y-axis coordinate calculations have been removed. Trade setup (entry/SL/TP) 
    is now calculated algorithmically using ATR-based risk management.
    """
    
    current_price = price_data.get("current_price", 0)
    rsi = indicators.get("rsi", {})
    ema = indicators.get("ema", {})
    macd = indicators.get("macd", {})
    divergence = indicators.get("divergence", {})
    bb = indicators.get("bollinger", {})
    atr = indicators.get("atr", 0)
    sr = indicators.get("support_resistance", {})
    fib = indicators.get("fibonacci", {})
    vol = indicators.get("volume", {})
    obv = indicators.get("obv", {})
    
    # Higher timeframe info
    htf_info = ""
    if higher_tf_trend:
        htf_info = f"""
=== HIGHER TIMEFRAME TREND (1D) ===
Daily Trend: {higher_tf_trend.get('trend', 'unknown')}
EMA Alignment: {higher_tf_trend.get('alignment', 'unknown')}
Allow Longs: {higher_tf_trend.get('allow_longs', True)}
"""
    
    prompt = f"""ROLE: You are an expert institutional technical analyst with 20+ years experience.
Analyze this {symbol} chart on {timeframe} timeframe using systematic Chain-of-Thought reasoning.

IMPORTANT: This system is for SPOT TRADING ONLY. Only recommend LONG (BUY) setups or NEUTRAL (no trade).
NEVER recommend SHORT positions as this is not supported.

=== VERIFIED MARKET DATA (FROM API) ===
Symbol: {symbol}
Current Price: ${current_price:,.4f}
24h Change: {price_data.get('price_change_pct', 0):.2f}%
24h High: ${price_data.get('high_24h', 0):,.4f}
24h Low: ${price_data.get('low_24h', 0):,.4f}
24h Volume: ${price_data.get('quote_volume', 0):,.0f}
{htf_info}
=== PRE-CALCULATED INDICATORS (VERIFIED) ===
RSI(14): {rsi.get('value', 50)} ({rsi.get('signal', 'neutral')})
EMA20: ${ema.get('ema20', 0):,.4f}
EMA50: ${ema.get('ema50', 0):,.4f}
EMA200: ${ema.get('ema200', 0):,.4f}
EMA Alignment: {ema.get('alignment', 'mixed')}
Price vs EMAs: {ema.get('price_vs_ema', 'mixed')}
MACD: {macd.get('macd', 0):.6f} | Signal: {macd.get('signal', 0):.6f} | Histogram: {macd.get('histogram', 0):.6f}
MACD Trend: {macd.get('trend', 'neutral')}
Divergence: {divergence.get('type', 'none')} ({divergence.get('signal', 'neutral')}) - Strength: {divergence.get('strength', 'none')}
Bollinger Bands: Upper ${bb.get('upper', 0):,.4f} | Middle ${bb.get('middle', 0):,.4f} | Lower ${bb.get('lower', 0):,.4f}
Bollinger Position: {bb.get('position', 'middle')}
ATR(14): ${atr:,.4f}
Volume Trend: {vol.get('trend', 'neutral')} (Ratio: {vol.get('ratio', 1.0):.2f}x)
OBV Trend: {obv.get('trend', 'neutral')} | Divergence: {obv.get('divergence', 'none')}

=== DETECTED SUPPORT/RESISTANCE ===
Support Levels: {json.dumps(sr.get('support', []))}
Resistance Levels: {json.dumps(sr.get('resistance', []))}

=== ORDER BOOK DATA ===
Largest Bid Wall: ${depth_data.get('largest_bid_wall', {}).get('price', 0):,.4f} ({depth_data.get('largest_bid_wall', {}).get('quantity', 0):,.2f} units)
Largest Ask Wall: ${depth_data.get('largest_ask_wall', {}).get('price', 0):,.4f} ({depth_data.get('largest_ask_wall', {}).get('quantity', 0):,.2f} units)

=== FIBONACCI LEVELS ===
Swing High: ${fib.get('swing_high', 0):,.4f}
Swing Low: ${fib.get('swing_low', 0):,.4f}
Key Levels: {json.dumps(fib.get('levels', {}))}

=== CHAIN-OF-THOUGHT ANALYSIS ===

STEP 1: TREND ASSESSMENT
Look at the EMA alignment and price structure. Is price making higher highs/lows or lower highs/lows?

STEP 2: MOMENTUM EVALUATION  
Evaluate RSI (oversold <30 is bullish, overbought >70 is bearish) and MACD (positive histogram is bullish).
Check for divergences: Bullish divergence (price lower low, indicator higher low) is very bullish.

STEP 3: CHART PATTERN RECOGNITION (CRITICAL - 15% of confluence)
Carefully examine the chart for any recognizable patterns. Look for:

REVERSAL PATTERNS:
- Head & Shoulders / Inverse Head & Shoulders
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Rising Wedge / Falling Wedge
- Cup & Handle
- Rounding Bottom / Rounding Top

CONTINUATION PATTERNS:
- Bull Flag / Bear Flag
- Pennant (Bullish/Bearish)
- Ascending/Descending/Symmetrical Triangle
- Rectangle / Trading Range

CANDLESTICK PATTERNS:
- Engulfing Patterns (Bullish/Bearish)
- Hammer / Hanging Man / Inverted Hammer
- Doji / Morning Star / Evening Star
- Three White Soldiers / Three Black Crows

If you identify ANY pattern:
1. Name the pattern exactly
2. Explain what it means (reversal/continuation, bullish/bearish)
3. Rate its reliability (strong/moderate/weak)
4. Note the pattern's completion status (forming/completed/failed)

STEP 4: SUPPORT/RESISTANCE ANALYSIS
Identify where price might bounce (support) or get rejected (resistance).

STEP 5: CONFLUENCE SCORING
Score each indicator 0-100 based on how bullish it is, then multiply by weight:
- Chart Pattern: weight 15%
- S/R Levels: weight 20%
- Divergence: weight 8%
- RSI: weight 10%
- MACD: weight 10%
- EMA Alignment: weight 10%
- Price vs EMA: weight 8%
- Fibonacci: weight 8%
- Bollinger: weight 8%
- Volume: weight 3%

STEP 6: TRADE RECOMMENDATION
Based on confluence score:
- If confluence >= 60 AND indicators are bullish: recommend "long"
- Otherwise: recommend "neutral" (no trade)
NEVER recommend "short" - this is for spot trading only.

=== OUTPUT FORMAT (JSON ONLY - NO MARKDOWN, NO CODE BLOCKS) ===

Return ONLY valid JSON with this exact structure:
{{
  "trend_analysis": {{
    "trend": "bullish/bearish/neutral",
    "reasoning": "explanation"
  }},
  "chart_pattern": {{
    "name": "<pattern name or 'none'>",
    "type": "reversal/continuation/none",
    "direction": "bullish/bearish/neutral",
    "reliability": "strong/moderate/weak/none",
    "status": "forming/completed/failed/none",
    "description": "<detailed explanation of the pattern>",
    "score": <0-100 based on bullishness>,
    "weight": 15,
    "weighted_score": <calculated>
  }},
  "indicators": {{
    "rsi": {{"value": {rsi.get('value', 50)}, "signal": "{rsi.get('signal', 'neutral')}", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "macd": {{"value": {{"macd": {macd.get('macd', 0)}, "signal": {macd.get('signal', 0)}, "histogram": {macd.get('histogram', 0)}}}, "signal": "{macd.get('trend', 'neutral')}", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "divergence": {{"type": "{divergence.get('type', 'none')}", "signal": "{divergence.get('signal', 'neutral')}", "strength": "{divergence.get('strength', 'none')}", "score": <0-100>, "weight": 8, "weighted_score": <calculated>, "explanation": "why"}},
    "ema_alignment": {{"value": "{ema.get('alignment', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "price_vs_ema": {{"value": "{ema.get('price_vs_ema', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 8, "weighted_score": <calculated>, "explanation": "why"}},
    "support_resistance": {{"nearest_support": <price>, "nearest_resistance": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 20, "weighted_score": <calculated>, "explanation": "why"}},
    "fibonacci": {{"key_level": "0.618", "price_at_level": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 8, "weighted_score": <calculated>, "explanation": "why"}},
    "bollinger": {{"position": "{bb.get('position', 'middle')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 8, "weighted_score": <calculated>, "explanation": "why"}},
    "volume": {{"trend": "{vol.get('trend', 'neutral')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 3, "weighted_score": <calculated>, "explanation": "why"}}
  }},
  "confluence_score": <sum of weighted_scores>,
  "confluence_breakdown": {{
    "Chart Pattern": <weighted_score>,
    "S/R Levels": <weighted_score>,
    "Divergence": <weighted_score>,
    "RSI": <weighted_score>,
    "MACD": <weighted_score>,
    "EMA Alignment": <weighted_score>,
    "Price vs EMA": <weighted_score>,
    "Fibonacci": <weighted_score>,
    "Bollinger": <weighted_score>,
    "Volume": <weighted_score>
  }},
  "recommended_bias": "long/neutral",
  "support_levels": [
    {{"price": <n>, "strength": "strong/moderate/weak"}}
  ],
  "resistance_levels": [
    {{"price": <n>, "strength": "strong/moderate/weak"}}
  ],
  "analysis_summary": "2-3 sentence summary",
  "trade_rationale": "detailed explanation of why to trade or not trade"
}}

CRITICAL RULES:
1. CAREFULLY examine the chart for patterns - this is worth 15% of confluence score
2. Use the EXACT indicator values provided above for analysis
3. Divergence detection: Bullish divergence = high score (80-100), Bearish = low score (0-20), None = 50
4. confluence_score MUST equal sum of all weighted_scores (totaling 100%)
5. Only recommend "long" if confluence_score >= 60 AND most indicators are bullish
6. NEVER recommend "short" - only "long" or "neutral"
7. Return ONLY valid JSON, no markdown code blocks
8. Trade setup prices (entry/SL/TP) will be calculated algorithmically - just provide analysis"""

    return prompt


# ============================================================
# PART 5: CHART ANNOTATION DRAWING
# ============================================================

def draw_annotations(image_bytes: bytes, analysis_data: Dict, price_scale: Dict, img_height: int) -> bytes:
    """Return original image without annotations - chart is only used for pattern recognition by Gemini Vision.
    TP/SL/Entry will be displayed in the UI, not drawn on the chart."""
    # Just return the original image bytes without any drawing
    # Pattern recognition is done by Gemini Vision, not by drawing lines
    return image_bytes

from typing import Any, Dict, List, Optional

# ============================================================
# PART 6: MAIN ANALYSIS ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze_chart(
    file: UploadFile = File(...),
    symbol: str = Form(default="BTCUSDT"),
    timeframe: str = Form(default="4h"),
    price_data_json: Optional[str] = Form(default=None),
    klines_data_json: Optional[str] = Form(default=None),
    depth_data_json: Optional[str] = Form(default=None)
):
    """POST /api/analyze - Full chart analysis with annotations"""
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size
        
        symbol = symbol.upper()
        
        # Use provided data (from frontend) or fetch with multi-source fallback
        if price_data_json:
            # Use data provided by frontend - handle multiple formats
            price_data_raw = json.loads(price_data_json)
            if "current_price" in price_data_raw:
                # Already in our standard format (CoinGecko/OKX/Kraken from frontend)
                price_data = {
                    "symbol": price_data_raw.get("symbol", symbol),
                    "current_price": float(price_data_raw.get("current_price", 0)),
                    "price_change_24h": float(price_data_raw.get("price_change_24h", 0)),
                    "price_change_pct": float(price_data_raw.get("price_change_pct", 0)),
                    "high_24h": float(price_data_raw.get("high_24h", 0)),
                    "low_24h": float(price_data_raw.get("low_24h", 0)),
                    "volume_24h": float(price_data_raw.get("volume_24h", 0)),
                    "quote_volume": float(price_data_raw.get("quote_volume", 0))
                }
            elif "list" in price_data_raw:  # Bybit format
                ticker = price_data_raw["list"][0]
                price_data = {
                    "symbol": ticker["symbol"],
                    "current_price": float(ticker["lastPrice"]),
                    "price_change_24h": (float(ticker.get("lastPrice", 0)) - float(ticker.get("prevPrice24h", ticker["lastPrice"]))),
                    "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
                    "high_24h": float(ticker.get("highPrice24h", ticker["lastPrice"])),
                    "low_24h": float(ticker.get("lowPrice24h", ticker["lastPrice"])),
                    "volume_24h": float(ticker.get("volume24h", 0)),
                    "quote_volume": float(ticker.get("turnover24h", 0))
                }
            elif "lastPrice" in price_data_raw:  # Binance format
                price_data = {
                    "symbol": price_data_raw.get("symbol", symbol),
                    "current_price": float(price_data_raw["lastPrice"]),
                    "price_change_24h": float(price_data_raw.get("priceChange", 0)),
                    "price_change_pct": float(price_data_raw.get("priceChangePercent", 0)),
                    "high_24h": float(price_data_raw.get("highPrice", 0)),
                    "low_24h": float(price_data_raw.get("lowPrice", 0)),
                    "volume_24h": float(price_data_raw.get("volume", 0)),
                    "quote_volume": float(price_data_raw.get("quoteVolume", 0))
                }
            else:
                # Unknown format, extract what we can
                price_data = {
                    "symbol": symbol,
                    "current_price": float(price_data_raw.get("price", price_data_raw.get("usd", 0))),
                    "price_change_24h": 0,
                    "price_change_pct": float(price_data_raw.get("usd_24h_change", 0)),
                    "high_24h": 0,
                    "low_24h": 0,
                    "volume_24h": float(price_data_raw.get("usd_24h_vol", 0)),
                    "quote_volume": 0
                }
        else:
            # Fetch with multi-source fallback (CoinGecko -> OKX -> Kraken -> Bybit -> Binance)
            print(f"Fetching price for analysis from multi-source fallback...")
            
            # Try CoinGecko first
            price_data = await fetch_coingecko_market(symbol)
            if not price_data:
                # Try OKX
                price_data = await fetch_okx_price(symbol)
            if not price_data:
                # Try Kraken
                price_data = await fetch_kraken_price(symbol)
            if not price_data:
                # Try Bybit
                bybit_data = await fetch_bybit("tickers", {"category": "spot", "symbol": symbol})
                if bybit_data and bybit_data.get("list"):
                    ticker = bybit_data["list"][0]
                    price_data = {
                        "symbol": ticker["symbol"],
                        "current_price": float(ticker["lastPrice"]),
                        "price_change_24h": (float(ticker.get("lastPrice", 0)) - float(ticker.get("prevPrice24h", ticker["lastPrice"]))),
                        "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
                        "high_24h": float(ticker.get("highPrice24h", ticker["lastPrice"])),
                        "low_24h": float(ticker.get("lowPrice24h", ticker["lastPrice"])),
                        "volume_24h": float(ticker.get("volume24h", 0)),
                        "quote_volume": float(ticker.get("turnover24h", 0))
                    }
            if not price_data:
                raise HTTPException(status_code=404, detail=f"Failed to fetch price for {symbol} from any source")
        
        # Use provided klines or fetch with multi-source fallback
        if klines_data_json:
            klines_raw = json.loads(klines_data_json)
            # Handle multiple formats
            if isinstance(klines_raw, list) and len(klines_raw) > 0:
                if isinstance(klines_raw[0], dict):
                    # Already in dict format (from CoinGecko or our API)
                    klines = [{
                        "open_time": k.get("open_time", 0),
                        "open": float(k.get("open", 0)),
                        "high": float(k.get("high", 0)),
                        "low": float(k.get("low", 0)),
                        "close": float(k.get("close", 0)),
                        "volume": float(k.get("volume", 0))
                    } for k in klines_raw]
                else:
                    # Array format (Binance style: [time, o, h, l, c, v, ...])
                    klines = [{
                        "open_time": k[0], "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])
                    } for k in klines_raw]
                # Ensure klines are sorted chronologically (oldest to newest)
                klines.sort(key=lambda x: x["open_time"])
            else:
                klines = []
        else:
            # Fetch with multi-source fallback (OKX -> CoinGecko -> Bybit -> Binance)
            # Use 500 candles for accurate EMA200 and other indicator calculations
            print(f"Fetching klines for analysis from multi-source fallback (500 candles for accuracy)...")
            
            klines = await fetch_okx_klines(symbol, timeframe, 500)
            if not klines:
                days = INTERVAL_TO_DAYS.get(timeframe.lower(), 1)
                klines = await fetch_coingecko_ohlc(symbol, days)
            if not klines:
                # Try Bybit
                bybit_interval = INTERVAL_MAP.get(timeframe.lower(), timeframe)
                bybit_data = await fetch_bybit("kline", {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": bybit_interval,
                    "limit": 500
                })
                if bybit_data and bybit_data.get("list"):
                    klines = []
                    for k in reversed(bybit_data["list"]):
                        klines.append({
                            "open_time": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })
            if not klines:
                raise HTTPException(status_code=404, detail=f"Failed to fetch klines for {symbol} from any source")
            
            # Ensure klines are sorted chronologically (oldest to newest) for accurate calculations
            klines.sort(key=lambda x: x["open_time"])
        
        # Use provided depth or fetch with multi-source fallback
        depth_data = {"largest_bid_wall": {"price": 0, "quantity": 0}, "largest_ask_wall": {"price": 0, "quantity": 0}}
        if depth_data_json:
            depth_raw = json.loads(depth_data_json)
            if depth_raw:
                # Handle multiple formats (Bybit b/a, Binance bids/asks, our format)
                bids_raw = depth_raw.get("bids", depth_raw.get("b", []))
                asks_raw = depth_raw.get("asks", depth_raw.get("a", []))
                bids = [{"price": float(b[0] if isinstance(b, list) else b.get("price", 0)), 
                         "quantity": float(b[1] if isinstance(b, list) else b.get("quantity", 0))} for b in bids_raw]
                asks = [{"price": float(a[0] if isinstance(a, list) else a.get("price", 0)), 
                         "quantity": float(a[1] if isinstance(a, list) else a.get("quantity", 0))} for a in asks_raw]
                if bids:
                    depth_data["largest_bid_wall"] = max(bids, key=lambda x: x["quantity"])
                if asks:
                    depth_data["largest_ask_wall"] = max(asks, key=lambda x: x["quantity"])
        else:
            # Fetch with multi-source fallback (OKX -> Kraken -> Bybit)
            print(f"Fetching depth for analysis from multi-source fallback...")
            
            # Try OKX
            okx_symbol = symbol_to_okx(symbol)
            okx_data = await fetch_okx("books", {"instId": okx_symbol, "sz": "20"})
            if okx_data and len(okx_data) > 0:
                book = okx_data[0]
                bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in book.get("bids", [])]
                asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in book.get("asks", [])]
                if bids:
                    depth_data["largest_bid_wall"] = max(bids, key=lambda x: x["quantity"])
                if asks:
                    depth_data["largest_ask_wall"] = max(asks, key=lambda x: x["quantity"])
            else:
                # Try Bybit
                bybit_data = await fetch_bybit("orderbook", {
                    "category": "spot",
                    "symbol": symbol,
                    "limit": 20
                })
                if bybit_data:
                    bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in bybit_data.get("b", [])]
                    asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in bybit_data.get("a", [])]
                    if bids:
                        depth_data["largest_bid_wall"] = max(bids, key=lambda x: x["quantity"])
                    if asks:
                        depth_data["largest_ask_wall"] = max(asks, key=lambda x: x["quantity"])
        
        # Calculate indicators
        indicators = calculate_all_indicators(klines)
        
        # Check higher timeframe trend for trade filtering
        higher_tf_trend = await check_higher_timeframe_trend(symbol, timeframe)
        
        # Build prompt and call Gemini (simplified - no Y-coordinate calculations)
        prompt = build_analysis_prompt(
            symbol, timeframe, price_data, indicators, depth_data,
            higher_tf_trend
        )
        
        analysis_data = {}
        
        if model:
            try:
                response = model.generate_content(
                    [prompt, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        top_p=0.95,
                        max_output_tokens=8192
                    )
                )
                
                response_text = response.text.strip()
                
                # Clean up response
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*', '', response_text)
                response_text = response_text.strip()
                
                # Parse JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
            except Exception as e:
                print(f"Gemini error: {e}")
        
        # Fallback: Generate basic analysis from calculated indicators if AI fails
        if not analysis_data or "recommended_bias" not in analysis_data:
            analysis_data = generate_fallback_analysis(indicators, price_data, higher_tf_trend)
        
        # Return original image (no annotations - pattern recognition is done by Gemini)
        annotated_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Calculate professional trade setup using ATR-based risk management
        current_price = price_data["current_price"]
        sr = indicators.get("support_resistance", {})
        supports = sr.get("support", [])
        resistances = sr.get("resistance", [])
        atr_value = indicators.get("atr", 0)
        
        # Get confluence score from AI analysis
        confluence_score = analysis_data.get("confluence_score", 50)
        ai_bias = analysis_data.get("recommended_bias", "neutral")
        
        # Calculate professional trade setup
        trade_setup = calculate_long_trade_setup(
        current_price=current_price,
        supports=supports,
        resistances=resistances,
        atr_value=atr_value,
        confluence_score=confluence_score,
        higher_tf_trend=higher_tf_trend.get("trend", "neutral") if isinstance(higher_tf_trend, dict) else (higher_tf_trend or "neutral")
        )

        
        # Determine final bias: Only "long" or "neutral" for spot trading
        # Require both AI recommendation and confluence threshold
        if ai_bias == "long" and confluence_score >= 60 and higher_tf_trend.get("allow_longs", True):
            final_bias = "long"
        else:
            final_bias = "neutral"
        trade_setup["bias"] = final_bias
        
        # Format support/resistance levels without Y coordinates
        support_levels = [{"price": s["price"], "strength": s.get("strength", "moderate")} for s in supports[:3]]
        resistance_levels = [{"price": r["price"], "strength": r.get("strength", "moderate")} for r in resistances[:3]]
        
        # Build response
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "binance_data": price_data,
            "calculated_indicators": indicators,
            "higher_tf_trend": higher_tf_trend,
            "chart_pattern": analysis_data.get("chart_pattern", {}),
            "indicators": analysis_data.get("indicators", {}),
            "trade_setup": trade_setup,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "confluence_score": confluence_score,
            "confluence_breakdown": analysis_data.get("confluence_breakdown", {}),
            "trend": analysis_data.get("trend_analysis", {}).get("trend", indicators.get("trend", "neutral")),
            "bias": final_bias,
            "risk_reward": trade_setup.get("tp2", {}).get("risk_reward", "1:2.5"),
            "analysis_summary": analysis_data.get("analysis_summary", ""),
            "trade_rationale": analysis_data.get("trade_rationale", ""),
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "api_cost": 0.00025
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def generate_fallback_analysis(indicators: Dict, price_data: Dict, higher_tf_trend: Dict = None) -> Dict:
    """Generate fallback analysis when AI fails
    
    Professional analysis with:
    - ATR-based risk management
    - Only LONG or NEUTRAL bias (no shorts for spot trading)
    - OBV indicator included
    - No Y coordinates (removed)
    """
    
    current_price = price_data["current_price"]
    
    sr = indicators.get("support_resistance", {})
    supports = sr.get("support", [])
    resistances = sr.get("resistance", [])
    
    # Support/Resistance levels (without Y coordinates)
    support_levels = [{"price": s["price"], "strength": s.get("strength", "moderate")} for s in supports[:3]]
    resistance_levels = [{"price": r["price"], "strength": r.get("strength", "moderate")} for r in resistances[:3]]
    
    # Get indicator data
    rsi = indicators.get("rsi", {})
    macd = indicators.get("macd", {})
    ema = indicators.get("ema", {})
    vol = indicators.get("volume", {})
    bb = indicators.get("bollinger", {})
    obv = indicators.get("obv", {})
    divergence = indicators.get("divergence", {})
    
    # Simple confluence scoring with updated weights
    score = 50  # Start neutral
    breakdown = {}
    
    # Chart Pattern (15%) - No pattern in fallback
    pattern_score = 50
    breakdown["Chart Pattern"] = round(pattern_score * 0.15, 1)
    score += (pattern_score - 50) * 0.15
    
    # S/R Levels (20% - increased)
    sr_score = 60 if supports else 50
    breakdown["S/R Levels"] = round(sr_score * 0.20, 1)
    score += (sr_score - 50) * 0.20
    
    # Divergence (8%)
    if divergence.get("signal") == "bullish":
        div_score = 85
    elif divergence.get("signal") == "bearish":
        div_score = 15
    else:
        div_score = 50
    breakdown["Divergence"] = round(div_score * 0.08, 1)
    score += (div_score - 50) * 0.08
    
    # RSI (10% - reduced)
    if rsi.get("signal") == "oversold":
        rsi_score = 80
    elif rsi.get("signal") == "overbought":
        rsi_score = 20
    else:
        rsi_score = 50
    breakdown["RSI"] = round(rsi_score * 0.10, 1)
    score += (rsi_score - 50) * 0.10
    
    # MACD (10% - reduced)
    if macd.get("trend") == "bullish":
        macd_score = 75
    elif macd.get("trend") == "bearish":
        macd_score = 25
    else:
        macd_score = 50
    breakdown["MACD"] = round(macd_score * 0.10, 1)
    score += (macd_score - 50) * 0.10
    
    # EMA Alignment (10% - reduced)
    if ema.get("alignment") == "bullish":
        ema_score = 80
    elif ema.get("alignment") == "bearish":
        ema_score = 20
    else:
        ema_score = 50
    breakdown["EMA Alignment"] = round(ema_score * 0.10, 1)
    score += (ema_score - 50) * 0.10
    
    # Price vs EMA (8%)
    if ema.get("price_vs_ema") == "above_all":
        pve_score = 75
    elif ema.get("price_vs_ema") == "below_all":
        pve_score = 25
    else:
        pve_score = 50
    breakdown["Price vs EMA"] = round(pve_score * 0.08, 1)
    score += (pve_score - 50) * 0.08
    
    # Fibonacci (8%)
    fib_score = 55
    breakdown["Fibonacci"] = round(fib_score * 0.08, 1)
    score += (fib_score - 50) * 0.08
    
    # Bollinger (8%)
    if bb.get("position") == "lower_band":
        bb_score = 75
    elif bb.get("position") == "upper_band":
        bb_score = 25
    else:
        bb_score = 50
    breakdown["Bollinger"] = round(bb_score * 0.08, 1)
    score += (bb_score - 50) * 0.08
    
    # Volume (3%)
    if vol.get("trend") == "bullish":
        vol_score = 70
    elif vol.get("trend") == "bearish":
        vol_score = 30
    else:
        vol_score = 50
    breakdown["Volume"] = round(vol_score * 0.03, 1)
    score += (vol_score - 50) * 0.03
    
    confluence_score = max(0, min(100, round(score, 1)))
    
    # Determine bias: Only "long" or "neutral" for spot trading
    # Require confluence >= 60 AND higher TF is bullish or neutral
    htf_trend = higher_tf_trend.get("trend", "neutral") if higher_tf_trend else "neutral"
    allow_longs = higher_tf_trend.get("allow_longs", True) if higher_tf_trend else True
    
    if confluence_score >= 60 and htf_trend != "bearish" and allow_longs:
        recommended_bias = "long"
    else:
        recommended_bias = "neutral"
    
    return {
        "chart_pattern": {
            "name": "none", 
            "type": "none", 
            "direction": "neutral", 
            "reliability": "none", 
            "status": "none", 
            "description": "No pattern detected in fallback analysis", 
            "score": 50, 
            "weight": 15, 
            "weighted_score": breakdown.get("Chart Pattern", 7.5)
        },
        "trend_analysis": {
            "trend": indicators.get("trend", "neutral"), 
            "reasoning": "Based on indicator analysis"
        },
        "indicators": {
            "rsi": {"value": rsi.get("value", 50), "signal": rsi.get("signal", "neutral"), "score": rsi_score, "weight": 10, "weighted_score": breakdown["RSI"], "explanation": "RSI analysis"},
            "macd": {"value": macd, "signal": macd.get("trend", "neutral"), "score": macd_score, "weight": 10, "weighted_score": breakdown["MACD"], "explanation": "MACD analysis"},
            "divergence": {"type": divergence.get("type", "none"), "signal": divergence.get("signal", "neutral"), "strength": divergence.get("strength", "none"), "score": div_score, "weight": 8, "weighted_score": breakdown["Divergence"], "explanation": "Divergence analysis"},
            "ema_alignment": {"value": ema.get("alignment", "mixed"), "signal": ema.get("alignment", "mixed"), "score": ema_score, "weight": 10, "weighted_score": breakdown["EMA Alignment"], "explanation": "EMA alignment"},
            "price_vs_ema": {"value": ema.get("price_vs_ema", "mixed"), "signal": ema.get("price_vs_ema", "mixed"), "score": pve_score, "weight": 8, "weighted_score": breakdown["Price vs EMA"], "explanation": "Price position vs EMAs"},
            "support_resistance": {"nearest_support": supports[0]["price"] if supports else current_price * 0.95, "nearest_resistance": resistances[0]["price"] if resistances else current_price * 1.05, "signal": "bullish" if supports else "neutral", "score": sr_score, "weight": 20, "weighted_score": breakdown["S/R Levels"], "explanation": "Support/Resistance levels"},
            "fibonacci": {"key_level": "0.618", "price_at_level": current_price * 0.98, "signal": "neutral", "score": 55, "weight": 8, "weighted_score": breakdown["Fibonacci"], "explanation": "Fibonacci retracement"},
            "bollinger": {"position": bb.get("position", "middle"), "signal": "bullish" if bb.get("position") == "lower_band" else "bearish" if bb.get("position") == "upper_band" else "neutral", "score": bb_score, "weight": 8, "weighted_score": breakdown["Bollinger"], "explanation": "Bollinger Bands position"},
            "volume": {"trend": vol.get("trend", "neutral"), "signal": vol.get("trend", "neutral"), "score": vol_score, "weight": 3, "weighted_score": breakdown["Volume"], "explanation": "Volume analysis"},
            "obv": {"trend": obv.get("trend", "neutral"), "divergence": obv.get("divergence", "none"), "signal": obv.get("signal", "neutral"), "explanation": "On-Balance Volume analysis"}
        },
        "confluence_score": confluence_score,
        "confluence_breakdown": breakdown,
        "recommended_bias": recommended_bias,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "analysis_summary": f"Analysis shows {indicators.get('trend', 'neutral')} trend with confluence score of {confluence_score}. Higher TF trend: {htf_trend}.",
        "trade_rationale": f"Based on RSI ({rsi.get('value', 50)}), MACD ({macd.get('trend', 'neutral')}), EMA alignment ({ema.get('alignment', 'mixed')}), and OBV ({obv.get('trend', 'neutral')})."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemini-2.5-flash" if model else "none",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment (Railway/Heroku) or default to 8002
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
