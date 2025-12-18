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
    klines = await fetch_okx_klines(symbol, interval, min(limit, 100))
    if klines and len(klines) > 0:
        print(f"Got {len(klines)} klines from OKX")
        return {"symbol": symbol, "interval": interval, "klines": klines, "source": "okx"}
    
    # 2. Try CoinGecko OHLC (limited intervals)
    print(f"OKX failed, trying CoinGecko OHLC for {symbol}...")
    days = INTERVAL_TO_DAYS.get(interval.lower(), 1)
    cg_klines = await fetch_coingecko_ohlc(symbol, days)
    if cg_klines and len(cg_klines) > 0:
        print(f"Got {len(cg_klines)} klines from CoinGecko")
        return {"symbol": symbol, "interval": interval, "klines": cg_klines, "source": "coingecko"}
    
    # 3. Try Bybit
    print(f"CoinGecko failed, trying Bybit for {symbol}...")
    bybit_interval = INTERVAL_MAP.get(interval.lower(), interval)
    data = await fetch_bybit("kline", {
        "category": "spot",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": min(limit, 200)
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
        print(f"Got {len(klines)} klines from Bybit")
        return {"symbol": symbol, "interval": interval, "klines": klines, "source": "bybit"}
    
    # 4. Try Binance (last resort)
    print(f"Bybit failed, trying Binance for {symbol} (last resort)...")
    data = await fetch_binance("klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
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
        """RSI (Relative Strength Index) - Wilder's smoothing"""
        if len(closes) < period + 1:
            return {"value": 50.0, "signal": "neutral"}
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = []
        losses = []
        for d in deltas:
            gains.append(d if d > 0 else 0)
            losses.append(-d if d < 0 else 0)
        
        # Initial average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Wilder's smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi = round(rsi, 2)
        
        if rsi < 30:
            signal = "oversold"
        elif rsi > 70:
            signal = "overbought"
        else:
            signal = "neutral"
        
        return {"value": rsi, "signal": signal, "period": period}
    
    @staticmethod
    def calculate_ema(closes: List[float], period: int) -> float:
        """EMA (Exponential Moving Average)"""
        if len(closes) < period:
            return closes[-1] if closes else 0
        
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first EMA value
        ema = sum(closes[:period]) / period
        
        # Calculate EMA for rest of data
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round(ema, 4)
    
    @staticmethod
    def calculate_sma(closes: List[float], period: int) -> float:
        """SMA (Simple Moving Average)"""
        if len(closes) < period:
            return closes[-1] if closes else 0
        return round(sum(closes[-period:]) / period, 4)
    
    @staticmethod
    def calculate_macd(closes: List[float]) -> Dict:
        """MACD (12, 26, 9)"""
        if len(closes) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        ema12 = IndicatorEngine.calculate_ema(closes, 12)
        ema26 = IndicatorEngine.calculate_ema(closes, 26)
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-period EMA of MACD)
        # For simplicity, approximate with recent MACD values
        macd_values = []
        for i in range(len(closes) - 26, len(closes)):
            e12 = IndicatorEngine.calculate_ema(closes[:i+1], 12)
            e26 = IndicatorEngine.calculate_ema(closes[:i+1], 26)
            macd_values.append(e12 - e26)
        
        signal_line = IndicatorEngine.calculate_ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
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
    def calculate_bollinger(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Bollinger Bands"""
        if len(closes) < period:
            price = closes[-1] if closes else 0
            return {"upper": price, "middle": price, "lower": price, "bandwidth": 0, "position": "middle"}
        
        sma = sum(closes[-period:]) / period
        variance = sum((p - sma) ** 2 for p in closes[-period:]) / period
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
        """ATR (Average True Range)"""
        if len(klines) < period + 1:
            return 0
        
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
        
        return round(sum(trs[-period:]) / period, 4)
    
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


def calculate_all_indicators(klines: List[Dict]) -> Dict:
    """Calculate all indicators from kline data"""
    if not klines:
        return {}
    
    closes = [k["close"] for k in klines]
    current_price = closes[-1]
    
    engine = IndicatorEngine()
    
    # RSI
    rsi = engine.calculate_rsi(closes, 14)
    
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
        "ema": {
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "alignment": ema_alignment,
            "price_vs_ema": price_vs_ema
        },
        "macd": macd,
        "bollinger": bollinger,
        "atr": atr,
        "support_resistance": sr_levels,
        "fibonacci": fib,
        "volume": volume,
        "trend": trend
    }


@app.get("/api/indicators/{symbol}/{interval}")
async def get_indicators(symbol: str, interval: str):
    """GET /api/indicators - Pre-calculated indicators"""
    klines_data = await get_klines(symbol, interval, 500)
    klines = klines_data["klines"]
    
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
    img_width: int,
    img_height: int,
    price_scale: Dict
) -> str:
    """Build Chain-of-Thought analysis prompt"""
    
    current_price = price_data.get("current_price", 0)
    rsi = indicators.get("rsi", {})
    ema = indicators.get("ema", {})
    macd = indicators.get("macd", {})
    bb = indicators.get("bollinger", {})
    atr = indicators.get("atr", 0)
    sr = indicators.get("support_resistance", {})
    fib = indicators.get("fibonacci", {})
    vol = indicators.get("volume", {})
    
    max_price = price_scale["max_price"]
    min_price = price_scale["min_price"]
    chart_height = img_height - 150
    
    prompt = f"""ROLE: You are an expert institutional technical analyst with 20+ years experience.
Analyze this {symbol} chart on {timeframe} timeframe using systematic Chain-of-Thought reasoning.

=== VERIFIED MARKET DATA (FROM BINANCE API) ===
Symbol: {symbol}
Current Price: ${current_price:,.4f}
24h Change: {price_data.get('price_change_pct', 0):.2f}%
24h High: ${price_data.get('high_24h', 0):,.4f}
24h Low: ${price_data.get('low_24h', 0):,.4f}
24h Volume: ${price_data.get('quote_volume', 0):,.0f}

=== PRE-CALCULATED INDICATORS (VERIFIED) ===
RSI(14): {rsi.get('value', 50)} ({rsi.get('signal', 'neutral')})
EMA20: ${ema.get('ema20', 0):,.4f}
EMA50: ${ema.get('ema50', 0):,.4f}
EMA200: ${ema.get('ema200', 0):,.4f}
EMA Alignment: {ema.get('alignment', 'mixed')}
Price vs EMAs: {ema.get('price_vs_ema', 'mixed')}
MACD: {macd.get('macd', 0):.6f} | Signal: {macd.get('signal', 0):.6f} | Histogram: {macd.get('histogram', 0):.6f}
MACD Trend: {macd.get('trend', 'neutral')}
Bollinger Bands: Upper ${bb.get('upper', 0):,.4f} | Middle ${bb.get('middle', 0):,.4f} | Lower ${bb.get('lower', 0):,.4f}
Bollinger Position: {bb.get('position', 'middle')}
ATR(14): ${atr:,.4f}
Volume Trend: {vol.get('trend', 'neutral')} (Ratio: {vol.get('ratio', 1.0):.2f}x)

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

=== CHART SPECIFICATIONS ===
Image Dimensions: {img_width}px × {img_height}px
Y-axis Range: ${min_price:,.4f} (bottom, Y≈{img_height - 80}) to ${max_price:,.4f} (top, Y≈70)
Chart Area: Y = 70 to Y = {img_height - 80}

=== Y-COORDINATE FORMULA ===
For any price P: Y = 70 + ((max_price - P) / (max_price - min_price)) × {chart_height}
Where max_price = {max_price:,.4f} and min_price = {min_price:,.4f}

=== CHAIN-OF-THOUGHT ANALYSIS ===

STEP 1: TREND ASSESSMENT
Look at the EMA alignment and price structure. Is price making higher highs/lows or lower highs/lows?

STEP 2: MOMENTUM EVALUATION  
Evaluate RSI (oversold <30 is bullish, overbought >70 is bearish) and MACD (positive histogram is bullish).

STEP 3: SUPPORT/RESISTANCE ANALYSIS
Identify where price might bounce (support) or get rejected (resistance).

STEP 4: CONFLUENCE SCORING
Score each indicator 0-100 based on how bullish it is, then multiply by weight:
- RSI: weight 15%
- MACD: weight 15%
- EMA Alignment: weight 15%
- Price vs EMA: weight 10%
- S/R Levels: weight 15%
- Fibonacci: weight 10%
- Bollinger: weight 10%
- Volume: weight 10%

STEP 5: TRADE SETUP
Only recommend trade if confluence ≥ 60. Entry at support, SL below support, TPs at resistance levels.

=== OUTPUT FORMAT (JSON ONLY - NO MARKDOWN, NO CODE BLOCKS) ===

Return ONLY valid JSON with this exact structure:
{{
  "price_scale": {{
    "max_price": {max_price},
    "min_price": {min_price}
  }},
  "trend_analysis": {{
    "trend": "bullish/bearish/neutral",
    "reasoning": "explanation"
  }},
  "indicators": {{
    "rsi": {{"value": {rsi.get('value', 50)}, "signal": "{rsi.get('signal', 'neutral')}", "score": <0-100>, "weight": 15, "weighted_score": <calculated>, "explanation": "why"}},
    "macd": {{"value": {{"macd": {macd.get('macd', 0)}, "signal": {macd.get('signal', 0)}, "histogram": {macd.get('histogram', 0)}}}, "signal": "{macd.get('trend', 'neutral')}", "score": <0-100>, "weight": 15, "weighted_score": <calculated>, "explanation": "why"}},
    "ema_alignment": {{"value": "{ema.get('alignment', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 15, "weighted_score": <calculated>, "explanation": "why"}},
    "price_vs_ema": {{"value": "{ema.get('price_vs_ema', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "support_resistance": {{"nearest_support": <price>, "nearest_resistance": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 15, "weighted_score": <calculated>, "explanation": "why"}},
    "fibonacci": {{"key_level": "0.618", "price_at_level": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "bollinger": {{"position": "{bb.get('position', 'middle')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}},
    "volume": {{"trend": "{vol.get('trend', 'neutral')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": 10, "weighted_score": <calculated>, "explanation": "why"}}
  }},
  "confluence_score": <sum of weighted_scores>,
  "confluence_breakdown": {{
    "RSI": <weighted_score>,
    "MACD": <weighted_score>,
    "EMA Alignment": <weighted_score>,
    "Price vs EMA": <weighted_score>,
    "S/R Levels": <weighted_score>,
    "Fibonacci": <weighted_score>,
    "Bollinger": <weighted_score>,
    "Volume": <weighted_score>
  }},
  "trade_setup": {{
    "bias": "long/short/neutral",
    "confidence": "high/medium/low",
    "entry": {{"price": <number>, "y": <Y coordinate using formula>, "reasoning": "why"}},
    "stop_loss": {{"price": <number>, "y": <Y coordinate>, "reasoning": "why"}},
    "tp1": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:1.5", "reasoning": "why"}},
    "tp2": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:2.5", "reasoning": "why"}},
    "tp3": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:4", "reasoning": "why"}}
  }},
  "support_levels": [
    {{"price": <n>, "y": <Y coordinate>, "strength": "strong/moderate/weak"}}
  ],
  "resistance_levels": [
    {{"price": <n>, "y": <Y coordinate>, "strength": "strong/moderate/weak"}}
  ],
  "risk_reward": "1:X",
  "analysis_summary": "2-3 sentence summary",
  "trade_rationale": "detailed explanation"
}}

CRITICAL RULES:
1. Use the EXACT indicator values provided above
2. Calculate Y coordinates using: Y = 70 + ((max_price - price) / (max_price - min_price)) × {chart_height}
3. confluence_score MUST equal sum of all weighted_scores
4. Only recommend trade if confluence_score ≥ 60
5. For SPOT trading, prefer LONG setups
6. Return ONLY valid JSON, no markdown code blocks"""

    return prompt


# ============================================================
# PART 5: CHART ANNOTATION DRAWING
# ============================================================

def draw_annotations(image_bytes: bytes, analysis_data: Dict, price_scale: Dict, img_height: int) -> bytes:
    """Draw all TA annotations on the chart"""
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    height, width = img.shape[:2]
    
    max_price = price_scale["max_price"]
    min_price = price_scale["min_price"]
    
    # Colors (BGR)
    colors = {
        "entry": (0, 200, 0),       # Green
        "stop_loss": (0, 0, 255),   # Red
        "tp1": (255, 200, 0),       # Cyan
        "tp2": (255, 180, 0),       # Lighter cyan
        "tp3": (255, 150, 0),       # Even lighter
        "support": (0, 200, 200),   # Yellow
        "resistance": (200, 0, 200) # Purple/Magenta
    }
    
    def get_y(price: float) -> int:
        return price_to_y(price, max_price, min_price, height)
    
    def draw_dashed_line(pt1: Tuple, pt2: Tuple, color: Tuple, thickness: int = 2):
        """Draw dashed line"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dash_length = 15
        gap_length = 8
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        dx, dy = dx / length, dy / length
        current = 0
        is_dash = True
        
        while current < length:
            seg_len = dash_length if is_dash else gap_length
            next_pos = min(current + seg_len, length)
            
            if is_dash:
                start = (int(x1 + current * dx), int(y1 + current * dy))
                end = (int(x1 + next_pos * dx), int(y1 + next_pos * dy))
                cv2.line(img, start, end, color, thickness)
            
            current = next_pos
            is_dash = not is_dash
    
    def draw_label(text: str, y: int, color: Tuple, offset: int = 0):
        """Draw colored label on right side"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        padding = 6
        label_width = text_width + padding * 2
        label_height = text_height + padding * 2
        
        x = width - label_width - 10
        y = max(label_height + 5, min(y + offset, height - 5))
        
        # Background
        cv2.rectangle(img, (x, y - label_height), (x + label_width, y), color, -1)
        
        # Text (black)
        cv2.putText(img, text, (x + padding, y - padding), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Track Y positions to avoid overlap
    used_y_positions = []
    
    def get_safe_y(target_y: int) -> int:
        """Get Y position that doesn't overlap with existing labels"""
        min_gap = 25
        for used_y in used_y_positions:
            if abs(target_y - used_y) < min_gap:
                target_y = used_y + min_gap
        used_y_positions.append(target_y)
        return target_y
    
    # Draw Support levels
    for support in analysis_data.get("support_levels", []):
        price = support.get("price", 0)
        if price > 0:
            y = support.get("y") or get_y(price)
            if 70 < y < height - 80:
                draw_dashed_line((0, y), (width - 170, y), colors["support"], 2)
                safe_y = get_safe_y(y)
                draw_label(f"SUPPORT ${price:,.1f}", safe_y, colors["support"])
    
    # Draw Resistance levels
    for resistance in analysis_data.get("resistance_levels", []):
        price = resistance.get("price", 0)
        if price > 0:
            y = resistance.get("y") or get_y(price)
            if 70 < y < height - 80:
                draw_dashed_line((0, y), (width - 170, y), colors["resistance"], 2)
                safe_y = get_safe_y(y)
                draw_label(f"RESISTANCE ${price:,.1f}", safe_y, colors["resistance"])
    
    # Draw Trade Setup
    trade_setup = analysis_data.get("trade_setup", {})
    
    # Entry
    entry = trade_setup.get("entry", {})
    if entry.get("price"):
        price = entry["price"]
        y = entry.get("y") or get_y(price)
        if 70 < y < height - 80:
            draw_dashed_line((0, y), (width - 170, y), colors["entry"], 2)
            # Circle marker
            cv2.circle(img, (width // 2, y), 8, colors["entry"], -1)
            cv2.circle(img, (width // 2, y), 10, (255, 255, 255), 2)
            safe_y = get_safe_y(y)
            draw_label(f"ENTRY ${price:,.1f}", safe_y, colors["entry"])
    
    # Stop Loss
    sl = trade_setup.get("stop_loss", {})
    if sl.get("price"):
        price = sl["price"]
        y = sl.get("y") or get_y(price)
        if 70 < y < height - 80:
            draw_dashed_line((0, y), (width - 170, y), colors["stop_loss"], 2)
            # X marker
            cv2.line(img, (width//2-8, y-8), (width//2+8, y+8), colors["stop_loss"], 3)
            cv2.line(img, (width//2-8, y+8), (width//2+8, y-8), colors["stop_loss"], 3)
            safe_y = get_safe_y(y)
            draw_label(f"STOP LOSS ${price:,.1f}", safe_y, colors["stop_loss"])
    
    # Take Profits
    for i, tp_key in enumerate(["tp1", "tp2", "tp3"]):
        tp = trade_setup.get(tp_key, {})
        if tp.get("price"):
            price = tp["price"]
            y = tp.get("y") or get_y(price)
            if 70 < y < height - 80:
                color = colors[tp_key]
                draw_dashed_line((0, y), (width - 170, y), color, 2)
                # Circle marker
                x_offset = (i + 1) * 20
                cv2.circle(img, (width // 2 + x_offset, y), 6, color, -1)
                cv2.circle(img, (width // 2 + x_offset, y), 8, (255, 255, 255), 2)
                safe_y = get_safe_y(y)
                rr = tp.get("risk_reward", f"TP{i+1}")
                draw_label(f"TP{i+1} ${price:,.1f}", safe_y, color)
    
    # Encode result
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


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
            else:
                klines = []
        else:
            # Fetch with multi-source fallback (OKX -> CoinGecko -> Bybit -> Binance)
            print(f"Fetching klines for analysis from multi-source fallback...")
            
            klines = await fetch_okx_klines(symbol, timeframe, 100)
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
                    "limit": 200
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
        
        # Get price scale
        price_scale = extract_price_scale(klines)
        
        # Build prompt and call Gemini
        prompt = build_analysis_prompt(
            symbol, timeframe, price_data, indicators, depth_data,
            img_width, img_height, price_scale
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
        if not analysis_data or "trade_setup" not in analysis_data:
            analysis_data = generate_fallback_analysis(indicators, price_data, price_scale, img_height)
        
        # Draw annotations
        annotated_bytes = draw_annotations(image_bytes, analysis_data, price_scale, img_height)
        annotated_b64 = base64.b64encode(annotated_bytes).decode('utf-8')
        
        # Build response
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": price_data["current_price"],
            "binance_data": price_data,
            "calculated_indicators": indicators,
            "indicators": analysis_data.get("indicators", {}),
            "trade_setup": analysis_data.get("trade_setup", {}),
            "support_levels": analysis_data.get("support_levels", []),
            "resistance_levels": analysis_data.get("resistance_levels", []),
            "confluence_score": analysis_data.get("confluence_score", 0),
            "confluence_breakdown": analysis_data.get("confluence_breakdown", {}),
            "trend": analysis_data.get("trend_analysis", {}).get("trend", indicators.get("trend", "neutral")),
            "bias": analysis_data.get("trade_setup", {}).get("bias", "neutral"),
            "risk_reward": analysis_data.get("risk_reward", "N/A"),
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


def generate_fallback_analysis(indicators: Dict, price_data: Dict, price_scale: Dict, img_height: int) -> Dict:
    """Generate fallback analysis when AI fails"""
    
    current_price = price_data["current_price"]
    max_price = price_scale["max_price"]
    min_price = price_scale["min_price"]
    
    sr = indicators.get("support_resistance", {})
    supports = sr.get("support", [])
    resistances = sr.get("resistance", [])
    
    # Calculate Y coordinates for levels
    def get_y(price: float) -> int:
        return price_to_y(price, max_price, min_price, img_height)
    
    support_levels = []
    for s in supports[:3]:
        support_levels.append({
            "price": s["price"],
            "y": get_y(s["price"]),
            "strength": s.get("strength", "moderate")
        })
    
    resistance_levels = []
    for r in resistances[:3]:
        resistance_levels.append({
            "price": r["price"],
            "y": get_y(r["price"]),
            "strength": r.get("strength", "moderate")
        })
    
    # Determine trade setup
    rsi = indicators.get("rsi", {})
    macd = indicators.get("macd", {})
    ema = indicators.get("ema", {})
    vol = indicators.get("volume", {})
    bb = indicators.get("bollinger", {})
    
    # Simple confluence scoring
    score = 50  # Start neutral
    breakdown = {}
    
    # RSI
    if rsi.get("signal") == "oversold":
        rsi_score = 80
    elif rsi.get("signal") == "overbought":
        rsi_score = 20
    else:
        rsi_score = 50
    breakdown["RSI"] = round(rsi_score * 0.15, 1)
    score += (rsi_score - 50) * 0.15
    
    # MACD
    if macd.get("trend") == "bullish":
        macd_score = 75
    elif macd.get("trend") == "bearish":
        macd_score = 25
    else:
        macd_score = 50
    breakdown["MACD"] = round(macd_score * 0.15, 1)
    score += (macd_score - 50) * 0.15
    
    # EMA
    if ema.get("alignment") == "bullish":
        ema_score = 80
    elif ema.get("alignment") == "bearish":
        ema_score = 20
    else:
        ema_score = 50
    breakdown["EMA Alignment"] = round(ema_score * 0.15, 1)
    score += (ema_score - 50) * 0.15
    
    # Price vs EMA
    if ema.get("price_vs_ema") == "above_all":
        pve_score = 75
    elif ema.get("price_vs_ema") == "below_all":
        pve_score = 25
    else:
        pve_score = 50
    breakdown["Price vs EMA"] = round(pve_score * 0.1, 1)
    score += (pve_score - 50) * 0.1
    
    # S/R
    sr_score = 60 if supports else 50
    breakdown["S/R Levels"] = round(sr_score * 0.15, 1)
    score += (sr_score - 50) * 0.15
    
    # Volume
    if vol.get("trend") == "bullish":
        vol_score = 70
    elif vol.get("trend") == "bearish":
        vol_score = 30
    else:
        vol_score = 50
    breakdown["Volume"] = round(vol_score * 0.1, 1)
    score += (vol_score - 50) * 0.1
    
    # Bollinger
    if bb.get("position") == "lower_band":
        bb_score = 75
    elif bb.get("position") == "upper_band":
        bb_score = 25
    else:
        bb_score = 50
    breakdown["Bollinger"] = round(bb_score * 0.1, 1)
    score += (bb_score - 50) * 0.1
    
    # Fibonacci
    breakdown["Fibonacci"] = 5.0
    
    confluence_score = max(0, min(100, round(score, 1)))
    
    # Trade setup
    bias = "long" if confluence_score >= 55 else "short" if confluence_score <= 45 else "neutral"
    
    # Entry at nearest support for long
    entry_price = supports[0]["price"] if supports else current_price * 0.99
    sl_price = entry_price * 0.97  # 3% below entry
    
    # TPs
    tp1_price = entry_price * 1.03
    tp2_price = entry_price * 1.05
    tp3_price = entry_price * 1.08
    
    if resistances:
        tp1_price = resistances[0]["price"] if len(resistances) > 0 else tp1_price
        tp2_price = resistances[1]["price"] if len(resistances) > 1 else tp2_price
        tp3_price = resistances[2]["price"] if len(resistances) > 2 else tp3_price
    
    trade_setup = {
        "bias": bias,
        "confidence": "medium" if confluence_score >= 60 else "low",
        "entry": {"price": round(entry_price, 4), "y": get_y(entry_price), "reasoning": "Near support level"},
        "stop_loss": {"price": round(sl_price, 4), "y": get_y(sl_price), "reasoning": "Below support"},
        "tp1": {"price": round(tp1_price, 4), "y": get_y(tp1_price), "risk_reward": "1:1.5", "reasoning": "First resistance"},
        "tp2": {"price": round(tp2_price, 4), "y": get_y(tp2_price), "risk_reward": "1:2.5", "reasoning": "Second resistance"},
        "tp3": {"price": round(tp3_price, 4), "y": get_y(tp3_price), "risk_reward": "1:4", "reasoning": "Extended target"}
    }
    
    return {
        "price_scale": price_scale,
        "trend_analysis": {"trend": indicators.get("trend", "neutral"), "reasoning": "Based on indicator analysis"},
        "indicators": {},
        "confluence_score": confluence_score,
        "confluence_breakdown": breakdown,
        "trade_setup": trade_setup,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "risk_reward": "1:2.5",
        "analysis_summary": f"Analysis shows {indicators.get('trend', 'neutral')} trend with confluence score of {confluence_score}.",
        "trade_rationale": f"Based on RSI ({rsi.get('value', 50)}), MACD ({macd.get('trend', 'neutral')}), and EMA alignment ({ema.get('alignment', 'mixed')})."
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
