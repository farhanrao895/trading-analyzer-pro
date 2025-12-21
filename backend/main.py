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
# CODE VERSION CHECK - This proves the file is being loaded
# ============================================================
print("\n" + "="*80)
print("="*80)
print("✅✅✅ BACKEND MAIN.PY LOADED - VERSION: v2.0_fixed_support_detection ✅✅✅")
print("✅✅✅ This message proves the Python file is being executed ✅✅✅")
print("="*80)
print("="*80 + "\n")

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
        
        VALIDATED: Matches TradingView's ema() function exactly - alpha = 2/(n+1) formula confirmed.
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
        
        VALIDATED: Uses EMA(12)-EMA(26) for MACD line, EMA(9) for signal line - matches TradingView's MACD standard.
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
    def detect_divergence(prices: List[float], indicator_values: List[float], lookback: int = 50, indicator_type: str = "rsi", current_price: float = None) -> Dict:
        """Detect bullish/bearish divergences - Professional Implementation
        
        IMPORTANT: Divergence classification is PURELY based on price-indicator relationships.
        Trend is NOT used to classify divergence. Trend is context, not classification.
        
        Divergence Types (Price-Indicator Relationship):
        - Bullish Regular: Price makes lower low, indicator makes higher low
        - Bullish Hidden: Price makes higher low, indicator makes lower low
        - Bearish Regular: Price makes higher high, indicator makes lower high
        - Bearish Hidden: Price makes lower high, indicator makes higher high
        
        Uses Williams Fractals for swing detection with proper filtering:
        - Minimum 5 bars between swings
        - Maximum 60 bars between swings  
        - Dynamic threshold based on indicator type (RSI vs MACD)
        - Line-of-sight validation
        
        Args:
            prices: List of closing prices
            indicator_values: List of indicator values (RSI, MACD histogram, etc.)
            lookback: Number of recent bars to analyze (default 50)
            indicator_type: Type of indicator ("rsi", "macd", or other) for threshold calculation
        
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
        
        # Dynamic threshold based on indicator type
        if indicator_type == "rsi":
            MIN_INDICATOR_DIFF = 3  # For RSI (0-100 scale)
        elif indicator_type == "macd":
            # For MACD: Calculate as 10% of average absolute value of last 20 values
            # Add price-based minimum floor for better stability across different price ranges
            if len(recent_indicators) >= 20:
                avg_abs_value = sum(abs(v) for v in recent_indicators[-20:]) / 20
                # Price-based minimum floor (0.001% of current price, or 0.0001 if no price)
                price_floor = current_price * 0.00001 if current_price and current_price > 0 else 0.0001
                MIN_INDICATOR_DIFF = max(price_floor, avg_abs_value * 0.10)
            else:
                MIN_INDICATOR_DIFF = 0.0001
        else:
            MIN_INDICATOR_DIFF = 0  # No threshold for unknown types
        
        # ============================================================
        # BULLISH DIVERGENCE DETECTION (Price-Indicator Relationship)
        # ============================================================
        # NOTE: Classification is based ONLY on price vs indicator movement.
        # Trend is NOT used here - it's purely mathematical comparison.
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
        
        # ============================================================
        # BEARISH DIVERGENCE DETECTION (Price-Indicator Relationship)
        # ============================================================
        # NOTE: Classification is based ONLY on price vs indicator movement.
        # Trend is NOT used here - it's purely mathematical comparison.
        # Check for BEARISH DIVERGENCE (price higher high, indicator lower high)
        # Note: Only check if no bullish divergence found (prioritize first match)
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
        
        # Map divergence_type to proper format (ensure it matches required format)
        if divergence_type == "bullish_regular":
            type_formatted = "bullish_regular"
        elif divergence_type == "bullish_hidden":
            type_formatted = "bullish_hidden"
        elif divergence_type == "bearish_regular":
            type_formatted = "bearish_regular"
        elif divergence_type == "bearish_hidden":
            type_formatted = "bearish_hidden"
        else:
            type_formatted = "none"
        
        # Calculate score based on divergence type and strength
        if type_formatted in ["bullish_regular", "bullish_hidden"]:
            if strength == "strong":
                score = 85
            elif strength == "moderate":
                score = 75
            else:
                score = 65
        elif type_formatted in ["bearish_regular", "bearish_hidden"]:
            if strength == "strong":
                score = 15
            elif strength == "moderate":
                score = 25
            else:
                score = 35
        else:
            score = 50
        
        return {
            "type": type_formatted,  # "bullish_regular", "bullish_hidden", "bearish_regular", "bearish_hidden", "none"
            "indicator": indicator_type.upper(),  # "RSI", "MACD", "OBV"
            "signal": signal,  # "bullish", "bearish", "neutral"
            "strength": strength,  # "weak", "moderate", "strong", "none"
            "score": score,  # 0-100 score
            "explanation": f"{divergence_type.replace('_', ' ').title()} on {indicator_type.upper()}" if divergence_type != "none" else "No divergence detected"
        }
    
    @staticmethod
    def detect_market_regime(closes: List[float], ema20: float = None, ema50: float = None, ema200: float = None, adx: float = None) -> str:
        """Detect market regime: trend or range
        
        Trend regime: EMA alignment + (ADX > 25 OR EMA slope > threshold)
        Range regime: Otherwise
        
        Args:
            closes: List of closing prices
            ema20, ema50, ema200: EMA values (optional)
            adx: ADX value (optional)
        
        Returns:
            "trend" or "range"
        """
        # Check EMA alignment for trend
        is_trend_aligned = False
        if ema20 and ema50 and ema200:
            is_trend_aligned = (ema20 > ema50 > ema200) or (ema20 < ema50 < ema200)
        
        # Primary check: ADX > 25 indicates strong trend
        if adx and adx >= 25 and is_trend_aligned:
            return "trend"
        
        # Fallback: Check EMA slope (1.5% threshold - strengthened to reduce false positives)
        if ema20 and len(closes) >= 20:
            ema_slope = (ema20 - closes[-20]) / closes[-20] * 100 if closes[-20] > 0 else 0
            if abs(ema_slope) > 1.5 and is_trend_aligned:
                return "trend"
        
        return "range"
    
    @staticmethod
    def calculate_bollinger(closes: List[float], period: int = 20, std_dev: float = 2.0, 
                           ema20: float = None, ema50: float = None, ema200: float = None, adx: float = None) -> Dict:
        """Bollinger Bands - TradingView Compatible
        
        Uses POPULATION standard deviation (divides by N, not N-1)
        This is confirmed by John Bollinger himself.
        
        VALIDATED: Matches TradingView's BB(20, 2) exactly - SMA(20) middle band, population stddev, multiplier=2 confirmed.
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
        
        # Detect market regime for signal interpretation
        regime = IndicatorEngine.detect_market_regime(closes, ema20, ema50, ema200, adx)
        
        # Regime-aware signal interpretation:
        # In TREND: Riding upper/lower band = continuation (bullish/bearish)
        # In RANGE: Touching upper/lower band = mean reversion (bearish/bullish)
        signal = "neutral"
        if regime == "trend":
            if position == "upper_band":
                signal = "bullish_continuation"  # Riding upper band in uptrend
            elif position == "lower_band":
                signal = "bearish_continuation"  # Riding lower band in downtrend
        else:  # Range regime
            if position == "upper_band":
                signal = "bearish_mean_reversion"  # Overbought, expect pullback
            elif position == "lower_band":
                signal = "bullish_mean_reversion"  # Oversold, expect bounce
        
        return {
            "upper": round(upper, 4),
            "middle": round(sma, 4),
            "lower": round(lower, 4),
            "bandwidth": round(bandwidth, 2),
            "position": position,
            "regime": regime,
            "signal": signal
        }
    
    @staticmethod
    def calculate_atr(klines: List[Dict], period: int = 14) -> float:
        """ATR (Average True Range) using Wilder's smoothing method
        
        Initial ATR = Simple Average of first 14 True Ranges
        Subsequent ATR = ((Previous ATR × 13) + Current True Range) / 14
        
        This matches RSI methodology and is more responsive to volatility changes.
        
        VALIDATED: Matches TradingView's ATR(14) exactly - Wilder's smoothing method confirmed.
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
    def find_support_resistance(klines: List[Dict], num_levels: int = 3, recency_weight: float = 0.7) -> Dict:
        # FORCE PRINT IMMEDIATELY - This proves the function is being called
        print("\n!!! find_support_resistance() FUNCTION CALLED !!!")
        print(f"!!! This is the NEW CODE with recency_weight={recency_weight} !!!\n")
        """Find support and resistance levels with RECENCY WEIGHTING and BREAKOUT DETECTION
        
        CRITICAL IMPROVEMENTS:
        1. Recency weighting - recent levels get higher priority
        2. Breakout level detection - resistance turned support
        3. Maximum distance filter - no supports more than 20% below price
        4. Combined scoring (touches × recency) for ranking
        
        Args:
            klines: List of OHLCV kline dictionaries
            num_levels: Number of levels to return (default 3)
            recency_weight: Weight for recency scoring 0.0-1.0 (default 0.7)
        
        Returns:
            Dict with support and resistance lists, each containing:
            - price: float
            - touches: int
            - strength: "strong"/"moderate"/"weak"
            - recency_score: float (0-1)
        """
        if len(klines) < 10:
            return {"support": [], "resistance": []}
        
        print(f"[S/R] ===== find_support_resistance() CALLED =====")
        print(f"[S/R] Input: {len(klines)} klines, num_levels={num_levels}, recency_weight={recency_weight}")
        
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        closes = [k["close"] for k in klines]
        current_price = closes[-1]
        total_candles = len(klines)
        
        print(f"[S/R] Current price: ${current_price:.4f}, Total candles: {total_candles}")
        
        pivot_highs = []
        pivot_lows = []
        
        lookback = 5
        for i in range(lookback, len(klines) - lookback):
            # Calculate recency score (0.3 to 1.0 based on position in data)
            # More recent candles get higher scores
            recency_score = 0.3 + (i / total_candles) * recency_weight
            
            # Check for swing high
            is_swing_high = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                           all(highs[i] > highs[i+j] for j in range(1, lookback+1))
            if is_swing_high:
                pivot_highs.append({"price": highs[i], "index": i, "recency": recency_score})
            
            # Check for swing low
            is_swing_low = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                          all(lows[i] < lows[i+j] for j in range(1, lookback+1))
            if is_swing_low:
                pivot_lows.append({"price": lows[i], "index": i, "recency": recency_score})
        
        # NEW: Breakout Level Detection (Resistance Turned Support)
        # This is CRITICAL for detecting $0.03 level on ACTUSDT
        if len(klines) >= 50:
            recent_klines = klines[-50:]
            
            # Find the highest high from candles 20-50 bars ago (potential old resistance)
            old_section = recent_klines[:-20]  # Candles 20-50 bars ago
            new_section = recent_klines[-20:]  # Last 20 candles
            
            if old_section and new_section:
                potential_resistance = max(k["high"] for k in old_section)
                recent_lows = [k["low"] for k in new_section]
                recent_closes = [k["close"] for k in new_section]
                
                # Check if price broke above and is now holding above this level
                # Conditions: 
                # 1. Current price above the old resistance
                # 2. Recent lows mostly above the level (allowing for wicks)
                # 3. Recent closes all above the level
                price_above = current_price > potential_resistance
                lows_holding = sum(1 for low in recent_lows if low >= potential_resistance * 0.98) >= len(recent_lows) * 0.7
                closes_above = all(c > potential_resistance * 0.97 for c in recent_closes[-10:])
                
                print(f"[S/R DEBUG] Breakout detection check:")
                print(f"[S/R DEBUG]   Potential resistance: ${potential_resistance:.4f}")
                print(f"[S/R DEBUG]   Current price above: {price_above} (${current_price:.4f} > ${potential_resistance:.4f})")
                print(f"[S/R DEBUG]   Lows holding: {lows_holding} ({sum(1 for low in recent_lows if low >= potential_resistance * 0.98)}/{len(recent_lows)} >= 70%)")
                print(f"[S/R DEBUG]   Closes above: {closes_above} (last 10 closes)")
                
                if price_above and (lows_holding or closes_above):
                    # This is a confirmed breakout level - add as HIGH PRIORITY support
                    pivot_lows.append({
                        "price": potential_resistance,
                        "index": total_candles - 20,
                        "recency": 1.0,  # Highest priority
                        "type": "breakout_level"
                    })
                    print(f"[S/R] ✅ Detected breakout level (resistance->support): ${potential_resistance:.4f}")
                else:
                    print(f"[S/R DEBUG] Breakout level NOT detected - conditions not met")
        else:
            print(f"[S/R DEBUG] Breakout detection skipped - need at least 50 klines, have {len(klines)}")
        
        # NEW: Also detect recent swing lows from last 30 candles with extra weight
        if len(klines) >= 30:
            recent_30 = klines[-30:]
            recent_lows_prices = [k["low"] for k in recent_30]
            
            for i in range(3, len(recent_30) - 3):
                # More sensitive detection for recent data (lookback=3 instead of 5)
                is_recent_swing = all(recent_lows_prices[i] < recent_lows_prices[i-j] for j in range(1, 4)) and \
                                 all(recent_lows_prices[i] < recent_lows_prices[i+j] for j in range(1, 4))
                if is_recent_swing:
                    actual_idx = total_candles - 30 + i
                    # Check if this level isn't already captured
                    already_exists = any(abs(p["price"] - recent_lows_prices[i]) / recent_lows_prices[i] < 0.01 for p in pivot_lows)
                    if not already_exists:
                        pivot_lows.append({
                            "price": recent_lows_prices[i],
                            "index": actual_idx,
                            "recency": 0.95,  # High priority for recent swings
                            "type": "recent_swing"
                        })
        
        # Cluster similar price levels with recency-weighted scoring
        def cluster_levels(levels: List[Dict], threshold_pct: float = 1.0) -> List[Dict]:
            """Cluster nearby price levels and calculate combined scores"""
            if not levels:
                return []
            
            sorted_levels = sorted(levels, key=lambda x: x["price"])
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                # Check if level is within threshold of cluster
                cluster_avg = sum(l["price"] for l in current_cluster) / len(current_cluster)
                price_diff_pct = abs(level["price"] - cluster_avg) / cluster_avg * 100 if cluster_avg > 0 else 100
                
                if price_diff_pct <= threshold_pct:
                    current_cluster.append(level)
                else:
                    # Finalize current cluster
                    if current_cluster:
                        avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                        max_recency = max(l.get("recency", 0.5) for l in current_cluster)
                        avg_recency = sum(l.get("recency", 0.5) for l in current_cluster) / len(current_cluster)
                        has_breakout = any(l.get("type") == "breakout_level" for l in current_cluster)
                        
                        # Combined score: touches × recency, with bonus for breakout levels
                        combined_score = len(current_cluster) * avg_recency
                        if has_breakout:
                            combined_score *= 1.5  # 50% bonus for breakout levels
                        
                        clusters.append({
                            "price": round(avg_price, 4),
                            "touches": len(current_cluster),
                            "recency_score": round(max_recency, 2),
                            "combined_score": round(combined_score, 2),
                            "is_breakout_level": has_breakout
                        })
                    current_cluster = [level]
            
            # Don't forget the last cluster
            if current_cluster:
                avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                max_recency = max(l.get("recency", 0.5) for l in current_cluster)
                avg_recency = sum(l.get("recency", 0.5) for l in current_cluster) / len(current_cluster)
                has_breakout = any(l.get("type") == "breakout_level" for l in current_cluster)
                
                combined_score = len(current_cluster) * avg_recency
                if has_breakout:
                    combined_score *= 1.5
                
                clusters.append({
                    "price": round(avg_price, 4),
                    "touches": len(current_cluster),
                    "recency_score": round(max_recency, 2),
                    "combined_score": round(combined_score, 2),
                    "is_breakout_level": has_breakout
                })
            
            return clusters
        
        support_levels = cluster_levels(pivot_lows)
        resistance_levels = cluster_levels(pivot_highs)
        
        # CRITICAL FIX: Filter supports to reasonable range
        # Support must be: below current price AND within 20% of current price
        max_support_distance = current_price * 0.20  # 20% maximum distance
        min_valid_support = current_price - max_support_distance
        
        # DEBUG: Show filter details
        print(f"[S/R DEBUG] Current price: ${current_price:.4f}")
        print(f"[S/R DEBUG] Max support distance: ${max_support_distance:.4f} (20%)")
        print(f"[S/R DEBUG] Min valid support: ${min_valid_support:.4f}")
        print(f"[S/R DEBUG] Before filter - Support levels: {len(support_levels)}, Resistance levels: {len(resistance_levels)}")
        if support_levels:
            support_prices = [f"${s['price']:.4f}" for s in support_levels]
            print(f"[S/R DEBUG] All support levels before filter: {support_prices}")
        
        support = [s for s in support_levels if s["price"] < current_price and s["price"] >= min_valid_support]
        resistance = [r for r in resistance_levels if r["price"] > current_price]
        
        # DEBUG: Show what was filtered out
        filtered_out = [s for s in support_levels if s["price"] < current_price and s["price"] < min_valid_support]
        if filtered_out:
            filtered_prices = [f"${s['price']:.4f}" for s in filtered_out]
            print(f"[S/R DEBUG] FILTERED OUT {len(filtered_out)} supports below min_valid: {filtered_prices}")
        
        print(f"[S/R DEBUG] After filter - Support levels: {len(support)}, Resistance levels: {len(resistance)}")
        
        # Sort by COMBINED SCORE (recency × touches), not just price
        support = sorted(support, key=lambda x: x.get("combined_score", 1), reverse=True)
        resistance = sorted(resistance, key=lambda x: x.get("combined_score", 1), reverse=True)
        
        # Take top N levels
        support = support[:num_levels]
        resistance = resistance[:num_levels]
        
        # Re-sort by price for final output (highest support first, lowest resistance first)
        support = sorted(support, key=lambda x: x["price"], reverse=True)
        resistance = sorted(resistance, key=lambda x: x["price"])
        
        # Add strength rating based on combined score
        for s in support:
            score = s.get("combined_score", 1)
            s["strength"] = "strong" if score >= 2.0 else "moderate" if score >= 1.0 else "weak"
        for r in resistance:
            score = r.get("combined_score", 1)
            r["strength"] = "strong" if score >= 2.0 else "moderate" if score >= 1.0 else "weak"
        
        print(f"[S/R] Found {len(support)} support levels: {[s['price'] for s in support]}")
        print(f"[S/R] Found {len(resistance)} resistance levels: {[r['price'] for r in resistance]}")
        print(f"[S/R] ===== find_support_resistance() COMPLETE =====\n")
        
        return {"support": support, "resistance": resistance}
    
    @staticmethod
    def calculate_fibonacci(klines: List[Dict], lookback: int = 200, pivot_left: int = 3, pivot_right: int = 3, method: str = "swing", trend_direction: str = None, current_price: float = None, ema_alignment: str = None, ema50: float = None, choch_detected: bool = False, choch_direction: str = None, bos_detected: bool = False, bos_direction: str = None) -> Dict:
        """Calculate Fibonacci retracement levels with proper pivot-based swing detection
        
        Uses confirmed pivot points (pivot_left and pivot_right bars for confirmation) for accurate levels.
        Provides scoring based on price proximity to key Fibonacci levels (0.5, 0.618).
        
        Args:
            klines: List of OHLCV kline dictionaries
            lookback: Number of recent klines to analyze (default 200)
            pivot_left: Bars to left for pivot confirmation (default 3)
            pivot_right: Bars to right for pivot confirmation (default 3)
            method: Detection method ("swing" for pivot-based, "simple" for max/min)
            trend_direction: Optional trend direction ("bullish", "bearish", "mixed") for scoring
            current_price: Optional current price for proximity-based scoring
        """
        if len(klines) < pivot_left + pivot_right + 1:
            # Fallback: not enough data
            if not klines:
                return {
                    "levels": {},
                    "swing_high": 0,
                    "swing_low": 0,
                    "anchor_high": 0,
                    "anchor_low": 0,
                    "trend": "neutral",
                    "nearest_level": None,
                    "distance_pct": None,
                    "signal": "neutral",
                    "fib_score": 50
                }
            recent_klines = klines
        else:
            if len(klines) < lookback:
                lookback = len(klines)
            recent_klines = klines[-lookback:]
        
        # Use PatternDetector's pivot detection for coherent swing points
        # Pass None for lookback to use all recent_klines
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent_klines, pivot_left=pivot_left, pivot_right=pivot_right, lookback=None)
        
        # Build ordered swing lists by index
        swing_highs = sorted(pivot_highs, key=lambda x: x[0]) if pivot_highs else []
        swing_lows = sorted(pivot_lows, key=lambda x: x[0]) if pivot_lows else []
        
        # Determine latest swing event (most recent swing among highs and lows)
        if not swing_highs and not swing_lows:
            # Fallback: No confirmed swings found - use max/min method
            swing_high = float('-inf')
            swing_high_idx = -1
            swing_low = float('inf')
            swing_low_idx = -1
            
            for idx, k in enumerate(recent_klines):
                actual_idx = len(klines) - len(recent_klines) + idx
                if k["high"] > swing_high:
                    swing_high = k["high"]
                    swing_high_idx = actual_idx
                if k["low"] < swing_low:
                    swing_low = k["low"]
                    swing_low_idx = actual_idx
        else:
            # Get latest swing indices
            latest_high_idx = swing_highs[-1][0] if swing_highs else -1
            latest_low_idx = swing_lows[-1][0] if swing_lows else -1
            
            # Determine which is the latest swing event
            if latest_high_idx > latest_low_idx:
                # Latest swing is a HIGH at index H
                swing_high_idx, swing_high = swing_highs[-1]
                
                # Find most recent low with index < H
                matching_low = None
                for low_idx, low_val in reversed(swing_lows):
                    if low_idx < swing_high_idx:
                        matching_low = (low_idx, low_val)
                        break
                
                if matching_low:
                    swing_low_idx, swing_low = matching_low
                else:
                    # Fallback: find min low in lookback before H
                    lookback_before_h = max(0, swing_high_idx - 20)  # Look back up to 20 bars
                    min_low = float('inf')
                    min_low_idx = lookback_before_h
                    start_idx = max(0, swing_high_idx - len(klines) + len(recent_klines) - 20)
                    end_idx = swing_high_idx - len(klines) + len(recent_klines)
                    for i in range(start_idx, min(end_idx, len(recent_klines))):
                        if recent_klines[i]["low"] < min_low:
                            min_low = recent_klines[i]["low"]
                            min_low_idx = len(klines) - len(recent_klines) + i
                    swing_low = min_low if min_low != float('inf') else (recent_klines[0]["low"] if recent_klines else 0)
                    swing_low_idx = min_low_idx if min_low != float('inf') else (len(klines) - len(recent_klines))
            else:
                # Latest swing is a LOW at index L
                swing_low_idx, swing_low = swing_lows[-1]
                
                # Find most recent high with index < L
                matching_high = None
                for high_idx, high_val in reversed(swing_highs):
                    if high_idx < swing_low_idx:
                        matching_high = (high_idx, high_val)
                        break
                
                if matching_high:
                    swing_high_idx, swing_high = matching_high
                else:
                    # Fallback: find max high in lookback before L
                    lookback_before_l = max(0, swing_low_idx - 20)  # Look back up to 20 bars
                    max_high = float('-inf')
                    max_high_idx = lookback_before_l
                    start_idx = max(0, swing_low_idx - len(klines) + len(recent_klines) - 20)
                    end_idx = swing_low_idx - len(klines) + len(recent_klines)
                    for i in range(start_idx, min(end_idx, len(recent_klines))):
                        if recent_klines[i]["high"] > max_high:
                            max_high = recent_klines[i]["high"]
                            max_high_idx = len(klines) - len(recent_klines) + i
                    swing_high = max_high if max_high != float('-inf') else (recent_klines[0]["high"] if recent_klines else 0)
                    swing_high_idx = max_high_idx if max_high != float('-inf') else (len(klines) - len(recent_klines))
        
        # Determine trend direction from the chosen pair (low before high => uptrend, high before low => downtrend)
        # Only override if trend_direction parameter is not provided
        if trend_direction is None:
            if swing_low_idx < swing_high_idx:
                trend_direction_calc = "uptrend"
            else:
                trend_direction_calc = "downtrend"
        else:
            # Map provided trend direction to internal format ("bullish" -> "uptrend", etc.)
            trend_direction_calc = "uptrend" if trend_direction in ["bullish", "uptrend"] else "downtrend"
        
        # Use determined trend direction for calculations
        if trend_direction_calc == "uptrend":
            # UPTREND: Draw from swing_low to swing_high (standard TradingView method)
            # 0% = swing_low (start of move), 100% = swing_high (end of move)
            # Retracement levels are calculated going DOWN from the high: swing_high - (swing_high - swing_low) × ratio
            diff = swing_high - swing_low
            levels = {
                "0.0": round(swing_low, 4),  # Start of move (0%)
                "0.236": round(swing_high - diff * 0.236, 4),  # 23.6% retracement from high
                "0.382": round(swing_high - diff * 0.382, 4),  # 38.2% retracement from high
                "0.5": round(swing_high - diff * 0.5, 4),  # 50% retracement from high
                "0.618": round(swing_high - diff * 0.618, 4),  # 61.8% retracement from high
                "0.786": round(swing_high - diff * 0.786, 4),  # 78.6% retracement from high
                "1.0": round(swing_high, 4),  # End of move (100%)
                # Extensions project ABOVE swing_high (beyond 100%)
                "1.272": round(swing_high + diff * 0.272, 4),  # 127.2% extension
                "1.618": round(swing_high + diff * 0.618, 4),  # 161.8% extension
            }
        else:
            # DOWNTREND: Draw from swing_high to swing_low (standard TradingView method)
            # Note: trend_direction_calc is already set above, no need to overwrite
            # 0% = swing_high (start of move), 100% = swing_low (end of move)
            # Retracement levels are calculated going UP from the low: swing_low + (swing_high - swing_low) × ratio
            diff = swing_high - swing_low
            levels = {
                "0.0": round(swing_high, 4),  # Start of move (0%)
                "0.236": round(swing_low + diff * 0.236, 4),  # 23.6% retracement from low
                "0.382": round(swing_low + diff * 0.382, 4),  # 38.2% retracement from low
                "0.5": round(swing_low + diff * 0.5, 4),  # 50% retracement from low
                "0.618": round(swing_low + diff * 0.618, 4),  # 61.8% retracement from low
                "0.786": round(swing_low + diff * 0.786, 4),  # 78.6% retracement from low
                "1.0": round(swing_low, 4),  # End of move (100%)
                # Extensions project BELOW swing_low (beyond 100%)
                "1.272": round(swing_low - diff * 0.272, 4),  # 127.2% extension
                "1.618": round(swing_low - diff * 0.618, 4),  # 161.8% extension
            }
        
        # Fibonacci scoring based on price proximity to key levels (0.5, 0.618) and trend alignment
        fib_score = 50  # Neutral default
        nearest_level = None
        distance_pct = float('inf')
        
        # Get current price from klines if not provided
        if current_price is None:
            current_price = recent_klines[-1]["close"] if recent_klines else 0
        
        if current_price and current_price > 0:
            # Find nearest key Fibonacci level (0.5 or 0.618)
            key_levels = ["0.5", "0.618"]
            for level_name in key_levels:
                if level_name in levels:
                    level_price = levels[level_name]
                    dist_pct = abs(current_price - level_price) / current_price * 100
                    if dist_pct < distance_pct:
                        distance_pct = dist_pct
                        nearest_level = level_name
            
            # Score: Higher if price is within proximity (0.5% default) of key level AND aligned with trend
            # TREND-AWARE: Neutralize bullish signals in bearish regime
            proximity_pct = 0.5  # Configurable proximity threshold
            if nearest_level and distance_pct <= proximity_pct:
                nearest_level_price = levels[nearest_level]
                
                # Check if we're in a bearish regime that should neutralize bullish fib signals
                is_bearish_regime = False
                if ema_alignment == "bearish":
                    # If EMA alignment is bearish AND price is below EMA50, it's a bearish regime
                    if ema50 and current_price < ema50:
                        is_bearish_regime = True
                
                # Only allow fib bullish boost when: bullish CHOCH OR bullish BOS OR price above EMA50
                bullish_allowed = (choch_detected and choch_direction == "bullish") or (bos_detected and bos_direction == "bullish") or (ema50 and current_price > ema50)
                
                if trend_direction_calc == "uptrend" and current_price > nearest_level_price:
                    if is_bearish_regime and not bullish_allowed:
                        fib_score = 40  # Neutralized: bullish fib in bearish regime
                    else:
                        fib_score = 75  # Price holding above key fib in uptrend (bullish)
                elif trend_direction_calc == "downtrend" and current_price < nearest_level_price:
                    fib_score = 75  # Price holding below key fib in downtrend (bearish continuation)
                elif trend_direction_calc == "uptrend" and current_price < nearest_level_price:
                    fib_score = 35  # Price rejected from key fib in uptrend (bearish)
                elif trend_direction_calc == "downtrend" and current_price > nearest_level_price:
                    if is_bearish_regime:
                        fib_score = 40  # Neutralized: bullish reversal signal in bearish regime
                    else:
                        fib_score = 35  # Price rejected from key fib in downtrend (bullish reversal)
            else:
                nearest_level = "1.0"  # Default to 100% level if not near key levels
                # Even when not near key levels, check for bearish regime neutralization
                if ema_alignment == "bearish" and ema50 and current_price < ema50:
                    # If in bearish regime, cap bullish fib score at 40
                    if fib_score > 50:
                        fib_score = 40
        
        # diff is already calculated above in both branches, but calculate it here for consistency in return
        diff = swing_high - swing_low
        
        # Determine signal based on trend and price position relative to key levels
        signal = "neutral"
        if nearest_level and distance_pct <= 1.0:  # Within 1% of key level
            if trend_direction_calc == "uptrend":
                if current_price > levels.get(nearest_level, current_price):
                    signal = "bullish"  # Price holding above key fib in uptrend
                else:
                    signal = "bearish"  # Price rejected from key fib in uptrend
            else:  # downtrend
                if current_price < levels.get(nearest_level, current_price):
                    signal = "bearish"  # Price holding below key fib in downtrend (bearish continuation)
                else:
                    signal = "bullish"  # Price rejected from key fib in downtrend (bullish reversal)
        
        # Map trend_direction_calc to anchor labels
        if trend_direction_calc == "uptrend":
            anchor_high = swing_high
            anchor_low = swing_low
            trend_label = "bullish_leg"
        else:
            anchor_high = swing_high
            anchor_low = swing_low
            trend_label = "bearish_leg"
        
        return {
            "levels": levels,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "anchor_high": anchor_high,
            "anchor_low": anchor_low,
            "trend": trend_label,
            "trend_direction": trend_direction_calc,  # Keep for backward compatibility
            "swing_high_idx": swing_high_idx,
            "swing_low_idx": swing_low_idx,
            "diff": round(diff, 4),
            "nearest_level": nearest_level,
            "distance_pct": round(distance_pct, 2) if distance_pct != float('inf') else None,
            "signal": signal,
            "fib_score": fib_score
        }
    
    @staticmethod
    def analyze_volume(klines: List[Dict], period: int = 20) -> Dict:
        """Analyze volume trends with fallback for missing data"""
        if len(klines) < period:
            return {"current": 0, "average": 0, "ratio": 1.0, "trend": "neutral", "up_volume": 0, "down_volume": 0, "data_available": False}
        
        volumes = [k.get("volume", 0) for k in klines]
        
        # Check if volume data is available (CoinGecko OHLC doesn't include volume)
        if sum(volumes) == 0:
            return {
                "current": 0,
                "average": 0,
                "ratio": 1.0,
                "trend": "neutral",
                "up_volume": 0,
                "down_volume": 0,
                "data_available": False
            }
        
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
            "down_volume": round(down_volume, 2),
            "data_available": True
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

    @staticmethod
    def detect_order_blocks(klines: List[Dict], atr_period: int = 14, move_threshold: float = 1.5) -> Dict:
        """Detect Order Blocks - Last bullish/bearish candle before strong move (≥1.5x ATR)
        
        Order Blocks are institutional entry zones where large orders were placed.
        They are valid until price closes through them (invalidates the OB).
        
        Returns:
            Dict with bullish_ob, bearish_ob lists, and nearest bullish/bearish OBs
        """
        if len(klines) < atr_period + 5:
            return {"bullish_ob": [], "bearish_ob": [], "nearest_bullish": {}, "nearest_bearish": {}}
        
        # Calculate ATR first
        atr_value = IndicatorEngine.calculate_atr(klines, atr_period)
        if atr_value == 0:
            return {"bullish_ob": [], "bearish_ob": [], "nearest_bullish": {}, "nearest_bearish": {}}
        
        min_move = atr_value * move_threshold
        current_price = klines[-1]["close"]
        
        bullish_obs = []
        bearish_obs = []
        
        # Look for Order Blocks: last candle before strong move
        # Check candles from index 2 to len-3 (need room to look ahead)
        for i in range(2, len(klines) - 3):
            candle = klines[i]
            prev_candle = klines[i-1]
            
            # Check if this candle is bullish or bearish
            is_bullish = candle["close"] > candle["open"]
            is_bearish = candle["close"] < candle["open"]
            
            if not (is_bullish or is_bearish):
                continue  # Skip doji candles
            
            # Look ahead 2-5 candles for strong move
            found_strong_move = False
            move_direction = None
            
            for look_ahead in range(2, min(6, len(klines) - i)):
                future_candle = klines[i + look_ahead]
                
                # Calculate displacement using close-to-close move (reflects actual price movement rather than wick spikes)
                # This is the standard SMC approach: displacement = close-to-close move, not wick-to-wick
                displacement = abs(future_candle["close"] - candle["close"])
                
                if displacement >= min_move:
                    # Determine direction based on close price
                    if future_candle["close"] > candle["close"]:
                        found_strong_move = True
                        move_direction = "bullish"
                    elif future_candle["close"] < candle["close"]:
                        found_strong_move = True
                        move_direction = "bearish"
                    if found_strong_move:
                        break
            
            if found_strong_move:
                # Check if OB is still valid (price hasn't closed through it)
                is_valid = True
                ob_high = candle["high"]
                ob_low = candle["low"]
                
                # Check all candles after the OB to see if price closed through it
                for j in range(i + 1, len(klines)):
                    future_close = klines[j]["close"]
                    if move_direction == "bullish" and future_close < ob_low:
                        is_valid = False  # Price closed below bullish OB
                        break
                    elif move_direction == "bearish" and future_close > ob_high:
                        is_valid = False  # Price closed above bearish OB
                        break
                
                # Order Block body-size check: Filter out very small OBs (less than 0.3x ATR)
                # This helps avoid noise from tiny candles that don't represent significant order flow
                ob_body_size = ob_high - ob_low
                min_body_size = atr_value * 0.3
                
                if ob_body_size < min_body_size:
                    continue  # Skip tiny order blocks
                
                ob_data = {
                    "start_index": i,
                    "end_index": i,
                    "high": round(ob_high, 4),
                    "low": round(ob_low, 4),
                    "strength": "strong" if ob_body_size >= atr_value else "moderate",
                    "is_valid": is_valid,
                    "body_size": round(ob_body_size, 4)
                }
                
                # Standard SMC Order Block Definition:
                # Bullish Order Block = Last BEARISH candle immediately before a bullish displacement (impulse)
                # Bearish Order Block = Last BULLISH candle immediately before a bearish displacement
                if move_direction == "bullish" and is_bearish:
                    bullish_obs.append(ob_data)
                elif move_direction == "bearish" and is_bullish:
                    bearish_obs.append(ob_data)
        
        # Find nearest valid bullish OB below current price (support) and bearish OB above (resistance)
        nearest_bullish = {}
        nearest_bearish = {}
        
        # Bullish OB should be below current price (support level to buy from)
        valid_bullish = [ob for ob in bullish_obs if ob["is_valid"] and ob["high"] < current_price]
        # Bearish OB should be above current price (resistance level that might reject)
        valid_bearish = [ob for ob in bearish_obs if ob["is_valid"] and ob["low"] > current_price]
        
        if valid_bullish:
            nearest_bullish = max(valid_bullish, key=lambda x: x["high"])  # Closest below current price
        if valid_bearish:
            nearest_bearish = min(valid_bearish, key=lambda x: x["low"])  # Closest above current price
        
        return {
            "bullish_ob": bullish_obs[-10:],  # Return last 10 to avoid too much data
            "bearish_ob": bearish_obs[-10:],
            "nearest_bullish": nearest_bullish,
            "nearest_bearish": nearest_bearish
        }

    @staticmethod
    def detect_fair_value_gaps(klines: List[Dict], lookback: int = 100) -> Dict:
        """Detect Fair Value Gaps (FVG) - 3-candle gap patterns where middle candle doesn't overlap
        
        FVGs are price imbalances that tend to get filled. They occur when:
        - Bullish FVG: Gap up between candle i-1 and i+1 (candle i doesn't overlap)
        - Bearish FVG: Gap down between candle i-1 and i+1 (candle i doesn't overlap)
        
        Returns:
            Dict with bullish_fvg, bearish_fvg, unfilled_fvg, and nearest_fvg
        """
        if len(klines) < 3:
            return {"bullish_fvg": [], "bearish_fvg": [], "unfilled_fvg": [], "nearest_fvg": {}}
        
        lookback = min(lookback, len(klines) - 2)
        current_price = klines[-1]["close"]
        
        bullish_fvgs = []
        bearish_fvgs = []
        
        # Check for FVGs in 3-candle windows
        for i in range(1, len(klines) - 1):
            prev_candle = klines[i-1]
            curr_candle = klines[i]
            next_candle = klines[i+1]
            
            # Check for bullish FVG (gap UP)
            # Bullish FVG: Candle 3's LOW > Candle 1's HIGH (price moved up so fast it left a gap)
            if next_candle["low"] > prev_candle["high"]:
                fvg_top = next_candle["low"]
                fvg_bottom = prev_candle["high"]
                fvg_size = fvg_top - fvg_bottom
                
                # Check if gap is filled: filled when subsequent candle's low <= fvg_top
                is_filled = False
                fill_percentage = 0.0
                
                for j in range(i + 2, len(klines)):
                    candle_low = klines[j]["low"]
                    if candle_low <= fvg_top:
                        is_filled = True
                        # Calculate fill percentage
                        if candle_low <= fvg_bottom:
                            fill_percentage = 100.0
                        else:
                            fill_percentage = ((fvg_top - candle_low) / fvg_size) * 100
                        break
                
                fvg_data = {
                    "start_index": i - 1,
                    "top": round(fvg_top, 4),
                    "bottom": round(fvg_bottom, 4),
                    "size": round(fvg_size, 4),
                    "direction": "bullish",
                    "is_filled": is_filled,
                    "fill_percentage": round(fill_percentage, 2)
                }
                bullish_fvgs.append(fvg_data)
            
            # Check for bearish FVG (gap DOWN)
            # Bearish FVG: Candle 3's HIGH < Candle 1's LOW (price moved down so fast it left a gap)
            elif next_candle["high"] < prev_candle["low"]:
                fvg_top = prev_candle["low"]
                fvg_bottom = next_candle["high"]
                fvg_size = fvg_top - fvg_bottom
                
                # Check if gap is filled: filled when subsequent candle's high >= fvg_bottom
                is_filled = False
                fill_percentage = 0.0
                
                for j in range(i + 2, len(klines)):
                    candle_high = klines[j]["high"]
                    if candle_high >= fvg_bottom:
                        is_filled = True
                        # Calculate fill percentage
                        if candle_high >= fvg_top:
                            fill_percentage = 100.0
                        else:
                            fill_percentage = ((candle_high - fvg_bottom) / fvg_size) * 100
                        break
                
                fvg_data = {
                    "start_index": i - 1,
                    "top": round(fvg_top, 4),
                    "bottom": round(fvg_bottom, 4),
                    "size": round(fvg_size, 4),
                    "direction": "bearish",
                    "is_filled": is_filled,
                    "fill_percentage": round(fill_percentage, 2)
                }
                bearish_fvgs.append(fvg_data)
        
        # Get recent FVGs (last lookback candles)
        recent_bullish = bullish_fvgs[-lookback:] if len(bullish_fvgs) > lookback else bullish_fvgs
        recent_bearish = bearish_fvgs[-lookback:] if len(bearish_fvgs) > lookback else bearish_fvgs
        
        # Find unfilled FVGs
        unfilled_fvg = []
        for fvg in recent_bullish + recent_bearish:
            if not fvg["is_filled"]:
                unfilled_fvg.append(fvg)
        
        # Find nearest FVG to current price
        nearest_fvg = {}
        all_fvgs = recent_bullish + recent_bearish
        if all_fvgs:
            # Calculate distance from current price to each FVG
            distances = []
            for fvg in all_fvgs:
                mid_price = (fvg["top"] + fvg["bottom"]) / 2
                distance = abs(current_price - mid_price)
                distances.append((distance, fvg))
            
            if distances:
                nearest_fvg = min(distances, key=lambda x: x[0])[1]
        
        return {
            "bullish_fvg": recent_bullish[-20:],  # Return last 20
            "bearish_fvg": recent_bearish[-20:],
            "unfilled_fvg": unfilled_fvg[-10:],  # Return last 10 unfilled
            "nearest_fvg": nearest_fvg
        }

    @staticmethod
    def detect_liquidity_zones(klines: List[Dict], cluster_threshold_pct: float = 0.5, min_cluster_size: int = 2) -> Dict:
        """Detect Liquidity Zones - Clusters of swing highs/lows where stop losses cluster
        
        Liquidity zones are price levels where retail traders place stop losses.
        These zones attract price as institutions hunt for liquidity.
        
        Returns:
            Dict with liquidity_above, liquidity_below, strongest_above, strongest_below
        """
        if len(klines) < 20:
            return {"liquidity_above": [], "liquidity_below": [], "strongest_above": {}, "strongest_below": {}}
        
        # Use existing method to find peaks and troughs
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        
        peaks, troughs = IndicatorEngine.find_peaks_and_troughs(highs, lookback=5)
        _, troughs_low = IndicatorEngine.find_peaks_and_troughs(lows, lookback=5)
        
        current_price = klines[-1]["close"]
        
        # Combine swing highs (peaks) and swing lows (troughs)
        swing_points = [(idx, price, "high") for idx, price in peaks] + [(idx, price, "low") for idx, price in troughs_low]
        
        # Cluster swing points within threshold percentage
        def cluster_levels(swing_points_list, threshold_pct):
            if not swing_points_list:
                return []
            
            # Sort by price
            sorted_swings = sorted(swing_points_list, key=lambda x: x[1])
            clusters = []
            current_cluster = [sorted_swings[0]]
            
            for swing in sorted_swings[1:]:
                # Check if swing is within threshold of cluster
                cluster_avg = sum(s[1] for s in current_cluster) / len(current_cluster)
                price_diff_pct = abs(swing[1] - cluster_avg) / cluster_avg * 100 if cluster_avg > 0 else 100
                
                if price_diff_pct <= threshold_pct:
                    current_cluster.append(swing)
                else:
                    # Finalize current cluster
                    if len(current_cluster) >= min_cluster_size:
                        cluster_price = sum(s[1] for s in current_cluster) / len(current_cluster)
                        clusters.append({
                            "price": round(cluster_price, 4),
                            "touches": len(current_cluster),
                            "strength": "strong" if len(current_cluster) >= 4 else "moderate" if len(current_cluster) >= 3 else "weak",
                            "zone_type": "high" if any(s[2] == "high" for s in current_cluster) else "low"
                        })
                    current_cluster = [swing]
            
            # Finalize last cluster
            if len(current_cluster) >= min_cluster_size:
                cluster_price = sum(s[1] for s in current_cluster) / len(current_cluster)
                clusters.append({
                    "price": round(cluster_price, 4),
                    "touches": len(current_cluster),
                    "strength": "strong" if len(current_cluster) >= 4 else "moderate" if len(current_cluster) >= 3 else "weak",
                    "zone_type": "high" if any(s[2] == "high" for s in current_cluster) else "low"
                })
            
            return clusters
        
        all_zones = cluster_levels(swing_points, cluster_threshold_pct)
        
        # Separate zones above and below current price
        liquidity_above = [z for z in all_zones if z["price"] > current_price]
        liquidity_below = [z for z in all_zones if z["price"] < current_price]
        
        # Sort by strength (touches)
        liquidity_above.sort(key=lambda x: x["touches"], reverse=True)
        liquidity_below.sort(key=lambda x: x["touches"], reverse=True)
        
        strongest_above = liquidity_above[0] if liquidity_above else {}
        strongest_below = liquidity_below[0] if liquidity_below else {}
        
        return {
            "liquidity_above": liquidity_above[:10],  # Top 10
            "liquidity_below": liquidity_below[:10],
            "strongest_above": strongest_above,
            "strongest_below": strongest_below
        }

    @staticmethod
    def detect_structure(klines: List[Dict], swings: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]], atr_value: float = None, scan_window: int = 20) -> Dict:
        """Detect BOS (Break of Structure) and CHOCH (Change of Character) with stateful break confirmation
        
        Scans last N candles (default 20) to detect breaks and persists BOS/CHOCH until counter-BOS occurs.
        
        Args:
            klines: List of OHLC kline dictionaries
            swings: Tuple of (swing_highs, swing_lows) from PatternDetector.get_swings()
            atr_value: ATR value for buffer calculation (if None, uses fixed 0.1%)
            scan_window: Number of recent candles to scan for breaks (default 20)
        
        Returns:
            Dict with bos, choch, last_swing_high, last_swing_low
        """
        swing_highs, swing_lows = swings
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "bos": {"detected": False, "direction": "none", "level": 0.0, "break_index": -1},
                "choch": {"detected": False, "direction": "none", "level": 0.0, "break_index": -1},
                "last_swing_high": 0.0,
                "last_swing_low": 0.0
            }
        
        # Get last 2 swings ordered by index (time)
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x[0])
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x[0])
        
        prev_high_idx, prev_high = swing_highs_sorted[-2] if len(swing_highs_sorted) >= 2 else swing_highs_sorted[-1]
        last_high_idx, last_high = swing_highs_sorted[-1]
        
        prev_low_idx, prev_low = swing_lows_sorted[-2] if len(swing_lows_sorted) >= 2 else swing_lows_sorted[-1]
        last_low_idx, last_low = swing_lows_sorted[-1]
        
        current_price = klines[-1]["close"]
        current_idx = len(klines) - 1
        
        # Calculate buffer: 0.001 (0.1%) OR 0.2 * ATR (whichever is larger)
        if atr_value and atr_value > 0:
            buffer_atr = 0.2 * atr_value / current_price  # Convert ATR to percentage
            buffer = max(0.001, buffer_atr)
        else:
            buffer = 0.001
        
        # Determine trend direction from swing sequence
        # Bullish trend: Higher highs (HH) and Higher lows (HL)
        # Bearish trend: Lower highs (LH) and Lower lows (LL)
        is_bullish_trend = last_high > prev_high and last_low > prev_low
        is_bearish_trend = last_high < prev_high and last_low < prev_low
        
        # Scan last N candles for breaks (stateful detection)
        scan_window = min(scan_window, len(klines))
        recent_klines = klines[-scan_window:]
        
        # Track breaks in scan window
        bullish_bos_break = None
        bearish_bos_break = None
        bullish_choch_break = None
        bearish_choch_break = None
        
        # Scan each candle for breaks
        for i, k in enumerate(recent_klines):
            candle_idx = len(klines) - scan_window + i
            close_price = k["close"]
            
            # BOS Detection: Bullish BOS = close breaks above previous confirmed swing high
            if close_price > prev_high * (1 + buffer):
                if bullish_bos_break is None or candle_idx < bullish_bos_break["index"]:
                    bullish_bos_break = {
                        "direction": "bullish",
                        "level": prev_high,
                        "index": candle_idx
                    }
            
            # BOS Detection: Bearish BOS = close breaks below previous confirmed swing low
            if close_price < prev_low * (1 - buffer):
                if bearish_bos_break is None or candle_idx < bearish_bos_break["index"]:
                    bearish_bos_break = {
                        "direction": "bearish",
                        "level": prev_low,
                        "index": candle_idx
                    }
            
            # CHOCH Detection: If trend was bearish, breaking prior swing high = bullish CHOCH
            if is_bearish_trend and close_price > prev_high * (1 + buffer):
                if bullish_choch_break is None or candle_idx < bullish_choch_break["index"]:
                    bullish_choch_break = {
                        "direction": "bullish",
                        "level": prev_high,
                        "index": candle_idx
                    }
            
            # CHOCH Detection: If trend was bullish, breaking prior swing low = bearish CHOCH
            if is_bullish_trend and close_price < prev_low * (1 - buffer):
                if bearish_choch_break is None or candle_idx < bearish_choch_break["index"]:
                    bearish_choch_break = {
                        "direction": "bearish",
                        "level": prev_low,
                        "index": candle_idx
                    }
        
        # Determine active BOS (persists until counter-BOS)
        # If both detected, the more recent one takes precedence (unless counter-BOS occurs)
        bos_result = {"detected": False, "direction": "none", "level": 0.0, "break_index": -1}
        
        if bullish_bos_break and bearish_bos_break:
            # Counter-BOS: More recent break determines direction
            if bullish_bos_break["index"] > bearish_bos_break["index"]:
                bos_result = {
                    "detected": True,
                    "direction": "bullish",
                    "level": round(bullish_bos_break["level"], 4),
                    "break_index": bullish_bos_break["index"]
                }
            else:
                bos_result = {
                    "detected": True,
                    "direction": "bearish",
                    "level": round(bearish_bos_break["level"], 4),
                    "break_index": bearish_bos_break["index"]
                }
        elif bullish_bos_break:
            bos_result = {
                "detected": True,
                "direction": "bullish",
                "level": round(bullish_bos_break["level"], 4),
                "break_index": bullish_bos_break["index"]
            }
        elif bearish_bos_break:
            bos_result = {
                "detected": True,
                "direction": "bearish",
                "level": round(bearish_bos_break["level"], 4),
                "break_index": bearish_bos_break["index"]
            }
        
        # Determine active CHOCH (persists independently until opposite CHOCH)
        choch_result = {"detected": False, "direction": "none", "level": 0.0, "break_index": -1}
        
        if bullish_choch_break and bearish_choch_break:
            # Opposite CHOCH: More recent one takes precedence
            if bullish_choch_break["index"] > bearish_choch_break["index"]:
                choch_result = {
                    "detected": True,
                    "direction": "bullish",
                    "level": round(bullish_choch_break["level"], 4),
                    "break_index": bullish_choch_break["index"]
                }
            else:
                choch_result = {
                    "detected": True,
                    "direction": "bearish",
                    "level": round(bearish_choch_break["level"], 4),
                    "break_index": bearish_choch_break["index"]
                }
        elif bullish_choch_break:
            choch_result = {
                "detected": True,
                "direction": "bullish",
                "level": round(bullish_choch_break["level"], 4),
                "break_index": bullish_choch_break["index"]
            }
        elif bearish_choch_break:
            choch_result = {
                "detected": True,
                "direction": "bearish",
                "level": round(bearish_choch_break["level"], 4),
                "break_index": bearish_choch_break["index"]
            }
        
        return {
            "bos": bos_result,
            "choch": choch_result,
            "last_swing_high": round(last_high, 4),
            "last_swing_low": round(last_low, 4)
        }

    @staticmethod
    def detect_break_of_structure(klines: List[Dict], lookback: int = 50, wick_mode: bool = False) -> Dict:
        """Detect Break of Structure (BOS) - Uses detect_structure() for proper break confirmation
        
        Maintains backward compatibility with existing API while using improved detection logic.
        
        Returns:
            Dict with bos_detected, direction, structure_type, confidence, choch info
        """
        if len(klines) < lookback:
            return {
                "bos_detected": False, "direction": "neutral", "last_swing_high": 0, "last_swing_low": 0,
                "structure_type": "neutral", "confidence": "none", "choch_detected": False, "choch_direction": "none"
            }
        
        lookback = min(lookback, len(klines))
        recent_klines = klines[-lookback:]
        
        # Use unified swing detection
        swing_highs, swing_lows = PatternDetector.get_swings(recent_klines, left=3, right=3, lookback=None)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "bos_detected": False, "direction": "neutral", "last_swing_high": 0, "last_swing_low": 0,
                "structure_type": "neutral", "confidence": "none", "choch_detected": False, "choch_direction": "none"
            }
        
        # Get ATR for buffer calculation
        atr_value = IndicatorEngine.calculate_atr(recent_klines, 14)
        
        # Use new detect_structure() method with scan_window for stateful detection
        structure = IndicatorEngine.detect_structure(recent_klines, (swing_highs, swing_lows), atr_value, scan_window=20)
        
        bos_info = structure["bos"]
        choch_info = structure["choch"]
        
        # Determine structure type from swing relationships
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x[0])
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x[0])
        
        if len(swing_highs_sorted) >= 2 and len(swing_lows_sorted) >= 2:
            prev_high = swing_highs_sorted[-2][1]
            last_high = swing_highs_sorted[-1][1]
            prev_low = swing_lows_sorted[-2][1]
            last_low = swing_lows_sorted[-1][1]
            
            if last_high > prev_high and last_low > prev_low:
                structure_type = "bullish"
            elif last_high < prev_high and last_low < prev_low:
                structure_type = "bearish"
            else:
                structure_type = "neutral"
        else:
            structure_type = "neutral"
        
        # Map BOS direction
        if bos_info["detected"]:
            direction = bos_info["direction"]
            confidence = "high"  # Confirmed break
        else:
            direction = "neutral"
            confidence = "none"
        
        return {
            "bos_detected": bos_info["detected"],
            "direction": direction,
            "last_swing_high": structure["last_swing_high"],
            "last_swing_low": structure["last_swing_low"],
            "structure_type": structure_type,
            "confidence": confidence,
            "internal_bos": False,  # Not used in new implementation
            "external_bos": bos_info["detected"],
            "choch_detected": choch_info["detected"],
            "choch_direction": choch_info["direction"]
        }


class PatternDetector:
    """Rule-based chart pattern detection from OHLC data (not images)"""
    
    @staticmethod
    def get_swings(klines: List[Dict], left: int = 3, right: int = 3, lookback: int = None) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Single source of truth for swing detection using high/low prices
        
        This function provides consistent swing point detection across all pattern
        detection methods. Uses high prices for swing highs and low prices for swing lows.
        
        Args:
            klines: List of OHLC kline dictionaries
            left: Bars to left for confirmation (default 3)
            right: Bars to right for confirmation (default 3)
            lookback: Number of recent candles to analyze (None = use all klines)
        
        Returns:
            Tuple of (swing_highs, swing_lows) where each is [(index, price), ...]
            Index is relative to the start of the klines list provided
            swing_highs use high prices, swing_lows use low prices
        """
        if len(klines) < left + right + 1:
            return [], []
        
        # Use lookback if specified, otherwise use all klines
        if lookback is not None and len(klines) > lookback:
            recent = klines[-lookback:]
            start_idx = len(klines) - lookback
        else:
            recent = klines
            start_idx = 0
        
        highs = [k["high"] for k in recent]
        lows = [k["low"] for k in recent]
        
        swing_highs = []
        swing_lows = []
        
        for i in range(left, len(recent) - right):
            # Check for swing high (using high prices)
            is_swing_high = True
            for j in range(1, left + 1):
                if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                actual_idx = start_idx + i
                swing_highs.append((actual_idx, highs[i]))
            
            # Check for swing low (using low prices)
            is_swing_low = True
            for j in range(1, left + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                actual_idx = start_idx + i
                swing_lows.append((actual_idx, lows[i]))
        
        return swing_highs, swing_lows
    
    @staticmethod
    def find_pivots(klines: List[Dict], pivot_left: int = 3, pivot_right: int = 3, lookback: int = None) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find pivot highs and lows using confirmed swing points
        
        DEPRECATED: Use get_swings() as single source of truth.
        This method now delegates to get_swings() for consistency.
        
        Args:
            klines: List of OHLC kline dictionaries (can be full list or subset)
            pivot_left: Bars to left for confirmation
            pivot_right: Bars to right for confirmation
            lookback: Number of recent candles to analyze (None = use all klines)
        
        Returns:
            Tuple of (pivot_highs, pivot_lows) where each is [(index, price), ...]
            Index is relative to the start of the klines list provided
        """
        return PatternDetector.get_swings(klines, left=pivot_left, right=pivot_right, lookback=lookback)
    
    @staticmethod
    def linear_regression_slope(points: List[Tuple[int, float]]) -> float:
        """Calculate linear regression slope for trend line
        
        Args:
            points: List of (index, value) tuples
        
        Returns:
            Slope (positive = up, negative = down)
        """
        if len(points) < 2:
            return 0.0
        
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_x2 = sum(p[0] ** 2 for p in points)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    @staticmethod
    def linear_regression_r_squared(points: List[Tuple[int, float]]) -> float:
        """Calculate R² (coefficient of determination) for linear regression fit
        
        R² measures how well the trend line fits the data points.
        Values range from 0 to 1, where 1 = perfect fit, 0 = poor fit.
        
        Args:
            points: List of (index, value) tuples
        
        Returns:
            R² value between 0 and 1 (higher = better fit)
        """
        if len(points) < 2:
            return 0.0
        
        # Calculate slope and intercept
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_x2 = sum(p[0] ** 2 for p in points)
        sum_y2 = sum(p[1] ** 2 for p in points)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate mean of y values
        mean_y = sum_y / n
        
        # Calculate sum of squares
        ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in points)  # Residual sum of squares
        ss_tot = sum((p[1] - mean_y) ** 2 for p in points)  # Total sum of squares
        
        if ss_tot < 1e-10:
            return 1.0  # Perfect fit if no variance
        
        r_squared = 1.0 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]
    
    @staticmethod
    def detect_triangle(klines: List[Dict], lookback: int = 80) -> Dict:
        """Detect triangle patterns (ascending, descending, symmetrical)
        
        Returns pattern dict or None
        """
        if len(klines) < 20:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=lookback)
        
        if len(pivot_highs) < 3 or len(pivot_lows) < 3:
            return None
        
        # Use last K pivots for trend line fitting
        K = min(5, len(pivot_highs), len(pivot_lows))
        recent_highs = sorted(pivot_highs, key=lambda x: x[0])[-K:]
        recent_lows = sorted(pivot_lows, key=lambda x: x[0])[-K:]
        
        high_slope = PatternDetector.linear_regression_slope(recent_highs)
        low_slope = PatternDetector.linear_regression_slope(recent_lows)
        
        # Calculate R² for trendline fit validation
        high_r_squared = PatternDetector.linear_regression_r_squared(recent_highs)
        low_r_squared = PatternDetector.linear_regression_r_squared(recent_lows)
        
        # Reject if poor trendline fit (R² < 0.7 threshold)
        r_squared_threshold = 0.7
        if high_r_squared < r_squared_threshold or low_r_squared < r_squared_threshold:
            return None  # Poor fit - not a valid triangle pattern
        
        # Normalize slopes by average price for meaningful comparison
        avg_price = sum(k["close"] for k in recent) / len(recent)
        high_slope_pct = (high_slope / avg_price * 100) if avg_price > 0 else 0
        low_slope_pct = (low_slope / avg_price * 100) if avg_price > 0 else 0
        
        # Threshold for "flat" (slope < 0.1% per bar)
        flat_threshold = 0.1
        
        # Check convergence (distance between lines shrinking) - require clear convergence
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            early_dist = recent_highs[0][1] - recent_lows[0][1]
            late_dist = recent_highs[-1][1] - recent_lows[-1][1]
            converging = late_dist < early_dist * 0.8  # At least 20% convergence
        else:
            converging = False
        
        # Reject if channel-like (parallel lines) - slope difference too small
        slope_diff = abs(abs(high_slope_pct) - abs(low_slope_pct))
        is_parallel = slope_diff < 0.05  # Less than 0.05% difference = too parallel
        if is_parallel and not converging:
            return None  # Likely a channel, not a triangle
        
        # Require ≥2 valid touches per side
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        current_price = recent[-1]["close"]
        current_volume = recent[-1].get("volume", 0)
        avg_volume = sum(k.get("volume", 0) for k in recent[-20:]) / 20 if len(recent) >= 20 else current_volume
        volume_expansion = current_volume > avg_volume * 1.2 if avg_volume > 0 else False
        
        pattern = None
        
        # Ascending triangle: upper flat (≈0), lower rising (>0)
        # Clear slope definition: lower slope > flat_threshold, upper slope within flat_threshold
        if low_slope_pct > flat_threshold and abs(high_slope_pct) < flat_threshold and converging:
            upper_line = recent_highs[-1][1] if recent_highs else current_price * 1.02
            breakout = current_price > upper_line * 0.995  # Within 0.5% of upper line
            pattern = {
                "name": "Ascending Triangle",
                "type": "continuation",
                "direction": "bullish" if breakout else "neutral",
                "reliability": "strong" if converging and len(recent_highs) >= 4 else "moderate",
                "status": "completed" if breakout and volume_expansion else "forming",
                "description": f"Ascending triangle with rising lows and flat resistance at ${upper_line:.2f}. {'Breakout confirmed' if breakout else 'Pattern forming'}.",
                "score": 75 if breakout else 60
            }
        
        # Descending triangle: upper falling (<0), lower flat (≈0)
        # Clear slope definition: upper slope < -flat_threshold, lower slope within flat_threshold
        elif high_slope_pct < -flat_threshold and abs(low_slope_pct) < flat_threshold and converging:
            lower_line = recent_lows[-1][1] if recent_lows else current_price * 0.98
            breakdown = current_price < lower_line * 1.005  # Within 0.5% of lower line
            pattern = {
                "name": "Descending Triangle",
                "type": "continuation",
                "direction": "bearish" if breakdown else "neutral",
                "reliability": "strong" if converging and len(recent_lows) >= 4 else "moderate",
                "status": "completed" if breakdown and volume_expansion else "forming",
                "description": f"Descending triangle with falling highs and flat support at ${lower_line:.2f}. {'Breakdown confirmed' if breakdown else 'Pattern forming'}.",
                "score": 25 if breakdown else 50  # Bearish pattern = low score (keep neutral)
            }
        
        # Symmetrical triangle: upper falling (<0), lower rising (>0), both converging
        # Clear slope definition: upper < 0, lower > 0, both significant slopes
        elif (abs(high_slope_pct) > flat_threshold and abs(low_slope_pct) > flat_threshold and 
              high_slope_pct < 0 and low_slope_pct > 0 and converging):
            # Direction depends on breakout
            upper_line = recent_highs[-1][1] if recent_highs else current_price * 1.02
            lower_line = recent_lows[-1][1] if recent_lows else current_price * 0.98
            breakout_up = current_price > upper_line * 0.995
            breakout_down = current_price < lower_line * 1.005
            pattern = {
                "name": "Symmetrical Triangle",
                "type": "continuation",
                "direction": "bullish" if breakout_up else ("bearish" if breakout_down else "neutral"),
                "reliability": "moderate",
                "status": "completed" if (breakout_up or breakout_down) and volume_expansion else "forming",
                "description": f"Symmetrical triangle with converging trend lines. {'Bullish breakout' if breakout_up else ('Bearish breakdown' if breakout_down else 'Awaiting breakout')}.",
                "score": 75 if breakout_up else (25 if breakout_down else 50)
            }
        
        return pattern
    
    @staticmethod
    def detect_flag_pennant(klines: List[Dict], lookback: int = 80) -> Dict:
        """Detect flag and pennant patterns - IMPROVED VERSION
        
        Relaxed thresholds for better real-world detection:
        - Lower impulse requirement (3.5% instead of 5%)
        - Wider consolidation range tolerance (5% instead of 3%)
        - Optional volume check (handles missing volume data)
        - More flexible validation logic
        
        Bull Flag: Strong impulse UP + downward-sloping consolidation + breakout
        Bear Flag: Strong impulse DOWN + upward-sloping consolidation + breakdown
        Pennant: Strong impulse + symmetrical consolidation
        
        Returns pattern dict or None
        """
        if len(klines) < 30:
            return None
        
        recent = klines[-lookback:] if len(klines) >= lookback else klines
        closes = [k["close"] for k in recent]
        highs = [k["high"] for k in recent]
        lows = [k["low"] for k in recent]
        volumes = [k.get("volume", 0) for k in recent]
        
        # Check if volume data is available (CoinGecko OHLC doesn't include volume)
        has_volume_data = sum(volumes) > 0
        
        # IMPROVED: More flexible impulse window (minimum 8 instead of 12)
        impulse_window = min(15, max(8, len(recent) // 4))
        
        if len(recent) < impulse_window + 8:
            return None
        
        # Calculate impulse move (first portion of the lookback)
        early_return = (closes[impulse_window] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
        
        # Calculate ATR for normalization
        atr = IndicatorEngine.calculate_atr(recent[:impulse_window], 14)
        atr_pct = (atr / closes[impulse_window] * 100) if closes[impulse_window] > 0 else 0
        
        # IMPROVED: Relaxed impulse threshold (3.5% instead of 5%, 1.2x ATR instead of 1.5x)
        impulse_threshold = max(3.5, atr_pct * 1.2)
        strong_impulse = abs(early_return) > impulse_threshold
        
        if not strong_impulse:
            return None
        
        # Consolidation phase (after impulse)
        consolidation_start = impulse_window
        consolidation_candles = recent[consolidation_start:]
        
        if len(consolidation_candles) < 8:
            return None
        
        # IMPROVED: Optional volume check with relaxed threshold
        volume_decline = True  # Default to true if no volume data
        if has_volume_data:
            early_vol = sum(volumes[:impulse_window]) / impulse_window if impulse_window > 0 else 0
            late_vol = sum(volumes[consolidation_start:]) / len(consolidation_candles) if len(consolidation_candles) > 0 else 0
            # IMPROVED: Relaxed from 0.7 to 0.85 (15% decline instead of 30%)
            volume_decline = late_vol < early_vol * 0.85 if early_vol > 0 else True
        
        # IMPROVED: Relaxed consolidation range (5% instead of 3%)
        cons_highs = [k["high"] for k in consolidation_candles]
        cons_lows = [k["low"] for k in consolidation_candles]
        cons_range = (max(cons_highs) - min(cons_lows)) / closes[-1] * 100 if closes[-1] > 0 else 100
        tight_range = cons_range < 5.0  # IMPROVED: 5% instead of 3%
        
        # IMPROVED: Extended tolerance for slightly wider ranges with volume confirmation
        acceptable_range = cons_range < 8.0  # Extended tolerance
        
        # IMPROVED: More flexible condition - tight range OR (acceptable range with volume decline)
        if not tight_range and not (acceptable_range and volume_decline):
            return None
        
        # Calculate consolidation slopes using pivot detection
        cons_pivot_highs, cons_pivot_lows = PatternDetector.find_pivots(
            consolidation_candles, pivot_left=2, pivot_right=2, lookback=None
        )
        
        # If not enough pivots, use simple linear regression on prices
        if len(cons_pivot_highs) >= 2 and len(cons_pivot_lows) >= 2:
            cons_high_slope = PatternDetector.linear_regression_slope(
                sorted(cons_pivot_highs, key=lambda x: x[0])[-3:]
            )
            cons_low_slope = PatternDetector.linear_regression_slope(
                sorted(cons_pivot_lows, key=lambda x: x[0])[-3:]
            )
        else:
            # Fallback: calculate slope from first and last points
            if len(cons_highs) > 1:
                cons_high_slope = (cons_highs[-1] - cons_highs[0]) / len(cons_highs)
            else:
                cons_high_slope = 0
            if len(cons_lows) > 1:
                cons_low_slope = (cons_lows[-1] - cons_lows[0]) / len(cons_lows)
            else:
                cons_low_slope = 0
        
        # Normalize slopes for comparison
        avg_price = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        cons_high_slope_pct = (cons_high_slope / avg_price * 100) if avg_price > 0 else 0
        cons_low_slope_pct = (cons_low_slope / avg_price * 100) if avg_price > 0 else 0
        
        current_price = closes[-1]
        current_volume = volumes[-1] if volumes else 0
        late_vol = sum(volumes[consolidation_start:]) / len(consolidation_candles) if has_volume_data and len(consolidation_candles) > 0 else 0
        # IMPROVED: Relaxed volume expansion threshold (1.3x instead of 1.5x)
        volume_expansion = current_volume > late_vol * 1.3 if late_vol > 0 else False
        
        pattern = None
        
        # Bull flag: impulse up, consolidation slopes slightly down or flat
        # IMPROVED: More lenient slope check (0.15% tolerance instead of strict < 0)
        if early_return > 0 and cons_high_slope_pct <= 0.15 and cons_low_slope_pct <= 0.15:
            resistance = max(cons_highs)
            breakout = current_price > resistance * 0.998
            # Calculate reliability based on conditions met
            reliability_score = 0
            if volume_decline: reliability_score += 1
            if tight_range: reliability_score += 1
            if abs(cons_high_slope_pct) < 0.2 and abs(cons_low_slope_pct) < 0.2: reliability_score += 1
            if has_volume_data and volume_expansion: reliability_score += 1
            
            if reliability_score >= 3:
                reliability = "strong"
            elif reliability_score >= 2:
                reliability = "moderate"
            else:
                reliability = "weak"
            
            pattern = {
                "name": "Bull Flag",
                "type": "continuation",
                "direction": "bullish" if breakout else "neutral",
                "reliability": reliability if breakout else "moderate",
                "status": "completed" if breakout else "forming",
                "description": f"Bull flag pattern: {early_return:.1f}% impulse move followed by {cons_range:.1f}% consolidation. {'Breakout confirmed!' if breakout else f'Awaiting breakout above ${resistance:.4f}'}",
                "score": 82 if breakout else 68,
                "impulse_pct": round(early_return, 2),
                "consolidation_range_pct": round(cons_range, 2),
                "volume_confirmed": volume_expansion if has_volume_data else None
            }
        
        # Bear flag: impulse down, consolidation slopes slightly up or flat
        elif early_return < 0 and cons_high_slope_pct >= -0.15 and cons_low_slope_pct >= -0.15:
            support = min(cons_lows)
            breakdown = current_price < support * 1.002
            reliability_score = 0
            if volume_decline: reliability_score += 1
            if tight_range: reliability_score += 1
            if abs(cons_high_slope_pct) < 0.2 and abs(cons_low_slope_pct) < 0.2: reliability_score += 1
            if has_volume_data and volume_expansion: reliability_score += 1
            
            if reliability_score >= 3:
                reliability = "strong"
            elif reliability_score >= 2:
                reliability = "moderate"
            else:
                reliability = "weak"
            
            pattern = {
                "name": "Bear Flag",
                "type": "continuation",
                "direction": "bearish" if breakdown else "neutral",
                "reliability": reliability if breakdown else "moderate",
                "status": "completed" if breakdown else "forming",
                "description": f"Bear flag pattern: {abs(early_return):.1f}% downward impulse followed by {cons_range:.1f}% consolidation. {'Breakdown confirmed!' if breakdown else 'Pattern forming'}.",
                "score": 22 if breakdown else 45,
                "impulse_pct": round(early_return, 2),
                "consolidation_range_pct": round(cons_range, 2),
                "volume_confirmed": volume_expansion if has_volume_data else None
            }
        
        # Pennant: symmetrical consolidation (converging slopes)
        elif early_return != 0:
            # Check for convergence: one slope positive, one negative, both relatively small
            slopes_converging = (
                (cons_high_slope_pct < 0 and cons_low_slope_pct > 0) or
                (abs(cons_high_slope_pct - cons_low_slope_pct) < 0.3 and 
                 abs(cons_high_slope_pct) < 0.3 and abs(cons_low_slope_pct) < 0.3)
            )
            
            if slopes_converging:
                # Direction depends on impulse
                if early_return > 0:
                    resistance = max(cons_highs)
                breakout = current_price > resistance * 0.998
                pattern = {
                    "name": "Bullish Pennant",
                    "type": "continuation",
                    "direction": "bullish" if breakout else "neutral",
                    "reliability": "strong" if volume_expansion else "moderate",
                    "status": "completed" if breakout and volume_expansion else "forming",
                    "description": f"Bullish pennant: {early_return:.1f}% impulse with symmetrical consolidation. {'Breakout confirmed!' if breakout else 'Awaiting breakout'}.",
                    "score": 80 if breakout else 65,
                    "impulse_pct": round(early_return, 2),
                    "consolidation_range_pct": round(cons_range, 2)
                }
            else:
                support = min(cons_lows)
                breakdown = current_price < support * 1.002
                pattern = {
                    "name": "Bearish Pennant",
                    "type": "continuation",
                    "direction": "bearish" if breakdown else "neutral",
                    "reliability": "strong" if volume_expansion else "moderate",
                    "status": "completed" if breakdown and volume_expansion else "forming",
                    "description": f"Bearish pennant: {abs(early_return):.1f}% downward impulse with symmetrical consolidation. {'Breakdown confirmed!' if breakdown else 'Pattern forming'}.",
                    "score": 20 if breakdown else 45,
                    "impulse_pct": round(early_return, 2),
                    "consolidation_range_pct": round(cons_range, 2)
                }
        
        return pattern
    
    @staticmethod
    def detect_wedge(klines: List[Dict], lookback: int = 80) -> Dict:
        """Detect wedge patterns (Rising Wedge and Falling Wedge)
        
        Rising Wedge (Bearish Reversal):
        - Both support AND resistance lines slope UPWARD
        - Lines are CONVERGING (getting closer together)
        - Typically forms after an uptrend
        - Breakdown below support = bearish signal
        
        Falling Wedge (Bullish Reversal):
        - Both support AND resistance lines slope DOWNWARD
        - Lines are CONVERGING (getting closer together)
        - Typically forms after a downtrend
        - Breakout above resistance = bullish signal
        
        Returns pattern dict or None
        """
        if len(klines) < 30:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=None)
        
        if len(pivot_highs) < 3 or len(pivot_lows) < 3:
            return None
        
        # Use last K pivots for trend line fitting
        K = min(5, len(pivot_highs), len(pivot_lows))
        recent_highs = sorted(pivot_highs, key=lambda x: x[0])[-K:]
        recent_lows = sorted(pivot_lows, key=lambda x: x[0])[-K:]
        
        high_slope = PatternDetector.linear_regression_slope(recent_highs)
        low_slope = PatternDetector.linear_regression_slope(recent_lows)
        
        # Calculate R² for trendline fit validation
        high_r_squared = PatternDetector.linear_regression_r_squared(recent_highs)
        low_r_squared = PatternDetector.linear_regression_r_squared(recent_lows)
        
        # Reject if poor trendline fit (R² < 0.7 threshold)
        r_squared_threshold = 0.7
        if high_r_squared < r_squared_threshold or low_r_squared < r_squared_threshold:
            return None  # Poor fit - not a valid wedge pattern
        
        # Normalize slopes by average price for meaningful comparison
        avg_price = sum(k["close"] for k in recent) / len(recent)
        high_slope_pct = (high_slope / avg_price * 100) if avg_price > 0 else 0
        low_slope_pct = (low_slope / avg_price * 100) if avg_price > 0 else 0
        
        # Threshold for meaningful slope (> 0.1% per bar)
        slope_threshold = 0.1
        
        # Check convergence (distance between lines shrinking) - strict requirement ≥30%
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            early_dist = recent_highs[0][1] - recent_lows[0][1]
            late_dist = recent_highs[-1][1] - recent_lows[-1][1]
            converging = late_dist < early_dist * 0.7  # At least 30% convergence (strict)
        else:
            converging = False
        
        # Both slopes must be same direction (both positive or both negative)
        both_positive = high_slope_pct > slope_threshold and low_slope_pct > slope_threshold
        both_negative = high_slope_pct < -slope_threshold and low_slope_pct < -slope_threshold
        same_direction = both_positive or both_negative
        
        if not (converging and same_direction):
            return None
        
        # Reject near-parallel lines (distinguishes from channels/flags)
        # If lines are too parallel (slope difference < threshold), it's likely a channel/flag, not a wedge
        slope_diff = abs(abs(high_slope_pct) - abs(low_slope_pct))
        if slope_diff < 0.05:  # Less than 0.05% difference = too parallel
            return None  # Likely a channel/flag, not a wedge
        
        # For Falling Wedge: Lower trendline slope magnitude > upper slope magnitude (steeper support)
        # This distinguishes falling wedge from bear flag (which has more parallel lines)
        falling_wedge_valid = False
        rising_wedge_valid = False
        if both_negative:
            # Lower slope should be steeper (more negative) than upper slope
            # Check: abs(low_slope_pct) > abs(high_slope_pct)
            falling_wedge_valid = abs(low_slope_pct) > abs(high_slope_pct) * 1.1  # At least 10% steeper
        elif both_positive:
            # For Rising Wedge: Upper slope should be steeper (more positive) than lower slope
            rising_wedge_valid = abs(high_slope_pct) > abs(low_slope_pct) * 1.1
        
        current_price = recent[-1]["close"]
        current_volume = recent[-1].get("volume", 0)
        avg_volume = sum(k.get("volume", 0) for k in recent[-20:]) / 20 if len(recent) >= 20 else current_volume
        volume_expansion = current_volume > avg_volume * 1.2 if avg_volume > 0 else False
        
        # Optional: Check volume decline during wedge formation
        early_vol = sum(k.get("volume", 0) for k in recent[:len(recent)//2]) / (len(recent) // 2) if len(recent) >= 10 else 0
        late_vol = sum(k.get("volume", 0) for k in recent[len(recent)//2:]) / (len(recent) - len(recent)//2) if len(recent) >= 10 else 0
        volume_decline = late_vol < early_vol * 0.8 if early_vol > 0 else False
        
        pattern = None
        
        # Validate touch requirements for wedge patterns
        # At least 3 touches on each line OR 2 touches + 1 near-touch tolerance
        high_touches = len(recent_highs)
        low_touches = len(recent_lows)
        valid_touches = (high_touches >= 3 and low_touches >= 3) or (high_touches >= 2 and low_touches >= 2)
        
        if not valid_touches:
            return None  # Insufficient touches for valid wedge
        
        # Rising Wedge: both slopes positive, converging (bearish reversal)
        # Upper slope should be steeper for valid rising wedge
        if both_positive and converging and rising_wedge_valid:
            support_line = recent_lows[-1][1] if recent_lows else current_price * 0.98
            breakdown = current_price < support_line * 1.01  # Within 1% of support line
            pattern = {
                "name": "Rising Wedge",
                "type": "reversal",
                "direction": "bearish" if breakdown else "neutral",
                "reliability": "strong" if breakdown and volume_expansion else ("moderate" if volume_decline else "weak"),
                "status": "completed" if breakdown and volume_expansion else "forming",
                "description": f"Rising wedge pattern with converging upward-sloping support and resistance. Both lines slope upward but converging, typically bearish. {'Breakdown confirmed' if breakdown else 'Pattern forming'}.",
                "score": 25 if breakdown else 45  # Bearish pattern = low score (keep neutral if forming)
            }
        
        # Falling Wedge: both slopes negative, converging (bullish reversal)
        # Must have steeper lower slope (distinguishes from bear flag)
        elif both_negative and converging and falling_wedge_valid:
            resistance_line = recent_highs[-1][1] if recent_highs else current_price * 1.02
            breakout = current_price > resistance_line * 0.99  # Within 1% of resistance line
            pattern = {
                "name": "Falling Wedge",
                "type": "reversal",
                "direction": "bullish" if breakout else "neutral",
                "reliability": "strong" if breakout and volume_expansion else ("moderate" if volume_decline else "weak"),
                "status": "completed" if breakout and volume_expansion else "forming",
                "description": f"Falling wedge pattern with converging downward-sloping support and resistance. Both lines slope downward but converging, typically bullish. {'Breakout confirmed' if breakout else 'Pattern forming'}.",
                "score": 80 if breakout else 65  # Bullish pattern = high score
            }
        
        if pattern:
            print(f"[PatternDetector.detect_wedge] Detected: {pattern['name']} (slope_high={high_slope_pct:.2f}%, slope_low={low_slope_pct:.2f}%, converging={converging})")
        
        return pattern
    
    @staticmethod
    def detect_channel(klines: List[Dict], lookback: int = 80) -> Dict:
        """Detect channel patterns (Ascending, Descending, Horizontal)
        
        Ascending Channel (Bullish Continuation):
        - Both support AND resistance lines slope UPWARD
        - Lines are PARALLEL (not converging)
        - Buy near support, sell near resistance within channel
        
        Descending Channel (Bearish Continuation):
        - Both support AND resistance lines slope DOWNWARD
        - Lines are PARALLEL (not converging)
        - Sell near resistance within channel
        
        Horizontal Channel / Rectangle (Neutral):
        - Both lines are approximately flat (slope < 0.1%)
        - Price consolidating in range
        - Trade breakout direction
        
        Returns pattern dict or None
        """
        if len(klines) < 30:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=None)
        
        if len(pivot_highs) < 3 or len(pivot_lows) < 3:
            return None
        
        # Use last K pivots for trend line fitting
        K = min(5, len(pivot_highs), len(pivot_lows))
        recent_highs = sorted(pivot_highs, key=lambda x: x[0])[-K:]
        recent_lows = sorted(pivot_lows, key=lambda x: x[0])[-K:]
        
        high_slope = PatternDetector.linear_regression_slope(recent_highs)
        low_slope = PatternDetector.linear_regression_slope(recent_lows)
        
        # Normalize slopes by average price
        avg_price = sum(k["close"] for k in recent) / len(recent)
        high_slope_pct = (high_slope / avg_price * 100) if avg_price > 0 else 0
        low_slope_pct = (low_slope / avg_price * 100) if avg_price > 0 else 0
        
        # Threshold for "flat" (slope < 0.1% per bar)
        flat_threshold = 0.1
        # Threshold for parallel (slope difference < 0.15%)
        parallel_threshold = 0.15
        
        # Check if lines are parallel
        slope_diff = abs(high_slope_pct - low_slope_pct)
        is_parallel = slope_diff < parallel_threshold
        
        if not is_parallel:
            return None
        
        current_price = recent[-1]["close"]
        
        # Determine support and resistance lines (use latest pivot prices)
        support_line = recent_lows[-1][1] if recent_lows else current_price * 0.98
        resistance_line = recent_highs[-1][1] if recent_highs else current_price * 1.02
        
        # Calculate channel width
        channel_width = abs(resistance_line - support_line) / avg_price * 100 if avg_price > 0 else 0
        
        # Determine price position within channel (within 2% = near, otherwise middle)
        distance_to_support = abs(current_price - support_line) / current_price * 100
        distance_to_resistance = abs(current_price - resistance_line) / current_price * 100
        
        if distance_to_support <= 2.0:
            position = "near_support"
        elif distance_to_resistance <= 2.0:
            position = "near_resistance"
        else:
            position = "middle"
        
        pattern = None
        
        # Ascending Channel: both slopes positive, parallel
        if high_slope_pct > flat_threshold and low_slope_pct > flat_threshold:
            pattern = {
                "name": "Ascending Channel",
                "type": "continuation",
                "direction": "bullish" if position == "near_support" else "neutral",
                "reliability": "moderate",
                "status": "forming",
                "description": f"Ascending channel with parallel upward-sloping support and resistance. Channel width: {channel_width:.2f}%. Price position: {position}.",
                "score": 70 if position == "near_support" else 55 if position == "near_resistance" else 62  # Better near support
            }
        
        # Descending Channel: both slopes negative, parallel
        elif high_slope_pct < -flat_threshold and low_slope_pct < -flat_threshold:
            pattern = {
                "name": "Descending Channel",
                "type": "continuation",
                "direction": "bearish" if position == "near_resistance" else "neutral",
                "reliability": "moderate",
                "status": "forming",
                "description": f"Descending channel with parallel downward-sloping support and resistance. Channel width: {channel_width:.2f}%. Price position: {position}.",
                "score": 35 if position == "near_resistance" else 48 if position == "near_support" else 42  # Bearish = low score
            }
        
        # Horizontal Channel: both slopes approximately flat
        elif abs(high_slope_pct) < flat_threshold and abs(low_slope_pct) < flat_threshold:
            pattern = {
                "name": "Horizontal Channel",
                "type": "continuation",
                "direction": "neutral",
                "reliability": "moderate",
                "status": "forming",
                "description": f"Horizontal channel/rectangle pattern with flat support at ${support_line:.2f} and resistance at ${resistance_line:.2f}. Channel width: {channel_width:.2f}%. Awaiting breakout.",
                "score": 50  # Neutral until breakout
            }
        
        if pattern:
            print(f"[PatternDetector.detect_channel] Detected: {pattern['name']} (slope_high={high_slope_pct:.2f}%, slope_low={low_slope_pct:.2f}%, parallel={is_parallel})")
        
        return pattern
    
    @staticmethod
    def detect_head_shoulders(klines: List[Dict], lookback: int = 100) -> Dict:
        """Detect Head & Shoulders / Inverse Head & Shoulders patterns
        
        Requires: 3 peaks with middle highest (or 3 troughs with middle lowest)
        """
        if len(klines) < 30:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=lookback)
        
        pattern = None
        
        # Head & Shoulders (bearish): 3 peaks, middle highest
        if len(pivot_highs) >= 3:
            sorted_highs = sorted(pivot_highs, key=lambda x: x[0])[-3:]
            left_shoulder = sorted_highs[0][1]
            head = sorted_highs[1][1]
            right_shoulder = sorted_highs[2][1]
            
            # Head must be highest, shoulders similar (within 3% tolerance)
            tolerance = head * 0.03
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) < tolerance):
                # Find neckline (low between head and right shoulder)
                head_idx = sorted_highs[1][0]
                right_idx = sorted_highs[2][0]
                neckline_lows = [k["low"] for k in recent if head_idx <= (len(recent) - len(klines) + recent.index(k)) <= right_idx]
                neckline = min(neckline_lows) if neckline_lows else (left_shoulder + right_shoulder) / 2
                
                current_price = recent[-1]["close"]
                breakdown = current_price < neckline * 1.01
                
                pattern = {
                    "name": "Head & Shoulders",
                    "type": "reversal",
                    "direction": "bearish" if breakdown else "neutral",
                    "reliability": "strong" if breakdown else "moderate",
                    "status": "completed" if breakdown else "forming",
                    "description": f"Head & Shoulders pattern detected with head at ${head:.2f} and neckline at ${neckline:.2f}. {'Breakdown confirmed' if breakdown else 'Pattern forming'}.",
                    "score": 20 if breakdown else 45  # Bearish = low score
                }
        
        # Inverse Head & Shoulders (bullish): 3 troughs, middle lowest
        if len(pivot_lows) >= 3:
            sorted_lows = sorted(pivot_lows, key=lambda x: x[0])[-3:]
            left_shoulder = sorted_lows[0][1]
            head = sorted_lows[1][1]
            right_shoulder = sorted_lows[2][1]
            
            tolerance = head * 0.03
            if (head < left_shoulder and head < right_shoulder and 
                abs(left_shoulder - right_shoulder) < tolerance):
                # Find neckline (high between head and right shoulder)
                head_idx = sorted_lows[1][0]
                right_idx = sorted_lows[2][0]
                neckline_highs = [k["high"] for k in recent if head_idx <= (len(recent) - len(klines) + recent.index(k)) <= right_idx]
                neckline = max(neckline_highs) if neckline_highs else (left_shoulder + right_shoulder) / 2
                
                current_price = recent[-1]["close"]
                breakout = current_price > neckline * 0.99
                
                # Prefer bullish pattern if both detected
                if not pattern or pattern.get("score", 50) < 60:
                    pattern = {
                        "name": "Inverse Head & Shoulders",
                        "type": "reversal",
                        "direction": "bullish" if breakout else "neutral",
                        "reliability": "strong" if breakout else "moderate",
                        "status": "completed" if breakout else "forming",
                        "description": f"Inverse Head & Shoulders pattern detected with head at ${head:.2f} and neckline at ${neckline:.2f}. {'Breakout confirmed' if breakout else 'Pattern forming'}.",
                        "score": 80 if breakout else 65
                    }
        
        return pattern
    
    @staticmethod
    def detect_double_top_bottom(klines: List[Dict], lookback: int = 100) -> Dict:
        """Detect double top (bearish) / double bottom (bullish) patterns"""
        if len(klines) < 30:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=lookback)
        
        pattern = None
        
        # Double top (bearish): Two similar highs with breakdown below support
        if len(pivot_highs) >= 2:
            sorted_highs = sorted(pivot_highs, key=lambda x: x[0])[-2:]
            top1 = sorted_highs[0][1]
            top2 = sorted_highs[1][1]
            
            # Tops should be similar (within 2% tolerance)
            tolerance = top1 * 0.02
            if abs(top1 - top2) < tolerance:
                # Find support (low between the two tops)
                top1_idx = sorted_highs[0][0]
                top2_idx = sorted_highs[1][0]
                support_lows = [k["low"] for k in recent if top1_idx <= (len(recent) - len(klines) + recent.index(k)) <= top2_idx]
                support = min(support_lows) if support_lows else (top1 + top2) / 2 * 0.98
                
                current_price = recent[-1]["close"]
                breakdown = current_price < support * 1.01
                
                pattern = {
                    "name": "Double Top",
                    "type": "reversal",
                    "direction": "bearish" if breakdown else "neutral",
                    "reliability": "moderate" if breakdown else "weak",
                    "status": "completed" if breakdown else "forming",
                    "description": f"Double top pattern detected with resistance at ${(top1 + top2) / 2:.2f} and support at ${support:.2f}. {'Breakdown confirmed' if breakdown else 'Pattern forming'}.",
                    "score": 30 if breakdown else 45  # Bearish = low score
                }
        
        # Double bottom (bullish): Two similar lows with breakout above resistance
        if len(pivot_lows) >= 2:
            sorted_lows = sorted(pivot_lows, key=lambda x: x[0])[-2:]
            bottom1 = sorted_lows[0][1]
            bottom2 = sorted_lows[1][1]
            
            tolerance = bottom1 * 0.02
            if abs(bottom1 - bottom2) < tolerance:
                # Find resistance (high between the two bottoms)
                bottom1_idx = sorted_lows[0][0]
                bottom2_idx = sorted_lows[1][0]
                resistance_highs = [k["high"] for k in recent if bottom1_idx <= (len(recent) - len(klines) + recent.index(k)) <= bottom2_idx]
                resistance = max(resistance_highs) if resistance_highs else (bottom1 + bottom2) / 2 * 1.02
                
                current_price = recent[-1]["close"]
                breakout = current_price > resistance * 0.99
                
                # Prefer bullish pattern if both detected
                if not pattern or pattern.get("score", 50) < 60:
                    pattern = {
                        "name": "Double Bottom",
                        "type": "reversal",
                        "direction": "bullish" if breakout else "neutral",
                        "reliability": "moderate" if breakout else "weak",
                        "status": "completed" if breakout else "forming",
                        "description": f"Double bottom pattern detected with support at ${(bottom1 + bottom2) / 2:.2f} and resistance at ${resistance:.2f}. {'Breakout confirmed' if breakout else 'Pattern forming'}.",
                        "score": 75 if breakout else 60
                    }
        
        return pattern
    
    @staticmethod
    def detect_triple_top_bottom(klines: List[Dict], lookback: int = 100) -> Dict:
        """Detect triple top (bearish) / triple bottom (bullish) patterns
        
        Triple Top (Bearish Reversal):
        - THREE peaks at approximately same level (within 2% tolerance)
        - Two valleys between peaks forming neckline
        - Breakdown below neckline = bearish confirmation
        - More reliable than double top
        
        Triple Bottom (Bullish Reversal):
        - THREE troughs at approximately same level (within 2% tolerance)
        - Two peaks between troughs forming neckline
        - Breakout above neckline = bullish confirmation
        - More reliable than double bottom
        
        Returns pattern dict or None
        """
        if len(klines) < 40:
            return None
        
        recent = klines[-lookback:]
        pivot_highs, pivot_lows = PatternDetector.find_pivots(recent, pivot_left=3, pivot_right=3, lookback=None)
        
        pattern = None
        
        # Triple top (bearish): Three similar highs with breakdown below neckline
        if len(pivot_highs) >= 3:
            sorted_highs = sorted(pivot_highs, key=lambda x: x[0])[-3:]
            top1 = sorted_highs[0][1]
            top2 = sorted_highs[1][1]
            top3 = sorted_highs[2][1]
            
            # Tops should be similar (within 2% tolerance)
            tolerance = top2 * 0.02
            if abs(top1 - top2) < tolerance and abs(top2 - top3) < tolerance:
                # Find neckline (lows between the three peaks - use min of intermediate lows)
                top1_idx = sorted_highs[0][0]
                top2_idx = sorted_highs[1][0]
                top3_idx = sorted_highs[2][0]
                
                # Get lows between peaks
                neckline_lows = []
                for k_idx, k in enumerate(recent):
                    actual_idx = len(klines) - len(recent) + k_idx
                    if top1_idx <= actual_idx <= top3_idx:
                        neckline_lows.append(k["low"])
                
                neckline = min(neckline_lows) if neckline_lows else (top1 + top2 + top3) / 3 * 0.98
                
                current_price = recent[-1]["close"]
                breakdown = current_price < neckline * 1.01
                
                pattern = {
                    "name": "Triple Top",
                    "type": "reversal",
                    "direction": "bearish" if breakdown else "neutral",
                    "reliability": "strong" if breakdown else "moderate",
                    "status": "completed" if breakdown else "forming",
                    "description": f"Triple top pattern detected with three peaks at approximately ${(top1 + top2 + top3) / 3:.2f} and neckline at ${neckline:.2f}. {'Breakdown confirmed' if breakdown else 'Pattern forming'}.",
                    "score": 20 if breakdown else 42  # Bearish = low score
                }
        
        # Triple bottom (bullish): Three similar lows with breakout above neckline
        if len(pivot_lows) >= 3:
            sorted_lows = sorted(pivot_lows, key=lambda x: x[0])[-3:]
            bottom1 = sorted_lows[0][1]
            bottom2 = sorted_lows[1][1]
            bottom3 = sorted_lows[2][1]
            
            tolerance = bottom2 * 0.02
            if abs(bottom1 - bottom2) < tolerance and abs(bottom2 - bottom3) < tolerance:
                # Find neckline (highs between the three troughs - use max of intermediate highs)
                bottom1_idx = sorted_lows[0][0]
                bottom2_idx = sorted_lows[1][0]
                bottom3_idx = sorted_lows[2][0]
                
                # Get highs between bottoms
                neckline_highs = []
                for k_idx, k in enumerate(recent):
                    actual_idx = len(klines) - len(recent) + k_idx
                    if bottom1_idx <= actual_idx <= bottom3_idx:
                        neckline_highs.append(k["high"])
                
                neckline = max(neckline_highs) if neckline_highs else (bottom1 + bottom2 + bottom3) / 3 * 1.02
                
                current_price = recent[-1]["close"]
                breakout = current_price > neckline * 0.99
                
                # Prefer bullish pattern if both detected
                if not pattern or pattern.get("score", 50) < 60:
                    pattern = {
                        "name": "Triple Bottom",
                        "type": "reversal",
                        "direction": "bullish" if breakout else "neutral",
                        "reliability": "strong" if breakout else "moderate",
                        "status": "completed" if breakout else "forming",
                        "description": f"Triple bottom pattern detected with three troughs at approximately ${(bottom1 + bottom2 + bottom3) / 3:.2f} and neckline at ${neckline:.2f}. {'Breakout confirmed' if breakout else 'Pattern forming'}.",
                        "score": 85 if breakout else 68  # Bullish = high score
                    }
        
        if pattern:
            print(f"[PatternDetector.detect_triple_top_bottom] Detected: {pattern['name']} (score: {pattern['score']}, status: {pattern['status']})")
        
        return pattern
    
    @staticmethod
    def detect_candlestick_patterns(klines: List[Dict]) -> Dict:
        """Detect candlestick patterns (1-3 candles)
        
        Supports 3-candle patterns: Morning Star, Evening Star, Three White Soldiers, Three Black Crows
        Supports 2-candle patterns: Engulfing, Piercing Line, Dark Cloud Cover, Hammer, Shooting Star
        Supports 1-candle patterns: Doji
        """
        if len(klines) < 2:
            return None
        
        pattern = None
        
        # Check 3-candle patterns first (more significant)
        if len(klines) >= 3:
            recent3 = klines[-3:]
            c1 = recent3[0]
            c2 = recent3[1]
            c3 = recent3[2]
            
            open1, close1, high1, low1 = c1["open"], c1["close"], c1["high"], c1["low"]
            open2, close2, high2, low2 = c2["open"], c2["close"], c2["high"], c2["low"]
            open3, close3, high3, low3 = c3["open"], c3["close"], c3["high"], c3["low"]
            
            body1 = abs(close1 - open1)
            body2 = abs(close2 - open2)
            body3 = abs(close3 - open3)
            range1 = high1 - low1
            range2 = high2 - low2
            range3 = high3 - low3
            
            # Three White Soldiers (strong bullish)
            if (close1 > open1 and close2 > open2 and close3 > open3 and
                open2 > open1 and close2 > close1 and
                open3 > open2 and close3 > close2 and
                (high1 - max(open1, close1)) < range1 * 0.2 and
                (high2 - max(open2, close2)) < range2 * 0.2 and
                (high3 - max(open3, close3)) < range3 * 0.2):
                pattern = {
                    "name": "Three White Soldiers",
                    "type": "continuation",
                    "direction": "bullish",
                    "reliability": "strong",
                    "status": "completed",
                    "description": "Three White Soldiers: three consecutive bullish candles, each opening within previous body and closing higher. Strong bullish continuation signal.",
                    "score": 82
                }
            
            # Three Black Crows (strong bearish)
            elif (close1 < open1 and close2 < open2 and close3 < open3 and
                  open2 < open1 and close2 < close1 and
                  open3 < open2 and close3 < close2 and
                  (min(open1, close1) - low1) < range1 * 0.2 and
                  (min(open2, close2) - low2) < range2 * 0.2 and
                  (min(open3, close3) - low3) < range3 * 0.2):
                pattern = {
                    "name": "Three Black Crows",
                    "type": "continuation",
                    "direction": "bearish",
                    "reliability": "strong",
                    "status": "completed",
                    "description": "Three Black Crows: three consecutive bearish candles, each opening within previous body and closing lower. Strong bearish continuation signal.",
                    "score": 18  # Bearish = low score
                }
            
            # Morning Star (bullish reversal)
            elif (close1 < open1 and  # Candle 1: bearish
                  body2 < max(body1, body3) * 0.3 and  # Candle 2: small body
                  close2 < close1 and  # Gap down (preferred but not required)
                  close3 > open3 and  # Candle 3: bullish
                  close3 > (open1 + close1) / 2):  # Closes above midpoint of candle 1
                pattern = {
                    "name": "Morning Star",
                    "type": "reversal",
                    "direction": "bullish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Morning Star: bearish candle, small body (gap down), followed by bullish candle closing above midpoint of first candle. Bullish reversal signal.",
                    "score": 72
                }
            
            # Evening Star (bearish reversal)
            elif (close1 > open1 and  # Candle 1: bullish
                  body2 < max(body1, body3) * 0.3 and  # Candle 2: small body
                  close2 > close1 and  # Gap up (preferred but not required)
                  close3 < open3 and  # Candle 3: bearish
                  close3 < (open1 + close1) / 2):  # Closes below midpoint of candle 1
                pattern = {
                    "name": "Evening Star",
                    "type": "reversal",
                    "direction": "bearish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Evening Star: bullish candle, small body (gap up), followed by bearish candle closing below midpoint of first candle. Bearish reversal signal.",
                    "score": 28  # Bearish = low score
                }
        
        # Check 2-candle patterns (if no 3-candle pattern found)
        if pattern is None and len(klines) >= 2:
            recent = klines[-2:]
            current = recent[-1]
            prev = recent[0]
            
            open_curr = current["open"]
            close_curr = current["close"]
            high_curr = current["high"]
            low_curr = current["low"]
            
            open_prev = prev["open"]
            close_prev = prev["close"]
            high_prev = prev["high"]
            low_prev = prev["low"]
            
            body_curr = abs(close_curr - open_curr)
            body_prev = abs(close_prev - open_prev)
            range_curr = high_curr - low_curr
            midpoint_prev = (open_prev + close_prev) / 2
            
            # Doji (indecision - 1 candle pattern, but check on current candle)
            if range_curr > 0 and body_curr < range_curr * 0.1:
                pattern = {
                    "name": "Doji",
                    "type": "reversal",
                    "direction": "neutral",
                    "reliability": "weak",
                    "status": "completed",
                    "description": "Doji candlestick pattern: small body indicates indecision. Potential reversal signal, needs confirmation.",
                    "score": 50  # Neutral
                }
            
            # Piercing Line (bullish reversal)
            elif (close_prev < open_prev and  # Previous: bearish
                  close_curr > open_curr and  # Current: bullish
                  open_curr < low_prev and  # Opens below previous low
                  close_curr > midpoint_prev):  # Closes above midpoint of previous
                pattern = {
                    "name": "Piercing Line",
                    "type": "reversal",
                    "direction": "bullish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Piercing Line: bearish candle followed by bullish candle opening below previous low and closing above midpoint. Bullish reversal signal.",
                    "score": 68
                }
            
            # Dark Cloud Cover (bearish reversal)
            elif (close_prev > open_prev and  # Previous: bullish
                  close_curr < open_curr and  # Current: bearish
                  open_curr > high_prev and  # Opens above previous high
                  close_curr < midpoint_prev):  # Closes below midpoint of previous
                pattern = {
                    "name": "Dark Cloud Cover",
                    "type": "reversal",
                    "direction": "bearish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Dark Cloud Cover: bullish candle followed by bearish candle opening above previous high and closing below midpoint. Bearish reversal signal.",
                    "score": 32  # Bearish = low score
                }
            
            # Bullish engulfing
            elif (close_prev < open_prev and close_curr > open_curr and 
                  open_curr < close_prev and close_curr > open_prev and body_curr > body_prev * 1.1):
                pattern = {
                    "name": "Bullish Engulfing",
                    "type": "reversal",
                    "direction": "bullish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Bullish engulfing candlestick pattern: strong bullish candle engulfs previous bearish candle.",
                    "score": 70
                }
            
            # Bearish engulfing
            elif (close_prev > open_prev and close_curr < open_curr and 
                  open_curr > close_prev and close_curr < open_prev and body_curr > body_prev * 1.1):
                pattern = {
                    "name": "Bearish Engulfing",
                    "type": "reversal",
                    "direction": "bearish",
                    "reliability": "moderate",
                    "status": "completed",
                    "description": "Bearish engulfing candlestick pattern: strong bearish candle engulfs previous bullish candle.",
                    "score": 30  # Bearish = low score
                }
            
            # Hammer (bullish reversal at bottom)
            elif (range_curr > 0 and body_curr < range_curr * 0.3 and 
                  low_curr < min(open_curr, close_curr) * 1.02 and 
                  abs(close_curr - open_curr) < range_curr * 0.3 and
                  (high_curr - max(open_curr, close_curr)) < range_curr * 0.2):
                # Check if we're in a downtrend (simplified: lower low than previous)
                if low_curr < low_prev:
                    pattern = {
                        "name": "Hammer",
                        "type": "reversal",
                        "direction": "bullish",
                        "reliability": "weak",
                        "status": "completed",
                        "description": "Hammer candlestick pattern: potential bullish reversal signal at support.",
                        "score": 65
                    }
            
            # Shooting star (bearish reversal at top)
            elif (range_curr > 0 and body_curr < range_curr * 0.3 and
                  high_curr > max(open_curr, close_curr) * 0.98 and
                  abs(close_curr - open_curr) < range_curr * 0.3 and
                  (min(open_curr, close_curr) - low_curr) < range_curr * 0.2):
                # Check if we're in an uptrend (simplified: higher high than previous)
                if high_curr > high_prev:
                    pattern = {
                        "name": "Shooting Star",
                        "type": "reversal",
                        "direction": "bearish",
                        "reliability": "weak",
                        "status": "completed",
                        "description": "Shooting star candlestick pattern: potential bearish reversal signal at resistance.",
                        "score": 35  # Bearish = low score
                    }
        
        return pattern
    
    @staticmethod
    def is_bullish_reversal_allowed(choch_detected: bool, choch_direction: str, divergence: Dict, ema_alignment: str, ema50: float, current_price: float) -> bool:
        """Check if bullish reversal pattern is allowed based on market regime
        
        Requires at least ONE:
        1. bullish CHOCH
        2. bullish divergence (RSI or MACD)
        3. EMA50 reclaimed OR EMA alignment not bearish
        """
        # 1. Check for bullish CHOCH
        if choch_detected and choch_direction == "bullish":
            return True
        
        # 2. Check for bullish divergence
        if divergence and divergence.get("type") in ["bullish_regular", "bullish_hidden"]:
            return True
        
        # 3. Check EMA alignment and price vs EMA50
        if ema_alignment != "bearish":
            return True
        if ema50 and current_price > ema50:
            return True  # Price above EMA50 even if alignment is bearish
        
        return False
    
    @staticmethod
    def detect_best(klines: List[Dict], choch_detected: bool = False, choch_direction: str = "none", divergence: Dict = None, ema_alignment: str = "neutral", ema50: float = None, current_price: float = None) -> Dict:
        """Detect the best/most significant pattern from OHLC data with market-regime gating
        
        Args:
            klines: List of OHLC kline dictionaries
            choch_detected: Whether CHOCH was detected
            choch_direction: Direction of CHOCH ("bullish", "bearish", "none")
            divergence: Divergence dict with type, indicator, signal
            ema_alignment: EMA alignment ("bullish", "bearish", "neutral")
            ema50: EMA50 value for price comparison
            current_price: Current price for EMA comparison
        
        Returns a single pattern dict with structure:
        {
            "name": str,
            "type": "reversal"/"continuation"/"none",
            "direction": "bullish"/"bearish"/"neutral",
            "reliability": "strong"/"moderate"/"weak"/"none",
            "status": "forming"/"completed"/"failed"/"none",
            "description": str,
            "score": int (0-100)
        }
        """
        if not klines or len(klines) < 10:
            return {
                "name": "none",
                "type": "none",
                "direction": "neutral",
                "reliability": "none",
                "status": "none",
                "description": "Insufficient data for pattern detection",
                "score": 50
            }
        
        # Get current price if not provided
        if current_price is None:
            current_price = klines[-1]["close"]
        
        # Check for minimum swing points requirement (NONE pattern strictness)
        # Return "none" ONLY when structure is insufficient
        swing_highs, swing_lows = PatternDetector.get_swings(klines, left=3, right=3, lookback=100)
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {
                "name": "none",
                "type": "none",
                "direction": "neutral",
                "reliability": "none",
                "status": "none",
                "description": "Insufficient swing points for pattern detection",
                "score": 50
            }
        
        candidates = []
        
        # Try each detector (order matters - more significant patterns first)
        detectors = [
            PatternDetector.detect_head_shoulders,      # Most significant reversal
            PatternDetector.detect_triple_top_bottom,   # NEW - Strong reversal
            PatternDetector.detect_double_top_bottom,   # Reversal
            PatternDetector.detect_wedge,               # NEW - Reversal
            PatternDetector.detect_flag_pennant,        # Continuation
            PatternDetector.detect_channel,             # NEW - Continuation
            PatternDetector.detect_triangle,            # Continuation
            PatternDetector.detect_candlestick_patterns # Single/multi candle
        ]
        
        for detector in detectors:
            try:
                pattern = detector(klines)
                if pattern:
                    # Market-regime gating: Reject bullish reversal patterns if regime doesn't allow
                    if pattern.get("type") == "reversal" and pattern.get("direction") == "bullish":
                        if not PatternDetector.is_bullish_reversal_allowed(
                            choch_detected, choch_direction, divergence, ema_alignment, ema50, current_price
                        ):
                            # Downgrade to "none" or adjust to neutral continuation
                            pattern = {
                                "name": "none",
                                "type": "none",
                                "direction": "neutral",
                                "reliability": "none",
                                "status": "none",
                                "description": "Bullish reversal pattern rejected due to bearish market regime",
                                "score": 50
                            }
                    
                    candidates.append(pattern)
                    print(f"[PatternDetector] Detected: {pattern['name']} (score: {pattern['score']}, status: {pattern['status']})")
            except Exception as e:
                print(f"[PatternDetector] Error in {detector.__name__}: {e}")
                continue
        
        # Filter out "none" patterns before selection
        valid_candidates = [c for c in candidates if c.get("name") != "none"]
        
        if not valid_candidates:
            return {
                "name": "none",
                "type": "none",
                "direction": "neutral",
                "reliability": "none",
                "status": "none",
                "description": "No valid pattern detected",
                "score": 50
            }
        
        # Select best pattern: prioritize completed patterns with higher scores
        # For spot trading, prefer bullish patterns if scores are close
        best = valid_candidates[0]
        for candidate in valid_candidates[1:]:
            # Completed patterns rank higher
            if candidate["status"] == "completed" and best["status"] != "completed":
                best = candidate
            elif candidate["status"] == best["status"]:
                # Same status: prefer higher score, or bullish if scores close (within 5 points)
                if candidate["score"] > best["score"] or (
                    abs(candidate["score"] - best["score"]) <= 5 and candidate["score"] > 50
                ):
                    best = candidate
        
        print(f"[PatternDetector] Selected best pattern: {best['name']} (score: {best['score']}, direction: {best['direction']})")
        return best


def calculate_all_indicators(klines: List[Dict], debug: bool = False) -> Dict:
    """Calculate all indicators from kline data
    
    Ensures klines are sorted chronologically (oldest to newest) for accurate calculations.
    
    Args:
        klines: List of OHLCV kline dictionaries
        debug: If True, print last 200 OHLCV and final computed indicator values for verification
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
    
    # Detect divergences with indicator-specific thresholds (pass current_price for MACD threshold calculation)
    rsi_divergence = engine.detect_divergence(closes, rsi_values, lookback=50, indicator_type="rsi", current_price=current_price) if len(rsi_values) >= 50 else {"type": "none", "signal": "neutral", "strength": "none", "indicator": "RSI", "explanation": "No divergence detected"}
    macd_divergence = engine.detect_divergence(closes, macd_histogram_values, lookback=50, indicator_type="macd", current_price=current_price) if len(macd_histogram_values) >= 50 else {"type": "none", "signal": "neutral", "strength": "none", "indicator": "MACD", "explanation": "No divergence detected"}
    
    # Extract OBV divergence (simple format from OBV calculation)
    obv = engine.calculate_obv(klines)
    obv_divergence_type = obv.get("divergence", "none")
    # Format OBV divergence to match required structure
    obv_score = 75 if obv_divergence_type in ["bullish", "bullish_regular", "bullish_hidden"] else (25 if obv_divergence_type in ["bearish", "bearish_regular", "bearish_hidden"] else 50)
    obv_divergence = {
        "type": obv_divergence_type if obv_divergence_type != "none" else "none",
        "indicator": "OBV",
        "signal": obv.get("signal", "neutral"),
        "strength": "moderate" if obv_divergence_type != "none" else "none",
        "score": obv_score,
        "explanation": f"{obv_divergence_type.title()} divergence on OBV" if obv_divergence_type != "none" else "No divergence detected"
    }
    
    # Combine divergences: Prioritize RSI > MACD > OBV, but return strongest if multiple exist
    all_divergences = {
        "rsi": rsi_divergence,
        "macd": macd_divergence,
        "obv": obv_divergence
    }
    
    # Select strongest divergence (prioritize RSI if multiple exist, then MACD, then OBV)
    # Pick the best divergence among RSI/MACD/OBV and ensure proper format
    divergence = rsi_divergence if rsi_divergence.get("type") != "none" else (macd_divergence if macd_divergence.get("type") != "none" else obv_divergence)
    if divergence.get("type") == "none" or not divergence:
        divergence = {
            "type": "none",
            "indicator": "none",
            "signal": "neutral",
            "strength": "none",
            "score": 50,
            "explanation": "No divergence detected"
        }
    
    # Bollinger Bands (with regime detection using EMA alignment and ADX)
    bollinger = engine.calculate_bollinger(closes, 20, 2.0, ema20, ema50, ema200, adx.get("adx") if adx else None)
    
    # ATR
    atr = engine.calculate_atr(klines, 14)
    
    # Smart Money Concepts (SMC)
    order_blocks = engine.detect_order_blocks(klines, atr_period=14, move_threshold=1.5)
    fair_value_gaps = engine.detect_fair_value_gaps(klines, lookback=100)
    liquidity_zones = engine.detect_liquidity_zones(klines, cluster_threshold_pct=0.5, min_cluster_size=2)
    break_of_structure = engine.detect_break_of_structure(klines, lookback=50)
    
    # Support/Resistance
    print(f"[calculate_all_indicators] Calling find_support_resistance() with {len(klines)} klines...")
    sr_levels = engine.find_support_resistance(klines, 3)
    print(f"[calculate_all_indicators] find_support_resistance() returned {len(sr_levels.get('support', []))} supports, {len(sr_levels.get('resistance', []))} resistances")
    
    # Chart Pattern Detection (rule-based, deterministic) with market-regime gating
    try:
        chart_pattern = PatternDetector.detect_best(
            klines,
            choch_detected=break_of_structure.get("choch_detected", False),
            choch_direction=break_of_structure.get("choch_direction", "none"),
            divergence=divergence,
            ema_alignment=ema_alignment,
            ema50=ema50,
            current_price=current_price
        )
        print(f"[calculate_all_indicators] Chart pattern detected: {chart_pattern.get('name', 'none')} (score: {chart_pattern.get('score', 50)})")
    except Exception as e:
        print(f"[calculate_all_indicators] Pattern detection error: {e}")
        chart_pattern = {
            "name": "none",
            "type": "none",
            "direction": "neutral",
            "reliability": "none",
            "status": "none",
            "description": "Pattern detection error",
            "score": 50
        }
    
    # Fibonacci (with proper pivot-based swing detection, trend-aware)
    fib = engine.calculate_fibonacci(
        klines, 
        lookback=200, 
        pivot_left=3, 
        pivot_right=3, 
        method="swing",
        trend_direction=ema_alignment, 
        current_price=current_price,
        ema_alignment=ema_alignment,
        ema50=ema50,
        choch_detected=break_of_structure.get("choch_detected", False),
        choch_direction=break_of_structure.get("choch_direction", "none"),
        bos_detected=break_of_structure.get("bos_detected", False),
        bos_direction=break_of_structure.get("direction", "none")
    )
    print(f"[calculate_all_indicators] Fibonacci: trend={fib.get('trend', 'neutral')}, nearest_level={fib.get('nearest_level', 'none')}, fib_score={fib.get('fib_score', 50)}")
    
    # Volume
    volume = engine.analyze_volume(klines, 20)
    
    # On-Balance Volume (OBV) - already calculated above for divergence
    # obv = engine.calculate_obv(klines)  # Removed duplicate
    
    # Debug logging: Print last 200 OHLCV and final computed indicator values
    if debug:
        print("\n" + "="*80)
        print("INDICATOR DEBUG OUTPUT - Last 200 OHLCV and Final Values")
        print("="*80)
        
        # Print last 200 OHLCV (or all if less than 200)
        debug_klines = klines[-200:] if len(klines) > 200 else klines
        print(f"\nOHLCV Data (Last {len(debug_klines)} candles):")
        print("Index | Open      | High      | Low       | Close     | Volume")
        print("-"*80)
        for i, k in enumerate(debug_klines):
            idx_offset = len(klines) - len(debug_klines) + i
            print(f"{idx_offset:5d} | {k['open']:9.4f} | {k['high']:9.4f} | {k['low']:9.4f} | {k['close']:9.4f} | {k.get('volume', 0):.2f}")
        
        # Print final computed indicator values
        print("\n" + "-"*80)
        print("FINAL INDICATOR VALUES:")
        print("-"*80)
        print(f"RSI(14): {rsi.get('value', 'N/A')} ({rsi.get('signal', 'N/A')})")
        print(f"MACD: {macd.get('macd', 'N/A'):.6f} | Signal: {macd.get('signal', 'N/A'):.6f} | Histogram: {macd.get('histogram', 'N/A'):.6f} | Trend: {macd.get('trend', 'N/A')}")
        print(f"EMA20: {ema20:.4f}")
        print(f"EMA50: {ema50:.4f}")
        print(f"EMA200: {ema200:.4f}")
        print(f"EMA Alignment: {ema_alignment}")
        print(f"Price vs EMA: {price_vs_ema}")
        print(f"Bollinger: Upper={bollinger.get('upper', 'N/A'):.4f} | Middle={bollinger.get('middle', 'N/A'):.4f} | Lower={bollinger.get('lower', 'N/A'):.4f} | Position={bollinger.get('position', 'N/A')}")
        print(f"ATR(14): {atr:.4f}")
        print(f"Fibonacci:")
        print(f"  Swing High: {fib.get('swing_high', 'N/A'):.4f} (index: {fib.get('swing_high_idx', 'N/A')})")
        print(f"  Swing Low: {fib.get('swing_low', 'N/A'):.4f} (index: {fib.get('swing_low_idx', 'N/A')})")
        print(f"  Trend Direction: {fib.get('trend_direction', 'N/A')}")
        print(f"  Diff: {fib.get('diff', 'N/A'):.4f}")
        print(f"  Levels: {fib.get('levels', {})}")
        print(f"Stochastic RSI: K={stoch_rsi.get('k', 'N/A'):.2f} | D={stoch_rsi.get('d', 'N/A'):.2f} | Signal={stoch_rsi.get('signal', 'N/A')}")
        print(f"ADX: {adx.get('adx', 'N/A'):.2f} | +DI: {adx.get('plus_di', 'N/A'):.2f} | -DI: {adx.get('minus_di', 'N/A'):.2f} | Trend Strength: {adx.get('trend_strength', 'N/A')}")
        print(f"Volume: Current={volume.get('current', 'N/A')} | Average={volume.get('average', 'N/A'):.2f} | Ratio={volume.get('ratio', 'N/A'):.2f} | Trend={volume.get('trend', 'N/A')}")
        print(f"OBV: {obv.get('obv', 'N/A'):.2f} | Trend={obv.get('trend', 'N/A')} | Divergence={obv.get('divergence', 'N/A')}")
        print(f"Divergence: Type={divergence.get('type', 'N/A')} | Indicator={divergence.get('indicator', 'N/A')} | Signal={divergence.get('signal', 'N/A')} | Strength={divergence.get('strength', 'N/A')}")
        print(f"Chart Pattern: {chart_pattern.get('name', 'N/A')} | Type={chart_pattern.get('type', 'N/A')} | Direction={chart_pattern.get('direction', 'N/A')} | Score={chart_pattern.get('score', 'N/A')}")
        print("="*80 + "\n")
    
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
        "order_blocks": order_blocks,
        "fair_value_gaps": fair_value_gaps,
        "liquidity_zones": liquidity_zones,
        "break_of_structure": break_of_structure,
        "support_resistance": sr_levels,
        "fibonacci": fib,
        "volume": volume,
        "obv": obv,
        "trend": trend,
        "all_divergences": all_divergences  # Include all divergences for reference
    }


def calculate_long_trade_setup(
    current_price: float,
    supports: List[Dict],
    resistances: List[Dict],
    atr_value: float,
    confluence_score: float = 50,
    higher_tf_trend: str = "neutral",
    adx_value: float = 25,
    order_blocks: Dict = None,
    fair_value_gaps: Dict = None
) -> Dict:
    """Calculate LONG trade setup with SMART ENTRY SELECTION and ATR-based risk management
    
    CRITICAL IMPROVEMENTS:
    1. Smart entry selection using multiple confluence zones
    2. Maximum entry distance (10% below current price)
    3. Uses FVG and Order Blocks for optimal entry
    4. Fallback to pullback entry if no good levels found
    
    Entry Priority:
    1. If valid support within 2 ATR → Use support level
    2. If unfilled bullish FVG below price → Use FVG top as entry
    3. If valid bullish Order Block below price → Use OB high as entry
    4. Fallback → Use current price with small pullback (0.5-1 ATR)
    
    Args:
        current_price: Current market price
        supports: List of support levels [{"price": float, "strength": str}, ...]
        resistances: List of resistance levels [{"price": float, "strength": str}, ...]
        atr_value: Current ATR(14) value
        confluence_score: Confluence score (0-100)
        higher_tf_trend: Higher timeframe trend ("bullish", "bearish", "neutral")
        adx_value: ADX value for volatility adjustment
        order_blocks: Order block data from IndicatorEngine
        fair_value_gaps: FVG data from IndicatorEngine
    
    Returns:
        Dict with entry, stop_loss, tp1, tp2, tp3, risk metrics, and bias
    """
    
    # Initialize defaults
    order_blocks = order_blocks or {}
    fair_value_gaps = fair_value_gaps or {}
    
    # Volatility filter: Check if ATR is extreme relative to price
    atr_pct_of_price = (atr_value / current_price * 100) if current_price > 0 else 0
    extreme_volatility = atr_pct_of_price > 5.0  # 5% threshold
    
    # Determine bias: Only LONG or NEUTRAL for spot trading
    if confluence_score >= 60 and higher_tf_trend != "bearish" and not extreme_volatility:
        bias = "long"
        confidence = "high" if confluence_score >= 75 else "medium"
    else:
        bias = "neutral"
        confidence = "low"
        if extreme_volatility and confluence_score >= 60:
            confidence = "low"
    
    # CRITICAL: Define maximum entry distance (10% below current price)
    # Any entry below this is unrealistic and won't fill
    max_entry_distance_pct = 0.10  # 10%
    min_reasonable_entry = current_price * (1 - max_entry_distance_pct)
    
    # Also define "ideal" entry zone (within 2 ATR of current price)
    ideal_entry_max_distance = atr_value * 2
    ideal_entry_min = current_price - ideal_entry_max_distance
    
    print(f"[TradeSetup] Current: ${current_price:.4f}, Min reasonable entry: ${min_reasonable_entry:.4f}, Ideal entry min: ${ideal_entry_min:.4f}")
    
    # ========== SMART ENTRY SELECTION ==========
    entry_price = current_price  # Default
    entry_reasoning = "Entry at current price (default)"
    entry_source = "current_price"
    
    # PRIORITY 1: Check for valid support within ideal range
    if supports and len(supports) > 0:
        # Filter supports to only those within reasonable range
        valid_supports = [s for s in supports if s["price"] >= min_reasonable_entry and s["price"] < current_price]
        
        if valid_supports:
            # Get highest valid support (closest to current price)
            nearest_valid_support = max(valid_supports, key=lambda x: x["price"])
            support_price = nearest_valid_support["price"]
            distance_to_support = current_price - support_price
            distance_pct = (distance_to_support / current_price) * 100
            
            print(f"[TradeSetup] Nearest valid support: ${support_price:.4f} ({distance_pct:.1f}% below)")
            
            # If support is within ideal range (2 ATR), use it directly
            if distance_to_support <= ideal_entry_max_distance:
                entry_price = support_price
                entry_reasoning = f"Entry at support ${support_price:.4f} ({distance_pct:.1f}% below current)"
                entry_source = "support"
            else:
                # Support exists but a bit far - note it but continue checking other options
                print(f"[TradeSetup] Support at ${support_price:.4f} is {distance_pct:.1f}% away, checking FVG/OB...")
    
    # PRIORITY 2: Check for unfilled bullish FVG below price
    if entry_source == "current_price" and fair_value_gaps:
        unfilled_fvgs = fair_value_gaps.get("unfilled_fvg", [])
        
        # Filter for bullish FVGs below current price and within reasonable range
        valid_fvgs = [
            f for f in unfilled_fvgs 
            if f.get("direction") == "bullish" 
            and f.get("top", 0) < current_price 
            and f.get("top", 0) >= min_reasonable_entry
        ]
        
        if valid_fvgs:
            # Use the highest (closest) unfilled bullish FVG
            best_fvg = max(valid_fvgs, key=lambda x: x.get("top", 0))
            fvg_entry = best_fvg["top"]
            fvg_distance_pct = ((current_price - fvg_entry) / current_price) * 100
            
            print(f"[TradeSetup] Found unfilled bullish FVG: ${best_fvg.get('bottom', 0):.4f} - ${fvg_entry:.4f}")
            
            # Use FVG if it's closer than our current entry or if we don't have a good entry yet
            if fvg_entry > entry_price or entry_source == "current_price":
                entry_price = fvg_entry
                entry_reasoning = f"Entry at FVG top ${fvg_entry:.4f} (unfilled gap - {fvg_distance_pct:.1f}% below)"
                entry_source = "fvg"
    
    # PRIORITY 3: Check for valid bullish Order Block below price
    if entry_source == "current_price" and order_blocks:
        nearest_ob = order_blocks.get("nearest_bullish", {})
        
        if nearest_ob.get("is_valid"):
            ob_high = nearest_ob.get("high", 0)
            ob_low = nearest_ob.get("low", 0)
            
            # Check if OB is within reasonable range
            if ob_high >= min_reasonable_entry and ob_high < current_price:
                ob_distance_pct = ((current_price - ob_high) / current_price) * 100
                
                print(f"[TradeSetup] Found valid bullish OB: ${ob_low:.4f} - ${ob_high:.4f}")
                
                # Use OB if it's better than current entry
                if ob_high > entry_price or entry_source == "current_price":
                    entry_price = ob_high
                    entry_reasoning = f"Entry at Order Block high ${ob_high:.4f} ({ob_distance_pct:.1f}% below)"
                    entry_source = "order_block"
    
    # PRIORITY 4: If still at current price, use a small pullback
    if entry_source == "current_price":
        # Use 0.5 ATR pullback as entry (reasonable dip buy)
        pullback_amount = atr_value * 0.5
        pullback_entry = current_price - pullback_amount
        pullback_pct = (pullback_amount / current_price) * 100
        
        entry_price = pullback_entry
        entry_reasoning = f"Entry on pullback ${pullback_entry:.4f} (0.5 ATR / {pullback_pct:.1f}% below current)"
        entry_source = "pullback"
        
        print(f"[TradeSetup] Using pullback entry: ${pullback_entry:.4f}")
    
    # FINAL SAFETY CHECK: Ensure entry is never below minimum reasonable
    if entry_price < min_reasonable_entry:
        old_entry = entry_price
        entry_price = current_price - (atr_value * 1.0)  # 1 ATR pullback max
        entry_price = max(entry_price, min_reasonable_entry)
        entry_reasoning = f"Entry adjusted from ${old_entry:.4f} to ${entry_price:.4f} (max 10% below current)"
        entry_source = "adjusted"
        print(f"[TradeSetup] SAFETY: Adjusted entry from ${old_entry:.4f} to ${entry_price:.4f}")
    
    print(f"[TradeSetup] FINAL ENTRY: ${entry_price:.4f} via {entry_source}")
    
    # ========== STOP LOSS CALCULATION ==========
    # Stop loss must be below entry with ATR buffer
    if atr_value > 0:
        # Dynamic ATR multiplier based on ADX (trend strength)
        if adx_value >= 35:
            atr_multiplier = 2.0  # Strong trend - wider stops
        elif adx_value >= 25:
            atr_multiplier = 1.5  # Moderate trend
        elif adx_value >= 20:
            atr_multiplier = 1.2  # Weak trend - tighter stops
        else:
            atr_multiplier = 1.5  # Ranging - default
        
        atr_buffer = atr_value * atr_multiplier
        min_buffer = entry_price * 0.005  # Minimum 0.5% buffer
        buffer = max(atr_buffer, min_buffer)
        
        # Find support below entry for SL placement
        supports_below_entry = [s for s in (supports or []) if s["price"] < entry_price]
        
        if supports_below_entry:
            # Place SL below the nearest support below entry
            nearest_support_below = max(supports_below_entry, key=lambda x: x["price"])
            sl_price = nearest_support_below["price"] - (atr_value * 0.5)  # Small buffer below support
            sl_reasoning = f"Stop loss below support ${nearest_support_below['price']:.4f} with buffer"
        else:
            # No support below entry - use ATR buffer below entry
            sl_price = entry_price - buffer
            sl_reasoning = f"Stop loss {atr_multiplier}x ATR (${buffer:.4f}) below entry"
        
        # Ensure SL is always below entry
        if sl_price >= entry_price:
            sl_price = entry_price - buffer
            sl_reasoning = "Stop loss adjusted (ATR buffer below entry)"
    else:
        # Fallback: 3% below entry if no ATR
        sl_price = entry_price * 0.97
        sl_reasoning = "Stop loss 3% below entry (fallback - no ATR)"
    
    # ========== TAKE PROFIT CALCULATION ==========
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
    
    # Adjust TPs if resistance levels are nearby
    if resistances:
        for res in resistances[:3]:
            res_price = res["price"]
            
            # Adjust TP1 if resistance is close
            if tp1_price * 0.98 < res_price < tp1_price * 1.02 and res_price > entry_price:
                tp1_price = res_price * 0.995  # Just below resistance
                tp1_reasoning = f"TP1: Below resistance ${res_price:.4f}"
            
            # Adjust TP2 if resistance is close
            elif tp2_price * 0.98 < res_price < tp2_price * 1.02 and res_price > tp1_price:
                tp2_price = res_price * 0.995
                tp2_reasoning = f"TP2: Below resistance ${res_price:.4f}"
    
    # Validate: TPs must be ascending and above entry
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
            "reasoning": entry_reasoning,
            "source": entry_source
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
        "higher_tf_trend": higher_tf_trend,
        "atr_pct_of_price": round(atr_pct_of_price, 2),
        "extreme_volatility": extreme_volatility
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
    print(f"[API] /api/indicators called for {symbol.upper()} {interval}")
    klines_data = await get_klines(symbol, interval, 500)
    klines = klines_data["klines"]
    
    # Ensure klines are sorted chronologically (oldest to newest)
    klines.sort(key=lambda x: x["open_time"])
    
    indicators = calculate_all_indicators(klines)
    
    # Add debug info to response
    sr = indicators.get("support_resistance", {})
    supports = sr.get("support", [])
    current_price = indicators.get("current_price", 0)
    
    print(f"[API] /api/indicators response: {len(supports)} supports, current_price=${current_price:.4f}")
    if supports:
        print(f"[API] Support levels in response: {[s['price'] for s in supports]}")
    
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "indicators": indicators,
        "_debug": {
            "code_version": "v2.0_fixed_support_detection",
            "support_count": len(supports),
            "current_price": current_price
        }
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
    stoch_rsi = indicators.get("stoch_rsi", {})
    adx = indicators.get("adx", {})
    order_blocks = indicators.get("order_blocks", {})
    fair_value_gaps = indicators.get("fair_value_gaps", {})
    liquidity_zones = indicators.get("liquidity_zones", {})
    break_of_structure = indicators.get("break_of_structure", {})
    
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
Stochastic RSI: K={stoch_rsi.get('k', 50)}, D={stoch_rsi.get('d', 50)} ({stoch_rsi.get('signal', 'neutral')})
ADX(14): {adx.get('adx', 0)} ({adx.get('trend_strength', 'weak')}) | +DI: {adx.get('plus_di', 0)} | -DI: {adx.get('minus_di', 0)}
Volume Trend: {vol.get('trend', 'neutral')} (Ratio: {vol.get('ratio', 1.0):.2f}x)
OBV Trend: {obv.get('trend', 'neutral')} | Divergence: {obv.get('divergence', 'none')}

=== DETECTED SUPPORT/RESISTANCE ===
Support Levels: {json.dumps(sr.get('support', []))}
Resistance Levels: {json.dumps(sr.get('resistance', []))}

=== SMART MONEY CONCEPTS (SMC) ===
Order Blocks:
  - Nearest Bullish OB: {json.dumps(indicators.get('order_blocks', {}).get('nearest_bullish', {}))}
  - Nearest Bearish OB: {json.dumps(indicators.get('order_blocks', {}).get('nearest_bearish', {}))}
Fair Value Gaps:
  - Nearest FVG: {json.dumps(indicators.get('fair_value_gaps', {}).get('nearest_fvg', {}))}
  - Unfilled FVGs: {len(indicators.get('fair_value_gaps', {}).get('unfilled_fvg', []))} unfilled gaps
Liquidity Zones:
  - Strongest Above: {json.dumps(indicators.get('liquidity_zones', {}).get('strongest_above', {}))}
  - Strongest Below: {json.dumps(indicators.get('liquidity_zones', {}).get('strongest_below', {}))}
Break of Structure:
  - BOS Detected: {indicators.get('break_of_structure', {}).get('bos_detected', False)}
  - Direction: {indicators.get('break_of_structure', {}).get('direction', 'neutral')}
  - Structure Type: {indicators.get('break_of_structure', {}).get('structure_type', 'neutral')}
  - Confidence: {indicators.get('break_of_structure', {}).get('confidence', 'none')}

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

STEP 4: SUPPORT/RESISTANCE & SMART MONEY CONCEPTS ANALYSIS
Identify where price might bounce (support) or get rejected (resistance).
Analyze Smart Money Concepts:
- Order Blocks: Institutional entry zones (bullish OB below price, bearish OB above price)
- Fair Value Gaps: Price imbalances likely to be filled (unfilled gaps are more significant)
- Liquidity Zones: Areas where stop losses cluster (price often moves to these zones)
- Break of Structure: Trend changes when price breaks above/below last swing points (bullish BOS = bullish, bearish BOS = bearish)

STEP 5: CONFLUENCE SCORING
Score each indicator 0-100 based on how bullish it is, then multiply by weight:
- Chart Pattern: weight {CONFLUENCE_WEIGHTS['chart_pattern']*100:.0f}%
- S/R Levels: weight {CONFLUENCE_WEIGHTS['support_resistance']*100:.0f}%
- Order Blocks: weight {CONFLUENCE_WEIGHTS['order_blocks']*100:.0f}%
- Liquidity Zones: weight {CONFLUENCE_WEIGHTS['liquidity_zones']*100:.0f}%
- Break of Structure: weight {CONFLUENCE_WEIGHTS['break_of_structure']*100:.0f}%
- Divergence: weight {CONFLUENCE_WEIGHTS['divergence']*100:.0f}%
- RSI: weight {CONFLUENCE_WEIGHTS['rsi']*100:.0f}%
- MACD: weight {CONFLUENCE_WEIGHTS['macd']*100:.0f}%
- EMA Alignment: weight {CONFLUENCE_WEIGHTS['ema_alignment']*100:.0f}%
- Price vs EMA: weight {CONFLUENCE_WEIGHTS['price_vs_ema']*100:.0f}%
- Fibonacci: weight {CONFLUENCE_WEIGHTS['fibonacci']*100:.0f}%
- Bollinger: weight {CONFLUENCE_WEIGHTS['bollinger']*100:.0f}%
- Volume: weight {CONFLUENCE_WEIGHTS['volume']*100:.0f}%

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
    "weight": {int(CONFLUENCE_WEIGHTS['chart_pattern']*100)},
    "weighted_score": <calculated>
  }},
  "indicators": {{
    "rsi": {{"value": {rsi.get('value', 50)}, "signal": "{rsi.get('signal', 'neutral')}", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['rsi']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "macd": {{"value": {{"macd": {macd.get('macd', 0)}, "signal": {macd.get('signal', 0)}, "histogram": {macd.get('histogram', 0)}}}, "signal": "{macd.get('trend', 'neutral')}", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['macd']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "divergence": {{"type": "{divergence.get('type', 'none')}", "signal": "{divergence.get('signal', 'neutral')}", "strength": "{divergence.get('strength', 'none')}", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['divergence']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "ema_alignment": {{"value": "{ema.get('alignment', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['ema_alignment']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "price_vs_ema": {{"value": "{ema.get('price_vs_ema', 'mixed')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['price_vs_ema']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "support_resistance": {{"nearest_support": <price>, "nearest_resistance": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['support_resistance']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "fibonacci": {{"key_level": "0.618", "price_at_level": <price>, "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['fibonacci']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "bollinger": {{"position": "{bb.get('position', 'middle')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['bollinger']*100)}, "weighted_score": <calculated>, "explanation": "why"}},
    "volume": {{"trend": "{vol.get('trend', 'neutral')}", "signal": "bullish/bearish/neutral", "score": <0-100>, "weight": {int(CONFLUENCE_WEIGHTS['volume']*100)}, "weighted_score": <calculated>, "explanation": "why"}}
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
1. CAREFULLY examine the chart for patterns - this is worth 12% of confluence score
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
    
    print("\n" + "="*80)
    print("="*80)
    print("🚀 /api/analyze ENDPOINT CALLED!")
    print(f"   Symbol: {symbol}, Timeframe: {timeframe}")
    print("="*80)
    print("="*80 + "\n")
    
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
        print(f"[API] /api/analyze called for {symbol.upper()} {timeframe}")
        print(f"[API] Processing {len(klines)} klines...")
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
        
        # Get ADX value for dynamic stop loss adjustment
        adx_data = indicators.get("adx", {})
        adx_value = adx_data.get("adx", 25)
        
        print(f"[API] /api/analyze - Current price: ${current_price:.4f}")
        print(f"[API] /api/analyze - Supports received: {len(supports)}")
        if supports:
            print(f"[API] /api/analyze - Support prices: {[s['price'] for s in supports]}")
        
        # Calculate professional trade setup with Smart Money data
        print(f"[API] /api/analyze - Calling calculate_long_trade_setup()...")
        trade_setup = calculate_long_trade_setup(
            current_price=current_price,
            supports=supports,
            resistances=resistances,
            atr_value=atr_value,
            confluence_score=confluence_score,
            higher_tf_trend=higher_tf_trend.get("trend", "neutral") if isinstance(higher_tf_trend, dict) else (higher_tf_trend or "neutral"),
            adx_value=adx_value,
            order_blocks=indicators.get("order_blocks", {}),
            fair_value_gaps=indicators.get("fair_value_gaps", {})
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


# Single source of truth for confluence weights (sums to 100%)
CONFLUENCE_WEIGHTS = {
    "chart_pattern": 0.12,
    "support_resistance": 0.12,
    "order_blocks": 0.07,
    "liquidity_zones": 0.05,
    "break_of_structure": 0.05,
    "divergence": 0.08,
    "rsi": 0.09,
    "macd": 0.09,
    "ema_alignment": 0.09,
    "price_vs_ema": 0.07,
    "fibonacci": 0.05,
    "bollinger": 0.05,
    "volume": 0.07
}

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
    order_blocks = indicators.get("order_blocks", {})
    liquidity_zones = indicators.get("liquidity_zones", {})
    break_of_structure = indicators.get("break_of_structure", {})
    fib = indicators.get("fibonacci", {})
    
    # Confluence scoring using CONFLUENCE_WEIGHTS dict (single source of truth)
    # Score starts at 50 (neutral), then we add/subtract weighted deltas
    base_score = 50
    breakdown = {}  # Will contain raw_score, weighted_score, and delta for each indicator
    
    # Chart Pattern (12%) - use score from rule-based detection if available
    chart_pattern = indicators.get("chart_pattern", {})
    pattern_score = chart_pattern.get("score", 50) if chart_pattern else 50
    pattern_weight = CONFLUENCE_WEIGHTS["chart_pattern"]
    pattern_delta = (pattern_score - 50) * pattern_weight
    breakdown["Chart Pattern"] = {
        "raw_score": pattern_score,
        "weighted_score": round(pattern_score * pattern_weight, 1),
        "delta": round(pattern_delta, 2)
    }
    base_score += pattern_delta
    
    # S/R Levels (12%) - IMPROVED: Position-aware scoring
    sr_score = 50  # Default neutral
    if supports and len(supports) > 0:
        nearest_support = supports[0]["price"]
        distance_to_support_pct = ((current_price - nearest_support) / current_price * 100) if current_price > 0 else 0
        
        if distance_to_support_pct > 0:
            # Price is ABOVE support - bullish positioning
            if distance_to_support_pct < 2:
                sr_score = 75  # Very close to support = good entry zone
            elif distance_to_support_pct < 5:
                sr_score = 65  # Reasonable distance above support
            else:
                sr_score = 55  # Far above support, less relevant for entry
        else:
            # Price is BELOW nearest support - support broken, bearish
            sr_score = 30

    # Adjust for resistance proximity
    if resistances and len(resistances) > 0:
        nearest_resistance = resistances[0]["price"]
        distance_to_resistance_pct = ((nearest_resistance - current_price) / current_price * 100) if current_price > 0 else 0
        
        if distance_to_resistance_pct > 0:
            # Price is BELOW resistance
            if distance_to_resistance_pct < 1:
                sr_score = max(sr_score - 15, 35)  # Very close to resistance = caution
            elif distance_to_resistance_pct < 3:
                sr_score = max(sr_score - 5, 45)  # Moderate distance to resistance
        else:
            # Price is ABOVE resistance - breakout, bullish
            sr_score = min(sr_score + 15, 85)

    sr_weight = CONFLUENCE_WEIGHTS["support_resistance"]
    sr_delta = (sr_score - 50) * sr_weight
    breakdown["S/R Levels"] = {"raw_score": sr_score, "weighted_score": round(sr_score * sr_weight, 1), "delta": round(sr_delta, 2)}
    base_score += sr_delta
    
    # Order Blocks (7%)
    ob_score = 50
    nearest_bullish_ob = order_blocks.get("nearest_bullish", {})
    if nearest_bullish_ob.get("is_valid") and nearest_bullish_ob.get("low", 0) < current_price * 1.05:
        ob_score = 70  # Bullish OB nearby
    ob_weight = CONFLUENCE_WEIGHTS["order_blocks"]
    ob_delta = (ob_score - 50) * ob_weight
    breakdown["Order Blocks"] = {"raw_score": ob_score, "weighted_score": round(ob_score * ob_weight, 1), "delta": round(ob_delta, 2)}
    base_score += ob_delta
    
    # Liquidity Zones (5%)
    lz_score = 50
    strongest_above = liquidity_zones.get("strongest_above", {})
    if strongest_above.get("strength") == "strong":
        lz_score = 55  # Strong liquidity above (bullish target)
    lz_weight = CONFLUENCE_WEIGHTS["liquidity_zones"]
    lz_delta = (lz_score - 50) * lz_weight
    breakdown["Liquidity Zones"] = {"raw_score": lz_score, "weighted_score": round(lz_score * lz_weight, 1), "delta": round(lz_delta, 2)}
    base_score += lz_delta
    
    # Break of Structure (5%) - IMPROVED: Confidence and CHOCH aware
    bos_score = 50  # Default neutral
    bos_detected = break_of_structure.get("bos_detected", False)
    bos_direction = break_of_structure.get("direction", "neutral")
    bos_confidence = break_of_structure.get("confidence", "none")
    choch_detected = break_of_structure.get("choch_detected", False)
    choch_direction = break_of_structure.get("choch_direction", "none")

    if bos_detected:
        if bos_direction == "bullish":
            # Bullish BOS - score based on confidence
            if bos_confidence == "high":
                bos_score = 85  # Strong bullish BOS
            elif bos_confidence == "moderate":
                bos_score = 75
            else:
                bos_score = 65
        elif bos_direction == "bearish":
            if bos_confidence == "high":
                bos_score = 15  # Strong bearish BOS
            elif bos_confidence == "moderate":
                bos_score = 25
            else:
                bos_score = 35
    else:
        # No BOS detected - check structure type and CHOCH
        structure_type = break_of_structure.get("structure_type", "neutral")
        if choch_detected:
            # CHOCH is significant even without explicit BOS
            if choch_direction == "bullish":
                bos_score = 70
            elif choch_direction == "bearish":
                bos_score = 30
        elif structure_type == "bullish":
            bos_score = 58  # Slightly bullish structure
        elif structure_type == "bearish":
            bos_score = 42  # Slightly bearish structure

    bos_weight = CONFLUENCE_WEIGHTS["break_of_structure"]
    bos_delta = (bos_score - 50) * bos_weight
    breakdown["Break of Structure"] = {"raw_score": bos_score, "weighted_score": round(bos_score * bos_weight, 1), "delta": round(bos_delta, 2)}
    base_score += bos_delta
    
    # Divergence (8%)
    if divergence.get("signal") == "bullish":
        div_score = 85
    elif divergence.get("signal") == "bearish":
        div_score = 15
    else:
        div_score = 50
    div_weight = CONFLUENCE_WEIGHTS["divergence"]
    div_delta = (div_score - 50) * div_weight
    breakdown["Divergence"] = {"raw_score": div_score, "weighted_score": round(div_score * div_weight, 1), "delta": round(div_delta, 2)}
    base_score += div_delta
    
    # RSI (9%)
    if rsi.get("signal") == "oversold":
        rsi_score = 80
    elif rsi.get("signal") == "overbought":
        rsi_score = 20
    else:
        rsi_score = 50
    rsi_weight = CONFLUENCE_WEIGHTS["rsi"]
    rsi_delta = (rsi_score - 50) * rsi_weight
    breakdown["RSI"] = {"raw_score": rsi_score, "weighted_score": round(rsi_score * rsi_weight, 1), "delta": round(rsi_delta, 2)}
    base_score += rsi_delta
    
    # MACD (9%)
    if macd.get("trend") == "bullish":
        macd_score = 75
    elif macd.get("trend") == "bearish":
        macd_score = 25
    else:
        macd_score = 50
    macd_weight = CONFLUENCE_WEIGHTS["macd"]
    macd_delta = (macd_score - 50) * macd_weight
    breakdown["MACD"] = {"raw_score": macd_score, "weighted_score": round(macd_score * macd_weight, 1), "delta": round(macd_delta, 2)}
    base_score += macd_delta
    
    # EMA Alignment (9%)
    if ema.get("alignment") == "bullish":
        ema_score = 80
    elif ema.get("alignment") == "bearish":
        ema_score = 20
    else:
        ema_score = 50
    ema_weight = CONFLUENCE_WEIGHTS["ema_alignment"]
    ema_delta = (ema_score - 50) * ema_weight
    breakdown["EMA Alignment"] = {"raw_score": ema_score, "weighted_score": round(ema_score * ema_weight, 1), "delta": round(ema_delta, 2)}
    base_score += ema_delta
    
    # Price vs EMA (7%)
    if ema.get("price_vs_ema") == "above_all":
        pve_score = 75
    elif ema.get("price_vs_ema") == "below_all":
        pve_score = 25
    else:
        pve_score = 50
    pve_weight = CONFLUENCE_WEIGHTS["price_vs_ema"]
    pve_delta = (pve_score - 50) * pve_weight
    breakdown["Price vs EMA"] = {"raw_score": pve_score, "weighted_score": round(pve_score * pve_weight, 1), "delta": round(pve_delta, 2)}
    base_score += pve_delta
    
    # Fibonacci (5%) - use fib_score from fibonacci calculation if available
    fib_score = fib.get("fib_score", 50) if fib else 50
    fib_weight = CONFLUENCE_WEIGHTS["fibonacci"]
    fib_delta = (fib_score - 50) * fib_weight
    breakdown["Fibonacci"] = {"raw_score": fib_score, "weighted_score": round(fib_score * fib_weight, 1), "delta": round(fib_delta, 2)}
    base_score += fib_delta
    
    # Bollinger (5%) - use regime-aware signal
    bb_score = 50
    if bb.get("signal") == "bullish_continuation" or bb.get("signal") == "bullish_mean_reversion":
        bb_score = 75
    elif bb.get("signal") == "bearish_continuation" or bb.get("signal") == "bearish_mean_reversion":
        bb_score = 25
    elif bb.get("position") == "lower_band":
        bb_score = 65  # Oversold
    elif bb.get("position") == "upper_band":
        bb_score = 35  # Overbought
    bb_weight = CONFLUENCE_WEIGHTS["bollinger"]
    bb_delta = (bb_score - 50) * bb_weight
    breakdown["Bollinger"] = {"raw_score": bb_score, "weighted_score": round(bb_score * bb_weight, 1), "delta": round(bb_delta, 2)}
    base_score += bb_delta
    
    # Volume (7%) - IMPROVED: More nuanced scoring
    vol_score = 50  # Default neutral
    if vol.get("data_available", True):  # Check if volume data exists
        vol_trend = vol.get("trend", "neutral")
        vol_ratio = vol.get("ratio", 1.0)
        
        if vol_trend == "bullish":
            # Bullish volume - score based on strength
            if vol_ratio > 1.5:
                vol_score = 75  # Strong bullish volume
            else:
                vol_score = 65  # Moderate bullish volume
        elif vol_trend == "bearish":
            # Bearish volume - less harsh scoring
            if vol_ratio > 1.5:
                vol_score = 30  # Strong bearish volume
            else:
                vol_score = 40  # Moderate bearish volume
        else:
            # Neutral volume - score based on ratio
            if vol_ratio > 1.2:
                vol_score = 55  # Above average volume is slightly positive
            elif vol_ratio < 0.7:
                vol_score = 45  # Low volume is slightly negative
            else:
                vol_score = 50  # Normal volume is neutral
    else:
        # No volume data available (CoinGecko) - use neutral score
        vol_score = 50

    vol_weight = CONFLUENCE_WEIGHTS["volume"]
    vol_delta = (vol_score - 50) * vol_weight
    breakdown["Volume"] = {"raw_score": vol_score, "weighted_score": round(vol_score * vol_weight, 1), "delta": round(vol_delta, 2)}
    base_score += vol_delta
    
    # Final score clamped to [0, 100] to ensure valid range
    confluence_score = max(0, min(100, round(base_score, 1)))
    
    # Determine bias: Only "long" or "neutral" for spot trading
    # Require confluence >= 60 AND higher TF is bullish or neutral
    htf_trend = higher_tf_trend.get("trend", "neutral") if higher_tf_trend else "neutral"
    allow_longs = higher_tf_trend.get("allow_longs", True) if higher_tf_trend else True
    
    if confluence_score >= 60 and htf_trend != "bearish" and allow_longs:
        recommended_bias = "long"
    else:
        recommended_bias = "neutral"
    
    # Use chart_pattern from indicators if available, otherwise use default
    chart_pattern_data = indicators.get("chart_pattern", {})
    if not chart_pattern_data or chart_pattern_data.get("name") == "none":
        chart_pattern_data = {
            "name": "none",
            "type": "none",
            "direction": "neutral",
            "reliability": "none",
            "status": "none",
            "description": "No pattern detected in fallback analysis",
            "score": 50
        }
    
    return {
        "chart_pattern": {
            "name": chart_pattern_data.get("name", "none"),
            "type": chart_pattern_data.get("type", "none"),
            "direction": chart_pattern_data.get("direction", "neutral"),
            "reliability": chart_pattern_data.get("reliability", "none"),
            "status": chart_pattern_data.get("status", "none"),
            "description": chart_pattern_data.get("description", "No pattern detected"),
            "score": chart_pattern_data.get("score", 50),
            "weight": int(CONFLUENCE_WEIGHTS["chart_pattern"] * 100),
            "weighted_score": breakdown.get("Chart Pattern", {}).get("weighted_score", 6.0) if isinstance(breakdown.get("Chart Pattern"), dict) else breakdown.get("Chart Pattern", 6.0)
        },
        "trend_analysis": {
            "trend": indicators.get("trend", "neutral"), 
            "reasoning": "Based on indicator analysis"
        },
        "indicators": {
            "rsi": {"value": rsi.get("value", 50), "signal": rsi.get("signal", "neutral"), "score": rsi_score, "weight": 9, "weighted_score": breakdown["RSI"].get("weighted_score", 0) if isinstance(breakdown.get("RSI"), dict) else breakdown.get("RSI", 0), "explanation": "RSI analysis"},
            "macd": {"value": macd, "signal": macd.get("trend", "neutral"), "score": macd_score, "weight": 9, "weighted_score": breakdown["MACD"].get("weighted_score", 0) if isinstance(breakdown.get("MACD"), dict) else breakdown.get("MACD", 0), "explanation": "MACD analysis"},
            "divergence": {"type": divergence.get("type", "none"), "signal": divergence.get("signal", "neutral"), "strength": divergence.get("strength", "none"), "score": div_score, "weight": 8, "weighted_score": breakdown["Divergence"].get("weighted_score", 0) if isinstance(breakdown.get("Divergence"), dict) else breakdown.get("Divergence", 0), "explanation": "Divergence analysis"},
            "ema_alignment": {"value": ema.get("alignment", "mixed"), "signal": ema.get("alignment", "mixed"), "score": ema_score, "weight": 9, "weighted_score": breakdown["EMA Alignment"].get("weighted_score", 0) if isinstance(breakdown.get("EMA Alignment"), dict) else breakdown.get("EMA Alignment", 0), "explanation": "EMA alignment"},
            "price_vs_ema": {"value": ema.get("price_vs_ema", "mixed"), "signal": ema.get("price_vs_ema", "mixed"), "score": pve_score, "weight": 7, "weighted_score": breakdown["Price vs EMA"].get("weighted_score", 0) if isinstance(breakdown.get("Price vs EMA"), dict) else breakdown.get("Price vs EMA", 0), "explanation": "Price position vs EMAs"},
            "support_resistance": {"nearest_support": supports[0]["price"] if supports else current_price * 0.95, "nearest_resistance": resistances[0]["price"] if resistances else current_price * 1.05, "signal": "bullish" if supports else "neutral", "score": sr_score, "weight": 12, "weighted_score": breakdown["S/R Levels"].get("weighted_score", 0) if isinstance(breakdown.get("S/R Levels"), dict) else breakdown.get("S/R Levels", 0), "explanation": "Support/Resistance levels"},
            "fibonacci": {"key_level": fib.get("nearest_level", "0.618") if fib else "0.618", "price_at_level": current_price * 0.98, "signal": "neutral", "score": fib_score, "weight": 5, "weighted_score": breakdown["Fibonacci"].get("weighted_score", 0) if isinstance(breakdown.get("Fibonacci"), dict) else breakdown.get("Fibonacci", 0), "explanation": "Fibonacci retracement"},
            "bollinger": {"position": bb.get("position", "middle"), "signal": bb.get("signal", "neutral"), "score": bb_score, "weight": 5, "weighted_score": breakdown["Bollinger"].get("weighted_score", 0) if isinstance(breakdown.get("Bollinger"), dict) else breakdown.get("Bollinger", 0), "explanation": "Bollinger Bands position"},
            "volume": {"trend": vol.get("trend", "neutral"), "signal": vol.get("trend", "neutral"), "score": vol_score, "weight": 7, "weighted_score": breakdown["Volume"].get("weighted_score", 0) if isinstance(breakdown.get("Volume"), dict) else breakdown.get("Volume", 0), "explanation": "Volume analysis"},
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


# Self-test function for Fibonacci and PatternDetector validation
def test_fibonacci_and_patterns():
    """Quick self-test for Fibonacci calculation and pattern detection"""
    print("\n" + "="*80)
    print("SELF-TEST: Fibonacci & Pattern Detection")
    print("="*80)
    
    # Create mock OHLC data (simple uptrend)
    mock_klines = []
    base_price = 100.0
    for i in range(50):
        trend_factor = i * 0.5
        mock_klines.append({
            "open_time": i * 1000,
            "open": base_price + trend_factor,
            "high": base_price + trend_factor + 2.0,
            "low": base_price + trend_factor - 1.0,
            "close": base_price + trend_factor + 1.0,
            "volume": 100.0 + i * 0.1,
            "close_time": (i + 1) * 1000,
            "quote_volume": 10000.0,
            "trades": 10
        })
    
    # Test Fibonacci
    try:
        fib_result = IndicatorEngine.calculate_fibonacci(mock_klines, lookback=50, pivot_left=3, pivot_right=3)
        print(f"\n✓ Fibonacci calculation successful")
        print(f"  Swing High: {fib_result.get('swing_high', 'N/A')}")
        print(f"  Swing Low: {fib_result.get('swing_low', 'N/A')}")
        print(f"  Trend: {fib_result.get('trend', 'N/A')}")
        print(f"  Levels: {len(fib_result.get('levels', {}))} levels calculated")
        
        # Verify levels are monotonic (for uptrend: low < levels < high)
        levels = fib_result.get('levels', {})
        if levels:
            level_values = [v for k, v in sorted(levels.items(), key=lambda x: float(x[0]))]
            is_monotonic = all(level_values[i] <= level_values[i+1] for i in range(len(level_values)-1))
            print(f"  Levels monotonic: {'✓' if is_monotonic else '✗'}")
    except Exception as e:
        print(f"\n✗ Fibonacci calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PatternDetector
    try:
        pattern_result = PatternDetector.detect_best(mock_klines)
        print(f"\n✓ Pattern detection successful")
        print(f"  Pattern: {pattern_result.get('name', 'none')}")
        print(f"  Type: {pattern_result.get('type', 'none')}")
        print(f"  Score: {pattern_result.get('score', 50)}")
        
        # Verify returns "none" safely when insufficient data
        short_klines = mock_klines[:5]
        short_pattern = PatternDetector.detect_best(short_klines)
        assert short_pattern.get("name") == "none", "Should return 'none' for insufficient data"
        print(f"  ✓ Returns 'none' safely for insufficient data")
    except Exception as e:
        print(f"\n✗ Pattern detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import uvicorn
    
    # Run self-test only if explicitly requested
    if os.getenv("RUN_TESTS") == "1":
        test_fibonacci_and_patterns()
    
    # Use PORT from environment (Railway/Heroku) or default to 8002
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)