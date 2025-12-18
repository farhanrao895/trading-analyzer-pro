"""
Vercel Serverless Function for Chart Analysis
Handles: Image processing, indicator calculation, Gemini AI, annotation drawing
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import base64
import io
import re
import math
import httpx
from typing import Dict, List, Optional
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np

# Import indicator functions (we'll create a shared module)
# For now, inline the key functions

def calculate_rsi(closes: List[float], period: int = 14) -> Dict:
    """RSI calculation"""
    if len(closes) < period + 1:
        return {"value": 50.0, "signal": "neutral"}
    
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    rsi = round(rsi, 2)
    signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    
    return {"value": rsi, "signal": signal, "period": period}

def calculate_ema(closes: List[float], period: int) -> float:
    """EMA calculation"""
    if len(closes) < period:
        return closes[-1] if closes else 0
    
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    
    return round(ema, 4)

def calculate_macd(closes: List[float]) -> Dict:
    """MACD calculation"""
    if len(closes) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
    
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    macd_line = ema12 - ema26
    
    # Simplified signal line
    macd_values = []
    for i in range(len(closes) - 26, len(closes)):
        e12 = calculate_ema(closes[:i+1], 12)
        e26 = calculate_ema(closes[:i+1], 26)
        macd_values.append(e12 - e26)
    
    signal_line = calculate_ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
    histogram = macd_line - signal_line
    
    trend = "bullish" if histogram > 0 and macd_line > signal_line else "bearish" if histogram < 0 else "neutral"
    
    return {
        "macd": round(macd_line, 6),
        "signal": round(signal_line, 6),
        "histogram": round(histogram, 6),
        "trend": trend
    }

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
    position = "lower_band" if current_price <= lower else "upper_band" if current_price >= upper else "middle"
    
    return {
        "upper": round(upper, 4),
        "middle": round(sma, 4),
        "lower": round(lower, 4),
        "bandwidth": round(bandwidth, 2),
        "position": position
    }

def calculate_atr(klines: List[Dict], period: int = 14) -> float:
    """ATR calculation"""
    if len(klines) < period + 1:
        return 0
    
    trs = []
    for i in range(1, len(klines)):
        high = klines[i]["high"]
        low = klines[i]["low"]
        prev_close = klines[i-1]["close"]
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    
    if len(trs) < period:
        return round(sum(trs) / len(trs), 4) if trs else 0
    
    return round(sum(trs[-period:]) / period, 4)

def find_support_resistance(klines: List[Dict], num_levels: int = 3) -> Dict:
    """Find S/R levels"""
    if len(klines) < 10:
        return {"support": [], "resistance": []}
    
    highs = [k["high"] for k in klines]
    lows = [k["low"] for k in klines]
    closes = [k["close"] for k in klines]
    current_price = closes[-1]
    
    pivot_highs = []
    pivot_lows = []
    lookback = 5
    
    for i in range(lookback, len(klines) - lookback):
        if all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, lookback+1)):
            pivot_highs.append({"price": highs[i], "index": i})
        
        if all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, lookback+1)):
            pivot_lows.append({"price": lows[i], "index": i})
    
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
    
    support = [s for s in support_levels if s["price"] < current_price]
    resistance = [r for r in resistance_levels if r["price"] > current_price]
    
    support = sorted(support, key=lambda x: x["price"], reverse=True)[:num_levels]
    resistance = sorted(resistance, key=lambda x: x["price"])[:num_levels]
    
    for s in support:
        s["strength"] = "strong" if s["touches"] >= 3 else "moderate" if s["touches"] >= 2 else "weak"
    for r in resistance:
        r["strength"] = "strong" if r["touches"] >= 3 else "moderate" if r["touches"] >= 2 else "weak"
    
    return {"support": support, "resistance": resistance}

def calculate_fibonacci(klines: List[Dict], lookback: int = 50) -> Dict:
    """Fibonacci levels"""
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
        "1.272": round(swing_low - diff * 0.272, 4),
        "1.618": round(swing_low - diff * 0.618, 4),
    }
    
    return {"levels": levels, "swing_high": swing_high, "swing_low": swing_low}

def analyze_volume(klines: List[Dict], period: int = 20) -> Dict:
    """Volume analysis"""
    if len(klines) < period:
        return {"current": 0, "average": 0, "ratio": 1.0, "trend": "neutral"}
    
    volumes = [k["volume"] for k in klines]
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-period:]) / period
    
    ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    recent_closes = [k["close"] for k in klines[-10:]]
    recent_volumes = volumes[-10:]
    
    up_volume = sum(v for i, v in enumerate(recent_volumes[1:], 1) 
                   if recent_closes[i] > recent_closes[i-1])
    down_volume = sum(v for i, v in enumerate(recent_volumes[1:], 1) 
                     if recent_closes[i] < recent_closes[i-1])
    
    trend = "bullish" if up_volume > down_volume * 1.2 else "bearish" if down_volume > up_volume * 1.2 else "neutral"
    
    return {
        "current": round(current_volume, 2),
        "average": round(avg_volume, 2),
        "ratio": round(ratio, 2),
        "trend": trend,
        "up_volume": round(up_volume, 2),
        "down_volume": round(down_volume, 2)
    }

def calculate_all_indicators(klines: List[Dict]) -> Dict:
    """Calculate all indicators"""
    if not klines:
        return {}
    
    closes = [k["close"] for k in klines]
    current_price = closes[-1]
    
    rsi = calculate_rsi(closes, 14)
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200) if len(closes) >= 200 else calculate_ema(closes, min(len(closes), 100))
    
    ema_alignment = "bullish" if ema20 > ema50 > ema200 else "bearish" if ema20 < ema50 < ema200 else "mixed"
    above_count = sum([1 for ema in [ema20, ema50, ema200] if current_price > ema])
    price_vs_ema = "above_all" if above_count == 3 else "below_all" if above_count == 0 else "mixed"
    
    macd = calculate_macd(closes)
    bollinger = calculate_bollinger(closes, 20, 2.0)
    atr = calculate_atr(klines, 14)
    sr_levels = find_support_resistance(klines, 3)
    fib = calculate_fibonacci(klines, 50)
    volume = analyze_volume(klines, 20)
    
    return {
        "current_price": current_price,
        "rsi": rsi,
        "ema": {"ema20": ema20, "ema50": ema50, "ema200": ema200, "alignment": ema_alignment, "price_vs_ema": price_vs_ema},
        "macd": macd,
        "bollinger": bollinger,
        "atr": atr,
        "support_resistance": sr_levels,
        "fibonacci": fib,
        "volume": volume,
        "trend": "neutral"
    }

def extract_price_scale(klines: List[Dict], buffer_pct: float = 0.05) -> Dict:
    """Extract price scale"""
    if not klines:
        return {"max_price": 100, "min_price": 0}
    
    highs = [k["high"] for k in klines[-100:]]
    lows = [k["low"] for k in klines[-100:]]
    
    max_price = max(highs)
    min_price = min(lows)
    buffer = (max_price - min_price) * buffer_pct
    
    return {
        "max_price": round(max_price + buffer, 4),
        "min_price": round(min_price - buffer, 4)
    }

def price_to_y(price: float, max_price: float, min_price: float, img_height: int) -> int:
    """Price to Y coordinate"""
    chart_top_y = 70
    chart_bottom_y = img_height - 80
    chart_height = chart_bottom_y - chart_top_y
    
    if max_price == min_price:
        return chart_top_y + chart_height // 2
    
    y = chart_top_y + ((max_price - price) / (max_price - min_price)) * chart_height
    return int(max(chart_top_y, min(chart_bottom_y, y)))

def build_analysis_prompt(symbol: str, timeframe: str, price_data: Dict, indicators: Dict, 
                         depth_data: Dict, img_width: int, img_height: int, price_scale: Dict) -> str:
    """Build Gemini prompt"""
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
    
    return f"""ROLE: You are an expert institutional technical analyst with 20+ years experience.
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
Largest Bid Wall: ${depth_data.get('largest_bid_wall', {}).get('price', 0):,.4f}
Largest Ask Wall: ${depth_data.get('largest_ask_wall', {}).get('price', 0):,.4f}

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

=== OUTPUT FORMAT (JSON ONLY - NO MARKDOWN) ===

Return ONLY valid JSON with this structure:
{{
  "price_scale": {{"max_price": {max_price}, "min_price": {min_price}}},
  "trend_analysis": {{"trend": "bullish/bearish/neutral", "reasoning": "explanation"}},
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
    "entry": {{"price": <number>, "y": <Y coordinate>, "reasoning": "why"}},
    "stop_loss": {{"price": <number>, "y": <Y coordinate>, "reasoning": "why"}},
    "tp1": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:1.5", "reasoning": "why"}},
    "tp2": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:2.5", "reasoning": "why"}},
    "tp3": {{"price": <number>, "y": <Y coordinate>, "risk_reward": "1:4", "reasoning": "why"}}
  }},
  "support_levels": [{{"price": <n>, "y": <Y coordinate>, "strength": "strong/moderate/weak"}}],
  "resistance_levels": [{{"price": <n>, "y": <Y coordinate>, "strength": "strong/moderate/weak"}}],
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

def draw_annotations(image_bytes: bytes, analysis_data: Dict, price_scale: Dict, img_height: int) -> bytes:
    """Draw annotations on chart"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    height, width = img.shape[:2]
    max_price = price_scale["max_price"]
    min_price = price_scale["min_price"]
    
    colors = {
        "entry": (0, 200, 0),
        "stop_loss": (0, 0, 255),
        "tp1": (255, 200, 0),
        "tp2": (255, 180, 0),
        "tp3": (255, 150, 0),
        "support": (0, 200, 200),
        "resistance": (200, 0, 200)
    }
    
    def get_y(price: float) -> int:
        return price_to_y(price, max_price, min_price, height)
    
    def draw_dashed_line(pt1: tuple, pt2: tuple, color: tuple, thickness: int = 2):
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
    
    def draw_label(text: str, y: int, color: tuple, offset: int = 0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding = 6
        label_width = text_width + padding * 2
        label_height = text_height + padding * 2
        
        x = width - label_width - 10
        y = max(label_height + 5, min(y + offset, height - 5))
        
        cv2.rectangle(img, (x, y - label_height), (x + label_width, y), color, -1)
        cv2.putText(img, text, (x + padding, y - padding), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    used_y_positions = []
    
    def get_safe_y(target_y: int) -> int:
        min_gap = 25
        for used_y in used_y_positions:
            if abs(target_y - used_y) < min_gap:
                target_y = used_y + min_gap
        used_y_positions.append(target_y)
        return target_y
    
    # Draw support/resistance
    for support in analysis_data.get("support_levels", []):
        price = support.get("price", 0)
        if price > 0:
            y = support.get("y") or get_y(price)
            if 70 < y < height - 80:
                draw_dashed_line((0, y), (width - 170, y), colors["support"], 2)
                safe_y = get_safe_y(y)
                draw_label(f"SUPPORT ${price:,.1f}", safe_y, colors["support"])
    
    for resistance in analysis_data.get("resistance_levels", []):
        price = resistance.get("price", 0)
        if price > 0:
            y = resistance.get("y") or get_y(price)
            if 70 < y < height - 80:
                draw_dashed_line((0, y), (width - 170, y), colors["resistance"], 2)
                safe_y = get_safe_y(y)
                draw_label(f"RESISTANCE ${price:,.1f}", safe_y, colors["resistance"])
    
    # Draw trade setup
    trade_setup = analysis_data.get("trade_setup", {})
    
    entry = trade_setup.get("entry", {})
    if entry.get("price"):
        price = entry["price"]
        y = entry.get("y") or get_y(price)
        if 70 < y < height - 80:
            draw_dashed_line((0, y), (width - 170, y), colors["entry"], 2)
            cv2.circle(img, (width // 2, y), 8, colors["entry"], -1)
            cv2.circle(img, (width // 2, y), 10, (255, 255, 255), 2)
            safe_y = get_safe_y(y)
            draw_label(f"ENTRY ${price:,.1f}", safe_y, colors["entry"])
    
    sl = trade_setup.get("stop_loss", {})
    if sl.get("price"):
        price = sl["price"]
        y = sl.get("y") or get_y(price)
        if 70 < y < height - 80:
            draw_dashed_line((0, y), (width - 170, y), colors["stop_loss"], 2)
            cv2.line(img, (width//2-8, y-8), (width//2+8, y+8), colors["stop_loss"], 3)
            cv2.line(img, (width//2-8, y+8), (width//2+8, y-8), colors["stop_loss"], 3)
            safe_y = get_safe_y(y)
            draw_label(f"STOP LOSS ${price:,.1f}", safe_y, colors["stop_loss"])
    
    for i, tp_key in enumerate(["tp1", "tp2", "tp3"]):
        tp = trade_setup.get(tp_key, {})
        if tp.get("price"):
            price = tp["price"]
            y = tp.get("y") or get_y(price)
            if 70 < y < height - 80:
                color = colors[tp_key]
                draw_dashed_line((0, y), (width - 170, y), color, 2)
                x_offset = (i + 1) * 20
                cv2.circle(img, (width // 2 + x_offset, y), 6, color, -1)
                cv2.circle(img, (width // 2 + x_offset, y), 8, (255, 255, 255), 2)
                safe_y = get_safe_y(y)
                draw_label(f"TP{i+1} ${price:,.1f}", safe_y, color)
    
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Parse multipart form data
            import cgi
            form = cgi.FieldStorage(
                fp=io.BytesIO(post_data),
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            file_item = form.getfirst('file')
            symbol = form.getfirst('symbol', 'BTCUSDT').upper()
            timeframe = form.getfirst('timeframe', '4h')
            
            if not file_item:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file provided"}).encode())
                return
            
            # Read image
            if hasattr(file_item, 'file'):
                image_bytes = file_item.file.read()
            else:
                image_bytes = file_item
            
            image = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = image.size
            
            # Fetch Binance data
            async def fetch_data():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Price data
                    price_res = await client.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
                    price_data = price_res.json() if price_res.status_code == 200 else {}
                    
                    # Klines
                    klines_res = await client.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit=500")
                    klines_raw = klines_res.json() if klines_res.status_code == 200 else []
                    
                    # Depth
                    depth_res = await client.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20")
                    depth_raw = depth_res.json() if depth_res.status_code == 200 else {}
                    
                    return price_data, klines_raw, depth_raw
            
            # For Vercel, we'll use sync calls (or use asyncio.run)
            import asyncio
            price_data, klines_raw, depth_raw = asyncio.run(fetch_data())
            
            if not price_data:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Symbol {symbol} not found"}).encode())
                return
            
            klines = [{
                "open_time": k[0], "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])
            } for k in klines_raw]
            
            price_data = {
                "symbol": price_data["symbol"],
                "current_price": float(price_data["lastPrice"]),
                "price_change_24h": float(price_data.get("priceChange", 0)),
                "price_change_pct": float(price_data.get("priceChangePercent", 0)),
                "high_24h": float(price_data.get("highPrice", 0)),
                "low_24h": float(price_data.get("lowPrice", 0)),
                "volume_24h": float(price_data.get("volume", 0)),
                "quote_volume": float(price_data.get("quoteVolume", 0))
            }
            
            depth_data = {"largest_bid_wall": {"price": 0, "quantity": 0}, "largest_ask_wall": {"price": 0, "quantity": 0}}
            if depth_raw:
                bids = [{"price": float(b[0]), "quantity": float(b[1])} for b in depth_raw.get("bids", [])]
                asks = [{"price": float(a[0]), "quantity": float(a[1])} for a in depth_raw.get("asks", [])]
                if bids:
                    depth_data["largest_bid_wall"] = max(bids, key=lambda x: x["quantity"])
                if asks:
                    depth_data["largest_ask_wall"] = max(asks, key=lambda x: x["quantity"])
            
            # Calculate indicators
            indicators = calculate_all_indicators(klines)
            price_scale = extract_price_scale(klines)
            
            # Call Gemini
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set")
            
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = build_analysis_prompt(symbol, timeframe, price_data, indicators, depth_data, img_width, img_height, price_scale)
            
            response = model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(
                temperature=0.1, top_p=0.95, max_output_tokens=8192
            ))
            
            response_text = response.text.strip()
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            analysis_data = json.loads(json_match.group()) if json_match else {}
            
            # Draw annotations
            annotated_bytes = draw_annotations(image_bytes, analysis_data, price_scale, img_height)
            annotated_b64 = base64.b64encode(annotated_bytes).decode('utf-8')
            
            # Build response
            result = {
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
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": error_msg}).encode())

