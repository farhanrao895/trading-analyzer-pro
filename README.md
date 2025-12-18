# Trading Analyzer Pro 📊

**Advanced Professional Trading Chart Analyzer with AI-Powered Technical Analysis**

An intelligent trading assistant that analyzes chart screenshots using Gemini AI and real Binance market data to provide comprehensive technical analysis, entry/exit points, support/resistance levels, and confluence scoring.

![Trading Analyzer Pro](https://img.shields.io/badge/Version-2.0.0-cyan)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Node.js](https://img.shields.io/badge/Node.js-18+-green)

---

## ✨ Features

### Backend (FastAPI + Binance Integration)
- **Live Binance Data**: Real-time price, 24h stats, historical klines, order book depth
- **Indicator Engine**: RSI, EMA (20/50/200), MACD, Bollinger Bands, ATR
- **Support/Resistance Detection**: Automatic swing high/low detection with strength rating
- **Fibonacci Levels**: Automatic retracement and extension calculation
- **Order Book Analysis**: Identify large bid/ask walls as potential S/R
- **Gemini AI Integration**: Chain-of-Thought analysis with structured JSON output
- **Chart Annotation**: Automatic drawing of entry, SL, TPs, support, resistance

### Frontend (Next.js + React)
- **Modern UI**: Dark theme, professional dashboard layout
- **Live Price Ticker**: Auto-refreshing current price and 24h change
- **Symbol Selection**: Popular pairs dropdown + custom symbol input
- **Timeframe Selection**: 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W
- **Drag & Drop Upload**: Easy chart screenshot upload
- **Annotated Chart**: Visual display with all TA levels marked
- **Indicators Dashboard**: 8 indicators with scores and explanations
- **Confluence Score**: Weighted scoring system with breakdown
- **Trade Setup Card**: Entry, Stop Loss, TP1/TP2/TP3 with R:R ratios
- **Support/Resistance Panel**: Listed levels with strength indicators
- **Market Data Panel**: 24h high/low/volume from Binance

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Gemini API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### 1. Clone/Download
```bash
cd "D:\Trading Analyzer"
```

### 2. Create Environment File
Create `.env` file in the root directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Install Backend Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
npm install
```

---

## 🚀 Running the Application

### Option 1: Start Both Servers (Recommended)
```bash
start_both.bat
```
This will:
1. Start the backend server (port 8001)
2. Start the frontend server (port 3000)
3. Open your browser automatically

### Option 2: Start Individually
```bash
# Terminal 1 - Backend
start_backend.bat

# Terminal 2 - Frontend
start_frontend.bat
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs

---

## 📖 Usage Guide

1. **Select Symbol**: Choose from popular pairs or enter a custom symbol
2. **Select Timeframe**: Pick your analysis timeframe (4H recommended)
3. **Upload Chart**: Drag & drop a chart screenshot or click to browse
4. **Click Analyze**: Wait for AI analysis (10-30 seconds)
5. **Review Results**:
   - Annotated chart with entry/SL/TP lines
   - Confluence score and breakdown
   - Trade setup with exact prices
   - Technical indicators with signals
   - Support/resistance levels

---

## 📊 Technical Indicators

| Indicator | Weight | Bullish Signal | Bearish Signal |
|-----------|--------|----------------|----------------|
| RSI (14) | 15% | < 30 (oversold) | > 70 (overbought) |
| MACD | 15% | MACD > Signal | MACD < Signal |
| EMA Alignment | 15% | 20 > 50 > 200 | 20 < 50 < 200 |
| Price vs EMA | 10% | Above all EMAs | Below all EMAs |
| Support/Resistance | 15% | At support | At resistance |
| Fibonacci | 10% | At 0.618/0.786 | At extensions |
| Bollinger Bands | 10% | At lower band | At upper band |
| Volume | 10% | Bullish volume | Bearish volume |

### Confluence Score Interpretation
- **≥ 70**: Strong signal - Trade recommended
- **50-69**: Moderate signal - Proceed with caution
- **< 50**: Weak signal - No trade recommendation

---

## 🎨 Chart Annotation Colors

| Element | Color | Description |
|---------|-------|-------------|
| Entry | 🟢 Green | Entry point |
| Stop Loss | 🔴 Red | Risk level |
| Take Profit | 🔵 Cyan | Target levels |
| Support | 🟡 Yellow | Buy zones |
| Resistance | 🟣 Purple | Sell zones |

---

## 🔧 API Endpoints

### Market Data
```
GET /api/symbols          - All USDT trading pairs
GET /api/price/{symbol}   - Real-time price + 24h stats
GET /api/klines/{symbol}/{interval}  - Historical candlesticks
GET /api/depth/{symbol}   - Order book depth
GET /api/indicators/{symbol}/{interval}  - Calculated indicators
```

### Analysis
```
POST /api/analyze         - Full chart analysis
  - file: Chart image (multipart)
  - symbol: Trading pair (e.g., BTCUSDT)
  - timeframe: Chart timeframe (e.g., 4h)
```

---

## 💰 Costs

- **Gemini API**: ~$0.00025 per analysis (very affordable)
- **Binance API**: Free (no API key required for public data)

---

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only. It does NOT constitute financial advice.**

- Trading cryptocurrencies involves significant risk
- Past performance does not guarantee future results
- Always do your own research (DYOR)
- Never invest more than you can afford to lose
- The AI analysis may contain errors - verify before trading

---

## 🐛 Troubleshooting

### Backend won't start
1. Check Python version: `python --version` (needs 3.10+)
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Check `.env` file has valid `GEMINI_API_KEY`

### Frontend won't start
1. Check Node version: `node --version` (needs 18+)
2. Delete `node_modules` and run `npm install`
3. Try `npm run dev -- -p 3001` if port 3000 is busy

### Analysis fails
1. Check backend is running on port 8002
2. Check Gemini API key is valid
3. Try smaller image files (< 5MB recommended)
4. Check browser console for errors

### Lines not aligned on chart
1. Upload chart with visible Y-axis price scale
2. Use standard TradingView-style charts
3. Avoid charts with many overlays/indicators

---

## 📝 License

MIT License - Feel free to use and modify for personal use.

---

## 🙏 Credits

- **Gemini AI** by Google for chart analysis
- **Binance API** for market data
- **Next.js** and **Tailwind CSS** for frontend
- **FastAPI** for backend

---

Built with ❤️ for traders by traders
