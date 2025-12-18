'use client'

import { useState, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

// Use environment variable for backend URL, or auto-detect
const getApiUrl = () => {
  if (typeof window !== 'undefined') {
    // Client-side: use environment variable or default to same origin
    return process.env.NEXT_PUBLIC_BACKEND_URL || window.location.origin
  }
  // Server-side: use environment variable or localhost
  return process.env.BACKEND_URL || 'http://localhost:8002'
}

const API_URL = getApiUrl()

// ============================================================
// TYPES
// ============================================================

interface IndicatorData {
  value?: number | string | object
  signal?: string
  score?: number
  weight?: number
  weighted_score?: number
  explanation?: string
  position?: string
  trend?: string
  key_level?: string
  price_at_level?: number
  nearest_support?: number
  nearest_resistance?: number
}

interface TradeSetup {
  bias?: string
  confidence?: string
  entry?: { price: number; y: number; reasoning: string }
  stop_loss?: { price: number; y: number; reasoning: string }
  tp1?: { price: number; y: number; risk_reward: string; reasoning: string }
  tp2?: { price: number; y: number; risk_reward: string; reasoning: string }
  tp3?: { price: number; y: number; risk_reward: string; reasoning: string }
}

interface SupportResistance {
  price: number
  y: number
  strength: string
}

interface BinanceData {
  current_price: number
  price_change_24h: number
  price_change_pct: number
  high_24h: number
  low_24h: number
  volume_24h: number
  quote_volume: number
}

interface AnalysisResult {
  success: boolean
  symbol: string
  timeframe: string
  current_price: number
  binance_data: BinanceData
  calculated_indicators: any
  indicators: Record<string, IndicatorData>
  trade_setup: TradeSetup
  support_levels: SupportResistance[]
  resistance_levels: SupportResistance[]
  confluence_score: number
  confluence_breakdown: Record<string, number>
  trend: string
  bias: string
  risk_reward: string
  analysis_summary: string
  trade_rationale: string
  annotated_image: string
}

// ============================================================
// HELPER COMPONENTS
// ============================================================

const LoadingSpinner = () => (
  <div className="flex items-center justify-center gap-3">
    <div className="w-6 h-6 border-3 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
    <span className="text-cyan-400 font-medium">Analyzing chart with AI...</span>
  </div>
)

const SignalBadge = ({ signal }: { signal: string }) => {
  const colors: Record<string, string> = {
    bullish: 'bg-green-500/20 text-green-400 border-green-500/30',
    oversold: 'bg-green-500/20 text-green-400 border-green-500/30',
    above_all: 'bg-green-500/20 text-green-400 border-green-500/30',
    bearish: 'bg-red-500/20 text-red-400 border-red-500/30',
    overbought: 'bg-red-500/20 text-red-400 border-red-500/30',
    below_all: 'bg-red-500/20 text-red-400 border-red-500/30',
    neutral: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    mixed: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    middle: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  }
  
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[signal?.toLowerCase()] || colors.neutral}`}>
      {signal?.toUpperCase() || 'N/A'}
    </span>
  )
}

const IndicatorCard = ({ name, indicator, icon }: { name: string; indicator: IndicatorData; icon: string }) => {
  const getDisplayValue = () => {
    if (typeof indicator.value === 'number') {
      return indicator.value.toFixed(2)
    }
    if (typeof indicator.value === 'object') {
      return JSON.stringify(indicator.value).substring(0, 30) + '...'
    }
    return indicator.value || indicator.position || indicator.trend || indicator.key_level || '-'
  }

  return (
    <div className="bg-slate-800/60 rounded-lg p-4 border border-slate-700/50 hover:border-cyan-500/30 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <span className="text-sm font-medium text-slate-300">{name}</span>
        </div>
        <SignalBadge signal={indicator.signal || 'neutral'} />
      </div>
      
      <div className="text-xl font-bold text-white mb-1">
        {getDisplayValue()}
      </div>
      
      {indicator.score !== undefined && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>Score</span>
            <span>{indicator.score}/100 × {indicator.weight}%</span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all ${
                indicator.score >= 60 ? 'bg-green-500' : 
                indicator.score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${indicator.score}%` }}
            />
          </div>
        </div>
      )}
      
      {indicator.explanation && (
        <p className="text-xs text-slate-500 mt-2 line-clamp-2">{indicator.explanation}</p>
      )}
    </div>
  )
}

const TradeSetupCard = ({ setup }: { setup: TradeSetup }) => {
  const isLong = setup.bias?.toLowerCase() === 'long'
  
  const formatPrice = (price?: number) => {
    if (!price) return '-'
    return price >= 1000 ? `$${price.toLocaleString()}` : `$${price.toFixed(4)}`
  }

  return (
    <div className={`rounded-xl p-5 border-2 ${isLong ? 'bg-green-900/20 border-green-500/40' : 'bg-red-900/20 border-red-500/40'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-lg text-white flex items-center gap-2">
          {isLong ? '🟢' : '🔴'} Trade Setup
        </h3>
        <div className="flex gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${isLong ? 'bg-green-500 text-black' : 'bg-red-500 text-white'}`}>
            {setup.bias?.toUpperCase()}
          </span>
          {setup.confidence && (
            <span className="px-3 py-1 rounded-full text-sm font-medium bg-slate-700 text-slate-300">
              {setup.confidence?.toUpperCase()}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Entry */}
        <div className="bg-slate-800/50 rounded-lg p-3">
          <div className="text-xs text-slate-400 mb-1">ENTRY</div>
          <div className="text-lg font-bold text-green-400">{formatPrice(setup.entry?.price)}</div>
          {setup.entry?.reasoning && (
            <div className="text-xs text-slate-500 mt-1">{setup.entry.reasoning}</div>
          )}
        </div>

        {/* Stop Loss */}
        <div className="bg-slate-800/50 rounded-lg p-3">
          <div className="text-xs text-slate-400 mb-1">STOP LOSS</div>
          <div className="text-lg font-bold text-red-400">{formatPrice(setup.stop_loss?.price)}</div>
          {setup.stop_loss?.reasoning && (
            <div className="text-xs text-slate-500 mt-1">{setup.stop_loss.reasoning}</div>
          )}
        </div>

        {/* Take Profits */}
        {setup.tp1?.price && (
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-slate-400 mb-1">TP1 <span className="text-cyan-400">({setup.tp1.risk_reward})</span></div>
            <div className="text-lg font-bold text-cyan-400">{formatPrice(setup.tp1.price)}</div>
          </div>
        )}
        {setup.tp2?.price && (
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-slate-400 mb-1">TP2 <span className="text-cyan-400">({setup.tp2.risk_reward})</span></div>
            <div className="text-lg font-bold text-cyan-400">{formatPrice(setup.tp2.price)}</div>
          </div>
        )}
        {setup.tp3?.price && (
          <div className="col-span-2 bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-slate-400 mb-1">TP3 (Extended) <span className="text-cyan-400">({setup.tp3.risk_reward})</span></div>
            <div className="text-lg font-bold text-cyan-400">{formatPrice(setup.tp3.price)}</div>
          </div>
        )}
      </div>
    </div>
  )
}

const ConfluenceScoreCard = ({ score, breakdown }: { score: number; breakdown: Record<string, number> }) => {
  const getScoreColor = () => {
    if (score >= 70) return 'text-green-400'
    if (score >= 50) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getScoreLabel = () => {
    if (score >= 70) return 'STRONG'
    if (score >= 50) return 'MODERATE'
    return 'WEAK'
  }

  return (
    <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
      <h3 className="font-bold text-white mb-4 flex items-center gap-2">
        <span className="text-xl">📊</span> Confluence Score
      </h3>

      <div className="flex items-center gap-4 mb-4">
        <div className={`text-5xl font-bold ${getScoreColor()}`}>
          {score.toFixed(0)}
        </div>
        <div className="flex-1">
          <div className={`text-sm font-bold ${getScoreColor()}`}>{getScoreLabel()}</div>
          <div className="text-xs text-slate-400">out of 100</div>
          <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all ${
                score >= 70 ? 'bg-green-500' : score >= 50 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="space-y-2">
        {Object.entries(breakdown || {}).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between text-sm">
            <span className="text-slate-400">{key}</span>
            <div className="flex items-center gap-2">
              <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-cyan-500 rounded-full"
                  style={{ width: `${Math.min(100, (value / 15) * 100)}%` }}
                />
              </div>
              <span className="text-cyan-400 w-8 text-right font-medium">+{value.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function TradingAnalyzerPro() {
  // State
  const [symbols, setSymbols] = useState<string[]>([])
  const [popularSymbols, setPopularSymbols] = useState<string[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT')
  const [customSymbol, setCustomSymbol] = useState('')
  const [timeframe, setTimeframe] = useState('4h')
  const [livePrice, setLivePrice] = useState<number | null>(null)
  const [priceChange, setPriceChange] = useState<number | null>(null)
  const [uploadedImage, setUploadedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '30m', label: '30m' },
    { value: '1h', label: '1H' },
    { value: '4h', label: '4H' },
    { value: '1d', label: '1D' },
    { value: '1w', label: '1W' },
  ]

  // Fetch symbols on mount
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/symbols`)
        setSymbols(res.data.symbols)
        setPopularSymbols(res.data.popular)
      } catch (e) {
        console.error('Failed to fetch symbols:', e)
      }
    }
    fetchSymbols()
  }, [])

  // Fetch live price
  useEffect(() => {
    const symbol = customSymbol || selectedSymbol
    if (!symbol) return

    const fetchPrice = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/price/${symbol}`)
        setLivePrice(res.data.current_price)
        setPriceChange(res.data.price_change_pct)
      } catch (e) {
        console.error('Failed to fetch price:', e)
      }
    }

    fetchPrice()
    const interval = setInterval(fetchPrice, 5000)
    return () => clearInterval(interval)
  }, [selectedSymbol, customSymbol])

  // Dropzone
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setUploadedImage(file)
      setImagePreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'] },
    maxFiles: 1
  })

  // Analyze
  const handleAnalyze = async () => {
    if (!uploadedImage) {
      setError('Please upload a chart image first')
      return
    }

    setAnalyzing(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', uploadedImage)
    formData.append('symbol', customSymbol || selectedSymbol)
    formData.append('timeframe', timeframe)

    try {
      const res = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000
      })
      setResult(res.data)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Trading Analyzer Pro
              </h1>
              <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs font-bold rounded border border-green-500/30">
                SPOT • LONG ONLY
              </span>
            </div>

            {/* Live Price */}
            {livePrice && (
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-xs text-slate-400">{customSymbol || selectedSymbol}</div>
                  <div className="text-xl font-bold text-white">
                    ${livePrice >= 1000 ? livePrice.toLocaleString() : livePrice.toFixed(4)}
                  </div>
                </div>
                {priceChange !== null && (
                  <span className={`px-3 py-1 rounded-lg font-bold ${
                    priceChange >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-6 py-6">
        {/* Settings Panel */}
        <div className="bg-slate-900/50 rounded-xl p-5 mb-6 border border-slate-800">
          <div className="flex flex-wrap items-end gap-4">
            {/* Symbol Select */}
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-slate-400 mb-2">Symbol</label>
              <select
                value={selectedSymbol}
                onChange={(e) => {
                  setSelectedSymbol(e.target.value)
                  setCustomSymbol('')
                }}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              >
                <optgroup label="Popular">
                  {popularSymbols.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </optgroup>
                <optgroup label="All USDT Pairs">
                  {symbols.filter(s => !popularSymbols.includes(s)).map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            {/* Custom Symbol */}
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-slate-400 mb-2">Custom Symbol</label>
              <input
                type="text"
                value={customSymbol}
                onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
                placeholder="e.g., PEPEUSDT"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
            </div>

            {/* Timeframe */}
            <div className="min-w-[180px]">
              <label className="block text-sm font-medium text-slate-400 mb-2">Timeframe</label>
              <div className="flex gap-1 bg-slate-800 p-1 rounded-lg">
                {timeframes.map(tf => (
                  <button
                    key={tf.value}
                    onClick={() => setTimeframe(tf.value)}
                    className={`px-3 py-2 rounded-md text-sm font-medium transition ${
                      timeframe === tf.value
                        ? 'bg-cyan-500 text-black'
                        : 'text-slate-400 hover:text-white hover:bg-slate-700'
                    }`}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={analyzing || !uploadedImage}
              className="px-8 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold rounded-lg hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg shadow-cyan-500/20"
            >
              {analyzing ? 'Analyzing...' : '🔍 Analyze Chart'}
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Chart Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload / Chart Display */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden">
              {!result?.annotated_image && (
                <div
                  {...getRootProps()}
                  className={`p-8 text-center cursor-pointer transition ${
                    isDragActive ? 'bg-cyan-500/10 border-cyan-500' : 'hover:bg-slate-800/50'
                  }`}
                >
                  <input {...getInputProps()} />
                  
                  {imagePreview ? (
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Chart Preview"
                        className="max-w-full max-h-[600px] mx-auto rounded-lg"
                      />
                      <div className="absolute top-4 right-4 px-3 py-1 bg-slate-900/80 text-slate-300 text-sm rounded-lg">
                        Click or drop to replace
                      </div>
                    </div>
                  ) : (
                    <div className="py-16">
                      <div className="text-6xl mb-4">📊</div>
                      <p className="text-xl font-medium text-white mb-2">
                        Drop your chart screenshot here
                      </p>
                      <p className="text-slate-400">
                        or click to browse • PNG, JPG, WebP supported
                      </p>
                    </div>
                  )}
                </div>
              )}

              {analyzing && (
                <div className="p-12 text-center">
                  <LoadingSpinner />
                  <p className="text-slate-400 mt-4">
                    Reading chart, calculating indicators, running AI analysis...
                  </p>
                </div>
              )}

              {result?.annotated_image && (
                <div className="p-4">
                  <img
                    src={result.annotated_image}
                    alt="Analyzed Chart"
                    className="w-full rounded-lg"
                  />
                  
                  {/* Legend */}
                  <div className="mt-4 flex flex-wrap gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-green-500"></div>
                      <span className="text-slate-400">Entry</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-red-500"></div>
                      <span className="text-slate-400">Stop Loss</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-cyan-400"></div>
                      <span className="text-slate-400">Take Profit</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-yellow-400"></div>
                      <span className="text-slate-400">Support</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-purple-500"></div>
                      <span className="text-slate-400">Resistance</span>
                    </div>
                  </div>

                  <button
                    onClick={() => {
                      setResult(null)
                      setImagePreview(null)
                      setUploadedImage(null)
                    }}
                    className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition"
                  >
                    Upload New Chart
                  </button>
                </div>
              )}
            </div>

            {/* Indicators Grid */}
            {result && result.indicators && Object.keys(result.indicators).length > 0 && (
              <div className="bg-slate-900/50 rounded-xl p-5 border border-slate-800">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                  <span className="text-xl">📈</span> Technical Indicators
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                  {result.indicators.rsi && (
                    <IndicatorCard name="RSI (14)" indicator={result.indicators.rsi} icon="📊" />
                  )}
                  {result.indicators.macd && (
                    <IndicatorCard name="MACD" indicator={result.indicators.macd} icon="📉" />
                  )}
                  {result.indicators.ema_alignment && (
                    <IndicatorCard name="EMA Alignment" indicator={result.indicators.ema_alignment} icon="📈" />
                  )}
                  {result.indicators.price_vs_ema && (
                    <IndicatorCard name="Price vs EMA" indicator={result.indicators.price_vs_ema} icon="🎯" />
                  )}
                  {result.indicators.support_resistance && (
                    <IndicatorCard name="S/R Levels" indicator={result.indicators.support_resistance} icon="🔒" />
                  )}
                  {result.indicators.fibonacci && (
                    <IndicatorCard name="Fibonacci" indicator={result.indicators.fibonacci} icon="🌀" />
                  )}
                  {result.indicators.bollinger && (
                    <IndicatorCard name="Bollinger Bands" indicator={result.indicators.bollinger} icon="📊" />
                  )}
                  {result.indicators.volume && (
                    <IndicatorCard name="Volume" indicator={result.indicators.volume} icon="📦" />
                  )}
                </div>
              </div>
            )}

            {/* Calculated Indicators (from backend) */}
            {result && result.calculated_indicators && (
              <div className="bg-slate-900/50 rounded-xl p-5 border border-slate-800">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                  <span className="text-xl">🔬</span> Calculated Indicators (Binance Data)
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">RSI (14)</div>
                    <div className="text-xl font-bold text-white">{result.calculated_indicators.rsi?.value || '-'}</div>
                    <SignalBadge signal={result.calculated_indicators.rsi?.signal || 'neutral'} />
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">EMA20</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.ema?.ema20?.toFixed(2) || '-'}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">EMA50</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.ema?.ema50?.toFixed(2) || '-'}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">EMA200</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.ema?.ema200?.toFixed(2) || '-'}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">MACD</div>
                    <div className="text-xl font-bold text-white">{result.calculated_indicators.macd?.macd?.toFixed(4) || '-'}</div>
                    <SignalBadge signal={result.calculated_indicators.macd?.trend || 'neutral'} />
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">Bollinger Upper</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.bollinger?.upper?.toFixed(2) || '-'}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">Bollinger Lower</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.bollinger?.lower?.toFixed(2) || '-'}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-slate-400">ATR (14)</div>
                    <div className="text-xl font-bold text-white">${result.calculated_indicators.atr?.toFixed(4) || '-'}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Analysis Results */}
          <div className="space-y-6">
            {error && (
              <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-400">
                <div className="font-bold mb-1">⚠️ Error</div>
                <p className="text-sm">{error}</p>
              </div>
            )}

            {result && (
              <>
                {/* Trade Setup */}
                {result.trade_setup && result.trade_setup.bias && (
                  <TradeSetupCard setup={result.trade_setup} />
                )}

                {/* Confluence Score */}
                {result.confluence_score !== undefined && (
                  <ConfluenceScoreCard 
                    score={result.confluence_score} 
                    breakdown={result.confluence_breakdown} 
                  />
                )}

                {/* Support & Resistance */}
                {(result.support_levels?.length > 0 || result.resistance_levels?.length > 0) && (
                  <div className="bg-slate-900/50 rounded-xl p-5 border border-slate-800">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                      <span className="text-xl">🔐</span> Support & Resistance
                    </h3>
                    
                    <div className="space-y-4">
                      {result.support_levels?.length > 0 && (
                        <div>
                          <div className="text-xs text-yellow-400 font-medium mb-2">SUPPORT LEVELS</div>
                          <div className="space-y-2">
                            {result.support_levels.map((level, idx) => (
                              <div key={idx} className="flex items-center justify-between bg-yellow-500/10 rounded-lg px-3 py-2">
                                <span className="text-white font-medium">
                                  ${level.price >= 1000 ? level.price.toLocaleString() : level.price.toFixed(4)}
                                </span>
                                <span className={`text-xs px-2 py-0.5 rounded ${
                                  level.strength === 'strong' ? 'bg-green-500/20 text-green-400' :
                                  level.strength === 'moderate' ? 'bg-yellow-500/20 text-yellow-400' :
                                  'bg-slate-500/20 text-slate-400'
                                }`}>
                                  {level.strength?.toUpperCase()}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {result.resistance_levels?.length > 0 && (
                        <div>
                          <div className="text-xs text-purple-400 font-medium mb-2">RESISTANCE LEVELS</div>
                          <div className="space-y-2">
                            {result.resistance_levels.map((level, idx) => (
                              <div key={idx} className="flex items-center justify-between bg-purple-500/10 rounded-lg px-3 py-2">
                                <span className="text-white font-medium">
                                  ${level.price >= 1000 ? level.price.toLocaleString() : level.price.toFixed(4)}
                                </span>
                                <span className={`text-xs px-2 py-0.5 rounded ${
                                  level.strength === 'strong' ? 'bg-green-500/20 text-green-400' :
                                  level.strength === 'moderate' ? 'bg-yellow-500/20 text-yellow-400' :
                                  'bg-slate-500/20 text-slate-400'
                                }`}>
                                  {level.strength?.toUpperCase()}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Market Data */}
                {result.binance_data && (
                  <div className="bg-slate-900/50 rounded-xl p-5 border border-slate-800">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                      <span className="text-xl">📈</span> Market Data
                    </h3>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-slate-400">24h High</div>
                        <div className="text-lg font-bold text-green-400">
                          ${result.binance_data.high_24h >= 1000 
                            ? result.binance_data.high_24h.toLocaleString() 
                            : result.binance_data.high_24h.toFixed(4)}
                        </div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-slate-400">24h Low</div>
                        <div className="text-lg font-bold text-red-400">
                          ${result.binance_data.low_24h >= 1000 
                            ? result.binance_data.low_24h.toLocaleString() 
                            : result.binance_data.low_24h.toFixed(4)}
                        </div>
                      </div>
                      <div className="col-span-2 bg-slate-800/50 rounded-lg p-3">
                        <div className="text-slate-400">24h Volume</div>
                        <div className="text-lg font-bold text-white">
                          ${result.binance_data.quote_volume?.toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Analysis Summary */}
                {(result.analysis_summary || result.trade_rationale) && (
                  <div className="bg-slate-900/50 rounded-xl p-5 border border-slate-800">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                      <span className="text-xl">📝</span> Analysis Summary
                    </h3>
                    {result.analysis_summary && (
                      <p className="text-slate-300 mb-4">{result.analysis_summary}</p>
                    )}
                    {result.trade_rationale && (
                      <div className="bg-slate-800/50 rounded-lg p-4">
                        <div className="text-xs text-cyan-400 font-medium mb-2">TRADE RATIONALE</div>
                        <p className="text-sm text-slate-400">{result.trade_rationale}</p>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Empty State */}
            {!result && !analyzing && !error && (
              <div className="bg-slate-900/50 rounded-xl p-8 border border-slate-800 text-center">
                <div className="text-5xl mb-4">📊</div>
                <h3 className="text-lg font-medium text-white mb-2">Ready to Analyze</h3>
                <p className="text-slate-400 text-sm">
                  Upload a chart screenshot and click Analyze to get professional-grade technical analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-12 py-6">
        <div className="max-w-[1800px] mx-auto px-6 text-center text-slate-500 text-sm">
          Trading Analyzer Pro • AI-Powered Technical Analysis • Not Financial Advice
        </div>
      </footer>
    </div>
  )
}
