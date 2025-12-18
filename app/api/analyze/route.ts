import { NextRequest, NextResponse } from 'next/server'

// Get backend URL from environment or use default
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8002'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const symbol = (formData.get('symbol') as string) || 'BTCUSDT'
    const timeframe = (formData.get('timeframe') as string) || '4h'

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    // Forward to Python backend
    const analyzeFormData = new FormData()
    analyzeFormData.append('file', file)
    analyzeFormData.append('symbol', symbol)
    analyzeFormData.append('timeframe', timeframe)

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
