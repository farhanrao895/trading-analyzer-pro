import { NextResponse } from 'next/server'

// Forward to Python backend for indicator calculation
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'

export async function GET(
  request: Request,
  { params }: { params: { symbol: string; interval: string } }
) {
  try {
    const { symbol, interval } = params
    
    console.log(`[Next.js API] Forwarding /api/indicators/${symbol}/${interval} to Python backend: ${BACKEND_URL}`)
    
    // Forward request to Python backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/indicators/${symbol}/${interval}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store'
    })
    
    if (!backendResponse.ok) {
      console.error(`[Next.js API] Backend returned ${backendResponse.status}`)
      return NextResponse.json(
        { error: `Backend returned ${backendResponse.status}` },
        { status: backendResponse.status }
      )
    }
    
    const data = await backendResponse.json()
    console.log(`[Next.js API] Got response from backend for ${symbol}/${interval}`)
    
    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'no-store, no-cache, must-revalidate',
      }
    })
  } catch (error: any) {
    console.error(`[Next.js API] Error forwarding to backend:`, error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch indicators from backend' },
      { status: 500 }
    )
  }
}

