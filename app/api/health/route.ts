import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    model: process.env.GEMINI_API_KEY ? 'gemini-2.5-flash' : 'none',
    timestamp: new Date().toISOString()
  })
}

