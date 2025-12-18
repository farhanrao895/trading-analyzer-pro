# 🚀 Deploy to Vercel (Shareable Link)

## Quick Deploy - 2 Steps

### 1️⃣ Deploy Backend to Railway (2 minutes)

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variable: `GEMINI_API_KEY` = your key
6. Copy the Railway URL (e.g., `https://your-app.railway.app`)

### 2️⃣ Deploy Frontend to Vercel (1 command)

```bash
npm i -g vercel
vercel login
vercel env add NEXT_PUBLIC_BACKEND_URL
# Paste your Railway URL when prompted
vercel --prod
```

**Done!** Share your Vercel link: `https://your-project.vercel.app`

---

## ✅ Cost: $0 (Both free tiers)

## ✅ No Hassle: Auto-detects everything

## ✅ Works: Exactly like it does now

