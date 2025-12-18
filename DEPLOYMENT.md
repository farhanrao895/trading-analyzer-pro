# Vercel Deployment Guide

## 📊 Cost Summary

### ✅ **NO ADDITIONAL VERCEL COSTS** (Free Tier)
- **Vercel Hobby (Free)**: Unlimited requests, 100GB bandwidth/month
- **Serverless Functions**: 10s execution limit (may need Pro for 60s)
- **Bandwidth**: 100GB/month free

### 💰 **Existing Costs** (Same as before)
- **Gemini API**: ~$0.00025 per analysis
- **Binance API**: Free (public data)

### ⚠️ **Potential Upgrade Needed**
If analysis takes > 10 seconds, you may need:
- **Vercel Pro**: $20/month (60s execution time, better performance)

---

## 🚀 Deployment Steps

### Option 1: Full Vercel Deployment (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Set Environment Variables**:
   ```bash
   vercel env add GEMINI_API_KEY
   # Paste your Gemini API key when prompted
   ```

4. **Deploy**:
   ```bash
   vercel
   ```

5. **For Production**:
   ```bash
   vercel --prod
   ```

### Option 2: Hybrid Deployment (Backend Separate)

If Python serverless functions are too slow:

1. **Deploy Frontend to Vercel** (as above)
2. **Deploy Backend to Railway/Render** (free tier available)
3. **Update API_URL** in frontend to point to backend

---

## 📁 Project Structure for Vercel

```
├── app/
│   ├── api/              # Next.js API routes (TypeScript)
│   │   ├── health/
│   │   ├── symbols/
│   │   ├── price/[symbol]/
│   │   ├── klines/[symbol]/[interval]/
│   │   ├── depth/[symbol]/
│   │   └── analyze/      # Proxies to Python function
│   ├── page.tsx
│   └── layout.tsx
├── api/
│   └── analyze.py        # Python serverless function
├── vercel.json          # Vercel configuration
├── package.json
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Configuration

### `vercel.json`
- Sets Python runtime for `api/analyze.py`
- Increases timeout to 30 seconds
- Configures CORS headers

### Environment Variables (Set in Vercel Dashboard)
- `GEMINI_API_KEY`: Your Gemini API key

---

## 🔧 Troubleshooting

### Issue: Function timeout (>10s)
**Solution**: Upgrade to Vercel Pro ($20/month) for 60s timeout

### Issue: Python dependencies not found
**Solution**: Create `requirements.txt` in root with all dependencies

### Issue: Image processing fails
**Solution**: Ensure OpenCV and PIL are in requirements.txt

### Issue: CORS errors
**Solution**: Check `vercel.json` headers configuration

---

## 📝 Notes

1. **First deployment** may take 5-10 minutes (build time)
2. **Cold starts** for Python functions: ~2-3 seconds
3. **Warm functions**: Much faster subsequent requests
4. **Monitor usage** in Vercel dashboard

---

## 🎯 Recommended Setup

For best performance and cost:
- **Frontend**: Vercel (free tier)
- **Backend API routes**: Vercel (free tier)
- **Heavy processing**: Vercel Pro or separate Python service

This gives you:
- ✅ Zero additional cost (or $20/month for Pro)
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Easy deployments
- ✅ No server management

