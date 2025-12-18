# ⚡ Quick Command Reference

## After Creating GitHub Repo:

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/trading-analyzer-pro.git
git branch -M main
git push -u origin main
```

## After Railway Deployment:

**Copy your Railway URL** (from Railway Settings → Public Domain)

## Vercel Deployment:

```bash
# Install Vercel CLI (if not installed)
npm i -g vercel

# Login
vercel login

# Add backend URL (paste Railway URL when prompted)
vercel env add NEXT_PUBLIC_BACKEND_URL production

# Deploy
vercel --prod
```

**Done!** You'll get your shareable link! 🎉

