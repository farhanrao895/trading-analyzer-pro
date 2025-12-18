# ▲ Deploy Frontend to Vercel - Step by Step

## ✅ **Prerequisites**

- ✅ Code is on GitHub
- ✅ Backend is deployed on Railway
- ✅ You have your Railway backend URL

---

## 📋 **Step-by-Step Vercel Deployment**

### Step 1: Install Vercel CLI

**Open terminal in your project folder:**

```bash
cd "D:\Trading Analyzer"
npm i -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

- Browser opens automatically
- **Click**: "Authorize Vercel"
- **Return to terminal** - you should see "Success!"

### Step 3: Set Backend URL

**Use your Railway URL from the backend deployment:**

```bash
vercel env add NEXT_PUBLIC_BACKEND_URL production
```

**When prompted:**
- **Paste your Railway URL** (e.g., `https://trading-analyzer-production.up.railway.app`)
- **Press Enter**
- **Select**: `Production` when asked

### Step 4: Deploy!

```bash
vercel --prod
```

**Answer the prompts:**
- "Set up and deploy?": Type **Y** and press Enter
- "Which scope?": **Select your account** (press Enter if only one)
- "Link to existing project?": Type **N** and press Enter
- "What's your project's name?": Press **Enter** (uses default: `trading-analyzer`)
- "In which directory is your code located?": Press **Enter** (uses `./`)

**Wait 2-3 minutes** for deployment...

### Step 5: Get Your Shareable Link!

**At the end, you'll see:**
```
✅ Production: https://trading-analyzer.vercel.app
```

**🎉 This is your shareable link!**

---

## ✅ **Frontend Deployed!**

Your app is now live at: `https://your-project.vercel.app`

**Share this link with your friends!** 🚀

---

## 🔄 **Auto-Deploy (Bonus)**

Both Railway and Vercel will automatically redeploy when you push to GitHub!

Just run:
```bash
git add .
git commit -m "Update"
git push
```

Both services will automatically update! ✨

---

## 🆘 **Troubleshooting**

**Can't connect to backend?**
- Check `NEXT_PUBLIC_BACKEND_URL` is set: `vercel env ls`
- Make sure Railway URL is correct (no trailing slash)
- Check Railway backend is running

**Deployment failed?**
- Check Vercel dashboard for error logs
- Make sure all files are committed to GitHub

**Need help?** Tell me what error you see!

