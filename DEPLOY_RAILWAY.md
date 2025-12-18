# 🚂 Deploy Backend to Railway - Step by Step

## ✅ **Status: Code is on GitHub!**

Your repository: https://github.com/farhanrao895/trading-analyzer-pro

---

## 📋 **Step-by-Step Railway Deployment**

### Step 1: Sign Up to Railway

1. **Go to**: https://railway.app
2. **Click**: "Login" (top right)
3. **Click**: "Login with GitHub"
4. **Authorize** Railway to access your GitHub account

### Step 2: Create New Project

1. **Click**: "New Project" (big button)
2. **Click**: "Deploy from GitHub repo"
3. **Select**: `farhanrao895/trading-analyzer-pro`
4. **Railway will automatically**:
   - Detect Python
   - Install dependencies from `requirements.txt`
   - Start the backend server

**Wait 2-3 minutes** for the first deployment...

### Step 3: Add Environment Variable

1. **Click** on your service (the deployed app)
2. **Go to**: "Variables" tab
3. **Click**: "New Variable"
4. **Add**:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: (paste your Gemini API key here)
5. **Click**: "Add"

**Railway will automatically redeploy** with the new variable!

### Step 4: Get Your Backend URL

1. **Go to**: "Settings" tab
2. **Scroll down** to "Networking"
3. **Click**: "Generate Domain" (if not already generated)
4. **Copy the URL** (e.g., `https://trading-analyzer-production.up.railway.app`)

**✅ Save this URL - you'll need it for Vercel!**

---

## ✅ **Backend Deployed!**

Your backend is now live at: `https://your-backend.railway.app`

**Next**: Deploy frontend to Vercel (see DEPLOY_VERCEL.md)

---

## 🆘 **Troubleshooting**

**Deployment failed?**
- Check "Deployments" tab → Click latest → View logs
- Make sure `requirements.txt` exists and has all dependencies

**Backend not working?**
- Check if `GEMINI_API_KEY` is set in Variables
- Check logs for errors

**Need help?** Tell me what error you see!

