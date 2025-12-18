# 🔧 Fix Railway Deployment

## ❌ **Problem**: Railway is trying to build Next.js instead of Python backend

## ✅ **Solution**: Configure Railway to use backend directory

---

## 📋 **Step 1: Update Railway Settings**

1. **In Railway dashboard**, click on your service ("web")
2. **Go to**: "Settings" tab
3. **Scroll down** to "Root Directory"
4. **Set Root Directory**: `backend`
5. **Click**: "Save"

---

## 📋 **Step 2: Update Start Command**

1. **Still in Settings**, scroll to "Deploy"
2. **Start Command**: 
   ```
   python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
3. **Click**: "Save"

---

## 📋 **Step 3: Redeploy**

1. **Go to**: "Deployments" tab
2. **Click**: "Redeploy" (or push a new commit to trigger redeploy)

---

## ✅ **Alternative: Delete and Recreate Service**

If the above doesn't work:

1. **Delete** the current "web" service
2. **Create new service**: "New" → "GitHub Repo"
3. **Select**: Your repo
4. **In service settings**:
   - **Root Directory**: `backend`
   - **Start Command**: `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Add variable**: `GEMINI_API_KEY`

---

## 🎯 **What Should Happen**

After fixing, Railway should:
- ✅ Install Python dependencies from `backend/requirements.txt`
- ✅ Start FastAPI server
- ✅ Show "Deployed" status

**Let me know if you need help with any step!**

