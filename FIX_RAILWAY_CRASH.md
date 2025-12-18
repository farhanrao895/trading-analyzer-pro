# 🔧 Fix Railway Backend Crash

## ❌ **Status**: Backend Crashed

Your Railway service "trading-analyzer-pro" has crashed. Let's fix it!

---

## 📋 **Step 1: Check Logs**

1. **In Railway dashboard**, click on "trading-analyzer-pro" service
2. **Go to**: "Deployments" tab
3. **Click** on the crashed deployment (the red one)
4. **Click**: "Deploy Logs" tab
5. **Scroll down** to see the error message

**Common errors:**
- Missing `GEMINI_API_KEY`
- Python import errors
- Port configuration issues
- Missing dependencies

---

## 📋 **Step 2: Check Environment Variables**

1. **Go to**: "Variables" tab
2. **Make sure** `GEMINI_API_KEY` is set
3. **If missing**, add it:
   - Click "New Variable"
   - Key: `GEMINI_API_KEY`
   - Value: (your Gemini API key)
   - Click "Add"

---

## 📋 **Step 3: Restart Service**

1. **Go back to**: "Deployments" tab
2. **Click**: Red "Restart" button
3. **Wait 2-3 minutes** for restart

---

## 📋 **Step 4: Verify It's Working**

1. **Test the health endpoint**:
   ```
   https://trading-analyzer-pro-production-236c.up.railway.app/api/health
   ```
2. **Should see**: `{"status":"healthy",...}`

---

## 🆘 **If Still Crashing**

**Share the error from the logs** and I'll help fix it!

Common fixes:
- Add missing environment variables
- Fix Python code errors
- Update dependencies

