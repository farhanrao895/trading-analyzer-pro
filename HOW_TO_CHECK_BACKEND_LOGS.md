# How to Check Backend Logs

## 🔍 Where to Find Backend Logs

The **browser console** (F12 → Console tab) only shows **frontend JavaScript logs**. 

**Python print statements appear in the BACKEND SERVER TERMINAL**, not in the browser!

---

## 📍 Step 1: Find the Backend Server Terminal

The backend runs in a **separate terminal/command prompt window**. Look for:

### Windows:
- A window titled **"Trading Analyzer - Backend"** or **"Backend"**
- Or a CMD/PowerShell window running `uvicorn`
- Usually shows: `INFO:     Uvicorn running on http://0.0.0.0:8002`

### How to Find It:
1. **Check all open terminal windows** - the backend runs in its own window
2. Look for a window showing:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8002
   INFO:     Application startup complete.
   ```
3. If you can't find it, **restart the backend**:
   - Run `start_backend.bat` or `start_both.bat`
   - Or manually: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload`

---

## 📋 Step 2: What to Look For in Backend Logs

When you click **"Analyze Chart"** in the UI, you should see these debug messages in the **backend terminal**:

```
[API] /api/analyze called for ACTUSDT 4h
[API] Processing 500 klines...
[S/R] ===== find_support_resistance() CALLED =====
[S/R] Input: 500 klines, num_levels=3, recency_weight=0.7
[S/R] Current price: $0.0392, Total candles: 500
[S/R DEBUG] Current price: $0.0392
[S/R DEBUG] Max support distance: $0.0078 (20%)
[S/R DEBUG] Min valid support: $0.0314
[S/R DEBUG] Before filter - Support levels: X, Resistance levels: Y
[S/R DEBUG] All support levels before filter: [...]
[S/R DEBUG] Breakout detection check:
[S/R DEBUG]   Potential resistance: $0.0XXX
[S/R DEBUG] FILTERED OUT X supports below min_valid: [...]
[S/R DEBUG] After filter - Support levels: X, Resistance levels: Y
[S/R] Found X support levels: [...]
[S/R] ===== find_support_resistance() COMPLETE =====
[API] /api/analyze - Current price: $0.0392
[API] /api/analyze - Supports received: X
[API] /api/analyze - Support prices: [...]
[API] /api/analyze - Calling calculate_long_trade_setup()...
[TradeSetup] Current: $0.0392, Min reasonable entry: $0.0353, Ideal entry min: $0.0XXX
[TradeSetup] FINAL ENTRY: $0.0XXX via support/fvg/order_block/pullback
[API] /api/analyze - Trade setup entry: $0.0XXX
```

---

## 🌐 Step 3: Check Which API Endpoint is Being Called

### In Browser DevTools:

1. **Open DevTools** (F12)
2. Go to **Network** tab
3. Click **"Analyze Chart"** in the UI
4. Look for API calls:
   - `/api/analyze` - Full analysis (should show our debug logs)
   - `/api/indicators/ACTUSDT/4h` - Just indicators (also has debug logs)

### Check the Response:

1. Click on the API call in Network tab
2. Go to **Response** tab
3. Look for `_debug` field (if present):
   ```json
   {
     "_debug": {
       "code_version": "v2.0_fixed_support_detection",
       "support_count": 3,
       "current_price": 0.0392
     }
   }
   ```

---

## 🔧 Step 4: Verify Backend is Running New Code

### Quick Test:
1. **Restart the backend server** (Ctrl+C, then restart)
2. Look for this in backend terminal when it starts:
   ```
   INFO:     Application startup complete.
   ```
3. If you see `[S/R] ===== find_support_resistance() CALLED =====` when analyzing, the new code is running!

### If You Don't See Debug Logs:

**Problem:** Backend server wasn't restarted after code changes.

**Solution:**
1. Stop the backend (Ctrl+C in backend terminal)
2. Restart it:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
   ```
3. The `--reload` flag should auto-reload, but sometimes you need a manual restart

---

## 🐛 Step 5: Common Issues

### Issue 1: No Backend Logs Appearing
- **Cause:** Backend server not running or wrong terminal
- **Fix:** Restart backend, check terminal window

### Issue 2: Old Support Levels Still Showing
- **Cause:** Frontend caching or wrong endpoint
- **Fix:** 
  - Hard refresh browser (Ctrl+Shift+R)
  - Check Network tab to see which endpoint is called
  - Verify backend logs show new code running

### Issue 3: Still Seeing $0.0207
- **Check backend logs for:**
  - `[S/R DEBUG] FILTERED OUT` - should show $0.0207 being filtered
  - `[S/R] Found X support levels:` - should NOT include $0.0207
  - If $0.0207 appears in "Found" but was "FILTERED OUT", there's a bug

---

## 📸 Quick Visual Guide

```
┌─────────────────────────────────────┐
│  Browser (Frontend)                 │
│  - F12 → Console: JavaScript logs   │
│  - F12 → Network: API calls          │
└─────────────────────────────────────┘
              ↓ HTTP requests
┌─────────────────────────────────────┐
│  Backend Terminal (Python)          │
│  - Shows [S/R] debug messages       │
│  - Shows [API] debug messages        │
│  - Shows [TradeSetup] messages       │
└─────────────────────────────────────┘
```

---

## ✅ Verification Checklist

- [ ] Backend terminal window is open and showing logs
- [ ] When clicking "Analyze Chart", backend terminal shows `[S/R] ===== find_support_resistance() CALLED =====`
- [ ] Backend logs show `[S/R DEBUG] FILTERED OUT` with old supports like $0.0207
- [ ] Backend logs show final supports are recent (above $0.0314 for ACTUSDT at $0.0392)
- [ ] Network tab shows `/api/analyze` or `/api/indicators` being called
- [ ] Response includes `_debug` field with `code_version: "v2.0_fixed_support_detection"`

---

## 🆘 Still Not Working?

If you still see old support levels after checking all the above:

1. **Take a screenshot** of:
   - Backend terminal logs (the full output)
   - Network tab showing the API call
   - Response JSON from the API

2. **Check if multiple backend instances are running:**
   - Close ALL terminal windows
   - Restart backend fresh
   - Make sure only ONE backend is running on port 8002

3. **Verify the code is actually saved:**
   - Check `backend/main.py` line ~1129 - should have `recency_weight` parameter
   - Check `backend/main.py` line ~1297 - should have `max_support_distance = current_price * 0.20`

