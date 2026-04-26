Here are the exact changes needed — **3 changes only**:

---

## Change 1 — Replace module-level startup block with lazy globals

**Where:** Find and replace this block (currently below `load_models()` function):

**Current:**
```python
# ── Load all market models once at startup ────────────────────
print("🚀 Loading all market models at startup...")
ALL_METADATA = load_metadata()
ALL_MODELS   = load_models(ALL_METADATA)
print(f"✅ Markets loaded: {list(ALL_MODELS.keys())}")
```

**Change to:**
```python
# ── Lazy-loaded globals — populated on first request ──────────
ALL_METADATA = None
ALL_MODELS   = None

def _ensure_models_loaded():
    """
    Load all market models if not already loaded.
    Called at the start of every request that needs models.
    Only loads once — guarded by None check.
    """
    global ALL_METADATA, ALL_MODELS
    if ALL_METADATA is not None:
        return
    print("🚀 Loading all market models...")
    ALL_METADATA = load_metadata()
    ALL_MODELS   = load_models(ALL_METADATA)
    print(f"✅ Markets loaded: {list(ALL_MODELS.keys())}")
```

---

## Change 2 — Call `_ensure_models_loaded()` at top of `process_data()`

**Where:** First line inside `process_data()`, before Step 1

**Current:**
```python
def process_data(config_params=None, market=None):
    start  = time.time()
    schema = 'T1_TXNANAL_PHY'

    # ── Step 1: Resolve market → pick correct metadata + models ──
    market_key = market.replace(" ", "_") if market else None
```

**Change to:**
```python
def process_data(config_params=None, market=None):
    start  = time.time()
    schema = 'T1_TXNANAL_PHY'

    # ── Step 1: Ensure models are loaded ─────────────────────
    _ensure_models_loaded()

    # ── Step 2: Resolve market → pick correct metadata + models ──
    market_key = market.replace(" ", "_") if market else None
```

---

## Change 3 — Call `_ensure_models_loaded()` in `get_metadata` route

**Where:** First line inside the `get_metadata()` route's try block

**Current:**
```python
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    """Return metadata for a specific market or all markets."""
    try:
        market = request.args.get('market', None)
```

**Change to:**
```python
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    """Return metadata for a specific market or all markets."""
    try:
        _ensure_models_loaded()
        market = request.args.get('market', None)
```

---

## Summary

| # | Where | What |
|---|---|---|
| 1 | Module-level startup block | Replace `ALL_METADATA = load_metadata()` with lazy globals + `_ensure_models_loaded()` function |
| 2 | `process_data()` top | Add `_ensure_models_loaded()` as first call |
| 3 | `get_metadata()` route | Add `_ensure_models_loaded()` as first call |

Nothing else changes. Every other route (`/v1/process`, `/v1/health`, `/v1/getConfig`, etc.) stays exactly the same.
