Looking at your `app.py`, here are the exact changes needed — **4 changes total**.

---

## Change 1 — Replace `load_metadata()` — scan all market subfolders

**Where:** Replace the entire `load_metadata()` function

**Current:**
```python
def load_metadata():
    local_config = os.path.join(MODEL_DIR, "model_config.json")
    if not os.path.exists(local_config):
        raise FileNotFoundError(f"Local model_config.json not found at {local_config}")
    print(f"📥 Loading metadata from local model folder: {local_config}")
    with open(local_config, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Metadata loaded: {len(metadata.get('models', {}))} models found")
    return metadata
```

**Change to:**
```python
def load_metadata():
    """
    Scan MODEL_DIR for market subfolders, load each market's model_config.json.
    Returns dict: { "Hong_Kong": metadata_dict, "Singapore": metadata_dict, ... }
    """
    all_metadata = {}
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    for entry in os.scandir(MODEL_DIR):
        if not entry.is_dir():
            continue
        config_path = os.path.join(entry.path, "model_config.json")
        if not os.path.exists(config_path):
            print(f"⚠️ No model_config.json in {entry.path}, skipping.")
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            all_metadata[entry.name] = json.load(f)
        print(f"✅ Metadata loaded for market: {entry.name} — {len(all_metadata[entry.name].get('models', {}))} models")

    if not all_metadata:
        raise FileNotFoundError(f"No market subfolders with model_config.json found in {MODEL_DIR}")

    return all_metadata
```

---

## Change 2 — Replace `load_models()` — load all markets

**Where:** Replace the entire `load_models()` function

**Current:**
```python
def load_models(metadata):
    models = {}
    for key, model_info in metadata.get("models", {}).items():
        model_file = model_info.get("model_file")
        if not model_file:
            print(f"⚠️ metadata for key '{key}' missing 'model_file', skipping")
            continue
        local_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file listed in metadata not found: {local_path}")
        print(f"📥 Loading local model: {local_path}")
        models[key] = joblib.load(local_path)
        print(f"✅ Loaded model for key: {key}")
    return models
```

**Change to:**
```python
def load_models(all_metadata):
    """
    Load all model files for all markets.
    Returns dict: { "Hong_Kong": { key: model }, "Singapore": { key: model }, ... }
    """
    all_models = {}
    for market, metadata in all_metadata.items():
        market_dir = os.path.join(MODEL_DIR, market)
        models     = {}
        for key, model_info in metadata.get("models", {}).items():
            model_file = model_info.get("model_file")
            if not model_file:
                print(f"⚠️ [{market}] key '{key}' missing model_file, skipping.")
                continue
            local_path = os.path.join(market_dir, model_file)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Model file not found: {local_path}")
            print(f"📥 Loading [{market}] model: {local_path}")
            models[key] = joblib.load(local_path)
            print(f"✅ Loaded [{market}] key: {key}")
        all_models[market] = models
    return all_models
```

---

## Change 3 — Add startup load below `MODEL_DIR` line

**Where:** Right after this existing line:
```python
MODEL_DIR = os.environ.get("MODEL_PATH", "/mnt/models")
```

**Add:**
```python
# ── Load all market models once at startup ────────────────────
print("🚀 Loading all market models at startup...")
ALL_METADATA = load_metadata()
ALL_MODELS   = load_models(ALL_METADATA)
print(f"✅ Markets loaded: {list(ALL_MODELS.keys())}")
```

---

## Change 4 — Update `process_data()` — replace load calls + add market lookup

**Where:** Inside `process_data()`, Step 1 and Step 2

**Current Step 1:**
```python
# ── Step 1: Load models + metadata ───────────────
metadata = load_metadata()
models   = load_models(metadata)
```

**Current Step 2 starts with:**
```python
TEXT_COL = metadata.get("text_column")
```

**Change Step 1 and Step 2 to:**
```python
# ── Step 1: Resolve market → pick correct metadata + models ──
market_key = market.replace(" ", "_") if market else None
if not market_key or market_key not in ALL_METADATA:
    available = list(ALL_METADATA.keys())
    raise ValueError(f"❌ Market '{market}' not found. Available markets: {available}")

metadata = ALL_METADATA[market_key]
models   = ALL_MODELS[market_key]

# ── Step 2: Extract ALL training values from metadata ─────
# rest of Step 2 stays exactly the same
TEXT_COL = metadata.get("text_column")
```

---

## Change 5 — Update `/v1/getMetadata` route

**Where:** The `get_metadata` route at the bottom

**Current:**
```python
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    try:
        metadata = load_metadata()
        return jsonify({"status": "success", "data": metadata})
    except FileNotFoundError as e:
        return make_response(jsonify({"status": "error", "message": str(e)}), 404)
    except Exception as e:
        return make_response(jsonify({"status": "error", "message": str(e)}), 500)
```

**Change to:**
```python
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    """Return metadata for a specific market or all markets."""
    try:
        market     = request.args.get('market', None)
        if market:
            market_key = market.replace(" ", "_")
            if market_key not in ALL_METADATA:
                return make_response(jsonify({
                    "status":  "error",
                    "message": f"Market '{market}' not found. Available: {list(ALL_METADATA.keys())}"
                }), 404)
            return jsonify({"status": "success", "data": ALL_METADATA[market_key]})
        # No market specified — return all
        return jsonify({"status": "success", "data": ALL_METADATA})
    except Exception as e:
        return make_response(jsonify({"status": "error", "message": str(e)}), 500)
```

---

## Summary

| # | Where | What |
|---|---|---|
| 1 | `load_metadata()` | Full replacement — scans market subfolders |
| 2 | `load_models()` | Full replacement — loads per market into nested dict |
| 3 | Below `MODEL_DIR =` line | Add `ALL_METADATA` + `ALL_MODELS` startup load |
| 4 | `process_data()` Step 1 | Replace `load_metadata/load_models` calls with dict lookup by market key |
| 5 | `/v1/getMetadata` route | Accept optional `?market=` param, return all if not specified |

Nothing else in `process_data()` changes — Steps 3 through 18, all overwrite rules, net off logic, and all other routes stay exactly the same.
