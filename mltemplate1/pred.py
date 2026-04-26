def load_metadata():
    """
    Scan MODEL_DIR for market subfolders, load each market's model_config.json.
    Returns dict: { "HongKong": metadata_dict, "Singapore": metadata_dict, ... }
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
        print(f"✅ Loaded metadata for market: {entry.name}")

    if not all_metadata:
        raise FileNotFoundError(f"No market subfolders with model_config.json found in {MODEL_DIR}")

    return all_metadata

def load_models(all_metadata):
    """
    Load all model files for all markets.
    Returns dict: { "HongKong": { key: model }, "Singapore": { key: model }, ... }
    """
    all_models = {}
    for market, metadata in all_metadata.items():
        market_dir = os.path.join(MODEL_DIR, market)
        models     = {}
        for key, model_info in metadata.get("models", {}).items():
            model_file = model_info.get("model_file")
            if not model_file:
                print(f"⚠️ Market '{market}' key '{key}' missing model_file, skipping.")
                continue
            local_path = os.path.join(market_dir, model_file)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Model file not found: {local_path}")
            print(f"📥 Loading [{market}] model: {local_path}")
            models[key] = joblib.load(local_path)
            print(f"✅ Loaded [{market}] key: {key}")
        all_models[market] = models

    return all_models


def process_data(config_params=None, market=None):

    # ── Resolve market key (folder name uses underscore) ─────────
    market_key = market.replace(" ", "_") if market else None
    if not market_key or market_key not in ALL_METADATA:
        available = list(ALL_METADATA.keys())
        raise ValueError(f"Market '{market}' not found. Available: {available}")

    metadata = ALL_METADATA[market_key]   # ← was: load_metadata()
    models   = ALL_MODELS[market_key]     # ← was: load_models(metadata)

    # rest of process_data stays exactly the same from here
    TEXT_COL = metadata.get("text_column")
    ...
