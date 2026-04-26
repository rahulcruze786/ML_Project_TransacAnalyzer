ALL_METADATA = None
ALL_MODELS   = None

@app.before_request
def load_models_once():
    global ALL_METADATA, ALL_MODELS
    if ALL_METADATA is None:
        print("🚀 Loading all market models...")
        ALL_METADATA = load_metadata()
        ALL_MODELS   = load_models(ALL_METADATA)
        print(f"✅ Markets loaded: {list(ALL_MODELS.keys())}")
