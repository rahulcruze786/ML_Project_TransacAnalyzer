# ── Standard library imports ──────────────────────────────────
import os      # used to read environment variables and build file paths
import json    # used to parse the .json config file into a Python dict
import joblib  # used to deserialize trained sklearn .pkl model files
               # at deployment time, SAP AI Core authenticates into
               # AWS S3 (ap-southeast-1) using the Object Store Secret,
               # resolves ai://default/<execution_id>/ALL_MARKET_NB_MODEL
               # and copies the .pkl files into /mnt/models/ inside the container

# ── Model directory path ──────────────────────────────────────
# SAP AI Core injects MODEL_PATH environment variable at container startup
# pointing to /mnt/models/ — where .pkl files were copied from
# AWS S3 bucket (ap-southeast-1) under path prefix ai/default/
# If MODEL_PATH is not set (e.g. running locally), falls back to "/mnt/models"
MODEL_DIR = os.environ.get("MODEL_PATH", "/mnt/models")


def load_metadata(market):
    """
    Reads the model configuration JSON file for the given market from MODEL_DIR.
    This file is produced by the trainer after a successful training run and
    stored in S3 — SAP AI Core copies it into /mnt/models/ at deployment time
    alongside the .pkl model files.

    Returns a Python dict containing:
        - text_column    : column name used as input text during training
        - target_column  : column name used as the label during training
        - key_cols       : columns used to build the model lookup key
        - process_types  : GL account process types to filter on
        - seasonal_words : comma-separated keywords for text preprocessing
        - allowed_class  : list of valid prediction output classes
        - models         : dict of { key: { model_file: "filename.pkl" } }
    """

    # Guard: market name is required to know which config file to load
    # e.g. "Singapore" → looks for model_config_Singapore.json
    if not market:
        raise ValueError("Market is required to load metadata")

    # Sanitize the market name so it is safe to use as part of a filename
    # spaces and slashes in market names would break the file path
    # e.g. "Hong Kong" → "Hong_Kong",   "SG/HK" → "SG_HK"
    safe_market = str(market).replace("/", "_").replace(" ", "_")

    # Build the full path to the market-specific config file inside the container
    # e.g. "/mnt/models/model_config_Singapore.json"
    # this path exists because SAP AI Core copied it from S3 at deployment time
    local_config = os.path.join(MODEL_DIR, f"model_config_{safe_market}.json")

    # Verify the config file was actually copied into the container before opening
    # gives a clear error message instead of a cryptic Python IOError later
    if not os.path.exists(local_config):
        raise FileNotFoundError(
            f"No metadata config found for market '{market}' at {local_config}"
        )

    # Open the config file in read mode with UTF-8 encoding
    # 'with' block ensures the file handle is automatically closed after reading
    # even if an exception occurs mid-read
    with open(local_config, "r", encoding="utf-8") as f:
        # Parse the JSON file contents into a Python dictionary
        # json.load() converts JSON → Python dict (not yet JSON again until jsonify())
        # e.g. {"text_column": "TEXT(S4Journal)", "models": {"TRAVEL": {...}}}
        metadata = json.load(f)

    # Log the number of models found in metadata for quick verification at startup
    # .get("models", {}) safely returns {} if "models" key is missing — avoids KeyError
    print(f"✅ Metadata loaded: {len(metadata.get('models', {}))} models found")

    # Return the metadata as a plain Python dict to the caller (pipeline.py)
    # it becomes JSON only later when jsonify() is called in the Flask route
    return metadata


def load_models(metadata):
    """
    Loads all trained sklearn pipeline (.pkl) files listed in the metadata dict.
    Each .pkl file was saved to S3 after training and copied into /mnt/models/
    by SAP AI Core at deployment time when resolving the artifact URL:
    ai://default/<execution_id>/ALL_MARKET_NB_MODEL

    Returns a dict: { key: <loaded sklearn pipeline object> }
    e.g. { "TRAVEL": <Pipeline>, "MEALS": <Pipeline>, "HOTEL": <Pipeline> }
    These pipeline objects are used directly in predict() to run inference.
    """

    # Empty dict to accumulate loaded models
    # final structure: { "TRAVEL": <pipeline>, "MEALS": <pipeline>, ... }
    models = {}

    # Loop over every model entry defined inside metadata["models"]
    # .get("models", {}) → safely returns {} if "models" key is missing (no KeyError)
    # .items()           → yields each entry as a (key, model_info) tuple
    # e.g. key="TRAVEL", model_info={"model_file": "model_travel.pkl"}
    for key, model_info in metadata.get("models", {}).items():

        # Extract the .pkl filename for this model from its metadata entry
        # e.g. "model_travel.pkl"
        model_file = model_info.get("model_file")

        # If model_file is missing or empty in metadata, skip this entry silently
        # prevents crash when metadata has an incomplete model entry
        if not model_file:
            print(f"⚠️ No model_file found for key '{key}', skipping")
            continue

        # Build the full path to the .pkl file inside the container
        # e.g. "/mnt/models/model_travel.pkl"
        # this file exists because SAP AI Core copied it from S3 at deployment time
        local_path = os.path.join(MODEL_DIR, model_file)

        # Verify the .pkl file was actually copied into the container before loading
        # gives a clear error instead of a cryptic joblib/pickle error later
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file not found: {local_path}")

        # Deserialize the sklearn pipeline from the .pkl file using joblib
        # joblib.load() reconstructs the full pipeline object in memory:
        # Pipeline(steps=[('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
        # this is the same object that was saved after training with joblib.dump()
        models[key] = joblib.load(local_path)

        # Confirm successful load — key name helps identify which model loaded
        print(f"✅ Loaded model: {key}")

    # Return the completed dict of all loaded pipeline objects to the caller
    # pipeline.py uses models[key].predict_proba() during inference
    return models
