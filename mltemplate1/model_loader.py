# ── Standard library imports ──────────────────────────────────
import os      # used to read environment variables and build file paths
import json    # used to parse the .json config file into a Python dict
import joblib  # used to load trained sklearn model (.pkl) files from disk

# ── Model directory path ──────────────────────────────────────
# Read MODEL_PATH from environment variable (set in Docker/deployment)
# If not set, fall back to "/mnt/models" as the default location
MODEL_DIR = os.environ.get("MODEL_PATH", "/mnt/models")


def load_metadata(market):
    # Guard: market is required — without it we don't know which config file to load
    if not market:
        raise ValueError("Market is required to load metadata")

    # Sanitize the market name so it's safe to use in a filename
    # e.g. "Hong Kong" → "Hong_Kong",  "SG/HK" → "SG_HK"
    safe_market = str(market).replace("/", "_").replace(" ", "_")

    # Build the full path to the config file
    # e.g. MODEL_DIR + "model_config_Hong_Kong.json"
    local_config = os.path.join(MODEL_DIR, f"model_config_{safe_market}.json")

    # Check the file actually exists before trying to open it
    # Raises a clear error instead of a cryptic Python FileNotFoundError later
    if not os.path.exists(local_config):
        raise FileNotFoundError(f"No metadata config found for market '{market}' at {local_config}")

    # Open the file in read mode with UTF-8 encoding
    # 'with' ensures the file is automatically closed after reading, even if an error occurs
    with open(local_config, "r", encoding="utf-8") as f:
        # Parse the JSON file contents into a Python dictionary
        # e.g. {"text_column": "TEXT(S4Journal)", "models": {"TRAVEL": {...}}}
        metadata = json.load(f)

    # Log how many models were found inside the metadata for quick verification
    print(f"✅ Metadata loaded: {len(metadata.get('models', {}))} models found")

    # Return the metadata dict to the caller (pipeline.py uses this)
    return metadata


def load_models(metadata):
    # Empty dict to collect loaded models — will be { "TRAVEL": <pipeline>, "MEALS": <pipeline>, ... }
    models = {}

    # Loop over every model entry inside metadata["models"]
    # .get("models", {}) safely returns {} if "models" key is missing — avoids KeyError
    # .items() gives each entry as (key, model_info) e.g. ("TRAVEL", {"model_file": "model_travel.pkl"})
    for key, model_info in metadata.get("models", {}).items():

        # Extract the filename of the model from its metadata entry
        # e.g. "model_travel.pkl"
        model_file = model_info.get("model_file")

        # If model_file is missing or empty in the metadata, skip this entry
        if not model_file:
            continue

        # Build the full path to the model file on disk
        # e.g. "/mnt/models/model_travel.pkl"
        local_path = os.path.join(MODEL_DIR, model_file)

        # Check the model file actually exists before trying to load it
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file not found: {local_path}")

        # Load the trained sklearn pipeline from disk using joblib
        # joblib.load deserializes the .pkl file back into a Python object (the pipeline)
        models[key] = joblib.load(local_path)

        # Confirm successful load with the model key name
        print(f"✅ Loaded model: {key}")

    # Return the completed dict of all loaded models to the caller
    return models
