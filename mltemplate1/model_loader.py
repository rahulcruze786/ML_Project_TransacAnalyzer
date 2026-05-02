# ── Standard library imports ──────────────────────────────────
import os      # used to read environment variables and build file paths
import json    # used to parse the .json config file into a Python dict
import joblib  # used to deserialize trained sklearn .pkl model files
               # at deployment time, SAP AI Core authenticates into
               # AWS S3 (ap-southeast-1) using the Object Store Secret,
               # resolves ai://default/<execution_id>/ALL_MARKET_NB_MODEL
               # and copies all files (.pkl + .json) into /mnt/models/
               # inside the running container

# ── Model directory path ──────────────────────────────────────
# SAP AI Core injects MODEL_PATH environment variable at container startup
# pointing to /mnt/models/ — where all model artifact files were copied from
# AWS S3 bucket (ap-southeast-1) under path prefix ai/default/
# trainer saves everything to MODEL_DIR (/tmp/models) during training,
# SAP AI Core uploads that directory to S3 as an artifact,
# and copies it back here at deployment time
# if MODEL_PATH is not set (e.g. running locally), falls back to "/mnt/models"
MODEL_DIR = os.environ.get("MODEL_PATH", "/mnt/models")


def load_metadata(market):
    """
    Reads the model configuration JSON file for the given market from MODEL_DIR.

    This file (model_config_{market}.json) is produced by train_models()
    in trainer.py at the end of a training run:

        config_path = os.path.join(model_dir, CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump(metadata, f)

    It is saved into MODEL_DIR alongside the .pkl files, picked up by
    SAP AI Core as part of the model artifact, and copied into /mnt/models/
    at deployment time — so both the JSON and .pkl files arrive together.

    Returns a Python dict containing:
        - text_column    : column name used as input text during training
        - target_column  : column name used as the label during training
        - key_cols       : columns used to build the model lookup key
        - process_types  : GL account process types the trainer filtered on
        - seasonal_words : comma-separated keywords for text preprocessing
        - allowed_class  : list of valid prediction output classes
        - models         : dict of { key: { model_file, metrics, train_size } }
    """

    # Guard: market name is required to know which config file to load
    # e.g. "Singapore" → looks for model_config_Singapore.json
    # e.g. "Hong Kong" → looks for model_config_Hong_Kong.json
    if not market:
        raise ValueError("Market is required to load metadata")

    # Sanitize the market name so it is safe to use as part of a filename
    # trainer applies the same sanitization when saving:
    #   safe_market = str(market).replace("/", "_").replace(" ", "_")
    # so we must match it exactly here to find the right file
    # e.g. "Hong Kong" → "Hong_Kong",   "SG/HK" → "SG_HK"
    safe_market = str(market).replace("/", "_").replace(" ", "_")

    # Build the full path to the market-specific config file inside the container
    # e.g. "/mnt/models/model_config_Hong_Kong.json"
    # this file was saved by trainer.py into MODEL_DIR during training,
    # uploaded to S3 as part of the model artifact,
    # and copied here by SAP AI Core at deployment time
    local_config = os.path.join(MODEL_DIR, f"model_config_{safe_market}.json")

    # Verify the config file exists inside the container before trying to open it
    # if missing it means either training never ran for this market,
    # or the artifact was not correctly registered in SAP AI Core
    if not os.path.exists(local_config):
        raise FileNotFoundError(
            f"No metadata config found for market '{market}' at {local_config}"
        )

    # Open the config file in read mode with UTF-8 encoding
    # 'with' block ensures the file handle is automatically closed after reading
    # even if an exception occurs mid-read
    with open(local_config, "r", encoding="utf-8") as f:
        # Parse the JSON file into a Python dictionary
        # json.load() converts the JSON file → Python dict (in memory)
        # it only becomes JSON again when jsonify() is called in the Flask route
        metadata = json.load(f)

    # Log how many models were found — quick sanity check at container startup
    # .get("models", {}) safely returns {} if "models" key is missing (no KeyError)
    print(f"✅ Metadata loaded: {len(metadata.get('models', {}))} models found")

    # Return the Python dict to the caller (pipeline.py uses this to
    # extract text_column, key_cols, process_types etc. before running inference)
    return metadata


def load_models(metadata):
    """
    Loads all trained sklearn pipeline (.pkl/.joblib) files listed in metadata.

    trainer.py saves each model file like this:
        model_filename = f"model_{safe_market}_{safe_key}.joblib"
        model_path     = os.path.join(model_dir, model_filename)
        joblib.dump(model, model_path)
        metadata["models"][str(key)] = {"model_file": model_filename, ...}

    All these .joblib files are saved to MODEL_DIR during training,
    uploaded to S3 as part of the model artifact alongside model_config.json,
    and copied into /mnt/models/ by SAP AI Core at deployment time.

    Returns a dict: { key: <loaded sklearn Pipeline object> }
    e.g. {
        "TRAVEL": Pipeline([tfidf, MultinomialNB]),
        "MEALS":  Pipeline([tfidf, MultinomialNB]),
    }
    These Pipeline objects are used directly in predict() for inference.
    """

    # Empty dict to accumulate loaded models as they are read from disk
    # final structure: { "TRAVEL": <Pipeline>, "MEALS": <Pipeline>, ... }
    models = {}

    # Loop over every model entry defined inside metadata["models"]
    # .get("models", {}) → safely returns {} if key is missing (no KeyError)
    #                       if {} then loop runs zero times — no crash
    # .items()            → yields (key, model_info) tuples e.g.
    #                       ("TRAVEL", {"model_file": "model_HongKong_TRAVEL.joblib",
    #                                   "metrics": {...}, "train_size": 120})
    for key, model_info in metadata.get("models", {}).items():

        # Extract the .joblib filename for this model from its metadata entry
        # this filename was set by trainer.py:
        #   model_filename = f"model_{safe_market}_{safe_key}.joblib"
        model_file = model_info.get("model_file")

        # If model_file key is missing or empty in this metadata entry, skip it
        # prevents crash when a metadata entry is incomplete
        if not model_file:
            print(f"⚠️ No model_file found for key '{key}', skipping")
            continue

        # Build the full path to the .joblib file inside the container
        # e.g. "/mnt/models/model_HongKong_TRAVEL.joblib"
        # this file was saved by trainer.py, uploaded to S3,
        # and copied here by SAP AI Core at deployment time
        local_path = os.path.join(MODEL_DIR, model_file)

        # Verify the .joblib file exists before trying to load it
        # gives a clear error instead of a cryptic joblib/pickle traceback
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file not found: {local_path}")

        # Deserialize the sklearn Pipeline object from the .joblib file
        # joblib.load() is the reverse of joblib.dump() used in trainer.py
        # reconstructs the full pipeline in memory:
        # Pipeline([("preprocess", FunctionTransformer),
        #           ("tfidf",      TfidfVectorizer),
        #           ("nb",         MultinomialNB)])
        models[key] = joblib.load(local_path)

        # Confirm each model loaded successfully — key name identifies which one
        print(f"✅ Loaded model: {key}")

    # Return the completed dict of all loaded Pipeline objects to pipeline.py
    # used as: models[key].predict_proba([raw_text]) during inference
    return models
