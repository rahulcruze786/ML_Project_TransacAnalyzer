from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
import sys
import os
import json
import importlib
import tempfile
import traceback

# ── Add shared folder to path ─────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))        # Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))  # Local: ../shared

import numpy as np
import pandas as pd
import time
import joblib

# ── Import shared modules once at module level ────────────────
_preprocess_module = importlib.import_module("preprocess_text")
preprocess_method  = _preprocess_module.preprocess_method
col_aggreate       = _preprocess_module.col_aggreate

_hdi_module        = importlib.import_module("hdi_data")
get_hdi_connection = _hdi_module.get_hdi_connection
read_hdi_table     = _hdi_module.read_hdi_table
write_hdi_table    = _hdi_module.write_hdi_table

app = Flask(__name__, static_folder="webapp", static_url_path="")
CORS(app)

# ── CHANGE 1: MODEL_DIR moved here — must be defined before any function uses it ──
# Previously MODEL_DIR was defined after @app.before_request which caused fragile ordering.
MODEL_DIR = os.environ.get("MODEL_PATH", "/mnt/models")

# ── CHANGE 2: Lazy-loaded globals — None until first request fires ─────────────
# Previously was module-level load which crashed because /mnt/models
# doesn't exist yet when the container starts (SAP AI Core mounts artifacts
# only after the container is running, not before).
ALL_METADATA = None
ALL_MODELS   = None


# ── CHANGE 3: New function — downloads all market artifacts from object store ──
# Scans all env vars starting with STORAGE_URI_ and downloads each one
# into MODEL_DIR/<market> subfolder dynamically.
# Zero code change needed when adding new markets — just add STORAGE_URI_<MARKET>
# to the serving YAML env section.
#
# Naming convention (must match trainer.py market subfolder):
#   STORAGE_URI_HONG_KONG  → /mnt/models/Hong_Kong/
#   STORAGE_URI_SINGAPORE  → /mnt/models/Singapore/
#   STORAGE_URI_MALAYSIA   → /mnt/models/Malaysia/
def _download_market_artifacts():
    import subprocess

    storage_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("STORAGE_URI_")
    }

    if not storage_vars:
        # Running locally or models already mounted — skip download silently
        print("⚠️ No STORAGE_URI_* env vars found. Assuming models already mounted locally.")
        return

    for env_key, uri in storage_vars.items():
        # Convert env key suffix to folder name matching trainer.py convention:
        # STORAGE_URI_HONG_KONG → "HONG_KONG" → "Hong Kong" → "Hong_Kong"
        market_folder = env_key.replace("STORAGE_URI_", "").replace("_", " ").title().replace(" ", "_")
        dest          = os.path.join(MODEL_DIR, market_folder)
        os.makedirs(dest, exist_ok=True)
        print(f"📥 Downloading [{market_folder}] from {uri} → {dest}")
        try:
            subprocess.run(
                ["aws", "s3", "sync", uri, dest, "--no-progress"],
                check=True
            )
            print(f"✅ [{market_folder}] downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download [{market_folder}]: {e}")
            raise


# ── CHANGE 4: @app.before_request replaces module-level startup load ──────────
# Flask calls this automatically before every request.
# The `if ALL_METADATA is None` guard ensures models load only ONCE
# (on the very first request), not on every request.
# This fires AFTER SAP AI Core has mounted the artifacts, so /mnt/models exists.
@app.before_request
def load_models_once():
    global ALL_METADATA, ALL_MODELS
    if ALL_METADATA is None:
        _download_market_artifacts()        # ← CHANGE 5: download artifacts first
        print("🚀 Loading all market models...")
        ALL_METADATA = load_metadata()
        ALL_MODELS   = load_models(ALL_METADATA)
        print(f"✅ Markets loaded: {list(ALL_MODELS.keys())}")


@app.route('/')
def root():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# ── CHANGE 6: load_metadata now scans market subfolders instead of flat dir ───
# Previously read a single model_config.json from MODEL_DIR root.
# Now scans MODEL_DIR for subdirectories (one per market) and loads
# each market's model_config.json.
# Returns: { "Hong_Kong": metadata_dict, "Singapore": metadata_dict, ... }
def load_metadata():
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


# ── CHANGE 7: load_models now loads all markets into nested dict ──────────────
# Previously loaded from a flat MODEL_DIR into { key: model }.
# Now loads per market from MODEL_DIR/<market>/ into
# { "Hong_Kong": { key: model }, "Singapore": { key: model }, ... }
def load_models(all_metadata):
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


# ─────────────────────────────────────────────────────────────
# Prediction Logic — NO CHANGE
# ─────────────────────────────────────────────────────────────

def predict(df, models, key_col):
    """
    Predict using loaded sklearn pipelines (preprocess + tfidf + nb)
    Pipeline is same as trainer's _build_pipeline()
    """
    predictions     = []
    confidences     = []
    influences      = []
    all_class_probs = []

    for _, row in df.iterrows():
        raw_text = str(row.get('ProcessedText', ''))
        key      = str(row[key_col])

        if key not in models:
            predictions.append(None)
            confidences.append(0.0)
            influences.append("")
            all_class_probs.append("")
            continue

        pipeline = models[key]
        probs    = pipeline.predict_proba([raw_text])[0]
        classes  = pipeline.classes_
        best_idx = np.argmax(probs)

        try:
            tfidf         = pipeline.named_steps['tfidf']
            feature_names = tfidf.get_feature_names_out()
            text_tokens   = raw_text.lower().split()
            matched_words = [w for w in feature_names if w in text_tokens]
        except Exception:
            matched_words = []

        predictions.append(classes[best_idx])
        confidences.append(float(probs[best_idx]))
        influences.append(", ".join(matched_words[:10]))
        all_class_probs.append(" | ".join(
            f"{cls}:{round(float(p), 4)}" for cls, p in zip(classes, probs)
        ))

    df["Prediction "]        = predictions
    df["Confidence"]         = confidences
    df["InflunceParameter"]  = influences
    df["ClassProbabilities"] = all_class_probs
    return df


# ─────────────────────────────────────────────────────────────
# HDI Config Loader — NO CHANGE
# ─────────────────────────────────────────────────────────────

def _load_hdi_config(conn, schema='T1_TXNANAL_PHY', market=None):
    """Load configuration from HDI Configuration table"""
    try:
        config_df = read_hdi_table(conn, schema, "Configuration")
        if market and 'Market' in config_df.columns:
            config_df = config_df[config_df['Market'] == market]
        return dict(zip(config_df['Parameter'], config_df['Value']))
    except Exception as e:
        print(f"⚠️ Could not load CONFIG table: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Main process_data
# ─────────────────────────────────────────────────────────────

def process_data(config_params=None, market=None):
    start  = time.time()
    schema = 'T1_TXNANAL_PHY'

    # ── CHANGE 8: Step 1 replaced — now resolves market key from ALL_METADATA ─
    # Previously called load_metadata() and load_models() on every request.
    # Now uses pre-loaded ALL_METADATA and ALL_MODELS globals,
    # picks the correct market's metadata and models by key.
    market_key = market.replace(" ", "_") if market else None
    if not market_key or market_key not in ALL_METADATA:
        available = list(ALL_METADATA.keys())
        raise ValueError(f"❌ Market '{market}' not found. Available markets: {available}")

    metadata = ALL_METADATA[market_key]
    models   = ALL_MODELS[market_key]

    # ── Step 2: Extract ALL training values from metadata — NO CHANGE ─────────
    TEXT_COL       = metadata.get("text_column")
    LABEL_COL      = metadata.get("target_column")
    seasonal_words = metadata.get("seasonal_words", "")
    KEY_COLS       = metadata.get("key_cols")
    process_types  = metadata.get("process_types")
    allowed_class  = metadata.get("allowed_class")

    if not TEXT_COL:
        raise ValueError("❌ Metadata missing 'text_column' - retrain the model!")
    if not KEY_COLS:
        raise ValueError("❌ Metadata missing 'key_cols' - retrain the model!")
    if not process_types:
        raise ValueError("❌ Metadata missing 'process_types' - retrain the model!")
    if not allowed_class:
        raise ValueError("❌ Metadata missing 'allowed_class' - retrain the model!")

    print(f"📋 Metadata → text_column   : {TEXT_COL}")
    print(f"📋 Metadata → target_column : {LABEL_COL}")
    print(f"📋 Metadata → key_cols      : {KEY_COLS}")
    print(f"📋 Metadata → process_types : {process_types}")
    print(f"📋 Metadata → seasonal_words: {seasonal_words}")
    print(f"📋 Metadata → allowed_class : {allowed_class}")

    # ── Steps 3–18: NO CHANGE ─────────────────────────────────────────────────

    # ── Step 3: Connect to HDI ────────────────────────────────
    conn = get_hdi_connection()

    # ── Step 4: Load HDI tables ───────────────────────────────
    category_mapping            = read_hdi_table(conn, schema, "Accountaxcode")
    category_mapping["Account"] = category_mapping["Account"].astype(str)

    predict_df = read_hdi_table(conn, schema, "JRNLINEITM")
    predict_df.drop(columns=[
        'IMPORTED', 'Source', 'BaseCurrency', 'Debit/Credit',
        'DocumentDate', 'GLTaxCode', 'GLTaxCountry/Region',
        'Product', 'Referencedocument', 'ReportingEntity', 'ReportingEntityCurrency'
    ], inplace=True, errors='ignore')

    # ── Step 5: Load HDI config + merge with user overrides ───
    default_config = _load_hdi_config(conn, schema, market)
    if config_params is None:
        config_params = {}
    final_config = {**default_config, **config_params}
    print(f"📋 Using HDI config keys: {list(final_config.keys())}")

    # ── Step 6: Extract overwrite rule params from HDI config ─
    OW_RL_FM_PROFIT_CENTRE = final_config.get('OW_RL_FM_PROFIT_CENTRE')
    OW_RL_FM_OPER_UNIT     = final_config.get('OW_RL_FM_OPER_UNIT')
    OW_RL_FM_SEGMENT       = final_config.get('OW_RL_FM_SEGMENT')

    OW_RL_FM_TEXT_CHATFIELD_COL_STR = final_config.get('OW_RL_FM_TEXT_CHATFIELD_COL', '')
    OW_RL_FM_TEXT_CHATFIELD_COL     = OW_RL_FM_TEXT_CHATFIELD_COL_STR.split(",") if isinstance(OW_RL_FM_TEXT_CHATFIELD_COL_STR, str) else OW_RL_FM_TEXT_CHATFIELD_COL_STR

    OW_OUTPUT_FM_TEXT_CHATFIELD_COL = final_config.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL')

    OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT_STR = final_config.get('OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT', '')
    OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT     = OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT_STR.split(",") if isinstance(OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT_STR, str) else OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT_STR

    OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT = final_config.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT')

    THRESHOLD_GROUPBY_STR = final_config.get('THRESHOLD_GROUPBY', '')
    THRESHOLD_GROUPBY     = THRESHOLD_GROUPBY_STR.split(",") if isinstance(THRESHOLD_GROUPBY_STR, str) else THRESHOLD_GROUPBY_STR

    THRESHOLD_SUMBY  = final_config.get('THRESHOLD_SUMBY')
    THRESHOLD_AMOUNT = float(final_config.get('THRESHOLD_AMOUNT', '50000'))

    NET_OFF_GROUPBY_STR = final_config.get('NET_OFF_GROUPBY', '')
    NET_OFF_GROUPBY     = NET_OFF_GROUPBY_STR.split(",") if isinstance(NET_OFF_GROUPBY_STR, str) else NET_OFF_GROUPBY_STR

    AMOUNT_COL = final_config.get('AMOUNT_COL')

    # ── Step 7: Filter accounts using process_types from metadata ─────────────
    print(f"📋 Filtering accounts by process_types from metadata: {process_types}")
    mask              = category_mapping['ProcessType'].isin(process_types)
    required_accounts = category_mapping.loc[mask, "Account"].astype(str).values.tolist()
    predict_df        = predict_df[predict_df['GLAccount'].astype(str).isin(required_accounts)].copy()

    # ── Step 8: Merge category mapping ───────────────────────
    predict_df = predict_df.merge(category_mapping, left_on="GLAccount", right_on="Account", how="left")
    predict_df.drop(columns=['ProcessType', 'Account', 'IMPORTED', 'Market'], inplace=True, errors='ignore')

    # ── Step 9: Preprocess text ───────────────────────────────
    predict_df["ProcessedText"] = predict_df[TEXT_COL].apply(
        lambda x: preprocess_method(x, seasonal_words)
    )
    predict_df['Amount(Base)'] = pd.to_numeric(predict_df["Amount(Base)"], errors="coerce")

    # ── Step 10: Generate key ─────────────────────────────────
    predict_df["key"] = col_aggreate(predict_df, KEY_COLS)

    # ── Step 11: Run prediction ───────────────────────────────
    predict_df = predict(predict_df, models, "key")

    # ── Step 12: Net Off Logic ────────────────────────────────
    group_sum            = predict_df.groupby(NET_OFF_GROUPBY)[AMOUNT_COL].transform('sum')
    predict_df['NetOff'] = np.where(group_sum == 0, 'Yes', 'No')

    data_no_netoff               = predict_df[predict_df['NetOff'] == 'No'].copy()
    data_no_netoff["abs_amount"] = data_no_netoff[AMOUNT_COL].abs()
    data_no_netoff['sign']       = np.where(data_no_netoff[AMOUNT_COL] >= 0, 1, -1)
    data_no_netoff['orig_index'] = data_no_netoff.index
    group_keys_extended          = NET_OFF_GROUPBY + ['abs_amount']

    def assign_net_off(group):
        n_positive = (group['sign'] == 1).sum()
        n_negative = (group['sign'] == -1).sum()
        n_pairs    = min(n_positive, n_negative)
        group      = group.copy()
        group['NetOff']     = 'No'
        pos_indices         = group[group['sign'] == 1].index
        neg_indices         = group[group['sign'] == -1].index
        group.loc[pos_indices, 'pos_cumcount'] = np.arange(1, len(pos_indices) + 1)
        group.loc[neg_indices, 'neg_cumcount'] = np.arange(1, len(neg_indices) + 1)
        group.loc[(group['sign'] == 1)  & (group['pos_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        group.loc[(group['sign'] == -1) & (group['neg_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        group = group.drop(columns=['pos_cumcount', 'neg_cumcount'])
        return group

    grouped        = data_no_netoff.groupby(group_keys_extended, as_index=False)
    data_no_netoff = grouped.apply(assign_net_off)
    predict_df.loc[data_no_netoff['orig_index'].to_numpy(), 'NetOff'] = data_no_netoff['NetOff'].to_numpy()

    # ── Step 13: Threshold Logic ──────────────────────────────
    grouped_sum                    = predict_df.groupby(THRESHOLD_GROUPBY)[THRESHOLD_SUMBY].transform('sum')
    predict_df['Threshold Amount'] = grouped_sum.apply(lambda x: 'Yes' if abs(x) < THRESHOLD_AMOUNT else 'No')

    # ── Step 14: Oper Unit Split ──────────────────────────────
    def get_oper_unit_split(x):
        try:
            val = int(float(str(x)[5:]))
            if (val in (784, 786, 789) or val >= 800) and val != 948:
                return 'Group'
        except Exception:
            pass
        return 'Country'

    predict_df['Oper Unit Split'] = predict_df['OperatingLocation'].apply(get_oper_unit_split)

    # ── Step 15: Overwrite Rules ──────────────────────────────
    for index, row in predict_df.iterrows():

        # Rule 1: PREV Journal Source → FX
        if row["GLTBSource"] == "PREV" and str(row["GLAccount"])[4:].startswith(('1', '2')):
            predict_df.at[index, "Prediction "]       = "FX"
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule since PREV in Journal Source"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        # Rule 2: Profit Centre + Oper Unit + Segment → Payment
        if (str(row["GLAccount"])[4:].replace(".0", "").isdigit()
                and str(row["GLAccount"])[4:].replace(".0", "").startswith(('1', '2'))
                and str(int(row["ProfitCenter"])) == OW_RL_FM_PROFIT_CENTRE
                and str(row["OperatingLocation"])[5:].replace(".0", "") == OW_RL_FM_OPER_UNIT
                and str(row["Segment"])[6:].replace(".0", "") == OW_RL_FM_SEGMENT):
            predict_df.at[index, "Prediction "]       = "Payment"
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule for Payment"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        # Rule 3: Text contains chatfield keywords → overwrite output
        if any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text Chatfield Column"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        # Rule 4: Threshold + Non-deductible + keyword → overwrite output
        if (row["Threshold Amount"] == "Yes"
                and row["Prediction "] == "Non-deductible"
                and any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT)):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text, Prediction and Threshold"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"

    # ── Step 16: Rename columns ───────────────────────────────
    predict_df.rename(columns={"Threshold Amount": "ThresholdAmountReached\t", "Oper Unit Split": "OperUnitSplit"}, inplace=True)

    print(predict_df.columns.tolist())

    # ── Step 17: Select final columns ────────────────────────
    predict_df = predict_df[[
        "MTDPeriod", "DocumentNumber", "GLTBSource", "Segment", "PostingDate",
        "GLAccount", "PartnerEntity", "PostingItem", TEXT_COL,
        "OperatingLocation", "Ledger", "CostCenter", "ProfitCenter",
        "Amount(Base)", "Amount(Transaction)", "TransactionCurrency",
        "Entity/BU", "SourceDocumentNo", "ExpenseType", "ProcessedText",
        "key", "Prediction ", "Confidence", "InflunceParameter",
        "ClassProbabilities", "NetOff", "ThresholdAmountReached\t", "OperUnitSplit"
    ]]

    # ── Step 18: Write to HDI JRNLOUTPUT ─────────────────────
    primary_keys = [
        "MTDPeriod", "DocumentNumber", "GLAccount",
        "PostingItem", "Ledger", "Entity/BU", "SourceDocumentNo"
    ]
    write_hdi_table(conn, predict_df, schema, "JRNLOUTPUT", primary_keys=primary_keys)
    conn.close()

    elapsed = time.time() - start
    print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds")

    return {
        "status":                 "success",
        "message":                "Data processed and written to JRNLOUTPUT table successfully.",
        "rows_processed":         len(predict_df),
        "execution_time_seconds": round(elapsed, 2),
        "data":                   predict_df.to_dict(orient='records')
    }


# ─────────────────────────────────────────────────────────────
# Flask Routes — NO CHANGE except get_metadata (CHANGE 9)
# ─────────────────────────────────────────────────────────────

@app.route('/v1/health')
def health():
    return jsonify({"status": "healthyUI"})


@app.route('/v1/getConfig', methods=['GET'])
def get_config():
    """Get configuration parameters filtered by market and processtype"""
    conn = None
    try:
        market       = request.args.get('market')
        process_type = request.args.get('processtype')
        schema       = 'T1_TXNANAL_PHY'
        conn         = get_hdi_connection()
        config_df    = read_hdi_table(conn, schema, "Configuration")

        if market and 'Market' in config_df.columns:
            config_df = config_df[config_df['Market'] == market]
        if process_type and 'ProcessType' in config_df.columns:
            config_df = config_df[config_df['ProcessType'] == process_type]

        return jsonify({"status": "success", "data": config_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except: pass


@app.route('/v1/getAccountMapping', methods=['GET'])
def get_account_mapping():
    """Get account mapping filtered by market and processtype"""
    conn = None
    try:
        market       = request.args.get('market')
        process_type = request.args.get('processtype')
        schema       = 'T1_TXNANAL_PHY'
        conn         = get_hdi_connection()
        mapping_df   = read_hdi_table(conn, schema, "Accountaxcode")

        if market and 'Market' in mapping_df.columns:
            mapping_df = mapping_df[mapping_df['Market'] == market]
        if process_type and 'ProcessType' in mapping_df.columns:
            process_types = [pt.strip() for pt in process_type.split(',')]
            mapping_df    = mapping_df[mapping_df['ProcessType'].isin(process_types)]

        return jsonify({"status": "success", "data": mapping_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except: pass


@app.route('/v1/OverwritePrediction', methods=['POST'])
def upsert_jrnoutput():
    """
    Accept a JSON object or list of objects with the exact JRNLOUTPUT columns,
    and write them to HDI table T1_TXNANAL_PHY.JRNLOUTPUT using the same primary keys.
    """
    conn = None
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

        records = payload if isinstance(payload, list) else [payload]
        if not records:
            return jsonify({"status": "error", "message": "No records provided"}), 400

        primary_keys = ["MTDPeriod", "DocumentNumber", "GLAccount", "PostingItem", "Ledger", "Entity/BU", "SourceDocumentNo"]

        missing = []
        for i, rec in enumerate(records):
            for pk in primary_keys:
                if pk not in rec:
                    missing.append({"index": i, "missing_key": pk})
        if missing:
            return jsonify({"status": "error", "message": "Missing primary key(s) in payload", "details": missing}), 400

        df     = pd.DataFrame(records)
        schema = 'T1_TXNANAL_PHY'
        conn   = get_hdi_connection()
        write_hdi_table(conn, df, schema, "JRNLOUTPUT", primary_keys=primary_keys)

        return jsonify({"status": "success", "message": f"Overwrite {len(df)} record(s) to JRNLOUTPUT"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except: pass


@app.route('/v1/OverwriteReview', methods=['POST'])
def upsert_jrnreview():
    """
    Accept a JSON object or list of objects with the exact JRNLOUTPUT columns,
    and write them to HDI table T1_TXNANAL_PHY.JRNLOPREVIEW using the same primary keys.
    """
    conn = None
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

        records = payload if isinstance(payload, list) else [payload]
        if not records:
            return jsonify({"status": "error", "message": "No records provided"}), 400

        primary_keys = ["MTDPeriod", "DocumentNumber", "GLAccount", "PostingItem", "Ledger", "Entity/BU", "SourceDocumentNo"]

        missing = []
        for i, rec in enumerate(records):
            for pk in primary_keys:
                if pk not in rec:
                    missing.append({"index": i, "missing_key": pk})
        if missing:
            return jsonify({"status": "error", "message": "Missing primary key(s) in payload", "details": missing}), 400

        df     = pd.DataFrame(records)
        schema = 'T1_TXNANAL_PHY'
        conn   = get_hdi_connection()
        write_hdi_table(conn, df, schema, "JRNLOPREVIEW", primary_keys=primary_keys)

        return jsonify({"status": "success", "message": f"Overwrite {len(df)} record(s) to JRNLOUTPUT"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except: pass


@app.route('/saveAccountMapping', methods=['GET', 'POST'])
def save_account_mapping():
    """Save updated account mapping back to HDI table"""
    conn = None
    try:
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}

        if data.get('payload'):
            updated_records = data.get('payload', [])
        else:
            account      = request.args.get('account')
            expense_type = request.args.get('expensetype')
            process_type = request.args.get('processtype')
            market       = request.args.get('market')

            if not all([account, expense_type, process_type, market]):
                return jsonify({
                    "status":  "error",
                    "message": "Provide 'payload' in JSON body OR URL params: account, expensetype, processtype, market"
                }), 400

            updated_records = [{
                'Account':     account,
                'ExpenseType': expense_type,
                'ProcessType': process_type,
                'Market':      market
            }]

        if not updated_records:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        schema       = 'T1_TXNANAL_PHY'
        conn         = get_hdi_connection()
        df           = pd.DataFrame(updated_records)
        primary_keys = ["Account", "ProcessType", "Market"]
        write_hdi_table(conn, df, schema, "Accountaxcode", primary_keys=primary_keys)

        return jsonify({
            "status":  "success",
            "message": f"Updated {len(updated_records)} record(s)",
            "data":    updated_records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except: pass


@app.route('/v1/process', methods=['POST'])
def process():
    """
    Process prediction pipeline.
    text_column, target_column, key_cols, allowed_class,
    seasonal_words, process_types → always from metadata (set during training).
    Overwrite rules & threshold params → from HDI config table (can be overridden via request body).
    """
    try:
        request_data  = request.json if request.json else {}
        market        = request_data.get('market', None)
        config_params = {k: v for k, v in request_data.items() if k != 'market'}

        print(f"📋 User provided parameters : {list(config_params.keys())}")
        print(f"📋 Market                   : {market}")

        result = process_data(config_params=config_params, market=market)
        return jsonify(result)
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"ERROR: {error_detail}")
        return jsonify({
            "status":    "error",
            "message":   str(e),
            "traceback": error_detail
        }), 500


# ── CHANGE 9: get_metadata updated — supports ?market= param or returns all ───
# Previously returned single flat metadata dict.
# Now returns metadata for a specific market (via ?market=Hong Kong)
# or all markets if no param provided.
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    """Return metadata for a specific market or all markets.
    Usage:
        GET /v1/getMetadata              → returns all markets
        GET /v1/getMetadata?market=Hong Kong → returns Hong Kong only
    """
    try:
        market = request.args.get('market', None)
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


'''
@app.route('/v1/reload-models', methods=['POST'])
def reload_models():
    """Force reload models from local MODEL_DIR (call after retraining)."""
    try:
        metadata = load_metadata()
        models = load_models(metadata)
        return jsonify({
            "status":  "success",
            "message": f"Reloaded {len(models)} models from local folder"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
