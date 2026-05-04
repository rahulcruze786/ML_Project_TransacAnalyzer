# ── Standard library imports ──────────────────────────────────────────────────
import numpy as np      # used for argmax (finding highest probability index)
                        # and np.where (vectorized if/else on DataFrame columns)
import pandas as pd     # used to build and manipulate DataFrames from payload data
import time             # used to measure total pipeline execution time
import importlib        # used to dynamically import preprocess_text from shared folder
import sys              # used to add shared folder to Python's module search path
import os               # used to build file paths for the shared folder

# ── Add shared folder to Python path ─────────────────────────────────────────
# preprocess_text.py lives in the shared/ folder, not in this directory
# sys.path.insert(0, ...) adds the shared folder so importlib can find it
# two paths are added to support both Docker (/app/shared) and local (../shared)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

# ── Import model loader functions ─────────────────────────────────────────────
# load_metadata() → reads model_config_{market}.json from /mnt/models/
# load_models()   → loads .joblib pipeline files from /mnt/models/
from model_loader import load_metadata, load_models

# ── Import shared preprocessing functions ────────────────────────────────────
# preprocess_method → cleans and normalizes raw journal text before prediction
# col_aggreate      → combines multiple key columns into a single lookup key string
_preprocess_module = importlib.import_module("preprocess_text")
preprocess_method  = _preprocess_module.preprocess_method
col_aggreate       = _preprocess_module.col_aggreate


# ─────────────────────────────────────────────────────────────────────────────
# predict()
# Runs inference on every row in the DataFrame using the loaded model pipelines.
# Each row is predicted independently using the pipeline matching its key.
# ─────────────────────────────────────────────────────────────────────────────

def predict(df, models, key_col):
    # Initialise four empty lists — one per output column
    # these are built row by row and assigned to the DataFrame at the end
    predictions     = []   # predicted class e.g. "Deductible"
    confidences     = []   # probability of the predicted class e.g. 0.91
    influences      = []   # tfidf words that matched the input text
    all_class_probs = []   # probabilities for ALL classes

    # Iterate over every row in the DataFrame
    # df.iterrows() yields (index, row) for each row
    # _ means the index is intentionally ignored — only row data is needed here
    for _, row in df.iterrows():

        # get the preprocessed text for this row as a plain string
        # .get() safely returns '' if 'ProcessedText' column is missing
        raw_text = str(row.get('ProcessedText', ''))

        # get the model lookup key for this row
        # e.g. "TRAVEL" or "MEALS" — used to find the right pipeline in models dict
        key = str(row[key_col])

        # if no trained model exists for this key, append empty/default values
        # and skip to the next row — avoids KeyError on models[key]
        if key not in models:
            predictions.append(None)
            confidences.append(0.0)
            influences.append("")
            all_class_probs.append("")
            continue

        # fetch the trained sklearn Pipeline for this key
        # e.g. models["TRAVEL"] → Pipeline([preprocess, tfidf, MultinomialNB])
        pipeline = models[key]

        # run prediction — predict_proba() returns 2D array e.g. [[0.91, 0.09]]
        # [0] gets the first (and only) row → [0.91, 0.09]
        probs    = pipeline.predict_proba([raw_text])[0]

        # get the list of class names the model was trained on
        # e.g. ["Deductible", "Non-deductible"]
        classes  = pipeline.classes_

        # find the index of the highest probability
        # np.argmax([0.91, 0.09]) → 0  (index of 0.91)
        best_idx = np.argmax(probs)

        # try to extract which tfidf vocabulary words appeared in the input text
        # these are the "influence words" — words that affected the prediction
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


# ─────────────────────────────────────────────────────────────────────────────
# process_data()
# Full end-to-end prediction pipeline.
# Receives the RAW payload from app.py — no pre-filtering done by app.py.
# pipeline.py is responsible for ALL data filtering and processing:
#   - filter account_mapping by market + process_types
#   - filter config by market + process_types
#   - convert config rows → flat config_params dict
#   - run the full ML prediction pipeline
#
# Expected payload keys (raw from UI, unfiltered):
#   market          : str             e.g. "Hong Kong"
#   journal_lines   : list of dicts   raw journal line rows (full, unfiltered)
#   account_mapping : list of dicts   full account mapping table (all markets)
#   config          : list of dicts   full config table rows (all markets)
# ─────────────────────────────────────────────────────────────────────────────

def process_data(payload: dict) -> dict:
    # record pipeline start time to measure total execution duration at the end
    start  = time.time()

    # ── Step 1: Load market, metadata and models ──────────────────────────────
    # extract market name from payload — determines which config + models to load
    market = payload.get("market")

    # load model_config_{market}.json from /mnt/models/
    # returns a Python dict with text_column, key_cols, process_types etc.
    # this file was saved by trainer.py and copied here by SAP AI Core from S3
    metadata: dict = load_metadata(market)

    # load all .joblib model files listed in metadata from /mnt/models/
    # returns { "TRAVEL": <Pipeline>, "MEALS": <Pipeline>, ... }
    models = load_models(metadata)

    # ── Step 2: Extract training configuration from metadata ──────────────────
    # these values ALWAYS come from metadata — set by the trainer at training time
    # they must NEVER come from config or the UI payload directly
    # because they must exactly match what the model was trained with
    TEXT_COL       = metadata.get("text_column")         # e.g. "TEXT(S4Journal)"
    seasonal_words = metadata.get("seasonal_words", "")  # e.g. "mbfc,cbp,rcls"
    KEY_COLS       = metadata.get("key_cols")            # e.g. ["ExpenseType"]
    process_types  = metadata.get("process_types")       # e.g. ["TRAVEL", "MEALS"]
    allowed_class  = metadata.get("allowed_class")       # e.g. ["Deductible","Non-deductible"]

    # safety checks — raise clear error if any required metadata field is missing
    if not TEXT_COL:      raise ValueError("❌ Metadata missing 'text_column' — retrain the model!")
    if not KEY_COLS:      raise ValueError("❌ Metadata missing 'key_cols' — retrain the model!")
    if not process_types: raise ValueError("❌ Metadata missing 'process_types' — retrain the model!")
    if not allowed_class: raise ValueError("❌ Metadata missing 'allowed_class' — retrain the model!")

    print(f"📋 text_column   : {TEXT_COL}")
    print(f"📋 key_cols      : {KEY_COLS}")
    print(f"📋 process_types : {process_types}")
    print(f"📋 allowed_class : {allowed_class}")

    # ── Step 3: Build and filter account_mapping ──────────────────────────────
    # UI sends the FULL account mapping table — all markets, all process types
    # We filter here because:
    #
    # MARKET filter:
    #   Same GL account number can exist in multiple markets with different
    #   ExpenseTypes — without filtering, Singapore account "99999" could
    #   incorrectly match a Hong Kong journal line → wrong model predicts
    #
    # PROCESS TYPE filter:
    #   Accounts from unrelated process types must not enter the pipeline —
    #   the model was only trained on specific process types from metadata
    account_mapping_data = payload.get("account_mapping", [])
    if not account_mapping_data:
        raise ValueError("❌ 'account_mapping' is required in the request body")

    # convert list of dicts → DataFrame for filtering
    category_mapping = pd.DataFrame(account_mapping_data)

    # filter by market — keep only rows matching the requested market
    # prevents accounts from other markets entering this market's pipeline
    if 'Market' in category_mapping.columns:
        category_mapping = category_mapping[category_mapping['Market'] == market]
        print(f"📋 account_mapping after market filter       : {len(category_mapping)} rows")

    # filter by process_types from metadata
    # .isin() keeps rows where ProcessType matches ANY value in the list
    # e.g. process_types=["TRAVEL","MEALS"] → keeps only TRAVEL and MEALS rows
    if 'ProcessType' in category_mapping.columns:
        category_mapping = category_mapping[
            category_mapping['ProcessType'].isin(process_types)
        ]
        print(f"📋 account_mapping after process_type filter : {len(category_mapping)} rows")

    # cast Account to string for consistent merging with GLAccount later
    category_mapping["Account"] = category_mapping["Account"].astype(str)

    # ── Step 4: Build and filter config → flat config_params dict ─────────────
    # UI sends the FULL config table — all markets, all process types
    # We filter here because:
    #
    # MARKET filter:
    #   Without it → dict(zip()) could pick up Singapore's THRESHOLD_AMOUNT
    #   instead of Hong Kong's — wrong value used in pipeline
    #
    # PROCESS TYPE filter:
    #   Without it → if two rows exist for THRESHOLD_AMOUNT (one for TRAVEL,
    #   one for HOTEL), dict(zip()) keeps whichever comes LAST in the list
    #   → unpredictable and wrong config value used in pipeline
    #
    # GLOBAL rows (no ProcessType value):
    #   Some config params apply to all process types — these rows have no
    #   ProcessType value e.g. OW_RL_FM_PROFIT_CENTRE applies globally
    #   → kept using .isna() check alongside the process_types filter
    config_rows = payload.get("config", [])
    if not config_rows:
        raise ValueError("❌ 'config' is required in the request body")

    # convert list of dicts → DataFrame for filtering
    config_df = pd.DataFrame(config_rows)

    # filter by market — keep only rows matching the requested market
    if 'Market' in config_df.columns:
        config_df = config_df[config_df['Market'] == market]
        print(f"📋 config after market filter       : {len(config_df)} rows")

    # filter by process_types — keep matching rows AND global rows (no ProcessType)
    # global rows are config params that apply to all process types
    if 'ProcessType' in config_df.columns:
        config_df = config_df[
            config_df['ProcessType'].isin(process_types) |  # matches process_types
            config_df['ProcessType'].isna()                  # OR global rows
        ]
        print(f"📋 config after process_type filter : {len(config_df)} rows")

    # validate Parameter and Value columns exist before converting
    if 'Parameter' not in config_df.columns or 'Value' not in config_df.columns:
        raise ValueError("❌ config rows must have 'Parameter' and 'Value' columns")

    # convert filtered config rows → flat key-value dict
    # zip() pairs Parameter and Value columns together row by row
    # dict() converts those pairs → flat key-value dict
    # e.g. Parameter=["THRESHOLD_AMOUNT"], Value=["30000"]
    #   → { "THRESHOLD_AMOUNT": "30000" }
    # now guaranteed to have only the correct market + process_type values
    config_params = dict(zip(config_df['Parameter'], config_df['Value']))
    print(f"📋 config_params built : {list(config_params.keys())}")

    # ── Step 5: Build journal lines DataFrame ─────────────────────────────────
    # journal_lines are NOT filtered here — the UI sends only the journal lines
    # relevant to this market already (one market per prediction run)
    journal_lines = payload.get("journal_lines", [])
    if not journal_lines:
        raise ValueError("❌ 'journal_lines' is required in the request body")

    # convert list of dicts → DataFrame
    predict_df = pd.DataFrame(journal_lines)

    # drop columns not needed for prediction
    # errors='ignore' means no crash if column doesn't exist in this payload
    predict_df.drop(columns=[
        'IMPORTED', 'Source', 'BaseCurrency', 'Debit/Credit',
        'DocumentDate', 'GLTaxCode', 'GLTaxCountry/Region',
        'Product', 'Referencedocument', 'ReportingEntity', 'ReportingEntityCurrency'
    ], inplace=True, errors='ignore')

    # ── Step 6: Extract individual config values from config_params ───────────

    # overwrite rule: Profit Centre + Oper Unit + Segment → "Payment"
    OW_RL_FM_PROFIT_CENTRE = config_params.get('OW_RL_FM_PROFIT_CENTRE')
    OW_RL_FM_OPER_UNIT     = config_params.get('OW_RL_FM_OPER_UNIT')
    OW_RL_FM_SEGMENT       = config_params.get('OW_RL_FM_SEGMENT')

    # helper: splits a comma-separated config string into a list
    # e.g. "chatfield,chat" → ["chatfield", "chat"]
    # if already a list, returns as-is
    def split_param(key, default=''):
        val = config_params.get(key, default)
        return val.split(",") if isinstance(val, str) else val

    OW_RL_FM_TEXT_CHATFIELD_COL         = split_param('OW_RL_FM_TEXT_CHATFIELD_COL')
    OW_OUTPUT_FM_TEXT_CHATFIELD_COL     = config_params.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL')
    OW_RL_FM_TEXT_CHATFIELD_COL_CND     = split_param('OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT')
    OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND = config_params.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT')
    THRESHOLD_GROUPBY                   = split_param('THRESHOLD_GROUPBY')
    THRESHOLD_SUMBY                     = config_params.get('THRESHOLD_SUMBY')
    THRESHOLD_AMOUNT                    = float(config_params.get('THRESHOLD_AMOUNT', '50000'))
    NET_OFF_GROUPBY                     = split_param('NET_OFF_GROUPBY')
    AMOUNT_COL                          = config_params.get('AMOUNT_COL')

    # ── Step 7: Filter journal lines to relevant GL accounts ──────────────────
    # keep only rows where GLAccount belongs to the filtered category_mapping
    # category_mapping is already filtered by market + process_types above
    # this ensures only eligible accounts go through prediction
    mask              = category_mapping['ProcessType'].isin(process_types)
    required_accounts = category_mapping.loc[mask, "Account"].astype(str).values.tolist()
    predict_df        = predict_df[predict_df['GLAccount'].astype(str).isin(required_accounts)].copy()

    # ── Step 8: Merge account mapping onto journal lines ──────────────────────
    # brings in ExpenseType from category_mapping based on GLAccount = Account
    # how="left" keeps all journal rows even if no mapping match found
    predict_df = predict_df.merge(
        category_mapping, left_on="GLAccount", right_on="Account", how="left"
    )
    predict_df.drop(
        columns=['ProcessType', 'Account', 'IMPORTED', 'Market'],
        inplace=True, errors='ignore'
    )

    # ── Step 9: Preprocess text ───────────────────────────────────────────────
    # apply the same text cleaning used during training
    # seasonal_words from metadata ensures consistency with training
    predict_df["ProcessedText"] = predict_df[TEXT_COL].apply(
        lambda x: preprocess_method(x, seasonal_words)
    )
    predict_df['Amount(Base)'] = pd.to_numeric(predict_df["Amount(Base)"], errors="coerce")

    # ── Step 10: Build model lookup key ──────────────────────────────────────
    # combines KEY_COLS into a single string key matching what the trainer used
    # e.g. KEY_COLS=["ExpenseType"] → key="TRAVEL"
    predict_df["key"] = col_aggreate(predict_df, KEY_COLS)

    # ── Step 11: Run ML prediction ────────────────────────────────────────────
    # calls predict() above — adds Prediction, Confidence, InflunceParameter,
    # ClassProbabilities columns to predict_df
    predict_df = predict(predict_df, models, "key")

    # ── Step 12: Net-off logic ────────────────────────────────────────────────
    # if positive and negative amounts cancel out within the same group
    # mark those rows as NetOff="Yes" — they offset each other, no tax impact

    # first pass: simple group sum — if total is exactly 0 the whole group nets off
    group_sum            = predict_df.groupby(NET_OFF_GROUPBY)[AMOUNT_COL].transform('sum')
    predict_df['NetOff'] = np.where(group_sum == 0, 'Yes', 'No')

    # second pass: partial net-off — pair positive and negative rows of equal amount
    data_no_netoff               = predict_df[predict_df['NetOff'] == 'No'].copy()
    data_no_netoff["abs_amount"] = data_no_netoff[AMOUNT_COL].abs()
    data_no_netoff['sign']       = np.where(data_no_netoff[AMOUNT_COL] >= 0, 1, -1)
    data_no_netoff['orig_index'] = data_no_netoff.index

    def assign_net_off(group):
        n_pairs = min((group['sign'] == 1).sum(), (group['sign'] == -1).sum())
        group   = group.copy()
        group['NetOff'] = 'No'
        pos_idx = group[group['sign'] == 1].index
        neg_idx = group[group['sign'] == -1].index
        group.loc[pos_idx, 'pos_cumcount'] = np.arange(1, len(pos_idx) + 1)
        group.loc[neg_idx, 'neg_cumcount'] = np.arange(1, len(neg_idx) + 1)
        group.loc[(group['sign'] == 1)  & (group['pos_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        group.loc[(group['sign'] == -1) & (group['neg_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        return group.drop(columns=['pos_cumcount', 'neg_cumcount'])

    data_no_netoff = data_no_netoff.groupby(
        NET_OFF_GROUPBY + ['abs_amount'], as_index=False
    ).apply(assign_net_off)
    predict_df.loc[
        data_no_netoff['orig_index'].to_numpy(), 'NetOff'
    ] = data_no_netoff['NetOff'].to_numpy()

    # ── Step 13: Threshold logic ──────────────────────────────────────────────
    # if absolute sum of a group is below THRESHOLD_AMOUNT → "Yes"
    grouped_sum                    = predict_df.groupby(THRESHOLD_GROUPBY)[THRESHOLD_SUMBY].transform('sum')
    predict_df['Threshold Amount'] = grouped_sum.apply(
        lambda x: 'Yes' if abs(x) < THRESHOLD_AMOUNT else 'No'
    )

    # ── Step 14: Operating unit split ─────────────────────────────────────────
    # classifies each row as "Group" or "Country" based on OperatingLocation code
    def get_oper_unit_split(x):
        try:
            val = int(float(str(x)[5:]))
            if (val in (784, 786, 789) or val >= 800) and val != 948:
                return 'Group'
        except Exception:
            pass
        return 'Country'

    predict_df['Oper Unit Split'] = predict_df['OperatingLocation'].apply(get_oper_unit_split)

    # ── Step 15: Overwrite rules ──────────────────────────────────────────────
    # these rules override ML prediction for specific business conditions
    # first matching rule wins — continue skips remaining rules for that row
    for index, row in predict_df.iterrows():

        # rule 1: PREV journal source + account starting with 1 or 2 → FX
        if row["GLTBSource"] == "PREV" and str(row["GLAccount"])[4:].startswith(('1', '2')):
            predict_df.at[index, "Prediction "]       = "FX"
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule since PREV in Journal Source"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        # rule 2: Profit Centre + Oper Unit + Segment match config values → Payment
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

        # rule 3: text contains chatfield keywords → overwrite with configured output
        if any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text Chatfield Column"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        # rule 4: threshold reached + Non-deductible + keyword → overwrite output
        if (row["Threshold Amount"] == "Yes"
                and row["Prediction "] == "Non-deductible"
                and any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL_CND)):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text, Prediction and Threshold"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"

    # ── Step 16: Rename + select final output columns ─────────────────────────
    predict_df.rename(columns={
        "Threshold Amount": "ThresholdAmountReached",
        "Oper Unit Split":  "OperUnitSplit"
    }, inplace=True)

    predict_df = predict_df[[
        "MTDPeriod", "DocumentNumber", "GLTBSource", "Segment", "PostingDate",
        "GLAccount", "PartnerEntity", "PostingItem", TEXT_COL,
        "OperatingLocation", "Ledger", "CostCenter", "ProfitCenter",
        "Amount(Base)", "Amount(Transaction)", "TransactionCurrency",
        "Entity/BU", "SourceDocumentNo", "ExpenseType", "ProcessedText",
        "key", "Prediction ", "Confidence", "InflunceParameter",
        "ClassProbabilities", "NetOff", "ThresholdAmountReached", "OperUnitSplit"
    ]]

    # ── Step 17: Return results to UI ─────────────────────────────────────────
    # results returned as JSON to UI — no HDI writes
    elapsed = time.time() - start
    print(f"✅ Pipeline completed in {elapsed:.1f} seconds")

    return {
        "status":                 "success",
        "message":                "Prediction pipeline completed successfully.",
        "rows_processed":         len(predict_df),
        "execution_time_seconds": round(elapsed, 2),
        "data":                   predict_df.to_dict(orient='records')
    }
