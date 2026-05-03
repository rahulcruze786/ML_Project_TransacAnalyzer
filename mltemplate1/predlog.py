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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))         # Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))   # Local:  ../shared

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
    all_class_probs = []   # probabilities for ALL classes e.g. "Deductible:0.91 | Non-deductible:0.09"

    # Iterate over every row in the DataFrame
    # df.iterrows() yields (index, row) for each row
    # _ means the index is intentionally ignored — only row data is needed here
    for _, row in df.iterrows():

        # Get the preprocessed text for this row as a plain string
        # .get() safely returns '' if 'ProcessedText' column is missing
        raw_text = str(row.get('ProcessedText', ''))

        # Get the model lookup key for this row
        # e.g. "TRAVEL" or "MEALS" — used to find the right pipeline in models dict
        key = str(row[key_col])

        # If no trained model exists for this key, append empty/default values
        # and skip to the next row — avoids KeyError on models[key]
        if key not in models:
            predictions.append(None)
            confidences.append(0.0)
            influences.append("")
            all_class_probs.append("")
            continue   # jump to next row immediately

        # Fetch the trained sklearn Pipeline for this key
        # e.g. models["TRAVEL"] → Pipeline([preprocess, tfidf, MultinomialNB])
        pipeline = models[key]

        # Run prediction — returns probability for each class
        # predict_proba() returns a 2D array e.g. [[0.91, 0.09]]
        # [0] gets the first (and only) row → [0.91, 0.09]
        probs = pipeline.predict_proba([raw_text])[0]

        # Get the list of class names the model was trained on
        # e.g. ["Deductible", "Non-deductible"]
        classes = pipeline.classes_

        # Find the index of the highest probability
        # np.argmax([0.91, 0.09]) → 0  (index of 0.91)
        best_idx = np.argmax(probs)

        # Try to extract which tfidf vocabulary words appeared in the input text
        # these are the "influence words" — words that affected the prediction
        try:
            tfidf         = pipeline.named_steps['tfidf']         # get tfidf step from pipeline
            feature_names = tfidf.get_feature_names_out()         # all words in trained vocabulary
            text_tokens   = raw_text.lower().split()              # split input text into words
            matched_words = [w for w in feature_names if w in text_tokens]  # words that match vocabulary
        except Exception:
            # if tfidf step fails for any reason, silently use empty list
            matched_words = []

        # Append results for this row to the output lists
        predictions.append(classes[best_idx])          # e.g. "Deductible"
        confidences.append(float(probs[best_idx]))     # e.g. 0.91
        influences.append(", ".join(matched_words[:10]))  # top 10 matched words only

        # Build a readable string of all class probabilities for this row
        # e.g. "Deductible:0.9100 | Non-deductible:0.0900"
        all_class_probs.append(" | ".join(
            f"{cls}:{round(float(p), 4)}" for cls, p in zip(classes, probs)
        ))

    # Assign all four output lists as new columns on the DataFrame
    df["Prediction "]        = predictions
    df["Confidence"]         = confidences
    df["InflunceParameter"]  = influences
    df["ClassProbabilities"] = all_class_probs

    # Return the DataFrame with prediction columns added
    return df


# ─────────────────────────────────────────────────────────────────────────────
# process_data()
# Full end-to-end prediction pipeline.
# All input data comes from the UI request payload — no HDI reads.
# Results are returned as a dict — no HDI writes.
#
# Expected payload keys:
#   market          : str             e.g. "Hong Kong"
#   journal_lines   : list of dicts   raw journal line rows (was JRNLINEITM)
#   account_mapping : list of dicts   GL account to expense type mapping
#   config_params   : dict            overwrite rules and threshold parameters
# ─────────────────────────────────────────────────────────────────────────────

def process_data(payload: dict) -> dict:
    # Record pipeline start time to measure total execution duration at the end
    start = time.time()

    # ── Step 1: Load market, metadata and models ──────────────────────────────
    # Extract market name from payload — determines which config + models to load
    market = payload.get("market")

    # Load model_config_{market}.json from /mnt/models/
    # returns a Python dict with text_column, key_cols, process_types etc.
    metadata: dict = load_metadata(market)

    # Load all .joblib model files listed in metadata from /mnt/models/
    # returns { "TRAVEL": <Pipeline>, "MEALS": <Pipeline>, ... }
    models = load_models(metadata)

    # ── Step 2: Extract training configuration from metadata ──────────────────
    # These values ALWAYS come from metadata — set by the trainer at training time
    # They must NEVER come from config_params or the UI payload
    # because they must exactly match what the model was trained with
    TEXT_COL       = metadata.get("text_column")         # e.g. "TEXT(S4Journal)"
    seasonal_words = metadata.get("seasonal_words", "")  # e.g. "mbfc,cbp,rcls"
    KEY_COLS       = metadata.get("key_cols")            # e.g. ["ExpenseType"]
    process_types  = metadata.get("process_types")       # e.g. ["Provisions and Payments..."]
    allowed_class  = metadata.get("allowed_class")       # e.g. ["Deductible","Non-deductible"]

    # Safety checks — raise a clear error if any required metadata field is missing
    # this means the model was saved without these fields and needs to be retrained
    if not TEXT_COL:      raise ValueError("❌ Metadata missing 'text_column' — retrain the model!")
    if not KEY_COLS:      raise ValueError("❌ Metadata missing 'key_cols' — retrain the model!")
    if not process_types: raise ValueError("❌ Metadata missing 'process_types' — retrain the model!")
    if not allowed_class: raise ValueError("❌ Metadata missing 'allowed_class' — retrain the model!")

    # ── Step 3: Build DataFrames from UI payload ──────────────────────────────
    # Previously these were read from HDI tables — now sent directly from the UI

    # journal_lines → was read from HDI JRNLINEITM table
    # each item in the list is one journal line row as a dict
    journal_lines = payload.get("journal_lines", [])
    if not journal_lines:
        raise ValueError("❌ 'journal_lines' is required in the request body")

    # Convert list of dicts → pandas DataFrame for processing
    predict_df = pd.DataFrame(journal_lines)

    # Drop columns not needed for prediction — errors='ignore' means
    # no crash if a column doesn't exist in this payload
    predict_df.drop(columns=[
        'IMPORTED', 'Source', 'BaseCurrency', 'Debit/Credit',
        'DocumentDate', 'GLTaxCode', 'GLTaxCountry/Region',
        'Product', 'Referencedocument', 'ReportingEntity', 'ReportingEntityCurrency'
    ], inplace=True, errors='ignore')

    # account_mapping → was read from HDI Accountaxcode table
    # maps GL accounts to expense types and process types
    account_mapping_data = payload.get("account_mapping", [])
    if not account_mapping_data:
        raise ValueError("❌ 'account_mapping' is required in the request body")

    # Convert to DataFrame and cast Account to string for consistent merging
    category_mapping = pd.DataFrame(account_mapping_data)
    category_mapping["Account"] = category_mapping["Account"].astype(str)

    # config_params → was read from HDI Configuration table
    # contains overwrite rules and threshold parameters
    config_params = payload.get("config_params", {})
    if not config_params:
        raise ValueError("❌ 'config_params' is required in the request body")

    # ── Step 4: Extract individual config values ──────────────────────────────

    # Overwrite rule: Profit Centre + Oper Unit + Segment → "Payment"
    OW_RL_FM_PROFIT_CENTRE = config_params.get('OW_RL_FM_PROFIT_CENTRE')
    OW_RL_FM_OPER_UNIT     = config_params.get('OW_RL_FM_OPER_UNIT')
    OW_RL_FM_SEGMENT       = config_params.get('OW_RL_FM_SEGMENT')

    # Helper: splits a comma-separated config string into a list
    # e.g. "chatfield,chat" → ["chatfield", "chat"]
    # if already a list, returns as-is
    def split_param(key, default=''):
        val = config_params.get(key, default)
        return val.split(",") if isinstance(val, str) else val

    # Overwrite rule: if text contains these keywords → set output to OW_OUTPUT
    OW_RL_FM_TEXT_CHATFIELD_COL         = split_param('OW_RL_FM_TEXT_CHATFIELD_COL')
    OW_OUTPUT_FM_TEXT_CHATFIELD_COL     = config_params.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL')

    # Overwrite rule: if threshold reached + Non-deductible + keyword → set output
    OW_RL_FM_TEXT_CHATFIELD_COL_CND     = split_param('OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT')
    OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND = config_params.get('OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT')

    # Threshold logic parameters
    THRESHOLD_GROUPBY = split_param('THRESHOLD_GROUPBY')  # columns to group by
    THRESHOLD_SUMBY   = config_params.get('THRESHOLD_SUMBY')  # column to sum
    THRESHOLD_AMOUNT  = float(config_params.get('THRESHOLD_AMOUNT', '50000'))  # threshold value

    # Net-off logic parameters
    NET_OFF_GROUPBY = split_param('NET_OFF_GROUPBY')  # columns to group by for net-off check
    AMOUNT_COL      = config_params.get('AMOUNT_COL') # column holding transaction amount

    # ── Step 5: Filter journal lines to relevant GL accounts ──────────────────
    # Keep only rows where GLAccount belongs to the process_types from metadata
    # process_types comes from metadata — not from config_params
    mask              = category_mapping['ProcessType'].isin(process_types)
    required_accounts = category_mapping.loc[mask, "Account"].astype(str).values.tolist()
    predict_df        = predict_df[predict_df['GLAccount'].astype(str).isin(required_accounts)].copy()

    # ── Step 6: Merge account mapping onto journal lines ──────────────────────
    # Brings in ExpenseType from category_mapping based on GLAccount = Account
    # how="left" keeps all journal rows even if no mapping match found
    predict_df = predict_df.merge(category_mapping, left_on="GLAccount", right_on="Account", how="left")
    predict_df.drop(columns=['ProcessType', 'Account', 'IMPORTED', 'Market'], inplace=True, errors='ignore')

    # ── Step 7: Preprocess text ───────────────────────────────────────────────
    # Apply the same text cleaning that was used during training
    # seasonal_words from metadata ensures consistency with training
    # result stored in new column "ProcessedText" — used by predict()
    predict_df["ProcessedText"] = predict_df[TEXT_COL].apply(
        lambda x: preprocess_method(x, seasonal_words)
    )

    # Cast Amount(Base) to numeric — coerce means invalid values become NaN
    predict_df['Amount(Base)'] = pd.to_numeric(predict_df["Amount(Base)"], errors="coerce")

    # ── Step 8: Build model lookup key ───────────────────────────────────────
    # Combines KEY_COLS into a single string key matching what the trainer used
    # e.g. KEY_COLS=["ExpenseType"] → key="TRAVEL"
    predict_df["key"] = col_aggreate(predict_df, KEY_COLS)

    # ── Step 9: Run ML prediction ─────────────────────────────────────────────
    # Calls predict() above — adds Prediction, Confidence, InflunceParameter,
    # ClassProbabilities columns to predict_df
    predict_df = predict(predict_df, models, "key")

    # ── Step 10: Net-off logic ────────────────────────────────────────────────
    # Net-off: if positive and negative amounts cancel out within the same group
    # mark those rows as NetOff="Yes" — they offset each other, no tax impact

    # First pass: simple group sum — if total is exactly 0 the whole group nets off
    group_sum            = predict_df.groupby(NET_OFF_GROUPBY)[AMOUNT_COL].transform('sum')
    predict_df['NetOff'] = np.where(group_sum == 0, 'Yes', 'No')

    # Second pass: partial net-off — pair positive and negative rows of equal amount
    data_no_netoff               = predict_df[predict_df['NetOff'] == 'No'].copy()
    data_no_netoff["abs_amount"] = data_no_netoff[AMOUNT_COL].abs()   # absolute value for grouping
    data_no_netoff['sign']       = np.where(data_no_netoff[AMOUNT_COL] >= 0, 1, -1)  # +1 or -1
    data_no_netoff['orig_index'] = data_no_netoff.index               # preserve original index for writing back

    def assign_net_off(group):
        # Count how many positive and negative rows exist in this group
        # n_pairs = number of pairs that can be netted off
        n_pairs = min((group['sign'] == 1).sum(), (group['sign'] == -1).sum())
        group   = group.copy()
        group['NetOff'] = 'No'

        # Get indices of positive and negative rows separately
        pos_idx = group[group['sign'] == 1].index
        neg_idx = group[group['sign'] == -1].index

        # Assign cumulative count within positive and negative rows
        # used to identify the first n_pairs rows on each side
        group.loc[pos_idx, 'pos_cumcount'] = np.arange(1, len(pos_idx) + 1)
        group.loc[neg_idx, 'neg_cumcount'] = np.arange(1, len(neg_idx) + 1)

        # Mark the first n_pairs positive and negative rows as NetOff="Yes"
        group.loc[(group['sign'] == 1)  & (group['pos_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        group.loc[(group['sign'] == -1) & (group['neg_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'

        # Remove helper columns before returning
        return group.drop(columns=['pos_cumcount', 'neg_cumcount'])

    # Apply assign_net_off to each group of same absolute amount
    data_no_netoff = data_no_netoff.groupby(
        NET_OFF_GROUPBY + ['abs_amount'], as_index=False
    ).apply(assign_net_off)

    # Write the updated NetOff values back to the main DataFrame using original index
    predict_df.loc[data_no_netoff['orig_index'].to_numpy(), 'NetOff'] = data_no_netoff['NetOff'].to_numpy()

    # ── Step 11: Threshold logic ──────────────────────────────────────────────
    # If the absolute sum of a group is below THRESHOLD_AMOUNT → "Yes" (threshold reached)
    # used in overwrite rule 4 below to handle small-amount Non-deductible rows
    grouped_sum                    = predict_df.groupby(THRESHOLD_GROUPBY)[THRESHOLD_SUMBY].transform('sum')
    predict_df['Threshold Amount'] = grouped_sum.apply(lambda x: 'Yes' if abs(x) < THRESHOLD_AMOUNT else 'No')

    # ── Step 12: Operating unit split ────────────────────────────────────────
    # Classifies each row as "Group" or "Country" based on OperatingLocation code
    # e.g. codes 784, 786, 789 or >= 800 (except 948) → "Group", all others → "Country"
    def get_oper_unit_split(x):
        try:
            val = int(float(str(x)[5:]))   # extract numeric part after first 5 chars
            if (val in (784, 786, 789) or val >= 800) and val != 948:
                return 'Group'
        except Exception:
            pass
        return 'Country'

    predict_df['Oper Unit Split'] = predict_df['OperatingLocation'].apply(get_oper_unit_split)

    # ── Step 13: Overwrite rules ──────────────────────────────────────────────
    # These rules override the ML prediction for specific business conditions
    # applied row by row — first matching rule wins (continue skips remaining rules)
    for index, row in predict_df.iterrows():
        # _ is NOT used here because index IS needed for predict_df.at[index, ...]

        # Rule 1: PREV journal source + account starting with 1 or 2 → FX
        if row["GLTBSource"] == "PREV" and str(row["GLAccount"])[4:].startswith(('1', '2')):
            predict_df.at[index, "Prediction "]      = "FX"
            predict_df.at[index, "InflunceParameter"] = "Overwrite rule since PREV in Journal Source"
            predict_df.at[index, "Confidence"]        = 1
            predict_df.at[index, "ClassProbabilities"]= "N/A"
            continue   # skip remaining rules for this row

        # Rule 2: Profit Centre + Oper Unit + Segment match config values → Payment
        if (str(row["GLAccount"])[4:].replace(".0", "").isdigit()
                and str(row["GLAccount"])[4:].replace(".0", "").startswith(('1', '2'))
                and str(int(row["ProfitCenter"])) == OW_RL_FM_PROFIT_CENTRE
                and str(row["OperatingLocation"])[5:].replace(".0", "") == OW_RL_FM_OPER_UNIT
                and str(row["Segment"])[6:].replace(".0", "") == OW_RL_FM_SEGMENT):
            predict_df.at[index, "Prediction "]      = "Payment"
            predict_df.at[index, "InflunceParameter"] = "Overwrite rule for Payment"
            predict_df.at[index, "Confidence"]        = 1
            predict_df.at[index, "ClassProbabilities"]= "N/A"
            continue   # skip remaining rules for this row

        # Rule 3: Text contains chatfield keywords → overwrite with configured output class
        if any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL):
            predict_df.at[index, "Prediction "]      = OW_OUTPUT_FM_TEXT_CHATFIELD_COL
            predict_df.at[index, "InflunceParameter"] = "Overwrite rule based on Text Chatfield Column"
            predict_df.at[index, "Confidence"]        = 1
            predict_df.at[index, "ClassProbabilities"]= "N/A"
            continue   # skip remaining rules for this row

        # Rule 4: Threshold reached + Non-deductible + keyword → overwrite output
        # only applies when all three conditions are true simultaneously
        if (row["Threshold Amount"] == "Yes"
                and row["Prediction "] == "Non-deductible"
                and any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL_CND)):
            predict_df.at[index, "Prediction "]      = OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND
            predict_df.at[index, "InflunceParameter"] = "Overwrite rule based on Text, Prediction and Threshold"
            predict_df.at[index, "Confidence"]        = 1
            predict_df.at[index, "ClassProbabilities"]= "N/A"
            # no continue needed — this is the last rule

    # ── Step 14: Rename columns ───────────────────────────────────────────────
    predict_df.rename(columns={
        "Threshold Amount": "ThresholdAmountReached",  # cleaner name for output
        "Oper Unit Split":  "OperUnitSplit"             # cleaner name for output
    }, inplace=True)

    # ── Step 15: Select and order final output columns ────────────────────────
    # TEXT_COL is dynamic (from metadata) — not hardcoded as "TEXT(S4Journal)"
    predict_df = predict_df[[
        "MTDPeriod", "DocumentNumber", "GLTBSource", "Segment", "PostingDate",
        "GLAccount", "PartnerEntity", "PostingItem", TEXT_COL,
        "OperatingLocation", "Ledger", "CostCenter", "ProfitCenter",
        "Amount(Base)", "Amount(Transaction)", "TransactionCurrency",
        "Entity/BU", "SourceDocumentNo", "ExpenseType", "ProcessedText",
        "key", "Prediction ", "Confidence", "InflunceParameter",
        "ClassProbabilities", "NetOff", "ThresholdAmountReached", "OperUnitSplit"
    ]]

    # ── Step 16: Return results to UI ─────────────────────────────────────────
    # Previously this step wrote to HDI JRNLOUTPUT table
    # Now results are returned directly as JSON to the UI — no HDI write
    elapsed = time.time() - start
    print(f"✅ Pipeline completed in {elapsed:.1f} seconds")

    return {
        "status":                 "success",
        "message":                "Prediction pipeline completed successfully.",
        "rows_processed":         len(predict_df),          # total rows predicted
        "execution_time_seconds": round(elapsed, 2),        # how long the pipeline took
        "data":                   predict_df.to_dict(orient='records')  # list of dicts → JSON
    }
