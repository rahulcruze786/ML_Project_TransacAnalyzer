import numpy as np
import pandas as pd
import time
import importlib

from model_loader import load_metadata, load_models

_preprocess_module = importlib.import_module("preprocess_text")
preprocess_method  = _preprocess_module.preprocess_method
col_aggreate       = _preprocess_module.col_aggreate


def predict(df, models, key_col):
    predictions, confidences, influences, all_class_probs = [], [], [], []

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


def process_data(payload):
    start    = time.time()
    market   = payload.get("market")
    metadata = load_metadata(market)
    models   = load_models(metadata)

    TEXT_COL       = metadata.get("text_column")
    seasonal_words = metadata.get("seasonal_words", "")
    KEY_COLS       = metadata.get("key_cols")
    process_types  = metadata.get("process_types")
    allowed_class  = metadata.get("allowed_class")

    if not TEXT_COL:    raise ValueError("❌ Metadata missing 'text_column'")
    if not KEY_COLS:    raise ValueError("❌ Metadata missing 'key_cols'")
    if not process_types: raise ValueError("❌ Metadata missing 'process_types'")
    if not allowed_class: raise ValueError("❌ Metadata missing 'allowed_class'")

    # Build DataFrames from payload
    journal_lines = payload.get("journal_lines", [])
    if not journal_lines:
        raise ValueError("❌ 'journal_lines' is required")

    predict_df = pd.DataFrame(journal_lines)
    predict_df.drop(columns=[
        'IMPORTED', 'Source', 'BaseCurrency', 'Debit/Credit',
        'DocumentDate', 'GLTaxCode', 'GLTaxCountry/Region',
        'Product', 'Referencedocument', 'ReportingEntity', 'ReportingEntityCurrency'
    ], inplace=True, errors='ignore')

    account_mapping_data = payload.get("account_mapping", [])
    if not account_mapping_data:
        raise ValueError("❌ 'account_mapping' is required")

    category_mapping = pd.DataFrame(account_mapping_data)
    category_mapping["Account"] = category_mapping["Account"].astype(str)

    config_params = payload.get("config_params", {})
    if not config_params:
        raise ValueError("❌ 'config_params' is required")

    # Extract config values
    OW_RL_FM_PROFIT_CENTRE = config_params.get('OW_RL_FM_PROFIT_CENTRE')
    OW_RL_FM_OPER_UNIT     = config_params.get('OW_RL_FM_OPER_UNIT')
    OW_RL_FM_SEGMENT       = config_params.get('OW_RL_FM_SEGMENT')

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

    # Filter + merge
    mask              = category_mapping['ProcessType'].isin(process_types)
    required_accounts = category_mapping.loc[mask, "Account"].astype(str).values.tolist()
    predict_df        = predict_df[predict_df['GLAccount'].astype(str).isin(required_accounts)].copy()
    predict_df        = predict_df.merge(category_mapping, left_on="GLAccount", right_on="Account", how="left")
    predict_df.drop(columns=['ProcessType', 'Account', 'IMPORTED', 'Market'], inplace=True, errors='ignore')

    # Preprocess + key + predict
    predict_df["ProcessedText"] = predict_df[TEXT_COL].apply(lambda x: preprocess_method(x, seasonal_words))
    predict_df['Amount(Base)']  = pd.to_numeric(predict_df["Amount(Base)"], errors="coerce")
    predict_df["key"]           = col_aggreate(predict_df, KEY_COLS)
    predict_df                  = predict(predict_df, models, "key")

    # Net Off
    group_sum            = predict_df.groupby(NET_OFF_GROUPBY)[AMOUNT_COL].transform('sum')
    predict_df['NetOff'] = np.where(group_sum == 0, 'Yes', 'No')

    data_no_netoff               = predict_df[predict_df['NetOff'] == 'No'].copy()
    data_no_netoff["abs_amount"] = data_no_netoff[AMOUNT_COL].abs()
    data_no_netoff['sign']       = np.where(data_no_netoff[AMOUNT_COL] >= 0, 1, -1)
    data_no_netoff['orig_index'] = data_no_netoff.index

    def assign_net_off(group):
        n_pairs     = min((group['sign'] == 1).sum(), (group['sign'] == -1).sum())
        group       = group.copy()
        group['NetOff'] = 'No'
        pos_idx     = group[group['sign'] == 1].index
        neg_idx     = group[group['sign'] == -1].index
        group.loc[pos_idx, 'pos_cumcount'] = np.arange(1, len(pos_idx) + 1)
        group.loc[neg_idx, 'neg_cumcount'] = np.arange(1, len(neg_idx) + 1)
        group.loc[(group['sign'] == 1)  & (group['pos_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        group.loc[(group['sign'] == -1) & (group['neg_cumcount'] <= n_pairs), 'NetOff'] = 'Yes'
        return group.drop(columns=['pos_cumcount', 'neg_cumcount'])

    data_no_netoff = data_no_netoff.groupby(
        NET_OFF_GROUPBY + ['abs_amount'], as_index=False
    ).apply(assign_net_off)
    predict_df.loc[data_no_netoff['orig_index'].to_numpy(), 'NetOff'] = data_no_netoff['NetOff'].to_numpy()

    # Threshold
    grouped_sum                    = predict_df.groupby(THRESHOLD_GROUPBY)[THRESHOLD_SUMBY].transform('sum')
    predict_df['Threshold Amount'] = grouped_sum.apply(lambda x: 'Yes' if abs(x) < THRESHOLD_AMOUNT else 'No')

    # Oper Unit Split
    def get_oper_unit_split(x):
        try:
            val = int(float(str(x)[5:]))
            if (val in (784, 786, 789) or val >= 800) and val != 948:
                return 'Group'
        except Exception:
            pass
        return 'Country'

    predict_df['Oper Unit Split'] = predict_df['OperatingLocation'].apply(get_oper_unit_split)

    # Overwrite Rules
    for index, row in predict_df.iterrows():
        if row["GLTBSource"] == "PREV" and str(row["GLAccount"])[4:].startswith(('1', '2')):
            predict_df.at[index, "Prediction "]       = "FX"
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule since PREV in Journal Source"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

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

        if any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text Chatfield Column"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"
            continue

        if (row["Threshold Amount"] == "Yes"
                and row["Prediction "] == "Non-deductible"
                and any(s in str(row[TEXT_COL]).lower() for s in OW_RL_FM_TEXT_CHATFIELD_COL_CND)):
            predict_df.at[index, "Prediction "]       = OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND
            predict_df.at[index, "InflunceParameter"]  = "Overwrite rule based on Text, Prediction and Threshold"
            predict_df.at[index, "Confidence"]         = 1
            predict_df.at[index, "ClassProbabilities"] = "N/A"

    # Final columns
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

    elapsed = time.time() - start
    print(f"✅ Pipeline completed in {elapsed:.1f} seconds")

    return {
        "status":                 "success",
        "message":                "Prediction pipeline completed successfully.",
        "rows_processed":         len(predict_df),
        "execution_time_seconds": round(elapsed, 2),
        "data":                   predict_df.to_dict(orient='records')
    }
