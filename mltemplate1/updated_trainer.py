import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import FunctionTransformer
import joblib
import sys
import os
import json
import importlib
#import boto
from ai_core_sdk.tracking import Tracking   
from ai_core_sdk.models import Metric, MetricLabel, MetricCustomInfo
from datetime import datetime, timezone


# Add shared folder to path so preprocess_text.py can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))        # ✅ Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))  # ✅ Local: ../shared
_preprocess_module = importlib.import_module("preprocess_text")

preprocess_transform = _preprocess_module.transform
col_aggreate = _preprocess_module.col_aggreate

CONFIG_FILENAME = "model_config.json"

def _normalize_key_cols(key_cols):
    if key_cols is None:
        return []
    if isinstance(key_cols, str):
        return [col.strip() for col in key_cols.split(",") if col.strip()]
    return list(key_cols)


def _build_pipeline(seasonal_words):
    return Pipeline([
        (
            "preprocess",
            FunctionTransformer(
                preprocess_transform,
                validate=False,
                kw_args={"seasonal_words": seasonal_words},
            ),
        ),
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])


def _map_expense_type_from_category_mapping(df,df_category_mapping,market,process_types,):
    # Change 1: derive ExpenseType from category mapping filtered by Market + ProcessType.

    filtered_mapping = df_category_mapping[(df_category_mapping["Market"].astype(str) == str(market)) & (df_category_mapping["ProcessType"].isin(process_types))].copy()

    account_to_expense = (filtered_mapping[["Account", "ExpenseType"]].dropna(subset=["Account"]).drop_duplicates(subset=["Account"], keep="first"))

    account_to_expense["Account"] = account_to_expense["Account"].astype(str)

    # Change 2: direct mapping from GLAccount (df) to Account (category mapping).
    expense_map = dict(zip(account_to_expense["Account"], account_to_expense["ExpenseType"]))
    df["ExpenseType"] = df["GLAccount"].astype(str).map(expense_map).fillna("")

    return df


def train_models(df, text_column, target_column,allowed_class, df_category_mapping, market, process_types, key_cols=None, seasonal_words=""):

    # ── Initialize SAP AI Core tracking ──────────────────────────────────────
    try:
        tracker = Tracking()
        tracking_enabled = True
        print("SAP AI Core tracking initialized.")
    except Exception as e:
        print(f"Tracking not available (running locally?): {e}")
        tracker = None
        tracking_enabled = False

    # ── 2. Filter data ────────────────────────────────────────────────────────
    df = df.copy()
    df = df[df[target_column].notna()]
    df = df[df[target_column].astype(str).str.strip() != ""]
    df = df[df[target_column].isin(allowed_class)]

    key_cols = _normalize_key_cols(key_cols)
    df = _map_expense_type_from_category_mapping(df=df,df_category_mapping=df_category_mapping,market=market,process_types=process_types)

    # Filter out rows where ExpenseType could not be mapped (empty string)
    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows had no matching ExpenseType and will be skipped. Unmatched GLAccounts: {unmatched['GLAccount'].unique().tolist()} Need to update the account mapping table")
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]

    # ── Log unmatched accounts metric ─────────────────────────────────────
    if tracking_enabled and not unmatched.empty:
        unmatched_accounts = ", ".join([f"'{a}'" for a in unmatched["GLAccount"].unique().tolist()])
        tracker.log_metrics(metrics=[Metric(
            name="unmatched_accounts",
            value=float(len(unmatched)),
            timestamp=datetime.now(timezone.utc),
            step=0,
            labels=[
                MetricLabel(name="remarks", value="accounts with no ExpenseType mapping — update account mapping table"),
                MetricLabel(name="accounts", value=unmatched_accounts),
            ]
        )])

    # Change 3: ExpenseType is always the default key; append optional user-provided keys.
    effective_key_cols = ["ExpenseType"] + key_cols
    df["key"] = col_aggreate(df, effective_key_cols)

    
    # ── Log dataset-level execution metrics (step 0) ─────────────────────────
    if tracking_enabled:
        tracker.set_custom_info([
            MetricCustomInfo(name="text_column",   value=str(text_column)),
            MetricCustomInfo(name="target_column", value=str(target_column)),
            MetricCustomInfo(name="allowed_class", value=",".join(allowed_class)),
            MetricCustomInfo(name="seasonal_words",value=str(seasonal_words)),
            MetricCustomInfo(name="market",        value=str(market)),
            MetricCustomInfo(name="process_types", value=" | ".join(process_types)),
            MetricCustomInfo(name="key_cols",      value=",".join(effective_key_cols)),
        ])
    

    metadata = {
        "key_cols": effective_key_cols,
        "text_column": text_column,
        "target_column": target_column,
        "allowed_class" : list(allowed_class),
        "seasonal_words": seasonal_words,
        "market": market,
        "process_types": list(process_types),
        "models": {}}

    saved_models = []
    skipped_groups = []
    total_train = 0
    total_test = 0
    key_results     = {}   # ← ADD: stores result per key, logged after loop

    #model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    # ✅ New: SAP AI Core mounts Object Store to /app/models at runtime
    model_dir = os.environ.get("MODEL_DIR", "/tmp/models")
    #model_dir = os.environ.get("MODEL_OUTPUT_PATH", "/tmp/models")
    print(f"ENV MODEL_DIR = {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    grouped = df.groupby("key")
    for key, group in grouped:
        texts = group[text_column].fillna("").astype(str)

        # ── Check 1: empty text ───────────────────────────────────────────────
        if texts.str.strip().eq("").all():
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            key_results[str(key)] = {"type": "skipped_empty_text"}
            continue

        X = texts
        y = group[target_column]

        num_classes = y.nunique()
        min_class_count = y.value_counts().min()

        metrics = {"test_accuracy": None, "test_f1_macro": None}
        train_size      = 0
        test_size       = 0

        if num_classes < 2:
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size

        # ── Check 3: multi class ──────────────────────────────────────────────
        else:
            step_trained += 1
            stratify_arg = y if min_class_count >= 5 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=stratify_arg, random_state=42
                )
                eval_model = _build_pipeline(seasonal_words)
                eval_model.fit(X_train, y_train)
                y_pred   = eval_model.predict(X_test)
                test_f1  = f1_score(y_test, y_pred, average="macro")
                test_acc = accuracy_score(y_test, y_pred)
                metrics  = {"test_accuracy": float(test_acc), "test_f1_macro": float(test_f1)}
                train_size   = len(X_train)
                test_size    = len(X_test)
                total_train += train_size
                total_test  += test_size
                print(f"{key} | Test F1: {test_f1:.3f} | Test Accuracy: {test_acc:.3f}")
                if tracking_enabled:
                    key_results[str(key)] = {
                    "type":         "trained",
                    "value":        float(test_acc),
                    "train_size":   train_size,
                    "test_size":    test_size,
                    "test_accuracy":str(round(test_acc, 4)),
                    "test_f1_macro":str(round(test_f1,  4)),
                }
            except ValueError as e:
                if "empty vocabulary" not in str(e).lower():
                    raise
                print(f"{key} --> Metrics skipped: empty vocabulary in eval.")

        # ── Train final model on full data ────────────────────────────────────
        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                key_results[str(key)] = {"type": "skipped_vocab"}   # ← overwrites any single_class entry
                continue
            raise
        # ── Save model ────────────────────────────────────────────────────────
        safe_key       = str(key).replace("/", "_").replace("\\", "_")
        model_filename = f"model_{safe_key}.joblib"
        model_path     = os.path.join(model_dir, model_filename)
        joblib.dump(model, model_path)
        metadata["models"][str(key)] = {
            "model_file": model_filename,
            "metrics":    metrics,
            "train_size": train_size,
            "test_size":  test_size,
        }
        saved_models.append(model_path)
        # store single_class result only if not already overwritten by skipped_vocab
        if str(key) not in key_results:
            key_results[str(key)] = {
                "type":       "single_class",
                "class_found": str(y.unique()[0]),
                "train_size": train_size,
            }

    # ── Log all key metrics after loop (correct order: trained → single → skipped) ──
    if tracking_enabled:
        step_trained = 0
        step_single  = 99
        step_skipped = 199

        for kg, result in key_results.items():
            if result["type"] == "trained":
                step_trained += 1
                tracker.log_metrics(metrics=[Metric(
                    name=kg,
                    value=result["value"],
                    timestamp=datetime.now(timezone.utc),
                    step=step_trained,
                    labels=[
                        MetricLabel(name="remarks",       value="TRAINED"),
                        MetricLabel(name="test_accuracy", value=result["test_accuracy"]),
                        MetricLabel(name="test_f1_macro", value=result["test_f1_macro"]),
                        MetricLabel(name="train_record",  value=str(result["train_size"])),
                        MetricLabel(name="test_record",   value=str(result["test_size"])),
                    ]
                )])
            elif result["type"] == "single_class":
                step_single += 1
                tracker.log_metrics(metrics=[Metric(
                    name=kg,
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_single,
                    labels=[
                        MetricLabel(name="remarks",      value=f"SINGLE_CLASS : only {result['class_found']} found"),
                        MetricLabel(name="train_record", value=str(result["train_size"])),
                    ]
                )])
            elif result["type"] in ("skipped_vocab", "skipped_empty_text"):
                step_skipped += 1
                reason = "SKIPPED : empty vocabulary after preprocessing" if result["type"] == "skipped_vocab" else "SKIPPED : empty text"
                tracker.log_metrics(metrics=[Metric(
                    name=kg,
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_skipped,
                    labels=[MetricLabel(name="remarks", value=reason)]
                )])


    if not saved_models:
        return (
            f"No models were saved for {len(df)} rows across {df['key'].nunique()} key group(s). "
            "Check input data, grouping keys, and text content."
        )

    
    # ── Log final summary execution metrics ──────────────────────────────────
    if tracking_enabled and saved_models:
        all_acc = [
            v["metrics"]["test_accuracy"]
            for v in metadata["models"].values()
            if v["metrics"]["test_accuracy"] is not None
        ]
        all_f1 = [
            v["metrics"]["test_f1_macro"]
            for v in metadata["models"].values()
            if v["metrics"]["test_f1_macro"] is not None
        ]
        summary_metrics = [
            Metric(name="Total_key_groups",
                value=float(df["key"].nunique()),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]),
            Metric(name="Total_models_saved",
                value=float(len(saved_models)),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]),
            Metric(name="Total_multiclass_groups",
                value=float(len(all_acc)),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY : key groups having multiclass")]),
            Metric(name="Total_singleclass_groups",
                value=float(sum(
                    1 for v in metadata["models"].values()
                    if v["metrics"]["test_accuracy"] is None
                )),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY : key groups having single class")]),
            Metric(name="Total_rows_loaded",
                value=float(len(df)),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]),
            Metric(name="Total_train_records",
                value=float(total_train),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]),
            Metric(name="Total_test_records",
                value=float(total_test),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]),
        ]
        if all_acc:
            summary_metrics.append(Metric(
                name="avg_test_accuracy",
                value=float(sum(all_acc) / len(all_acc)),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]
            ))
        if all_f1:
            summary_metrics.append(Metric(
                name="avg_test_f1_macro",
                value=float(sum(all_f1) / len(all_f1)),
                timestamp=datetime.now(timezone.utc), step=9999,
                labels=[MetricLabel(name="remarks", value="SUMMARY")]
            ))
        tracker.log_metrics(metrics=summary_metrics)
        print("Execution metrics logged to SAP AI Core.")

    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n===== Metadata saved to config =====")
    print(json.dumps(metadata, indent=2))
    print("=====================================\n")



    return f"Saved {len(saved_models)} model(s): {saved_models}. Metadata saved to: {config_path}"


if __name__ == "__main__":
    #df = pd.read_excel(r"C:\Users\2016565\Downloads\input-data\Training_data_2021_2022_2023 - SAP Format.xlsx")
    #df_category_mapping = pd.read_excel(r"C:\Users\2016565\Downloads\input-data\Tax_code_mapping.xlsx")
    # Import hdi_data from shared folder
    hdi_data = importlib.import_module("hdi_data")

    # Connect to HDI
    conn = hdi_data.get_hdi_connection()
    schema = "T1_TXNANAL_PHY"

    # Load data from HDI tables
    df = hdi_data.read_hdi_table(conn, schema, "JRNLFEATURE")
    df_category_mapping = hdi_data.read_hdi_table(conn, schema, "Accountaxcode")

    conn.close()
    
    text_column = "TEXT(S4Journal)"
    target_column = "taxcategory"
    allowed_class = ['Deductible','Non-deductible']
    seasonal_words = "mbfc,cbp,rcls"
    key_cols = None #["GLAccount"]
    market = "Hong Kong"
    process_types = ["Provisions and Payments of Operating Losses (B/S)","Provisions and Payments of Operating Losses (P/L)"]

    #print(
    train_models(df,text_column=text_column,target_column=target_column,allowed_class = allowed_class,df_category_mapping=df_category_mapping,
            key_cols=key_cols,seasonal_words=seasonal_words,market=market,process_types=process_types)
    #)



