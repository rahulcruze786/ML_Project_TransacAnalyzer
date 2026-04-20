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
from ai_core_sdk.models import Metric, MetricLabel
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
        if tracking_enabled:
            from ai_core_sdk.models import MetricTag
            tracker.log_metrics(
                metrics=[
                    Metric(
                        name="training_config",
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=0,
                        labels=[MetricLabel(name="phase", value="config")]
                    )
                ],
                tags=[
                    MetricTag(name="text_column",    value=str(text_column)),
                    MetricTag(name="target_column",  value=str(target_column)),
                    MetricTag(name="allowed_class",  value=",".join(allowed_class)),
                    MetricTag(name="seasonal_words", value=str(seasonal_words)),
                    MetricTag(name="market",         value=str(market)),
                    MetricTag(name="process_types",  value=" | ".join(process_types)),
                    MetricTag(name="key_cols",       value=",".join(effective_key_cols)),
                ]
            )
        tracking_enabled = True
        print("SAP AI Core tracking initialized.")
    except Exception as e:
        print(f"Tracking not available (running locally?): {e}")
        tracker = None
        tracking_enabled = False

    
    # step counter for per-key metric logs
    step_idx = 0
    
    
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

    # Change 3: ExpenseType is always the default key; append optional user-provided keys.
    effective_key_cols = ["ExpenseType"] + key_cols
    df["key"] = col_aggreate(df, effective_key_cols)

    
    # ── Log dataset-level execution metrics (step 0) ─────────────────────────
    if tracking_enabled:
        tracker.log_metrics(metrics=[
            Metric(
                name="total_rows",
                value=float(len(df)),
                timestamp=datetime.now(timezone.utc),
                step=0,
                labels=[MetricLabel(name="phase", value="data_load")]
            ),
            Metric(
                name="num_key_groups",
                value=float(df["key"].nunique()),
                timestamp=datetime.now(timezone.utc),
                step=0,
                labels=[MetricLabel(name="phase", value="data_load")]
            ),
            Metric(
                name="unmatched_rows",
                value=float(len(unmatched)),
                timestamp=datetime.now(timezone.utc),
                step=0,
                labels=[MetricLabel(name="phase", value="data_load")]
            ),
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
    #model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    # ✅ New: SAP AI Core mounts Object Store to /app/models at runtime
    model_dir = os.environ.get("MODEL_DIR", "/tmp/models")
    #model_dir = os.environ.get("MODEL_OUTPUT_PATH", "/tmp/models")
    print(f"ENV MODEL_DIR = {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    grouped = df.groupby("key")
    for key, group in grouped:
        # advance per-key step counter (used for tracking)
        step_idx += 1
        texts = group[text_column].fillna("").astype(str)
        #if texts.str.strip().eq("").all():
        #    print(f"{key} --> No model created and saved: all raw text entries are empty.")
        #    continue

        if texts.str.strip().eq("").all():
            print(f"{key} --> No model created and saved: all raw text entries are empty.")
            if tracking_enabled:
                tracker.log_metrics(metrics=[
                    Metric(
                        name="skipped_reason",
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_idx,
                        labels=[
                            MetricLabel(name="key_group", value=str(key)),
                            MetricLabel(name="reason", value="empty_text")
                        ]
                    )
                ])
            continue

        X = texts
        y = group[target_column]

        num_classes = y.nunique()
        min_class_count = y.value_counts().min()

        metrics = {"test_accuracy": None, "test_f1_macro": None}

        if num_classes >= 2:
            stratify_arg = y if min_class_count >= 5 else None

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=stratify_arg, random_state=42
                )

                eval_model = _build_pipeline(seasonal_words)
                eval_model.fit(X_train, y_train)

                y_pred = eval_model.predict(X_test)
                test_f1 = f1_score(y_test, y_pred, average="macro")
                test_acc = accuracy_score(y_test, y_pred)
                metrics = {"test_accuracy": float(test_acc), "test_f1_macro": float(test_f1)}

                print(f"{key} | Test F1: {test_f1:.3f} | Test Accuracy: {test_acc:.3f}")
                    
                # ── Log per-key execution metrics (step = key group index) ───
                if tracking_enabled:
                    tracker.log_metrics(metrics=[
                        Metric(
                            name="test_accuracy",
                            value=float(test_acc),
                            timestamp=datetime.now(timezone.utc),
                            step=step_idx,
                            labels=[MetricLabel(name="key_group", value=str(key))]
                        ),
                        Metric(
                            name="test_f1_macro",
                            value=float(test_f1),
                            timestamp=datetime.now(timezone.utc),
                            step=step_idx,
                            labels=[MetricLabel(name="key_group", value=str(key))]
                        ),
                        Metric(
                            name="train_size",
                            value=float(len(X_train)),
                            timestamp=datetime.now(timezone.utc),
                            step=step_idx,
                            labels=[MetricLabel(name="key_group", value=str(key))]
                        ),
                    ])
                
            except ValueError as e:
                if "empty vocabulary" not in str(e).lower():
                    raise
                print(f"{key} --> Metrics skipped: empty vocabulary.")
        #else:
        #    print(f"{key} | Single class only, test metrics skipped.")

        else:
            print(f"{key} | Single class only, test metrics skipped.")
            if tracking_enabled:
                tracker.log_metrics(metrics=[
                    Metric(
                        name="skipped_reason",
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_idx,
                        labels=[
                            MetricLabel(name="key_group", value=str(key)),
                            MetricLabel(name="reason", value="single_class_only"),
                            MetricLabel(name="class_found", value=str(y.unique()[0]))
                        ]
                    )
                ])

        model = _build_pipeline(seasonal_words)

        try:
            model.fit(X, y)
        #except ValueError as e:
        #    if "empty vocabulary" in str(e).lower():
        #        print(f"{key} --> No model created and saved: empty vocabulary after preprocessing/vectorization.")
        #        continue
        #    raise
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> No model created and saved: empty vocabulary after preprocessing/vectorization.")
                if tracking_enabled:
                    tracker.log_metrics(metrics=[
                        Metric(
                            name="skipped_reason",
                            value=0.0,
                            timestamp=datetime.now(timezone.utc),
                            step=step_idx,
                            labels=[
                                MetricLabel(name="key_group", value=str(key)),
                                MetricLabel(name="reason", value="empty_vocabulary")
                            ]
                        )
                    ])
                continue
            raise

         # ✅ Fix model_path to use just filename, not absolute local path
        safe_key = str(key).replace("/", "_").replace("\\", "_")
        #model_path = os.path.join(model_dir, f"model_{safe_key}.joblib")
        model_filename = f"model_{safe_key}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        #joblib.dump(model, model_path)
        #os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        #metadata["models"][str(key)] = {"model_path": model_path, "metrics": metrics}
        # ✅ Store only filename in metadata, not full local path
        metadata["models"][str(key)] = {
            "model_file": model_filename,   # ✅ just filename
            "metrics": metrics}
        saved_models.append(model_path)

    if not saved_models:
        return (
            f"No models were saved for {len(df)} rows across {df['key'].nunique()} key group(s). "
            "Check input data, grouping keys, and text content."
        )

    
    # ── Log final summary execution metrics ──────────────────────────────────
    if tracking_enabled and saved_models:
        # Collect only non-None accuracies and F1s for summary
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
            Metric(
                name="models_saved",
                value=float(len(saved_models)),
                timestamp=datetime.now(timezone.utc),
                step=9999,
                labels=[MetricLabel(name="phase", value="summary")]
            ),
        ]
        if all_acc:
            summary_metrics.append(Metric(
                name="avg_test_accuracy",
                value=float(sum(all_acc) / len(all_acc)),
                timestamp=datetime.now(timezone.utc),
                step=9999,
                labels=[MetricLabel(name="phase", value="summary")]
            ))
        if all_f1:
            summary_metrics.append(Metric(
                name="avg_test_f1_macro",
                value=float(sum(all_f1) / len(all_f1)),
                timestamp=datetime.now(timezone.utc),
                step=9999,
                labels=[MetricLabel(name="phase", value="summary")]
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



