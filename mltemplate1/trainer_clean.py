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

# Add shared folder to path so preprocess_text.py can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))        # Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))  # Local: ../shared
_preprocess_module = importlib.import_module("preprocess_text")

preprocess_transform = _preprocess_module.transform
col_aggreate         = _preprocess_module.col_aggreate

CONFIG_FILENAME = "model_config.json"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        ("nb",    MultinomialNB()),
    ])


def _map_expense_type_from_category_mapping(df, df_category_mapping, market, process_types):
    """Derive ExpenseType from category mapping filtered by Market + ProcessType."""
    filtered_mapping = df_category_mapping[
        (df_category_mapping["Market"].astype(str) == str(market)) &
        (df_category_mapping["ProcessType"].isin(process_types))
    ].copy()

    account_to_expense = (
        filtered_mapping[["Account", "ExpenseType"]]
        .dropna(subset=["Account"])
        .drop_duplicates(subset=["Account"], keep="first")
    )
    account_to_expense["Account"] = account_to_expense["Account"].astype(str)

    expense_map       = dict(zip(account_to_expense["Account"], account_to_expense["ExpenseType"]))
    df["ExpenseType"] = df["GLAccount"].astype(str).map(expense_map).fillna("")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# Returns: (message, metadata, key_results, unmatched_accounts, total_train, total_test)
# ─────────────────────────────────────────────────────────────────────────────

def train_models(df, text_column, target_column, allowed_class, df_category_mapping,
                 market, process_types, key_cols=None, seasonal_words=""):

    # ── Filter data ───────────────────────────────────────────────────────────
    df = df.copy()
    df = df[df[target_column].notna()]
    df = df[df[target_column].astype(str).str.strip() != ""]
    df = df[df[target_column].isin(allowed_class)]

    key_cols = _normalize_key_cols(key_cols)
    df       = _map_expense_type_from_category_mapping(
        df=df, df_category_mapping=df_category_mapping,
        market=market, process_types=process_types,
    )

    # ── Unmatched accounts ────────────────────────────────────────────────────
    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(
            f"Warning: {len(unmatched)} rows had no matching ExpenseType and will be skipped. "
            f"Unmatched GLAccounts: {unmatched['GLAccount'].unique().tolist()} "
            "Need to update the account mapping table"
        )
    unmatched_accounts = unmatched["GLAccount"].unique().tolist() if not unmatched.empty else []
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]

    # ── Build key column ──────────────────────────────────────────────────────
    effective_key_cols = ["ExpenseType"] + key_cols
    df["key"]          = col_aggreate(df, effective_key_cols)

    # ── Init metadata & counters ──────────────────────────────────────────────
    metadata = {
        "key_cols":       effective_key_cols,
        "text_column":    text_column,
        "target_column":  target_column,
        "allowed_class":  list(allowed_class),
        "seasonal_words": seasonal_words,
        "market":         market,
        "process_types":  list(process_types),
        "models":         {},
    }

    saved_models   = []
    skipped_groups = []
    total_train    = 0
    total_test     = 0
    key_results    = {}

    model_dir = os.environ.get("MODEL_DIR", "/tmp/models")
    print(f"ENV MODEL_DIR = {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for key, group in df.groupby("key"):
        texts = group[text_column].fillna("").astype(str)

        # Skip: all text empty
        if texts.str.strip().eq("").all():
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            key_results[str(key)] = {"type": "skipped_empty_text"}
            continue

        X = texts
        y = group[target_column]

        num_classes     = y.nunique()
        min_class_count = y.value_counts().min()
        metrics         = {"test_accuracy": None, "test_f1_macro": None}
        train_size      = 0
        test_size       = 0

        if num_classes < 2:
            # Single class — no eval split possible
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size

        else:
            # Multi-class: evaluate on held-out split
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
                key_results[str(key)] = {
                    "type":          "trained",
                    "value":         float(test_acc),
                    "train_size":    train_size,
                    "test_size":     test_size,
                    "test_accuracy": str(round(test_acc, 4)),
                    "test_f1_macro": str(round(test_f1,  4)),
                }
            except ValueError as e:
                if "empty vocabulary" not in str(e).lower():
                    raise
                print(f"{key} --> Metrics skipped: empty vocabulary in eval.")

        # Train final model on full data
        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                key_results[str(key)] = {"type": "skipped_vocab"}
                continue
            raise

        # Save model
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

        if str(key) not in key_results:
            key_results[str(key)] = {
                "type":        "single_class",
                "class_found": str(y.unique()[0]),
                "train_size":  train_size,
            }

    # ── Early exit: nothing saved ─────────────────────────────────────────────
    if not saved_models:
        return (
            f"No models were saved for {len(df)} rows across {df['key'].nunique()} key group(s). "
            "Check input data, grouping keys, and text content.",
            None, {}, unmatched_accounts, 0, 0,
        )

    # ── Save config ───────────────────────────────────────────────────────────
    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n===== Metadata saved to config =====")
    print(json.dumps(metadata, indent=2))
    print("=====================================\n")

    message = f"Saved {len(saved_models)} model(s): {saved_models}. Metadata saved to: {config_path}"
    return message, metadata, key_results, unmatched_accounts, total_train, total_test


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint — triggered directly via YAML when metrics are NOT needed
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hdi_data = importlib.import_module("hdi_data")

    conn   = hdi_data.get_hdi_connection()
    schema = "T1_TXNANAL_PHY"

    df                  = hdi_data.read_hdi_table(conn, schema, "JRNLFEATURE")
    df_category_mapping = hdi_data.read_hdi_table(conn, schema, "Accountaxcode")
    conn.close()

    text_column    = "TEXT(S4Journal)"
    target_column  = "taxcategory"
    allowed_class  = ["Deductible", "Non-deductible"]
    seasonal_words = "mbfc,cbp,rcls"
    key_cols       = None
    market         = "Hong Kong"
    process_types  = [
        "Provisions and Payments of Operating Losses (B/S)",
        "Provisions and Payments of Operating Losses (P/L)",
    ]

    message, *_ = train_models(
        df, text_column=text_column, target_column=target_column,
        allowed_class=allowed_class, df_category_mapping=df_category_mapping,
        key_cols=key_cols, seasonal_words=seasonal_words,
        market=market, process_types=process_types,
    )
    print(message)
