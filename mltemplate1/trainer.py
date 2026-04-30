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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))        # ✅ Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))  # ✅ Local: ../shared
_preprocess_module = importlib.import_module("preprocess_text")

preprocess_transform = _preprocess_module.transform
col_aggreate = _preprocess_module.col_aggreate

#CONFIG_FILENAME = "model_config.json"

# ________________________Internal Helper Functions_____________________________________
#If there are keys other than default(ExpenseType) coming from the user via UI then splits the keys(if more than one) column and clear it by strip 
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


def _map_expense_type_from_category_mapping(df,df_category_mapping,market,process_types):
    # 1: Create a filtered account_expense mapping dataframe basis on Market + Process_Type
    filtered_mapping = df_category_mapping[(df_category_mapping["Market"].astype(str) == str(market)) & (df_category_mapping["ProcessType"].isin(process_types))].copy()
    #Keep the two columns "Account", "ExpenseType" from the filtered_mapping and drop rows where account is duplicate
    account_to_expense = (filtered_mapping[["Account", "ExpenseType"]].dropna(subset=["Account"]).drop_duplicates(subset=["Account"], keep="first"))
    # Cast Amount to str datatype if there any
    account_to_expense["Account"] = account_to_expense["Account"].astype(str)

    # 2: Direct mapping from GLAccount (df) to Account (category mapping).
    expense_map = dict(zip(account_to_expense["Account"], account_to_expense["ExpenseType"]))
    df["ExpenseType"] = df["GLAccount"].astype(str).map(expense_map).fillna("")
    return df

# ______________________Main training function_____________________________________
def train_models(df, text_column, target_column,allowed_class, df_category_mapping, market, process_types, key_cols=None, seasonal_words=""):

    # ── 1. Filter dataframe basis on allowed classes from target column ────────────────────────────────────────────────────────
    df = df.copy()
    df = df[df[target_column].notna()]
    df = df[df[target_column].astype(str).str.strip() != ""]
    df = df[df[target_column].isin(allowed_class)]

    key_cols = _normalize_key_cols(key_cols) # creating "key_cols" list of clean key column given by the USER by calling function 
    
    # ── 2. Creating dataframe by mapping expenseType to training dataset along with Market+ProcessType specific Records only
    df = _map_expense_type_from_category_mapping(df=df,df_category_mapping=df_category_mapping,market=market,process_types=process_types)

    # Filter out rows where ExpenseType could not be mapped (empty string) : To Highlight this to trainer who is doing training. so that if required the trainer can update the AccontExpense mapping table.
    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows had no matching ExpenseType and will be skipped. Unmatched GLAccounts: {unmatched['GLAccount'].unique().tolist()} Need to update the account mapping table")
    unmatched_accounts = unmatched["GLAccount"].unique().tolist() if not unmatched.empty else []
    # ── 3. Training Data in which account are matched with Account Expense mapping table will go the training:
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]

    # ── 4. Build key column ---> ExpenseType is always the default key; append optional user-provided keys.
    effective_key_cols = ["ExpenseType"] + key_cols
    df["key"] = col_aggreate(df, effective_key_cols)
    
    # ──5. Init metadata & counters ──────────────────────────────────────────────
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
    #model_dir = os.path.join(os.environ.get("MODEL_DIR", "/tmp/models"), market.replace(" ","_"))
    model_dir = os.environ.get("MODEL_DIR", "/tmp/models")
    print(f"ENV MODEL_DIR = {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

     # ──6. Training loop ─────────────────────────────────────────────────────────
    grouped = df.groupby("key")
    for key, group in grouped:
        texts = group[text_column].fillna("").astype(str)
        X = texts
        y = group[target_column]

        # ── Gate 1: check preprocessed vocabulary is non-empty ────────────────
        # Run the same cleaning that TfidfVectorizer will see. If nothing survives
        # stop here with one clear message empty vocabulary after preprocessing
        cleaned = preprocess_transform(X.tolist(), seasonal_words) #returns a list of strings — one cleaned string per row
        if not any(t.strip() for t in cleaned): # True  → all empty  (any --> any empty, not any() --> all empty) # if t.strip() is not generated any value for all row in cleaned then TRUE
            print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
            skipped_groups.append(str(key))
            key_results[str(key)] = {"type": "skipped_vocab"}
            continue

        # ── Gate 2: check raw text is not entirely blank ──────────────────────
        if X.str.strip().eq("").all():
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            key_results[str(key)] = {"type": "skipped_empty_text"}
            continue


        num_classes = y.nunique()
        min_class_count = y.value_counts().min()
        metrics = {"test_accuracy": None, "test_f1_macro": None}
        train_size      = 0
        test_size       = 0

        if num_classes < 2:
            # Single class — no eval split needed, go straight to final fit
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size

        # ── Check 3: multi class evaluate on held-out split ──────────────────────────────────────────────
        else:
            stratify_arg = y if min_class_count >= 5 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_arg, random_state=42)
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
                    "type":         "trained",
                    "value":        float(test_acc),
                    "train_size":   train_size,
                    "test_size":    test_size,
                    "test_accuracy":str(round(test_acc, 4)),
                    "test_f1_macro":str(round(test_f1,  4)),
                }

        # ── Train final model on full data except gate1 and gate2────────────────────────────────────
        model = _build_pipeline(seasonal_words)
        model.fit(X, y)

        # ── Save model ────────────────────────────────────────────────────────
        #safe_key       = str(key).replace("/", "_").replace("\\", "_")
        #model_filename = f"model_{safe_key}.joblib"
        safe_market       = str(market).replace("/", "_").replace(" ","_")
        safe_key       = str(key).replace("/", "_").replace("\\", "_")
        model_filename = f"model_{safe_market}_{safe_key}.joblib"
        model_filename_sg = f"model_Singapore_{safe_key}.joblib" ##################################### TEMP ##########################################################################################
        model_path     = os.path.join(model_dir, model_filename)
        model_path_sg     = os.path.join(model_dir, model_filename_sg) ################################TEMP ##########################################################################################
        joblib.dump(model, model_path)
        joblib.dump(model, model_path_sg) ############################################################TEMP ###########################################################################################
        metadata["models"][str(key)] = {
            "model_file": model_filename,
            "metrics":    metrics,
            "train_size": train_size,
            "test_size":  test_size,
            "model_file_sg":model_filename_sg,
        }
        saved_models.append(model_path)
        # store single_class result only if not already overwritten by skipped_vocab
        if str(key) not in key_results:
            key_results[str(key)] = {
                "type": "single_class",
                "class_found": str(y.unique()[0]),
                "train_size": train_size,
            }


    if not saved_models:
        return (
            f"No models were saved for {len(df)} rows across {df['key'].nunique()} key group(s). "
            "Check input data, grouping keys, and text content."
        )

    safe_market = str(market).replace("/", "_").replace(" ","_")
    CONFIG_FILENAME = f"model_config_{safe_market}.json"
    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n===== Metadata saved to config =====")
    print(json.dumps(metadata, indent=2))
    print("=====================================\n")

    # TEMP: duplicate config for Singapore
    sg_metadata = {**metadata, "market": "Singapore"}
    sg_metadata["models"] = {k: {**v, "model_file": v['model_file_sg']} for k, v in metadata["models"].items()}
    with open(os.path.join(model_dir, "model_config_Singapore.json"), "w", encoding="utf-8") as f:
        json.dump(sg_metadata, f, indent=2)
    # END TEMP



    message = f"Saved {len(saved_models)} model(s): {saved_models}. Metadata saved to: {config_path}"
    return message, metadata, key_results, unmatched_accounts, total_train, total_test


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



