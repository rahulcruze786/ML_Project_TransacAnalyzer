"""
metrics.py  —  SAP AI Core entrypoint (with metrics)

This file is triggered by the SAP AI Core YAML workflow when metric logging is required.
It calls train_models() from trainer.py and logs all results to SAP AI Core Tracking.

YAML command when metrics ARE needed:
    command: ["python", "metrics.py"]

YAML command when metrics are NOT needed:
    command: ["python", "trainer.py"]
"""

import importlib
import sys
import os
from datetime import datetime, timezone

from ai_core_sdk.tracking import Tracking
from ai_core_sdk.models import Metric, MetricLabel, MetricCustomInfo

from trainer import train_models

# Add shared folder to path so hdi_data.py can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))        # Docker: /app/shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))  # Local: ../shared
hdi_data = importlib.import_module("hdi_data")


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

conn   = hdi_data.get_hdi_connection()
schema = "T1_TXNANAL_PHY"

df                  = hdi_data.read_hdi_table(conn, schema, "JRNLFEATURE")
df_category_mapping = hdi_data.read_hdi_table(conn, schema, "Accountaxcode")
conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

message, metadata, key_results, unmatched_accounts, total_train, total_test = train_models(
    df,
    text_column         = text_column,
    target_column       = target_column,
    allowed_class       = allowed_class,
    df_category_mapping = df_category_mapping,
    key_cols            = key_cols,
    seasonal_words      = seasonal_words,
    market              = market,
    process_types       = process_types,
)
print(message)

if metadata is None:
    print("Training produced no models. Skipping metric logging.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# SAP AI Core — log all metrics
# ─────────────────────────────────────────────────────────────────────────────

tracker            = Tracking()
effective_key_cols = metadata["key_cols"]

# ── Custom run info ───────────────────────────────────────────────────────────
tracker.set_custom_info([
    MetricCustomInfo(name="text_column",    value=str(text_column)),
    MetricCustomInfo(name="target_column",  value=str(target_column)),
    MetricCustomInfo(name="allowed_class",  value=",".join(allowed_class)),
    MetricCustomInfo(name="seasonal_words", value=str(seasonal_words)),
    MetricCustomInfo(name="market",         value=str(market)),
    MetricCustomInfo(name="process_types",  value=" | ".join(process_types)),
    MetricCustomInfo(name="key_cols",       value=",".join(effective_key_cols)),
])

# ── Unmatched accounts ────────────────────────────────────────────────────────
if unmatched_accounts:
    accounts_str = ", ".join([f"'{a}'" for a in unmatched_accounts])
    tracker.log_metrics(metrics=[Metric(
        name="unmatched_accounts",
        value=float(len(unmatched_accounts)),
        timestamp=datetime.now(timezone.utc),
        step=0,
        labels=[
            MetricLabel(name="remarks",  value="accounts with no ExpenseType mapping — update account mapping table"),
            MetricLabel(name="accounts", value=accounts_str),
        ],
    )])

# ── Per-key metrics ───────────────────────────────────────────────────────────
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
            ],
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
            ],
        )])

    elif result["type"] in ("skipped_vocab", "skipped_empty_text"):
        step_skipped += 1
        reason = (
            "SKIPPED : empty vocabulary after preprocessing"
            if result["type"] == "skipped_vocab"
            else "SKIPPED : empty text"
        )
        tracker.log_metrics(metrics=[Metric(
            name=kg,
            value=0.0,
            timestamp=datetime.now(timezone.utc),
            step=step_skipped,
            labels=[MetricLabel(name="remarks", value=reason)],
        )])

# ── Summary metrics ───────────────────────────────────────────────────────────
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
           value=float(len(key_results)),
           timestamp=datetime.now(timezone.utc), step=9999,
           labels=[MetricLabel(name="remarks", value="SUMMARY")]),
    Metric(name="Total_models_saved",
           value=float(len(metadata["models"])),
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
           value=float(sum(
               v["train_size"] + v["test_size"] for v in metadata["models"].values()
           )),
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
        labels=[MetricLabel(name="remarks", value="SUMMARY")],
    ))
if all_f1:
    summary_metrics.append(Metric(
        name="avg_test_f1_macro",
        value=float(sum(all_f1) / len(all_f1)),
        timestamp=datetime.now(timezone.utc), step=9999,
        labels=[MetricLabel(name="remarks", value="SUMMARY")],
    ))

tracker.log_metrics(metrics=summary_metrics)
print("Execution metrics logged to SAP AI Core.")
