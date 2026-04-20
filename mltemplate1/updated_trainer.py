Good thinking. Here are the exact changes needed in `trainer.py`:

## Change 1 — Log reason when single class (no test metrics)

```python
# BEFORE
else:
    print(f"{key} | Single class only, test metrics skipped.")

# AFTER
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
```

---

## Change 2 — Log reason when key group skipped (empty text)

```python
# BEFORE
if texts.str.strip().eq("").all():
    print(f"{key} --> No model created and saved: all raw text entries are empty.")
    continue

# AFTER
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
```

---

## Change 3 — Log reason when skipped due to empty vocabulary

```python
# BEFORE
except ValueError as e:
    if "empty vocabulary" in str(e).lower():
        print(f"{key} --> No model created and saved: empty vocabulary after preprocessing/vectorization.")
        continue
    raise

# AFTER
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
```

---

## Change 4 — Log all training metadata as Custom Tags at the very start

Add this right after `tracker = Tracking()` initializes, before any filtering:

```python
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
```

> **Note:** `MetricTag` appears in the **Tags section** in AI Launchpad (separate from the metrics table) — it's specifically designed for this kind of key-value configuration metadata. It's persistent and searchable, which makes it perfect for training config.

---

## What you'll see in AI Launchpad after this

**Metrics (27+):**

| Name | Value | Step | Labels |
|---|---|---|---|
| `skipped_reason` | 0.0 | 3 | `key_group: XYZ`, `reason: single_class_only`, `class_found: Deductible` |
| `skipped_reason` | 0.0 | 9 | `key_group: ABC`, `reason: empty_text` |
| `training_config` | 0.0 | 0 | `phase: config` |

**Tags:**

| Name | Value |
|---|---|
| `text_column` | `TEXT(S4Journal)` |
| `target_column` | `taxcategory` |
| `allowed_class` | `Deductible,Non-deductible` |
| `seasonal_words` | `mbfc,cbp,rcls` |
| `market` | `Hong Kong` |
| `process_types` | `Provisions and Payments... (B/S) \| Provisions and Payments... (P/L)` |
| `key_cols` | `ExpenseType` |

Rebuild as `v1.9`, push, and you'll have complete visibility into every training run.
