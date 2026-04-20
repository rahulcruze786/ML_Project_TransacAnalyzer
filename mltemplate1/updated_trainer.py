Looking at your current code, only **2 things** need to change:

---

## Change 1 — Remove the old data-level metrics block (lines ~88-97)

```python
# FIND AND DELETE THIS ENTIRE BLOCK
        # ── 5. Log data-level metrics ─────────────────────────────────────────────
        if tracking_enabled:
            tracker.log_metrics(metrics=[
                Metric(name="total_rows",     value=float(len(df)),
                    timestamp=datetime.now(timezone.utc), step=0,
                    labels=[MetricLabel(name="phase", value="data_load")]),
                Metric(name="num_key_groups", value=float(df["key"].nunique()),
                    timestamp=datetime.now(timezone.utc), step=0,
                    labels=[MetricLabel(name="phase", value="data_load")]),
                Metric(name="unmatched_rows", value=float(len(unmatched)),
                    timestamp=datetime.now(timezone.utc), step=0,
                    labels=[MetricLabel(name="phase", value="data_load")]),
            ])
```

These are now replaced by `Total_rows_loaded`, `Total_key_groups` in the summary block. No need to log them twice.

---

## Change 2 — Remove the `training_config` metric (keep only `set_custom_info`)

```python
# FIND AND DELETE JUST THESE LINES (keep the set_custom_info call below it)
        # Call 1 - log the metric
        tracker.log_metrics(metrics=[
            Metric(
                name="training_config",
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                step=0,
                labels=[MetricLabel(name="phase", value="config")]
            )
        ])
```

So after this change, the entire `if tracking_enabled:` block before the loop should look like **only this**:

```python
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
```

---

## That's it — everything else in your code is already correct ✅

The loop, the summary block, the per-key metrics are all good. Just those 2 removals and you'll see all 16 key groups by name in the metrics table with no extra noise rows.
