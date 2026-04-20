## Problem 1 — Double logging for same key group

The issue is that when a key group has **single class**, your code logs the `single_class` metric, then **falls through to `model.fit()`** which hits empty vocabulary and logs **again**. You need to **override** the previous metric by logging only `empty_vocabulary` and skipping the single class log entirely when empty vocab is detected.

**Find this block:**
```python
        # ── Train final model on full data ────────────────────────────────────
        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                if tracking_enabled:
                    tracker.log_metrics(metrics=[Metric(
                        name=str(key),
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_idx,
                        labels=[MetricLabel(name="remarks", value="SKIPPED : empty vocabulary after preprocessing")]
                    )])
                continue
            raise
```

**Replace with:**
```python
        # ── Train final model on full data ────────────────────────────────────
        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                if tracking_enabled:
                    # ← modify existing metric for this key to override single_class log
                    tracker.modify(metrics=[Metric(
                        name=str(key),
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_idx,
                        labels=[MetricLabel(name="remarks", value="SKIPPED : empty vocabulary after preprocessing")]
                    )])
                continue
            raise
```

> `tracker.modify()` **overwrites** the previously logged metric for the same `name` + `step`, so the single_class entry gets replaced by empty_vocabulary. No double entry.

---

## Problem 2 — Ordering: Summary → Trained → Single Class → Skipped

SAP AI Launchpad sorts metrics by **step number**. So assign step ranges to control order:

| Group | Step range |
|---|---|
| Summary | `9999` (already correct) |
| TRAINED groups | `1–99` |
| SINGLE_CLASS groups | `100–199` |
| SKIPPED groups | `200–299` |

**Find this line before the loop:**
```python
    step_idx = 0
```

**Replace with:**
```python
    step_trained    = 0
    step_single     = 99
    step_skipped    = 199
```

Then **find and replace `step_idx += 1`** at the top of the loop — remove it entirely, and instead increment the correct counter inside each branch:

```python
    grouped = df.groupby("key")
    for key, group in grouped:
        # ← REMOVE step_idx += 1 from here
        texts = group[text_column].fillna("").astype(str)

        # ── Check 1: empty text ───────────────────────────────────────────────
        if texts.str.strip().eq("").all():
            step_skipped += 1                          # ← ADD
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            if tracking_enabled:
                tracker.log_metrics(metrics=[Metric(
                    name=str(key),
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_skipped,                 # ← CHANGE
                    labels=[MetricLabel(name="remarks", value="SKIPPED : empty text")]
                )])
            continue

        X = texts
        y = group[target_column]
        num_classes     = y.nunique()
        min_class_count = y.value_counts().min()
        metrics         = {"test_accuracy": None, "test_f1_macro": None}
        train_size      = 0
        test_size       = 0

        if num_classes < 2:
            step_single += 1                           # ← ADD
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size
            if tracking_enabled:
                tracker.log_metrics(metrics=[Metric(
                    name=str(key),
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_single,                  # ← CHANGE
                    labels=[
                        MetricLabel(name="remarks",      value=f"SINGLE_CLASS : only {str(y.unique()[0])} found"),
                        MetricLabel(name="train_record", value=str(train_size)),
                    ]
                )])

        else:
            step_trained += 1                          # ← ADD
            stratify_arg = y if min_class_count >= 5 else None
            try:
                ...
                if tracking_enabled:
                    tracker.log_metrics(metrics=[Metric(
                        name=str(key),
                        value=float(test_acc),
                        timestamp=datetime.now(timezone.utc),
                        step=step_trained,             # ← CHANGE
                        labels=[...]
                    )])
            except ValueError as e:
                ...

        # ── Train final model on full data ────────────────────────────────────
        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                step_skipped += 1                      # ← ADD
                # remove the single_class step from single counter since this overrides it
                step_single -= 1                       # ← ADD
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                if tracking_enabled:
                    tracker.modify(metrics=[Metric(    # ← modify to override single_class log
                        name=str(key),
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_skipped,             # ← CHANGE to skipped range
                        labels=[MetricLabel(name="remarks", value="SKIPPED : empty vocabulary after preprocessing")]
                    )])
                continue
            raise
```

---

## Result in Metrics table (ordered correctly)

| Name | Value | Step | Labels |
|---|---|---|---|
| `Total_key_groups` | 16 | 9999 | `remarks: SUMMARY` |
| `Total_models_saved` | 15 | 9999 | `remarks: SUMMARY` |
| ... other summary rows ... | | 9999 | |
| `Other Costs` | 0.9996 | 1 | `remarks: TRAINED` ... |
| `Amt w-off w/o Pro-Op Loss Ext` | 0.9888 | 2 | `remarks: TRAINED` ... |
| ... other trained groups ... | | 1-99 | |
| `ECL Allowances - Stage 1 to 3` | 0.0 | 100 | `remarks: SINGLE_CLASS...` |
| ... other single class groups ... | | 100-199 | |
| `P/L-Pro no long req-Op.Loss In` | 0.0 | 200 | `remarks: SKIPPED : empty vocabulary...` |
| ... other skipped groups ... | | 200-299 | |
