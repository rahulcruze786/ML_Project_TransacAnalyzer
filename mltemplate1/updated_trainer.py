Looking at your current code, here are the **exact changes** needed — nothing else touches:

---

## Change 1 — Initialize `skipped_groups`, `total_train`, `total_test` before the loop

```python
# FIND THIS LINE
saved_models = []

# REPLACE WITH
saved_models   = []
skipped_groups = []
total_train    = 0
total_test     = 0
```

---

## Change 2 — Replace the entire `for key, group in grouped:` loop

```python
# REMOVE EVERYTHING FROM HERE
for key, group in grouped:
    # advance per-key step counter (used for tracking)
    step_idx += 1
    texts = group[text_column].fillna("").astype(str)
    ...
    ...
    saved_models.append(model_path)
# TO HERE (just before "if not saved_models:")


# REPLACE WITH THIS ENTIRE BLOCK
for key, group in grouped:
    step_idx += 1
    texts = group[text_column].fillna("").astype(str)

    # ── Check 1: empty text ───────────────────────────────────────────────
    if texts.str.strip().eq("").all():
        print(f"{key} --> Skipped: all text empty.")
        skipped_groups.append(str(key))
        if tracking_enabled:
            tracker.log_metrics(metrics=[Metric(
                name=str(key),
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                step=step_idx,
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

    # ── Check 2: single class ─────────────────────────────────────────────
    if num_classes < 2:
        print(f"{key} | Single class only.")
        train_size   = len(X)
        total_train += train_size
        if tracking_enabled:
            tracker.log_metrics(metrics=[Metric(
                name=str(key),
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                step=step_idx,
                labels=[
                    MetricLabel(name="remarks",      value=f"SINGLE_CLASS : only {str(y.unique()[0])} found"),
                    MetricLabel(name="train_record", value=str(train_size)),
                ]
            )])

    # ── Check 3: multi class ──────────────────────────────────────────────
    else:
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
                tracker.log_metrics(metrics=[Metric(
                    name=str(key),
                    value=float(test_acc),
                    timestamp=datetime.now(timezone.utc),
                    step=step_idx,
                    labels=[
                        MetricLabel(name="remarks",       value="TRAINED"),
                        MetricLabel(name="test_accuracy", value=str(round(test_acc, 4))),
                        MetricLabel(name="test_f1_macro", value=str(round(test_f1,  4))),
                        MetricLabel(name="train_record",  value=str(train_size)),
                        MetricLabel(name="test_record",   value=str(test_size)),
                    ]
                )])
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
```

---

## Change 3 — Replace the entire summary metrics block at the bottom

```python
# REMOVE EVERYTHING FROM HERE
# ── Log final summary execution metrics ──────────────────────────────────
if tracking_enabled and saved_models:
    ...
    tracker.log_metrics(metrics=summary_metrics)
    print("Execution metrics logged to SAP AI Core.")
# TO HERE


# REPLACE WITH THIS
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
```

---

## Summary — only 3 places changed

| Change | Location | What |
|---|---|---|
| 1 | After `saved_models = []` | Add `skipped_groups`, `total_train`, `total_test` |
| 2 | Entire `for key, group in grouped:` loop | Restructured with correct flow + new metric names |
| 3 | Summary metrics block at bottom | New names + `remarks` label + totals |
