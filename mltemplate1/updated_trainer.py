The issue is `tracker.modify()` — that method likely doesn't exist in your SDK version (same problem as `tags` earlier). We need a different approach.

**The real fix: don't log single_class metric at all when it might hit empty vocab. Instead, collect all results first, then log at the end.**

---

## Change 1 — Remove the single_class metric log from inside the loop

```python
# FIND THIS
        if num_classes < 2:
            step_single += 1
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size
            if tracking_enabled:
                tracker.log_metrics(metrics=[Metric(
                    name=str(key),
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_single, 
                    labels=[
                        MetricLabel(name="remarks",      value=f"SINGLE_CLASS : only {str(y.unique()[0])} found"),
                        MetricLabel(name="train_record", value=str(train_size)),
                    ]
                )])

# REPLACE WITH (no tracking call here — just collect the data)
        if num_classes < 2:
            print(f"{key} | Single class only.")
            train_size   = len(X)
            total_train += train_size
```

---

## Change 2 — Collect results into a dict instead of logging inside loop

Add this dict before the loop:

```python
# FIND
    saved_models = []
    skipped_groups = []
    total_train = 0
    total_test = 0

# REPLACE WITH
    saved_models    = []
    skipped_groups  = []
    total_train     = 0
    total_test      = 0
    key_results     = {}   # ← ADD: stores result per key, logged after loop
```

Then inside each branch, **store to `key_results` instead of calling `tracker.log_metrics`**:

### In the multi-class `else` block:
```python
# FIND
                if tracking_enabled:
                    tracker.log_metrics(metrics=[Metric(
                        name=str(key),
                        value=float(test_acc),
                        timestamp=datetime.now(timezone.utc),
                        step=step_trained,
                        labels=[
                            MetricLabel(name="remarks",       value="TRAINED"),
                            MetricLabel(name="test_accuracy", value=str(round(test_acc, 4))),
                            MetricLabel(name="test_f1_macro", value=str(round(test_f1,  4))),
                            MetricLabel(name="train_record",  value=str(train_size)),
                            MetricLabel(name="test_record",   value=str(test_size)),
                        ]
                    )])

# REPLACE WITH
                key_results[str(key)] = {
                    "type":         "trained",
                    "value":        float(test_acc),
                    "train_size":   train_size,
                    "test_size":    test_size,
                    "test_accuracy":str(round(test_acc, 4)),
                    "test_f1_macro":str(round(test_f1,  4)),
                }
```

### In the empty vocabulary `except` block:
```python
# FIND
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                step_skipped += 1
                step_single -= 1
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                if tracking_enabled:
                    tracker.modify(metrics=[Metric(
                        name=str(key),
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        step=step_skipped,
                        labels=[MetricLabel(name="remarks", value="SKIPPED : empty vocabulary after preprocessing")]
                    )])
                continue
            raise

# REPLACE WITH
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary after preprocessing.")
                skipped_groups.append(str(key))
                key_results[str(key)] = {"type": "skipped_vocab"}   # ← overwrites any single_class entry
                continue
            raise
```

### After the empty text `continue`, store result too:
```python
# FIND
        if texts.str.strip().eq("").all():
            step_skipped += 1
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            if tracking_enabled:
                tracker.log_metrics(metrics=[Metric(
                    name=str(key),
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    step=step_skipped,
                    labels=[MetricLabel(name="remarks", value="SKIPPED : empty text")]
                )])
            continue

# REPLACE WITH
        if texts.str.strip().eq("").all():
            print(f"{key} --> Skipped: all text empty.")
            skipped_groups.append(str(key))
            key_results[str(key)] = {"type": "skipped_empty_text"}
            continue
```

### After saving model successfully, store single_class result:
```python
# FIND
        saved_models.append(model_path)

# REPLACE WITH
        saved_models.append(model_path)
        # store single_class result only if not already overwritten by skipped_vocab
        if str(key) not in key_results:
            key_results[str(key)] = {
                "type":       "single_class",
                "class_found": str(y.unique()[0]),
                "train_size": train_size,
            }
```

---

## Change 3 — Log all key_results AFTER the loop in correct order

Add this block **just before `if not saved_models:`**:

```python
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
```

---

## Also remove the now-unused step counters at the top of the function

```python
# FIND AND REMOVE THESE 3 LINES (no longer needed at top)
    step_trained    = 0
    step_single     = 99
    step_skipped    = 199
```

They are now defined locally inside the post-loop logging block.

---

## Why this fully solves both problems

| Problem | Old approach | New approach |
|---|---|---|
| Double logging | `tracker.modify()` (didn't work) | `key_results[key]` dict — last write wins, so `skipped_vocab` overwrites `single_class` naturally |
| Wrong order | Logged during loop (alphabetical order) | Logged after loop in explicit order: trained → single → skipped |
