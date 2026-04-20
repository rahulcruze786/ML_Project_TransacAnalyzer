for key, group in grouped:
    step_idx += 1
    texts = group[text_column].fillna("").astype(str)

    # ── Check 1: empty text ───────────────────────────────────────────────
    if texts.str.strip().eq("").all():
        print(f"{key} --> Skipped: all text empty.")
        if tracking_enabled:
            tracker.log_metrics(metrics=[Metric(
                name="skipped_reason", value=0.0,
                timestamp=datetime.now(timezone.utc), step=step_idx,
                labels=[
                    MetricLabel(name="key_group", value=str(key)),
                    MetricLabel(name="reason",    value="empty_text")
                ]
            )])
        continue                          # ← skip everything below

    X = texts
    y = group[target_column]
    num_classes = y.nunique()
    min_class_count = y.value_counts().min()
    metrics = {"test_accuracy": None, "test_f1_macro": None}

    # ── Check 2: single class — log AND still train, but skip test metrics
    if num_classes < 2:
        print(f"{key} | Single class only, test metrics skipped.")
        if tracking_enabled:
            tracker.log_metrics(metrics=[Metric(
                name="skipped_reason", value=0.0,
                timestamp=datetime.now(timezone.utc), step=step_idx,
                labels=[
                    MetricLabel(name="key_group",   value=str(key)),
                    MetricLabel(name="reason",      value="single_class_only"),
                    MetricLabel(name="class_found", value=str(y.unique()[0]))
                ]
            )])
        # ← NO continue here — we still want to train and save the model

    # ── Check 3: multi class — run test metrics ───────────────────────────
    else:
        stratify_arg = y if min_class_count >= 5 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=stratify_arg, random_state=42
            )
            eval_model = _build_pipeline(seasonal_words)
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)
            test_f1  = f1_score(y_test, y_pred, average="macro")
            test_acc = accuracy_score(y_test, y_pred)
            metrics  = {"test_accuracy": float(test_acc), "test_f1_macro": float(test_f1)}
            print(f"{key} | Test F1: {test_f1:.3f} | Test Accuracy: {test_acc:.3f}")
            if tracking_enabled:
                tracker.log_metrics(metrics=[
                    Metric(name="test_accuracy", value=float(test_acc),
                           timestamp=datetime.now(timezone.utc), step=step_idx,
                           labels=[MetricLabel(name="key_group", value=str(key))]),
                    Metric(name="test_f1_macro", value=float(test_f1),
                           timestamp=datetime.now(timezone.utc), step=step_idx,
                           labels=[MetricLabel(name="key_group", value=str(key))]),
                    Metric(name="train_size",    value=float(len(X_train)),
                           timestamp=datetime.now(timezone.utc), step=step_idx,
                           labels=[MetricLabel(name="key_group", value=str(key))]),
                ])
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
            if tracking_enabled:
                tracker.log_metrics(metrics=[Metric(
                    name="skipped_reason", value=0.0,
                    timestamp=datetime.now(timezone.utc), step=step_idx,
                    labels=[
                        MetricLabel(name="key_group", value=str(key)),
                        MetricLabel(name="reason",    value="empty_vocabulary")
                    ]
                )])
            continue                      # ← skip saving, move to next key
        raise

    # ── Save model ────────────────────────────────────────────────────────
    safe_key       = str(key).replace("/", "_").replace("\\", "_")
    model_filename = f"model_{safe_key}.joblib"
    model_path     = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    metadata["models"][str(key)] = {
        "model_file": model_filename,
        "metrics":    metrics
    }
    saved_models.append(model_path)
