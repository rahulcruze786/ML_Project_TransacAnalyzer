def train_models(df, text_column, target_column, allowed_class, df_category_mapping,
                 market, process_types, key_cols=None, seasonal_words=""):
    
    # ── Initialize SAP AI Core tracking ──────────────────────────────────────
    try:
        tracker = Tracking()
        tracking_enabled = True
        print("SAP AI Core tracking initialized.")
    except Exception as e:
        print(f"Tracking not available (running locally?): {e}")
        tracker = None
        tracking_enabled = False

    df = df.copy()
    df = df[df[target_column].notna()]
    df = df[df[target_column].astype(str).str.strip() != ""]
    df = df[df[target_column].isin(allowed_class)]
    key_cols = _normalize_key_cols(key_cols)
    df = _map_expense_type_from_category_mapping(
        df=df, df_category_mapping=df_category_mapping,
        market=market, process_types=process_types
    )

    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows skipped - no matching ExpenseType.")
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]

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
        "allowed_class": list(allowed_class),
        "seasonal_words": seasonal_words,
        "market": market,
        "process_types": list(process_types),
        "models": {}
    }

    saved_models = []
    model_dir = os.environ.get("MODEL_DIR", "/tmp/models")
    print(f"ENV MODEL_DIR = {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    grouped = df.groupby("key")
    
    for step_idx, (key, group) in enumerate(grouped, start=1):
        texts = group[text_column].fillna("").astype(str)
        if texts.str.strip().eq("").all():
            print(f"{key} --> Skipped: all text empty.")
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
                metrics = {
                    "test_accuracy": float(test_acc),
                    "test_f1_macro": float(test_f1)
                }
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
        else:
            print(f"{key} | Single class only, metrics skipped.")

        model = _build_pipeline(seasonal_words)
        try:
            model.fit(X, y)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                print(f"{key} --> Skipped: empty vocabulary.")
                continue
            raise

        safe_key = str(key).replace("/", "_").replace("\\", "_")
        model_filename = f"model_{safe_key}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model, model_path)

        metadata["models"][str(key)] = {
            "model_file": model_filename,
            "metrics": metrics
        }
        saved_models.append(model_path)

    if not saved_models:
        return (
            f"No models saved for {len(df)} rows across {df['key'].nunique()} key group(s)."
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

    return f"Saved {len(saved_models)} model(s): {saved_models}. Config: {config_path}"
