def train_models(df, text_column, target_column, allowed_class, df_category_mapping,
                 market, process_types, key_cols=None, seasonal_words=""):

    # ── 1. Initialize tracker ─────────────────────────────────────────────────
    try:
        tracker = Tracking()
        tracking_enabled = True
        print("SAP AI Core tracking initialized.")
    except Exception as e:
        print(f"Tracking not available (running locally?): {e}")
        tracker = None
        tracking_enabled = False

    # ── 2. Filter data ────────────────────────────────────────────────────────
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

    # ── 3. effective_key_cols now defined ─────────────────────────────────────
    effective_key_cols = ["ExpenseType"] + key_cols
    df["key"] = col_aggreate(df, effective_key_cols)

    # ── 4. Log config tags (effective_key_cols now exists) ────────────────────
    if tracking_enabled:
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

    # ── 6. Rest of your training loop (no changes needed there) ──────────────
    ...
