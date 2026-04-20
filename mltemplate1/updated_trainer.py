# REMOVE THIS ENTIRE BLOCK
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
        tags=[                          # ← this parameter doesn't exist
            MetricTag(name="text_column", value=str(text_column)),
            ...
        ]
    )


# REPLACE WITH THESE TWO SEPARATE CALLS
if tracking_enabled:
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

    # Call 2 - log tags separately
    tracker.set_custom_info([
        MetricCustomInfo(name="text_column",   value=str(text_column)),
        MetricCustomInfo(name="target_column", value=str(target_column)),
        MetricCustomInfo(name="allowed_class", value=",".join(allowed_class)),
        MetricCustomInfo(name="seasonal_words",value=str(seasonal_words)),
        MetricCustomInfo(name="market",        value=str(market)),
        MetricCustomInfo(name="process_types", value=" | ".join(process_types)),
        MetricCustomInfo(name="key_cols",      value=",".join(effective_key_cols)),
    ])
