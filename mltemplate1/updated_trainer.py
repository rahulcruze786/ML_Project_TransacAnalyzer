Only one change needed — find this block in your code:

```python
# FIND THIS
    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows had no matching ExpenseType and will be skipped. Unmatched GLAccounts: {unmatched['GLAccount'].unique().tolist()} Need to update the account mapping table")
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]
```

```python
# REPLACE WITH
    unmatched = df[df["ExpenseType"].astype(str).str.strip() == ""]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows had no matching ExpenseType and will be skipped. Unmatched GLAccounts: {unmatched['GLAccount'].unique().tolist()} Need to update the account mapping table")
    df = df[df["ExpenseType"].astype(str).str.strip() != ""]

    # ── Log unmatched accounts metric ─────────────────────────────────────
    if tracking_enabled and not unmatched.empty:
        unmatched_accounts = ", ".join([f"'{a}'" for a in unmatched["GLAccount"].unique().tolist()])
        tracker.log_metrics(metrics=[Metric(
            name="unmatched_accounts",
            value=float(len(unmatched)),
            timestamp=datetime.now(timezone.utc),
            step=0,
            labels=[
                MetricLabel(name="remarks", value="accounts with no ExpenseType mapping — update account mapping table"),
                MetricLabel(name="accounts", value=unmatched_accounts),
            ]
        )])
```

---

## What you'll see in Metrics table

| Name | Value | Step | Labels |
|---|---|---|---|
| `unmatched_accounts` | 1 | 0 | `remarks: accounts with no ExpenseType mapping — update account mapping table`, `accounts: '0000273001'` |

The `value` = count of unmatched **rows**, and the `accounts` label lists the actual GL account numbers like `'0000273001', '0000273002'`.
