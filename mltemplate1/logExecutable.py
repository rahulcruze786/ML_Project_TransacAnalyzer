Let me read all the uploaded files carefully to diagnose properly.Now I have read everything. I can see all the problems clearly. Let me give you a complete diagnosis.

---

**There are 3 bugs — here they are in order of severity:**

---

**Bug 1 — CRITICAL: `load_metadata()` has misplaced indentation — the file is never read**

This is the root cause of everything. Look at `model_loader.py` lines 56-65:

```python
if not os.path.exists(local_config):
    raise FileNotFoundError(...)

    # ← THESE LINES ARE INDENTED INSIDE THE if block
    # They only run when the file DOES NOT exist
    # When the file DOES exist → raise fires → execution stops
    # When raise fires → the with open() below NEVER runs
    with open(local_config, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Metadata loaded: ...")
    return metadata  # ← this return is also inside the if block
```

The `with open()`, `json.load()`, and `return metadata` are all indented **one level too deep** — they sit inside the `if not os.path.exists` block. This means:

- When the file **doesn't exist** → `raise FileNotFoundError` fires correctly
- When the file **does exist** → the `if` block is skipped entirely → `with open()` never runs → function falls off the end → returns `None`

So `load_metadata()` always returns `None` for any market whose config file **exists**. That's why you saw `Models found: 0` and `Process types: None` — the file was found but never read.

**The fix:**

```python
if not os.path.exists(local_config):
    raise FileNotFoundError(f"No metadata config found for market {market} at {local_config}")

# ← these lines must be at THIS indentation level, outside the if block
with open(local_config, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"✅ Metadata loaded: {len(metadata.get('models', {}))} models found")
return metadata
```

---

**Bug 2 — CRITICAL: Tab character in column rename causes KeyError**

In `predict.py` line 390 and 403:

```python
# line 390 — rename
predict_df.rename(columns={
    "Threshold Amount": "ThresholdAmountReached\t",  # ← TAB CHARACTER after name
    "Oper Unit Split":  "OperUnitSplit"
}, inplace=True)

# line 403 — select columns
predict_df = predict_df[[
    ...
    "ThresholdAmountReached\t",  # ← TAB CHARACTER — must match exactly
    "OperUnitSplit"
]]
```

There is a hidden tab character `\t` in the column name `"ThresholdAmountReached	"`. This will cause a `KeyError` when the column is selected or returned in JSON because the tab makes the name look different from what the caller expects.

**The fix:**

```python
predict_df.rename(columns={
    "Threshold Amount": "ThresholdAmountReached",   # ← no tab
    "Oper Unit Split":  "OperUnitSplit"
}, inplace=True)
```

---

**Bug 3 — journal_lines column names don't match what the pipeline expects**

Your test script journal_lines use wrong column names. The pipeline expects the exact column names the trainer used:

```python
# your test script          →   what pipeline expects
"TEXT_S4Journal"            →   "TEXT(S4Journal)"
"Amount_Base"               →   "Amount(Base)"
"Amount_Trans"              →   "Amount(Transaction)"
"DocumentID"                →   "DocumentNumber"
"OperatingID"               →   "OperatingLocation"
"DebitCredit"               →   "Debit/Credit"
"GLTaxCountry"              →   "GLTaxCountry/Region"
"Entity_BU"                 →   "Entity/BU"
"PostingIter"               →   "PostingItem"
"SourceDoc"                 →   "SourceDocumentNo"
"ReportingE1"               →   "ReportingEntity"
"ReportingE2"               →   "ReportingEntityCurrency"
"Reference"                 →   "Referencedocument"
```

Also the GLAccount `"111101"` in your test data does not exist in your `account_mapping` — so after filtering, zero rows would pass through the pipeline producing empty results even if bugs 1 and 2 are fixed.

---

**Summary of all fixes needed**

```
Bug 1  model_loader.py     CRITICAL   with open() indented inside
       load_metadata()                if block — file never read
                                      Fix: dedent with open() block

Bug 2  predict.py          CRITICAL   tab character \t in column
       line 390 + 403                 name "ThresholdAmountReached"
                                      Fix: remove the tab character

Bug 3  test script         MEDIUM     wrong column names in
       journal_lines                  journal_lines dict
                                      Fix: use exact column names
                                      from trainer
```

Fix Bug 1 first — that alone will resolve the `Models found: 0` and the 500 error. Bug 2 will then surface as the next error once the pipeline actually runs. Fix both before redeploying.
