I can see **2 bugs** in your updated code:

---

## Bug 1 — `step_trained += 1` still in the loop (should be removed)

```python
# FIND AND REMOVE THIS LINE (inside the else block in the loop)
        else:
            step_trained += 1     # ← DELETE THIS LINE
            stratify_arg = y if min_class_count >= 5 else None
```

`step_trained` is no longer defined at the top of the function, so this will cause a `NameError` crash.

---

## Bug 2 — `key_results` store for `trained` is inside `if tracking_enabled:` — it should NOT be

```python
# FIND THIS
                if tracking_enabled:
                    key_results[str(key)] = {
                        "type":         "trained",
                        ...
                    }

# REPLACE WITH (remove the if tracking_enabled wrapper)
                key_results[str(key)] = {
                    "type":         "trained",
                    "value":        float(test_acc),
                    "train_size":   train_size,
                    "test_size":    test_size,
                    "test_accuracy":str(round(test_acc, 4)),
                    "test_f1_macro":str(round(test_f1,  4)),
                }
```

`key_results` is used for both tracking AND for the summary metrics calculation — wrapping it in `if tracking_enabled` means locally (no tracking) the dict stays empty and summary metrics break.

---

## Everything else is correct ✅

| Section | Status |
|---|---|
| `key_results = {}` initialized | ✅ |
| Empty text → `key_results[key] = skipped_empty_text` | ✅ |
| Single class → no log, falls through | ✅ |
| Empty vocab → `key_results[key] = skipped_vocab` overwrites single_class | ✅ |
| After save → `if str(key) not in key_results` for single_class | ✅ |
| Post-loop logging in correct order | ✅ |
| Summary metrics block | ✅ |
| `set_custom_info` | ✅ |
| `unmatched_accounts` metric | ✅ |

Just fix those 2 lines, rebuild as `v2.2`, push and run.
