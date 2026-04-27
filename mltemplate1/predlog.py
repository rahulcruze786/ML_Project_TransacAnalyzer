The error is crystal clear from the logs:

```
PermissionError: [Errno 13] Permission denied: '/mnt/models'
```

---

## Root cause

`/mnt/models` is **SAP AI Core's read-only artifact mount point**. You cannot create subfolders or write anything into it. Our `_download_market_artifacts()` was trying to do `os.makedirs('/mnt/models/Hong_Kong')` — which is denied.

---

## The fix

**Stop trying to download into `/mnt/models`.** SAP AI Core already mounts the artifact there automatically when you use `STORAGE_URI`. The models are already at `/mnt/models` — you just need to read from it directly.

---

## What actually happens in SAP AI Core

```
STORAGE_URI_HONGKONG = "..."  →  SAP AI Core mounts it at MODEL_PATH = /mnt/models
```

But since you're using a custom env var name `STORAGE_URI_HONGKONG` instead of the special `STORAGE_URI`, SAP AI Core **does not auto-mount it**. That's why models aren't there.

---

## The real solution — use `STORAGE_URI` (the special one)

SAP AI Core only auto-mounts when the env var is **exactly** `STORAGE_URI`. So change your serving YAML:

**Current:**
```yaml
env:
  - name: STORAGE_URI_HONGKONG
    value: "{{inputs.artifacts.multinomialNB_HongKong}}"
  - name: MODEL_PATH
    value: "/mnt/models"
```

**Change to:**
```yaml
env:
  - name: STORAGE_URI
    value: "{{inputs.artifacts.multinomialNB_HongKong}}"
  - name: MODEL_PATH
    value: "/mnt/models"
```

SAP AI Core sees `STORAGE_URI` → downloads the artifact → mounts it at `MODEL_PATH` (`/mnt/models`) automatically. No `aws s3 sync`, no `makedirs`, no permission issues.

---

## But then what about multiple markets?

Since `/mnt/models` is read-only and only one `STORAGE_URI` is auto-mounted, the correct pattern for multiple markets is to use a **writable temp directory** for additional markets and only `/mnt/models` for the primary one.

**Update `_download_market_artifacts()` in `app.py`:**

```python
def _download_market_artifacts():
    """
    SAP AI Core auto-mounts STORAGE_URI artifact into MODEL_PATH (/mnt/models).
    Additional markets use STORAGE_URI_* and are downloaded into /tmp/models/<market>
    which is writable.
    Primary market (auto-mounted) is read directly from /mnt/models.
    """
    import subprocess

    additional_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("STORAGE_URI_")   # custom vars only, not STORAGE_URI
    }

    if not additional_vars:
        print("ℹ️ No additional STORAGE_URI_* vars found. Only primary market will be loaded.")
        return

    for env_key, uri in additional_vars.items():
        market_folder = env_key.replace("STORAGE_URI_", "").replace("_", " ").title().replace(" ", "_")
        dest          = os.path.join("/tmp/models", market_folder)   # ← writable path
        os.makedirs(dest, exist_ok=True)
        print(f"📥 Downloading [{market_folder}] → {dest}")
        try:
            subprocess.run(["aws", "s3", "sync", uri, dest, "--no-progress"], check=True)
            print(f"✅ [{market_folder}] downloaded.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed [{market_folder}]: {e}")
            raise
```

And update `load_metadata()` to scan **both** `/mnt/models` and `/tmp/models`:

```python
def load_metadata():
    all_metadata = {}

    # Scan both mount path (primary, read-only) and tmp path (additional markets)
    scan_dirs = [
        os.environ.get("MODEL_PATH", "/mnt/models"),  # primary — auto-mounted by AI Core
        "/tmp/models",                                  # additional — downloaded by app
    ]

    for base_dir in scan_dirs:
        if not os.path.exists(base_dir):
            continue
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            config_path = os.path.join(entry.path, "model_config.json")
            if not os.path.exists(config_path):
                # Also check flat config (primary market mounted directly)
                flat_config = os.path.join(base_dir, "model_config.json")
                if os.path.exists(flat_config):
                    market_name = os.environ.get("PRIMARY_MARKET", "Hong_Kong")
                    with open(flat_config, "r", encoding="utf-8") as f:
                        all_metadata[market_name] = json.load(f)
                    print(f"✅ Metadata loaded for primary market: {market_name}")
                break
            with open(config_path, "r", encoding="utf-8") as f:
                all_metadata[entry.name] = json.load(f)
            print(f"✅ Metadata loaded for market: {entry.name}")

    if not all_metadata:
        raise FileNotFoundError(f"No market models found in {scan_dirs}")

    return all_metadata
```

And update `load_models()` to look in both paths:

```python
def load_models(all_metadata):
    all_models = {}
    search_dirs = [
        os.environ.get("MODEL_PATH", "/mnt/models"),
        "/tmp/models",
    ]
    for market, metadata in all_metadata.items():
        models = {}
        for base_dir in search_dirs:
            market_dir = os.path.join(base_dir, market)
            if os.path.exists(market_dir):
                break
            # Also check flat (primary market mounted directly at base_dir)
            if os.path.exists(os.path.join(base_dir, "model_config.json")):
                market_dir = base_dir
                break
        for key, model_info in metadata.get("models", {}).items():
            model_file = model_info.get("model_file")
            if not model_file:
                continue
            local_path = os.path.join(market_dir, model_file)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Model file not found: {local_path}")
            print(f"📥 Loading [{market}] {local_path}")
            models[key] = joblib.load(local_path)
            print(f"✅ Loaded [{market}] key: {key}")
        all_models[market] = models
    return all_models
```

---

## Also add `PRIMARY_MARKET` env to serving YAML

```yaml
env:
  - name: STORAGE_URI
    value: "{{inputs.artifacts.multinomialNB_HongKong}}"
  - name: MODEL_PATH
    value: "/mnt/models"
  - name: PRIMARY_MARKET
    value: "Hong_Kong"
```

---

## Summary of what changes

| What | Change |
|---|---|
| Serving YAML | `STORAGE_URI_HONGKONG` → `STORAGE_URI` |
| Serving YAML | Add `PRIMARY_MARKET = Hong_Kong` |
| `_download_market_artifacts()` | Only downloads `STORAGE_URI_*` vars into `/tmp/models` (writable) |
| `load_metadata()` | Scans both `/mnt/models` and `/tmp/models` |
| `load_models()` | Searches both paths per market |
