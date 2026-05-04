import requests

# ── Fill these from your AI Core Service Key ──────────────────────────────────
CLIENT_ID       = "sb-524de71c-6fdc-42f9-9ca3-1aa049f9521f!b37640|xsuaa_std!b15301"
CLIENT_SECRET   = "56e3a705-fe31-4cd8-adab-b9b49790abbe$VkGaPU4HAATczI9yH00PmPRSFS4QU-dJCJw9XzuED8o="
AUTH_URL        = "https://fin-analytical-svc-rnd.authentication.ap11.hana.ondemand.com"
DEPLOYMENT_URL  = "https://api.ai.prod-ap11.ap-southeast-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d0a8f994887c6669"
RESOURCE_GROUP  = "default"
# ─────────────────────────────────────────────────────────────────────────────

MARKET = "Singapore"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Get Bearer Token
# SAP AI Core uses OAuth2 client_credentials flow for authentication
# every API call needs a valid Bearer token in the Authorization header
# the token expires after a short time — always fetch fresh before calling
# ─────────────────────────────────────────────────────────────────────────────

def get_token():
    response = requests.post(
        url=f"{AUTH_URL}/oauth/token",
        data={
            "grant_type"   : "client_credentials",
            "client_id"    : CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
    )
    # raise_for_status() throws an exception if status code is 4xx or 5xx
    response.raise_for_status()
    # extract the token string from the response JSON
    return response.json()["access_token"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Check Metadata
# Calls GET /v1/getMetadata?market=Singapore
# Returns model info: process_types, text_column, key_cols, model files
# Useful to confirm the right model is loaded before running prediction
# ─────────────────────────────────────────────────────────────────────────────

def check_metadata(token, market):
    headers = {
        "Authorization"    : f"Bearer {token}",
        "AI-Resource-Group": RESOURCE_GROUP,
    }
    response = requests.get(
        url=f"{DEPLOYMENT_URL}/v1/getMetadata?market={market}",
        headers=headers
    )

    if response.status_code == 200:
        metadata = response.json().get("data", {})
        models   = metadata.get('models', {})
        print(f"\n{'='*55}")
        print(f"  🌏 Market        : {market}")
        print(f"  📂 Config file   : model_config_{market.replace(' ', '_')}.json")
        print(f"  📦 Models found  : {len(models)}")
        print(f"  📋 Process types : {metadata.get('process_types')}")
        print(f"  📝 Text column   : {metadata.get('text_column')}")
        print(f"  🔑 Key cols      : {metadata.get('key_cols')}")
        print(f"  ✅ Allowed class : {metadata.get('allowed_class')}")
        print(f"\n  🔑 Model files used:")
        for key, info in models.items():
            print(f"     {key} → {info.get('model_file')}")
        print(f"{'='*55}\n")
    else:
        print(f"❌ Metadata not found for market '{market}': {response.text}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build the test payload
#
# The payload has 4 keys:
#   market          → string: which market to run prediction for
#   journal_lines   → list of dicts: raw transaction rows (one dict per row)
#   account_mapping → list of dicts: GL account → ExpenseType mapping
#                     pipeline.py will filter this by market + process_types
#   config          → list of dicts: config parameter rows
#                     pipeline.py will filter this by market + process_types
#                     and convert to flat config_params dict
#
# Note: account_mapping and config contain rows for ALL markets here
# to simulate what the real UI sends — pipeline.py filters them correctly
# ─────────────────────────────────────────────────────────────────────────────

def build_payload(market):

    # ── journal_lines ─────────────────────────────────────────────────────────
    # small sample of 3 journal transaction rows for testing
    # column names must exactly match what the trainer used
    # TEXT(S4Journal) is the text column the model was trained on
    journal_lines = [
        {
            "MTDPeriod"            : "2024-01",
            "DocumentNumber"       : "1000001",
            "GLAccount"            : "0010123456",
            "GLTBSource"           : "MANUAL",
            "Segment"              : "SG0014098",
            "PostingDate"          : "2024-01-15",
            "PartnerEntity"        : "PE001",
            "PostingItem"          : "1",
            "TEXT(S4Journal)"      : "grab taxi to airport for client meeting",
            "OperatingLocation"    : "SG000800",
            "Ledger"               : "0L",
            "CostCenter"           : "CC001",
            "ProfitCenter"         : "301537",
            "Amount(Base)"         : "500.00",
            "Amount(Transaction)"  : "500.00",
            "TransactionCurrency"  : "SGD",
            "Entity/BU"            : "SG01",
            "SourceDocumentNo"     : "SD001",
        },
        {
            "MTDPeriod"            : "2024-01",
            "DocumentNumber"       : "1000002",
            "GLAccount"            : "0010123456",
            "GLTBSource"           : "MANUAL",
            "Segment"              : "SG0014098",
            "PostingDate"          : "2024-01-16",
            "PartnerEntity"        : "PE002",
            "PostingItem"          : "1",
            "TEXT(S4Journal)"      : "business lunch with client at restaurant",
            "OperatingLocation"    : "SG000801",
            "Ledger"               : "0L",
            "CostCenter"           : "CC002",
            "ProfitCenter"         : "301538",
            "Amount(Base)"         : "-500.00",
            "Amount(Transaction)"  : "-500.00",
            "TransactionCurrency"  : "SGD",
            "Entity/BU"            : "SG01",
            "SourceDocumentNo"     : "SD002",
        },
        {
            "MTDPeriod"            : "2024-01",
            "DocumentNumber"       : "1000003",
            "GLAccount"            : "0010123456",
            "GLTBSource"           : "PREV",
            "Segment"              : "SG0014098",
            "PostingDate"          : "2024-01-17",
            "PartnerEntity"        : "PE003",
            "PostingItem"          : "1",
            "TEXT(S4Journal)"      : "provision for operating loss settlement",
            "OperatingLocation"    : "SG000800",
            "Ledger"               : "0L",
            "CostCenter"           : "CC003",
            "ProfitCenter"         : "301537",
            "Amount(Base)"         : "75000.00",
            "Amount(Transaction)"  : "75000.00",
            "TransactionCurrency"  : "SGD",
            "Entity/BU"            : "SG01",
            "SourceDocumentNo"     : "SD003",
        },
    ]

    # ── account_mapping ───────────────────────────────────────────────────────
    # maps GL accounts to ExpenseType and ProcessType
    # pipeline.py filters this by market="Singapore" and process_types from metadata
    # including a row for a different market to verify pipeline filters correctly
    account_mapping = [
        {
            "Account"    : "0010123456",
            "ExpenseType": "Provisions and Payments of Operating Losses",
            "ProcessType": "Provisions and Payments of Operating Losses (B/S)",
            "Market"     : "Singapore"
        },
        {
            "Account"    : "0010123456",
            "ExpenseType": "Provisions and Payments of Operating Losses",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Market"     : "Singapore"
        },
        {
            "Account"    : "0010999999",
            "ExpenseType": "Provisions and Payments of Operating Losses",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Market"     : "Hong Kong"       # ← different market — pipeline filters this out
        },
    ]

    # ── config ────────────────────────────────────────────────────────────────
    # configuration parameters for the pipeline
    # pipeline.py filters by market + process_types and converts to flat dict
    # including rows for multiple process types and a different market
    # to verify pipeline filters correctly
    config = [
        # ── Overwrite rule: Profit Centre + Oper Unit + Segment → Payment ─────
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_RL_FM_PROFIT_CENTRE",
            "Value"      : "301537"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_RL_FM_OPER_UNIT",
            "Value"      : "800"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_RL_FM_SEGMENT",
            "Value"      : "4098"
        },

        # ── Overwrite rule: Text contains keywords → Non-deductible ───────────
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_RL_FM_TEXT_CHATFIELD_COL",
            "Value"      : "lb,non-lb,goodwill,opt loss,cems,incident recovery,refund"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_OUTPUT_FM_TEXT_CHATFIELD_COL",
            "Value"      : "Non-deductible"
        },

        # ── Overwrite rule: Threshold + Non-deductible + keyword → Deductible ─
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT",
            "Value"      : "goodwill,cems,refund"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT",
            "Value"      : "Deductible"
        },

        # ── Threshold logic ───────────────────────────────────────────────────
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "THRESHOLD_GROUPBY",
            "Value"      : "GLAccount,OperatingLocation,ProfitCenter,TEXT(S4Journal)"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "THRESHOLD_SUMBY",
            "Value"      : "Amount(Base)"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "THRESHOLD_AMOUNT",
            "Value"      : "50000"
        },

        # ── Net-off logic ─────────────────────────────────────────────────────
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "NET_OFF_GROUPBY",
            "Value"      : "GLAccount,OperatingLocation,ProfitCenter,TEXT(S4Journal)"
        },
        {
            "Market"     : "Singapore",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "AMOUNT_COL",
            "Value"      : "Amount(Base)"
        },

        # ── Row for different market — pipeline should filter this out ─────────
        {
            "Market"     : "Hong Kong",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Parameter"  : "THRESHOLD_AMOUNT",
            "Value"      : "99999"            # ← should NOT be used — wrong market
        },
    ]

    # assemble and return the complete payload dict
    return {
        "market"         : market,
        "journal_lines"  : journal_lines,
        "account_mapping": account_mapping,
        "config"         : config,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Call /v1/process — run the full prediction pipeline
# Sends the complete payload to the deployment URL
# pipeline.py handles all filtering, conversion, and prediction
# ─────────────────────────────────────────────────────────────────────────────

def call_process(token, market):
    headers = {
        "Authorization"    : f"Bearer {token}",
        "AI-Resource-Group": RESOURCE_GROUP,
        "Content-Type"     : "application/json"
    }

    # build the payload with hardcoded sample data
    payload = build_payload(market)

    print(f"\n📤 Sending payload:")
    print(f"   market          : {payload['market']}")
    print(f"   journal_lines   : {len(payload['journal_lines'])} rows")
    print(f"   account_mapping : {len(payload['account_mapping'])} rows")
    print(f"   config          : {len(payload['config'])} rows")

    response = requests.post(
        url=f"{DEPLOYMENT_URL}/v1/process",
        headers=headers,
        json=payload        # requests serializes the dict → JSON body automatically
    )

    print(f"\n📥 Response status : {response.status_code}")

    # raise_for_status() throws an exception for 4xx or 5xx responses
    # so we see a clear error instead of silently getting wrong data
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Print prediction results
# Loops over each predicted row and prints the key output columns
# ─────────────────────────────────────────────────────────────────────────────

def print_results(result):
    print(f"\n{'='*70}")
    print(f"  ✅ Status                  : {result.get('status')}")
    print(f"  📊 Rows processed          : {result.get('rows_processed')}")
    print(f"  ⏱️  Execution time          : {result.get('execution_time_seconds')}s")
    print(f"{'='*70}")

    data = result.get("data", [])

    if not data:
        print("  ⚠️  No data returned in response")
        return

    print(f"\n  📋 Prediction results ({len(data)} rows):\n")

    # print each predicted row — key columns only for readability
    for i, row in enumerate(data, start=1):
        print(f"  Row {i}:")
        print(f"    DocumentNumber       : {row.get('DocumentNumber')}")
        print(f"    GLAccount            : {row.get('GLAccount')}")
        print(f"    TEXT(S4Journal)      : {row.get('TEXT(S4Journal)')}")
        print(f"    ExpenseType          : {row.get('ExpenseType')}")
        print(f"    Prediction           : {row.get('Prediction ')}")
        print(f"    Confidence           : {row.get('Confidence')}")
        print(f"    InflunceParameter    : {row.get('InflunceParameter')}")
        print(f"    ClassProbabilities   : {row.get('ClassProbabilities')}")
        print(f"    NetOff               : {row.get('NetOff')}")
        print(f"    ThresholdAmountReached: {row.get('ThresholdAmountReached')}")
        print(f"    OperUnitSplit        : {row.get('OperUnitSplit')}")
        print()

    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main — runs when script is executed directly: python test_prediction.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Get OAuth token ───────────────────────────────────────────────
    print("Getting token...")
    token = get_token()
    print("Token retrieved ✅")

    # ── Step 2: Check metadata — confirm right model is loaded ───────────────
    print(f"\nChecking metadata for market: {MARKET}")
    check_metadata(token, MARKET)

    # ── Step 3: Run prediction ────────────────────────────────────────────────
    print(f"Running prediction for market: {MARKET}...")
    result = call_process(token, MARKET)

    # ── Step 4: Print results ─────────────────────────────────────────────────
    print_results(result)
