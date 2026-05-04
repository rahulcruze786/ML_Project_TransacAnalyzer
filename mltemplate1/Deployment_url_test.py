import requests

# ── Fill these from your AI Core Service Key ──────────────────
CLIENT_ID       = "sb-524de71c-6fdc-42f9-9ca3-1aa049f9521f!b37640|xsuaa_std!b15301"
CLIENT_SECRET   = "56e3a705-fe31-4cd8-adab-b9b49790abbe$VkGaPU4HAATczI9yH00PmPRSFS4QU-dJCJw9XzuED8o="
AUTH_URL        = "https://fin-analytical-svc-rnd.authentication.ap11.hana.ondemand.com"
DEPLOYMENT_URL  = "https://api.ai.prod-ap11.ap-southeast-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d008e4949af330f2"   # from AI Launchpad
RESOURCE_GROUP  = "default"                       # your resource group name

MARKET = "Hong Kong"


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
# Calls GET /v1/getMetadata?market=Hong Kong
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
        metadata = response.json().get("data") or {}
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
            "MTDPeriod"   : "2024012",
            "Entity_BU"   : "9380",
            "Source"      : "SYS",
            "GLAccount"   : "111101",
            "BaseCurrency": "HKD",
            "CostCenter"  : "#",
            "DebitCredit" : "H",
            "PostingDate" : "12/4/2024",
            "DocumentID"  : "3.1E+09",
            "GLTaxCode"   : "#",
            "GLTaxCountry": "#",
            "GLTBSource"  : "ZREV",
            "Ledger"      : "AL",
            "OperatingID" : "93800178#",
            "PartnerEntity": "#",
            "PostingDate2": "12/3/2024",
            "PostingIter" : "24",
            "Product"     : "6091",
            "ProfitCenter": "301429",
            "Reference"   : "3.1E+09",
            "ReportingE1" : "9380",
            "ReportingE2" : "HKD",
            "Segment"     : "1110#",
            "SourceDoc"   : "#",
            "TEXT_S4Journal": "#",
            "Transaction" : "#",
            "Amount_Base" : "CAD",
            "Amount_Trans": "0",
        },
        {
            "MTDPeriod"   : "2024012",
            "Entity_BU"   : "9380",
            "Source"      : "SYS",
            "GLAccount"   : "111101",
            "BaseCurrency": "HKD",
            "CostCenter"  : "#",
            "DebitCredit" : "H",
            "PostingDate" : "12/4/2024",
            "DocumentID"  : "3.1E+09",
            "GLTaxCode"   : "#",
            "GLTaxCountry": "#",
            "GLTBSource"  : "ZREV",
            "Ledger"      : "AL",
            "OperatingID" : "93800307#",
            "PartnerEntity": "#",
            "PostingDate2": "12/3/2024",
            "PostingIter" : "25",
            "Product"     : "6091",
            "ProfitCenter": "301429",
            "Reference"   : "3.1E+09",
            "ReportingE1" : "9380",
            "ReportingE2" : "HKD",
            "Segment"     : "1110#",
            "SourceDoc"   : "#",
            "TEXT_S4Journal": "#",
            "Transaction" : "#",
            "Amount_Base" : "CAD",
            "Amount_Trans": "0",
        },
        {
            "MTDPeriod"   : "2024012",
            "Entity_BU"   : "9380",
            "Source"      : "SYS",
            "GLAccount"   : "111101",
            "BaseCurrency": "HKD",
            "CostCenter"  : "#",
            "DebitCredit" : "H",
            "PostingDate" : "12/4/2024",
            "DocumentID"  : "3.1E+09",
            "GLTaxCode"   : "#",
            "GLTaxCountry": "#",
            "GLTBSource"  : "ZREV",
            "Ledger"      : "AL",
            "OperatingID" : "93800350#",
            "PartnerEntity": "#",
            "PostingDate2": "12/3/2024",
            "PostingIter" : "28",
            "Product"     : "6091",
            "ProfitCenter": "301429",
            "Reference"   : "3.1E+09",
            "ReportingE1" : "9380",
            "ReportingE2" : "HKD",
            "Segment"     : "1110#",
            "SourceDoc"   : "#",
            "TEXT_S4Journal": "#",
            "Transaction" : "#",
            "Amount_Base" : "CAD",
            "Amount_Trans": "0",
        },
        {
            "MTDPeriod"   : "2024012",
            "Entity_BU"   : "9380",
            "Source"      : "SYS",
            "GLAccount"   : "111101",
            "BaseCurrency": "HKD",
            "CostCenter"  : "#",
            "DebitCredit" : "H",
            "PostingDate" : "12/4/2024",
            "DocumentID"  : "3.1E+09",
            "GLTaxCode"   : "#",
            "GLTaxCountry": "#",
            "GLTBSource"  : "ZREV",
            "Ledger"      : "AL",
            "OperatingID" : "93800351#",
            "PartnerEntity": "#",
            "PostingDate2": "12/3/2024",
            "PostingIter" : "29",
            "Product"     : "6091",
            "ProfitCenter": "301429",
            "Reference"   : "3.1E+09",
            "ReportingE1" : "9380",
            "ReportingE2" : "HKD",
            "Segment"     : "1110#",
            "SourceDoc"   : "#",
            "TEXT_S4Journal": "#",
            "Transaction" : "#",
            "Amount_Base" : "CAD",
            "Amount_Trans": "0",
        },
        {
            "MTDPeriod"   : "2024012",
            "Entity_BU"   : "9380",
            "Source"      : "SYS",
            "GLAccount"   : "111101",
            "BaseCurrency": "HKD",
            "CostCenter"  : "#",
            "DebitCredit" : "H",
            "PostingDate" : "12/4/2024",
            "DocumentID"  : "3.1E+09",
            "GLTaxCode"   : "#",
            "GLTaxCountry": "#",
            "GLTBSource"  : "ZREV",
            "Ledger"      : "AL",
            "OperatingID" : "93800402#",
            "PartnerEntity": "#",
            "PostingDate2": "12/3/2024",
            "PostingIter" : "31",
            "Product"     : "6091",
            "ProfitCenter": "301429",
            "Reference"   : "3.1E+09",
            "ReportingE1" : "9380",
            "ReportingE2" : "HKD",
            "Segment"     : "1110#",
            "SourceDoc"   : "#",
            "TEXT_S4Journal": "#",
            "Transaction" : "#",
            "Amount_Base" : "CAD",
            "Amount_Trans": "0",
        },
    ]

    # ── account_mapping ───────────────────────────────────────────────────────
    # maps GL accounts to ExpenseType and ProcessType
    # pipeline.py filters this by market="Singapore" and process_types from metadata
    # including a row for a different market to verify pipeline filters correctly
    account_mapping = [
    {"Account": "0000123456", "ExpenseType": "Travel",                              "ProcessType": "Operating Losses",                                            "Market": "Hong Kong"},
    {"Account": "0000123457", "ExpenseType": "Travel",                              "ProcessType": "Operating Losses",                                            "Market": "Hong Kong"},
    {"Account": "0000268109", "ExpenseType": "Sundry Accruals & Deferred Inc",      "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273501", "ExpenseType": "Prov-Op Losses & Other Liabs-Ext",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273503", "ExpenseType": "Prov-Op Losses & Other Liabs-Ext",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273504", "ExpenseType": "Provision Utilised - External",       "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273505", "ExpenseType": "Prov-Op Losses & Other Liabs-Ext",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273506", "ExpenseType": "Prov-Op Losses & Other Liabs-Ext",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273511", "ExpenseType": "Prov-Op Losses & Other Liabs-Int",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273513", "ExpenseType": "Prov-Op Losses & Other Liabs-Int",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273515", "ExpenseType": "Prov-Op Losses & Other Liabs-Int",    "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000658012", "ExpenseType": "Amt w-off w/o Pro-Op Loss Ext",       "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658013", "ExpenseType": "P/L-Pro no long req-Op.Loss Ex",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658014", "ExpenseType": "Rec amt prev w/off-Op.Loss Ext",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658021", "ExpenseType": "P/L-New Prov Creat-Op.Loss Int",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658022", "ExpenseType": "Amt w-off w/o Pro-Op Loss Ext",       "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658024", "ExpenseType": "Rec amt prev w/off.Op.Loss Int",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658031", "ExpenseType": "Frauds, Shortages & Losses",          "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658032", "ExpenseType": "Operational Losses (Ext & Int)",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000713001", "ExpenseType": "ECL Allowances - Stage 1 to 3",       "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000665049", "ExpenseType": "Other Costs",                         "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658011", "ExpenseType": "P/L-New Prov Creat-Op.Loss Ext",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658023", "ExpenseType": "P/L-Pro no long req-Op.Loss In",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000658034", "ExpenseType": "Rec amt prev w/off-OpLoss GSSC",      "ProcessType": "Provisions and Payments of Operating Losses (P/L)",           "Market": "Hong Kong"},
    {"Account": "0000268101", "ExpenseType": "Prov-Op Losses & Other Liabs",        "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000273514", "ExpenseType": "Provision Utilised - Internal",       "ProcessType": "Provisions and Payments of Operating Losses (B/S)",           "Market": "Hong Kong"},
    {"Account": "0000652151", "ExpenseType": "Prof-Consultant Costs & Others",      "ProcessType": "Professional and Legal fees",                                 "Market": "Hong Kong"},
    {"Account": "0000652519", "ExpenseType": "Prof-Consultant Costs & Others",      "ProcessType": "Professional and Legal fees",                                 "Market": "Hong Kong"},
    {"Account": "0000652520", "ExpenseType": "Prof-Consultant Costs & Others",      "ProcessType": "Professional and Legal fees",                                 "Market": "Hong Kong"},
    ]

    # ── config ────────────────────────────────────────────────────────────────
    # configuration parameters for the pipeline
    # pipeline.py filters by market + process_types and converts to flat dict
    # including rows for multiple process types and a different market
    # to verify pipeline filters correctly
    config = [
        # ── Overwrite rule: Profit Centre + Oper Unit + Segment → Payment ─────
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_RL_FM_PROFIT_CENTRE",
            "Value"      : "301537"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_RL_FM_OPER_UNIT",
            "Value"      : "800"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_RL_FM_SEGMENT",
            "Value"      : "4098"
        },

        # ── Overwrite rule: Text contains keywords → Non-deductible ───────────
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_RL_FM_TEXT_CHATFIELD_COL",
            "Value"      : "lb,non-lb,goodwill,opt loss,cems,incident recovery,refund"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_OUTPUT_FM_TEXT_CHATFIELD_COL",
            "Value"      : "Non-deductible"
        },

        # ── Overwrite rule: Threshold + Non-deductible + keyword → Deductible ─
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT",
            "Value"      : "goodwill,cems,refund"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT",
            "Value"      : "Deductible"
        },

        # ── Threshold logic ───────────────────────────────────────────────────
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "THRESHOLD_GROUPBY",
            "Value"      : "GLAccount,OperatingLocation,ProfitCenter,TEXT(S4Journal)"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "THRESHOLD_SUMBY",
            "Value"      : "Amount(Base)"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "THRESHOLD_AMOUNT",
            "Value"      : "50000"
        },

        # ── Net-off logic ─────────────────────────────────────────────────────
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "NET_OFF_GROUPBY",
            "Value"      : "GLAccount,OperatingLocation,ProfitCenter,TEXT(S4Journal)"
        },
        {
            "Market"     : "Hong Kong",
            "Parameter"  : "AMOUNT_COL",
            "Value"      : "Amount(Base)"
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
