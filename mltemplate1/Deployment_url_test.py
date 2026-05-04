import requests
import pandas as pd

# ── Fill these from your AI Core Service Key ──────────────────
CLIENT_ID       = "sb-524de71c-6fdc-42f9-9ca3-1aa049f9521f!b37640|xsuaa_std!b15301"
CLIENT_SECRET   = "56e3a705-fe31-4cd8-adab-b9b49790abbe$VkGaPU4HAATczI9yH00PmPRSFS4QU-dJCJw9XzuED8o="
AUTH_URL        = "https://fin-analytical-svc-rnd.authentication.ap11.hana.ondemand.com"
DEPLOYMENT_URL  = "https://api.ai.prod-ap11.ap-southeast-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d0a8f994887c6669"   # from AI Launchpad
RESOURCE_GROUP  = "default"                       # your resource group name
# ─────────────────────────────────────────────────────────────
MARKET = "Singapore"

# Step 1: Get Bearer Token
def get_token():
    response = requests.post(
        url=f"{AUTH_URL}/oauth/token",
        data={
            "grant_type"   : "client_credentials",
            "client_id"    : CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]


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
        models = metadata.get('models',{})
        print(f"\n{'='*55}")
        print(f"  🌏 Market       : {market}")
        print(f"  📂 Config file  : model_config_{market.replace(' ','_')}.json")
        print(f"  📦 Models found : {len(models)}")
        print(f"  📋 Process types: {metadata.get('process_types')}")
        print(f"\n 🔑 Model files used:")
        for key, info in models.items():
            print(f" {key} -> {info.get('model_file')}")
        print(f"{'='*55}\n")
    else:
        print(f"❌ Metadata not found for market '{market}': {response.text}")



# Step 2: Call Health Endpoint
def call_process(token, market):
    headers = {
        "Authorization"    : f"Bearer {token}",
        "AI-Resource-Group": RESOURCE_GROUP,
        "Content-Type"     : "application/json"
    }
    body = {"TEXT_COL":"TEXT(S4Journal)","LABEL_COL":"taxcategory","CONSIDER_CLASS_TARGET_COL":["Deductible","Non-deductible"],"OW_RL_FM_PROFIT_CENTRE":"301537","OW_RL_FM_OPER_UNIT":"800","OW_RL_FM_SEGMENT":"4098","OW_RL_FM_TEXT_CHATFIELD_COL":["lb","non-lb","goodwill","opt loss","cems","incident recovery","refund"],"OW_OUTPUT_FM_TEXT_CHATFIELD_COL":"Non-deductible","OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT":["goodwill","cems","refund"],"OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT":"Deductible","THRESHOLD_GROUPBY":["GLAccount","OperatingLocation","ProfitCenter","TEXT(S4Journal)"],"THRESHOLD_SUMBY":"Amount(Base)","THRESHOLD_AMOUNT":"50000","NET_OFF_GROUPBY":["GLAccount","OperatingLocation","ProfitCenter","TEXT(S4Journal)"],"AMOUNT_COL":"Amount(Base)","KEY_COLS":["ExpenseType"],"Acct_filter_type":"Provisions and Payments of Operating Losses (B/S),Provisions and Payments of Operating Losses (P/L)","market": market}
    response = requests.post(
        url=f"{DEPLOYMENT_URL}/v1/process",
        headers=headers,
        json=body)

    print(f"Status Code : {response.status_code}")
    response.raise_for_status()
    return response.json()

# Main
if __name__ == "__main__":
    print("Getting token...")
    token = get_token()
    print("Token retrieved ✅")

    # ── Step 1: Show which metadata + models will be used ──
    print(f"\nChecking metadata for market: {MARKET}")
    check_metadata(token, MARKET)
    
    # ── Step 2: Run prediction ─────────────────────────────
    print(f"Running prediction for market: {MARKET}...")
    result = call_process(token, MARKET)
    print(f"✅ Status : {result.get('status')}")
    print(f"📊 Rows   : {result.get('rows_processed')}")
    print(f"⏱️  Time   : {result.get('execution_time_seconds')}s")

    # ── Step 3: Export to Excel ────────────────────────────
    data = result.get("data", [])
    if data:
        df = pd.DataFrame(data)
        output_file = f"predictions_{MARKET.replace(' ','_')}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n✅ Excel exported: {output_file}  ({len(df)} rows)")
    else:
        print("⚠️ No data in response")
