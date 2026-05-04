def call_process(token, market):
    headers = {
        "Authorization"    : f"Bearer {token}",
        "AI-Resource-Group": RESOURCE_GROUP,
        "Content-Type"     : "application/json"
    }

    payload = build_payload(market)

    print(f"\n📤 Sending payload:")
    print(f"   market          : {payload['market']}")
    print(f"   journal_lines   : {len(payload['journal_lines'])} rows")
    print(f"   account_mapping : {len(payload['account_mapping'])} rows")
    print(f"   config          : {len(payload['config'])} rows")

    response = requests.post(
        url=f"{DEPLOYMENT_URL}/v1/process",
        headers=headers,
        json=payload
    )

    print(f"\n📥 Response status : {response.status_code}")

    # ── Always print the response body BEFORE raising ─────────────────────────
    # When status is 500, the server returns a JSON body with:
    # { "status": "error", "message": "...", "traceback": "..." }
    # We must read this BEFORE raise_for_status() throws
    # otherwise we lose the actual error details
    try:
        response_json = response.json()

        if response.status_code != 200:
            # print the full error details from the server
            print(f"\n❌ Server returned error:")
            print(f"   message  : {response_json.get('message', 'N/A')}")
            print(f"\n   traceback:")
            # traceback is a multi-line string — print each line separately
            tb = response_json.get('traceback', 'N/A')
            for line in tb.splitlines():
                print(f"   {line}")

    except Exception:
        # if response body is not JSON, print raw text instead
        print(f"\n❌ Raw response body: {response.text}")

    # now raise if error — after we have already printed the details
    response.raise_for_status()
    return response_json
