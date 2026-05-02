payload = {
    # ── used in this section ──────────────────────────
    "market": "Hong Kong",

    # ── used further down in process_data() ──────────
    "journal_lines": [
        {
            "MTDPeriod":           "2024-01",
            "DocumentNumber":      "1000001",
            "GLAccount":           "12345",
            "GLTBSource":          "PREV",
            "Segment":             "SEG001",
            "PostingDate":         "2024-01-15",
            "PartnerEntity":       "PE001",
            "PostingItem":         "1",
            "TEXT(S4Journal)":     "grab taxi to airport for client meeting",
            "OperatingLocation":   "OL00784",
            "Ledger":              "0L",
            "CostCenter":          "CC001",
            "ProfitCenter":        "1234",
            "Amount(Base)":        "500.00",
            "Amount(Transaction)": "500.00",
            "TransactionCurrency": "HKD",
            "Entity/BU":           "HK01",
            "SourceDocumentNo":    "SD001",
            "ExpenseType":         "TRAVEL"
        },
        # ... more rows
    ],

    "account_mapping": [
        {
            "Account":     "12345",
            "ExpenseType": "TRAVEL",
            "ProcessType": "Provisions and Payments of Operating Losses (P/L)",
            "Market":      "Hong Kong"
        },
        # ... more rows
    ],

    "config_params": {
        "OW_RL_FM_PROFIT_CENTRE":                   "1234",
        "OW_RL_FM_OPER_UNIT":                       "784",
        "OW_RL_FM_SEGMENT":                         "001",
        "OW_RL_FM_TEXT_CHATFIELD_COL":              "chatfield,chat",
        "OW_OUTPUT_FM_TEXT_CHATFIELD_COL":          "Deductible",
        "OW_RL_FM_TEXT_CHATFIELD_COL_CND_TH_AMT":  "keyword1,keyword2",
        "OW_OUTPUT_FM_TEXT_CHATFIELD_COL_CND_TH_AMT": "Non-deductible",
        "THRESHOLD_GROUPBY":                        "DocumentNumber,GLAccount",
        "THRESHOLD_SUMBY":                          "Amount(Base)",
        "THRESHOLD_AMOUNT":                         "50000",
        "NET_OFF_GROUPBY":                          "DocumentNumber,GLAccount",
        "AMOUNT_COL":                               "Amount(Base)"
    }
}
