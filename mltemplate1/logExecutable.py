user: projects $ /home/user/.asdf-inst/shims/python3.13 "/home/user/projects/Prediction_Test copy.py"
Getting token...
Token retrieved ✅

Checking metadata for market: Hong Kong

=======================================================
  🌏 Market        : Hong Kong
  📂 Config file   : model_config_Hong_Kong.json
  📦 Models found  : 15
  📋 Process types : ['Provisions and Payments of Operating Losses (B/S)', 'Provisions and Payments of Operating Losses (P/L)']
  📝 Text column   : TEXT(S4Journal)
  🔑 Key cols      : ['ExpenseType']
  ✅ Allowed class : ['Deductible', 'Non-deductible']

  🔑 Model files used:
     Amt w-off w/o Pro-Op Loss Ext → model_Hong_Kong_Amt w-off w_o Pro-Op Loss Ext.joblib
     ECL Allowances - Stage 1 to 3 → model_Hong_Kong_ECL Allowances - Stage 1 to 3.joblib
     Frauds, Shortages & Losses → model_Hong_Kong_Frauds, Shortages & Losses.joblib
     Operational Losses (Ext & Int) → model_Hong_Kong_Operational Losses (Ext & Int).joblib
     Other Costs → model_Hong_Kong_Other Costs.joblib
     P/L-New Prov Creat-Op.Loss Int → model_Hong_Kong_P_L-New Prov Creat-Op.Loss Int.joblib
     P/L-Pro no long req-Op.Loss Ex → model_Hong_Kong_P_L-Pro no long req-Op.Loss Ex.joblib
     Prov-Op Losses & Other Liabs → model_Hong_Kong_Prov-Op Losses & Other Liabs.joblib
     Prov-Op Losses & Other Liabs-Ext → model_Hong_Kong_Prov-Op Losses & Other Liabs-Ext.joblib
     Prov-Op Losses & Other Liabs-Int → model_Hong_Kong_Prov-Op Losses & Other Liabs-Int.joblib
     Provision Utilised - External → model_Hong_Kong_Provision Utilised - External.joblib
     Rec amt prev w/off-Op.Loss Ext → model_Hong_Kong_Rec amt prev w_off-Op.Loss Ext.joblib
     Rec amt prev w/off-Op.Loss Int → model_Hong_Kong_Rec amt prev w_off-Op.Loss Int.joblib
     Rec amt prev w/off-OpLoss GSSC → model_Hong_Kong_Rec amt prev w_off-OpLoss GSSC.joblib
     Sundry Accruals & Deferred Inc → model_Hong_Kong_Sundry Accruals & Deferred Inc.joblib
=======================================================

Running prediction for market: Hong Kong...

📤 Sending payload:
   market          : Hong Kong
   journal_lines   : 5 rows
   account_mapping : 29 rows
   config          : 12 rows

📥 Response status : 500
Traceback (most recent call last):
  File "/home/user/projects/Prediction_Test copy.py", line 460, in <module>
    result = call_process(token, MARKET)
  File "/home/user/projects/Prediction_Test copy.py", line 400, in call_process
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/user/.asdf-inst/installs/python/3.13.1/lib/python3.13/site-packages/requests/models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://api.ai.prod-ap11.ap-southeast-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d757226da788f839/v1/process
user: projects $ 
