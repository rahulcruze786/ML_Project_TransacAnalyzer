# UI wants model metadata
@app.route('/v1/getMetadata', methods=['GET'])
def get_metadata():
    # job: read model_config.json → return as JSON

# UI wants config table filtered
@app.route('/v1/getConfig', methods=['POST'])
def get_config():
    # job: receive config rows → filter by market/processtype → return JSON

# UI wants account mapping filtered
@app.route('/v1/getAccountMapping', methods=['POST'])
def get_account_mapping():
    # job: receive mapping rows → filter by market/processtype → return JSON

# UI wants to run prediction
@app.route('/v1/process', methods=['POST'])
def process():
    # job: receive all data → call process_data() → return predictions

# reviewer corrects a prediction
@app.route('/v1/OverwritePrediction', methods=['POST'])
def upsert_jrnoutput():
    # job: validate records → return accepted records

# operator submits review
@app.route('/v1/OverwriteReview', methods=['POST'])
def upsert_jrnreview():
    # job: validate records → return accepted records

# admin updates account mapping
@app.route('/saveAccountMapping', methods=['POST'])
def save_account_mapping():
    # job: validate records → return accepted records
