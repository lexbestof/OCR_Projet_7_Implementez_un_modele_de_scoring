
import urllib.request
import json
import os
import ssl




def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

def request_api(client_id, data_array):

    data =  {
    "input_data": {
        "columns": [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "OCCUPATION_TYPE",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
        "REG_CITY_NOT_LIVE_CITY",
        "ORGANIZATION_TYPE",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "FLAG_DOCUMENT_3",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
        "BURO_DAYS_CREDIT_MIN",
        "BURO_DAYS_CREDIT_MAX",
        "BURO_DAYS_CREDIT_MEAN",
        "BURO_DAYS_CREDIT_VAR",
        "BURO_DAYS_CREDIT_ENDDATE_MIN",
        "BURO_DAYS_CREDIT_ENDDATE_MAX",
        "BURO_DAYS_CREDIT_ENDDATE_MEAN",
        "BURO_DAYS_CREDIT_UPDATE_MEAN",
        "BURO_CREDIT_DAY_OVERDUE_MAX",
        "BURO_AMT_CREDIT_SUM_MAX",
        "BURO_AMT_CREDIT_SUM_MEAN",
        "BURO_AMT_CREDIT_SUM_SUM",
        "BURO_AMT_CREDIT_SUM_DEBT_MAX",
        "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
        "BURO_AMT_CREDIT_SUM_DEBT_SUM",
        "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN",
        "BURO_AMT_CREDIT_SUM_LIMIT_MEAN",
        "BURO_AMT_CREDIT_SUM_LIMIT_SUM",
        "BURO_CNT_CREDIT_PROLONG_SUM",
        "BURO_BB_MONTHS_BALANCE_SIZE_SUM",
        "PREV_AMT_ANNUITY_MIN",
        "PREV_AMT_ANNUITY_MAX",
        "PREV_AMT_ANNUITY_MEAN",
        "PREV_AMT_APPLICATION_MIN",
        "PREV_AMT_APPLICATION_MAX",
        "PREV_AMT_APPLICATION_MEAN",
        "PREV_AMT_CREDIT_MIN",
        "PREV_AMT_CREDIT_MAX",
        "PREV_AMT_CREDIT_MEAN",
        "PREV_APP_CREDIT_PERC_MIN",
        "PREV_APP_CREDIT_PERC_MAX",
        "PREV_APP_CREDIT_PERC_MEAN",
        "PREV_APP_CREDIT_PERC_VAR",
        "PREV_AMT_DOWN_PAYMENT_MIN",
        "PREV_AMT_DOWN_PAYMENT_MAX",
        "PREV_AMT_DOWN_PAYMENT_MEAN",
        "PREV_AMT_GOODS_PRICE_MIN",
        "PREV_AMT_GOODS_PRICE_MEAN",
        "PREV_HOUR_APPR_PROCESS_START_MIN",
        "PREV_HOUR_APPR_PROCESS_START_MAX",
        "PREV_HOUR_APPR_PROCESS_START_MEAN",
        "PREV_RATE_DOWN_PAYMENT_MIN",
        "PREV_RATE_DOWN_PAYMENT_MAX",
        "PREV_RATE_DOWN_PAYMENT_MEAN",
        "PREV_DAYS_DECISION_MIN",
        "PREV_DAYS_DECISION_MAX",
        "PREV_DAYS_DECISION_MEAN",
        "PREV_CNT_PAYMENT_MEAN",
        "PREV_CNT_PAYMENT_SUM",
        "POS_MONTHS_BALANCE_MAX",
        "POS_MONTHS_BALANCE_MEAN",
        "POS_MONTHS_BALANCE_SIZE",
        "POS_SK_DPD_MAX",
        "POS_SK_DPD_DEF_MAX",
        "POS_SK_DPD_DEF_MEAN",
        "INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE",
        "INSTAL_DPD_MAX",
        "INSTAL_DPD_MEAN",
        "INSTAL_DPD_SUM",
        "INSTAL_DBD_MAX",
        "INSTAL_DBD_MEAN",
        "INSTAL_DBD_SUM",
        "INSTAL_PAYMENT_PERC_MAX",
        "INSTAL_PAYMENT_PERC_MEAN",
        "INSTAL_PAYMENT_PERC_SUM",
        "INSTAL_PAYMENT_PERC_VAR",
        "INSTAL_PAYMENT_DIFF_MAX",
        "INSTAL_PAYMENT_DIFF_MEAN",
        "INSTAL_PAYMENT_DIFF_SUM",
        "INSTAL_PAYMENT_DIFF_VAR",
        "INSTAL_AMT_INSTALMENT_MAX",
        "INSTAL_AMT_INSTALMENT_MEAN",
        "INSTAL_AMT_INSTALMENT_SUM",
        "INSTAL_AMT_PAYMENT_MIN",
        "INSTAL_AMT_PAYMENT_MAX",
        "INSTAL_AMT_PAYMENT_MEAN",
        "INSTAL_AMT_PAYMENT_SUM",
        "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
        "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
        "INSTAL_DAYS_ENTRY_PAYMENT_SUM",
        "INSTAL_COUNT"
        ],
        "index": [],
        "data": []
    },
    "params": {}
    }

    body = str.encode(json.dumps(data))

    url = 'https://projet-7-ocr-scoring-model.eastus2.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = 'GUR5dawhrjWNB5p8JAICSnm4P3oMqkkz'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'scoring-model-2-1' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        print(result)

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))