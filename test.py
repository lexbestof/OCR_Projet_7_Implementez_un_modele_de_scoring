import streamlit as st
import unittest
from tableau import optimal_threshold, metier_cost, request_api
import unittest

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
        "index": [client_id],
        "data": [data_array]
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

client_id = 0
data_array = [0.0,
 1.0,
 0.0,
 180000.0,
 114682.5,
 13099.5,
 99000.0,
 7.0,
 4.0,
 0.0,
 1.0,
 0.018634,
 -10840.0,
 -1598.0,
 -4623.0,
 -3335.0,
 0.0,
 0.0,
 1.0,
 8.0,
 2.0,
 2.0,
 0.0,
 16.0,
 1.0,
 5.0,
 0.2886110122159613,
 0.4812493411434029,
 0.0,
 0.0,
 0.0,
 0.0,
 1.0,
 2.0,
 2.0,
 0.1474169741697417,
 1.5695507161075142,
 90000.0,
 0.072775,
 0.11422405336472435,
 -732.0,
 -318.0,
 -459.3333333333333,
 55785.33333333333,
 -396.0,
 897.0,
 423.0,
 -200.0,
 0.0,
 569160.0,
 390816.0,
 1172448.0,
 447484.5,
 149161.5,
 447484.5,
 0.0,
 0.0,
 0.0,
 0.0,
 47.0,
 9285.255,
 22663.755,
 14677.79625,
 0.0,
 87165.0,
 50588.55,
 0.0,
 87165.0,
 48244.5,
 0.9323091694671147,
 1.250129132231405,
 1.1081257169535554,
 0.02763807862960677,
 0.0,
 8716.5,
 4357.6875,
 43562.25,
 63235.6875,
 10.0,
 17.0,
 12.8,
 0.0,
 0.2179081800339934,
 0.10894284082107372,
 -1063.0,
 -52.0,
 -569.6,
 4.5,
 18.0,
 -1.0,
 -16.0,
 18.0,
 0.0,
 0.0,
 0.0,
 2.0,
 10.0,
 1.1333333333333333,
 17.0,
 47.0,
 21.8,
 327.0,
 1.0,
 0.8666666666666667,
 13.0,
 0.11151995991489609,
 9007.244999999999,
 1238.0339999999997,
 18570.509999999995,
 9614800.449507857,
 44971.92,
 16698.041999999998,
 250470.62999999998,
 278.01,
 44971.92,
 15460.008,
 231900.12,
 -52.0,
 -524.2666666666667,
 -7864.0,
 15.0]
print(request_api(client_id, data_array))
client_id = 5
data_array = [0, 1, 2, 3, 4] # données fictives

print(request_api(client_id, data_array))

class TestMyStreamlitApp(unittest.TestCase):
    def setUp(self):
        # Configuration spécifique à Streamlit pour les tests
        st.set_page_config(  # configure_page()
            page_title='Test Dashboard',
            page_icon="📊",
        )

    def test_dashboard_elements(self):
        # Tester les éléments spécifiques de notre tableau de bord
        with st.sidebar:
            # Tester le contenu de la barre latérale
            pass

        with st.container():
            # Tester le contenu principal du tableau de bord
            pass

class TestOptimalThresholdFunction(unittest.TestCase):
    def test_optimal_threshold(self):
        # Remplacez ces valeurs par celles que vous attendez
        min_seuil_val = 0.79
        expected_result = 0.79  # La valeur de seuil attendue

        result = optimal_threshold(min_seuil_val)

        self.assertEqual(result, expected_result, "La fonction optimal_threshold ne renvoie pas la valeur attendue.")

if __name__ == '__main__':
    unittest.main()
