# Importation des bibliothèques nécessaires
import ast
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import urllib.request
import json
import os
import ssl
import platform
import streamlit as st
from streamlit_shap import st_shap
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import requests
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import socket
import os

def configure_page():
    st.set_page_config(
        page_title='Tableau de bord de la prédiction',
        page_icon="📊",
    )

    st.markdown(""" 
        <style>
        body {font-family:'Roboto Condensed';}
        h1 {font-family:'Roboto Condensed';}
        h2 {font-family:'Roboto Condensed';}
        p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
        .css-18e3th9 {padding-top: 1rem; 
                    padding-right: 1rem; 
                    padding-bottom: 1rem; 
                    padding-left: 1rem;}
        .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1rem;}
        .css-1offfwp li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
        </style> """,
        unsafe_allow_html=True)
def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

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
    api_key = 'GUR5dawhrjWNB5p8JAICSnm4P3oMqkkz'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
               'azureml-model-deployment': 'scoring-model-2-1'}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
    
    # Décoder les bytes en chaîne de caractères en utilisant UTF-8
        result_str = result.decode("utf-8")
    
        print(result)

        predictions = json.loads(result.decode("utf-8"))
        print("Predictions from API:", predictions)
        if predictions is not None:
            return predictions
        else:
            raise ValueError("Predictions not found in API response")


    except urllib.error.HTTPError as error:
        print("La requête a échoué avec le code d'état : " + str(error.code))
    
    # Imprimer les en-têtes - ils incluent l'ID de la requête et l'horodatage, qui sont utiles pour le débogage de l'échec
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))




def download_blob_as_bytes(blob_client):
    return blob_client.download_blob().readall()

def load_data_from_blob(blob_service_client, container_name, data_name, loader_func):
    blob_client_data = blob_service_client.get_blob_client(container=container_name, blob=data_name)
    data_bytes = download_blob_as_bytes(blob_client_data)

    # Check if loader_func is pickle.load and handle accordingly
    if loader_func == pickle.load:
        with BytesIO(data_bytes) as file:
            return loader_func(file)
    # Check if loader_func is np.load and handle accordingly
    elif loader_func == np.load:
        with BytesIO(data_bytes) as file:
            return loader_func(file, allow_pickle=True)
    # Check if loader_func is pd.read_csv and handle accordingly
    elif loader_func == pd.read_csv:
        with BytesIO(data_bytes) as file:
            return loader_func(file, encoding='utf-8')
    # Check if loader_func is joblib.load and handle accordingly
    elif loader_func == joblib.load:
        with BytesIO(data_bytes) as file:
            return loader_func(file)
    else:
        return loader_func(data_bytes)
    
def load_model_from_blob(blob_service_client, container_name, model_name):
    blob_client_model = blob_service_client.get_blob_client(container=container_name, blob=model_name)
    model_bytes = download_blob_as_bytes(blob_client_model)
    return pickle.loads(model_bytes)


def load_model_and_data():
    # Load other artifacts from Azure Blob Storage
    account_name = "projet7ocrscor9718876976"
    account_key = "KjFREPbKDYnZoilBu8J4g4O9UeAWnjQqkmaUnh9uhcAsE3Bab+YFycRASbHm8B/y0LimiIffjEVX+AStTMv6QQ=="
    container_name = "mlflow-model"

    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

    importance_results = load_data_from_blob(blob_service_client, container_name, "results_permutation_importance.pkl", pickle.load)
    X_validation = load_data_from_blob(blob_service_client, container_name, "X_validation_np.npy", np.load)
    y_validation = load_data_from_blob(blob_service_client, container_name, "y_validation_np.npy", np.load)
    X_validation_df = load_data_from_blob(blob_service_client, container_name, "X_validation.csv", pd.read_csv)
    y_validation_df = load_data_from_blob(blob_service_client, container_name, "y_validation.csv", pd.read_csv)

    # Load the MLflow model from Azure Blob Storage
    model_path = "mlflow_model/model.pkl"
    model = load_model_from_blob(blob_service_client, container_name, model_path)

    df_val_sample = load_data_from_blob(blob_service_client, container_name, "df_val_sample.joblib", joblib.load)
    feature_importance = load_data_from_blob(blob_service_client, container_name, "figure_summary_plot_shap.joblib", joblib.load)
    val_set_pred_proba = load_data_from_blob(blob_service_client, container_name, "val_set_pred_proba.csv", lambda x: pd.read_csv(BytesIO(x)))
    # Réinitialiser l'index
    val_set_pred_proba = val_set_pred_proba.reset_index()
  

    # Charger min_seuil_val directement dans la fonction
    model_name = "optimum_threshold.joblib"
    min_seuil_val = load_data_from_blob(blob_service_client, container_name, model_name, joblib.load)

    # Charger df_predictproba directement dans la fonctionn
    data_name = "df_predictproba.csv"
    df_predictproba = load_data_from_blob(blob_service_client, container_name, data_name, pd.read_csv)

    return model, X_validation, y_validation, X_validation_df, y_validation_df, val_set_pred_proba, importance_results,feature_importance, min_seuil_val, df_val_sample, df_predictproba

#Fonction de calcul de probabilité
def calculate_probabilities_api(client_id, data_array):
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
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
               'azureml-model-deployment': 'scoring-model-2-1'}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print(result)
        predictions = json.loads(result.decode("utf-8"))

        if predictions is not None:
            y_proba_validation = np.array(predictions)[:]
            return y_proba_validation
        else:
            raise ValueError("Predictions not found in API response")

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))


def optimal_threshold(min_seuil_val):
    return min_seuil_val

#Fonction du coût métier
def metier_cost(y_true, y_pred, cout_fn=10, cout_fp=1):
    vp = np.sum((y_true == 1) & (y_pred == 1))
    vn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cout = cout_fn * fn + cout_fp * fp
    return cout


def display_model_results(model, X_validation, y_validation,y_proba_validation, X_validation_df, y_validation_df,feature_importance, val_set_pred_proba, min_seuil_val, df_predictproba):
    
    #Titre du dashboard
    st.title("Tableau de bord intéractif d'évaluation de notre modèle")
    st.subheader("Résultats du Modèle")

    st.subheader("Seuil Optimal")
    st.markdown(f"Seuil optimal : {min_seuil_val}")
    st.info("Le seuil optimal est calculé pour minimiser les coûts métier. Plus le seuil est bas, plus le modèle est conservateur.")


    pourcentage_score = int(y_proba_validation[0] * 100)

    # Utiliser le widget progress pour afficher la jauge
    score_jauge = st.progress(pourcentage_score)
    score_jauge.progress(int(min_seuil_val * 100))
    

    selected_client = st.selectbox("Sélectionnez un client :", X_validation_df.index)
    
    # Obtenir la probabilité de prédiction depuis l'API pour le client choisi
    selected_client_data = X_validation[selected_client].tolist()
    api_prediction_proba = request_api(selected_client, selected_client_data)
    
    # obtenir les probabilités depuis le fichier local
    prediction_value = int(val_set_pred_proba.loc[selected_client, 'pred_proba'] > min_seuil_val)
    prediction_proba = val_set_pred_proba.loc[selected_client, 'pred_proba']

    
    st.header('Résultat de la demande de prêt')
   
    if api_prediction_proba == [0.0]:
        st.success(
            f"  \n __CREDIT ACCORDÉ__  \n  \nLa probabilité de défaut de remboursement pour le crédit demandé est de __{round(100*prediction_proba,1)}__% (inférieur aux {100*optimal_threshold(min_seuil_val)}% pour l'obtention d'un prêt).  \n "
        )
    else:
        st.error(
            f"__CREDIT REFUSÉ__  \nLa probabilité de défaut de remboursement pour le crédit demandé est de __{round(100*prediction_proba,1)}__ \n "
        )
        #% (supérieur aux {100*optimal_threshold(min_seuil_val)}% pour l'obtention d'un prêt).  


    st.subheader("Importance des variables")
    st.info("L'Importance des variables est une mesure qui permet de quantifier l'importance relative de chaque variable dans un modèle de prédiction. Cette mesure permet de comprendre quelles variables ont le plus grand impact sur les prédictions du modèle et donc de mieux comprendre les relations entre les variables et les prédictions.")

    st.title("Importance Globale des variables")
     # Display the Matplotlib Figure
    st.pyplot(feature_importance)

    st.subheader("Importance de variable locale")
    st.info("Quelles sont le variables qui ont conditionné la prise de décision pour le client sélectionné?")   
    
    explainer = shap.TreeExplainer(model)
    X_val_new_df = X_validation_df  # Use X_validation_df directly

    sample_idx = selected_client
    predicted_class = int(model.predict(X_validation_df.loc[[sample_idx]]))
    shap_values = explainer.shap_values(X_validation_df.loc[[sample_idx]])[predicted_class]

    plot_shap_bar_plot(shap_values[0], X_validation_df, X_validation_df.columns, max_display=10)


    st.subheader("Analyse des variables")
    st.subheader(f"Caractéristiques du Client {selected_client}")

    feature_options = X_validation_df.columns.tolist()
    selected_feature_1 = st.selectbox("Sélectionnez la première caractéristique :", feature_options)
    selected_feature_2 = st.selectbox("Sélectionnez la deuxième caractéristique :", feature_options)

    fig_dist = px.histogram(
        X_validation_df,
        x=selected_feature_1,
        color=y_validation_df["TARGET"].map({0: 'Non défaut', 1: 'Défaut'}),
        marginal="rug",
        nbins=30,
        title=f"Distribution de {selected_feature_1}"
    )
    st.plotly_chart(fig_dist)

    fig_dist_2 = px.histogram(
        X_validation_df,
        x=selected_feature_2,
        color=y_validation_df["TARGET"].map({0: 'Non défaut', 1: 'Défaut'}),
        marginal="rug",
        nbins=30,
        title=f"Distribution de {selected_feature_2}"
    )
    st.plotly_chart(fig_dist_2)

    fig_bivariate = px.scatter(
        X_validation_df,
        x=selected_feature_1,
        y=selected_feature_2,
        color=y_validation_df["TARGET"].map({0: 'Non défaut', 1: 'Défaut'}),
        color_continuous_scale="Viridis",
        title=f"Analyse Bi-Variée ({selected_feature_1} vs {selected_feature_2})"
    )

# Mettre à jour la légende en fonction de y_validation_df["TARGET"]
    fig_bivariate.update_layout(
        legend=dict(
            title="Défaut",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig_bivariate)

    ser_predictproba_true0 = df_predictproba.loc[df_predictproba['y_true'] == 0, 'y_predict_proba']
    ser_predictproba_true1 = df_predictproba.loc[df_predictproba['y_true'] == 1, 'y_predict_proba']

    fig, ax = plt.subplots()
    ser_predictproba_true0.plot(kind='kde', c='g', label='Clients sans défaut', bw_method=0.15, ind=1000)
    ser_predictproba_true1.plot(kind='kde', c='r', label='Clients avec défaut', bw_method=0.15)
    ax.set_title('Distribution des Probabilités selon la Vraie Classe du Client')
    ax.axvline(x=min_seuil_val, color='darkorange', linestyle='--', label='Seuil Optimal')
    ax.legend()
    ax.set_xlabel('Probabilité de Défaut (calculée par LGBM)')
    ax.set_ylabel('Densité de Probabilité')
    #st.pyplot(fig)


def plot_shap_bar_plot(shap_values, features, feature_names, max_display=10):
    plt.figure(edgecolor='black', linewidth=4)
    shap.bar_plot(shap_values, feature_names=feature_names, max_display=max_display)
    st.pyplot(plt.gcf())


def get_predictions_api(client_id, data_array):
    predictions = request_api(client_id, data_array)
    # Décoder les résultats en utilisant ast.literal_eval pour obtenir une liste de nombres
    return [ast.literal_eval(prediction) for prediction in predictions]

def main():
    configure_page()
    model, X_validation, y_validation, X_validation_df, y_validation_df, val_set_pred_proba, importance_results,feature_importance, min_seuil_val, df_val_sample, df_predictproba = load_model_and_data()

    min_seuil_val = optimal_threshold(min_seuil_val)
    y_true = y_validation.flatten()

    allowSelfSignedHttps(True)

    # Cette partie me sert de voir la structure de quelques prédictions
    for index in X_validation_df[:10].index:
        client_id = index


    # Extraire le data_array depuis X_validation
        data_to_send = X_validation[client_id].tolist()


    # appeler la fonction request_api  avec client_id et data_to_send
        predictions = request_api(client_id, data_to_send)
    #Afficher les valeurs de prédiction afin de voir leurs structures
    
        print(f"Valeurs de api_prediction_proba pour le client {client_id} : {predictions}")

    # Calculer le coût métier
    # Utiliser min_seuil_val directement pour calculer le coût
        #cout = metier_cost(y_true, predictions)

    # Afficher les autres résultats du modèle
    display_model_results(model, X_validation, y_validation, predictions, X_validation_df, y_validation_df,feature_importance, val_set_pred_proba, min_seuil_val, df_predictproba)

    #st.write(f"Coût métier : {cout}")

if __name__ == "__main__":
    main()