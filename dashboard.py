# Importation des bibliothèques nécessaires
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import platform
import streamlit as st
from streamlit_shap import st_shap
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
import shutil

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

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    if isinstance(data, pd.DataFrame):
        input_data = {"instances": data.to_dict(orient='records')}
    elif isinstance(data, np.ndarray):
        input_data = {"instances": data.tolist()}
    else:
        raise ValueError("Unsupported data type. Use DataFrame or numpy array.")

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=input_data)
    
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}")

    return response.json()

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
    val_set_pred_proba = load_data_from_blob(blob_service_client, container_name, "val_set_pred_proba.csv", lambda x: pd.read_csv(BytesIO(x)))
    # Réinitialiser l'index
    val_set_pred_proba = val_set_pred_proba.reset_index()
  
    # Load min_seuil_val directly in the function
    model_name = "optimum_threshold.joblib"
    min_seuil_val = load_data_from_blob(blob_service_client, container_name, model_name, joblib.load)

    # Load df_predictproba directly in the function
    data_name = "df_predictproba.csv"
    df_predictproba = load_data_from_blob(blob_service_client, container_name, data_name, pd.read_csv)

    return model, X_validation, y_validation, X_validation_df, y_validation_df, val_set_pred_proba, importance_results, min_seuil_val, df_val_sample, df_predictproba


def calculate_probabilities(model_uri, X_validation_df):
    headers = {"Content-Type": "application/json"}

    if isinstance(X_validation_df, pd.DataFrame):
        input_data = {"instances": X_validation_df.to_dict(orient='records')}
    elif isinstance(X_validation_df, np.ndarray):
        input_data = {"instances": X_validation_df.tolist()}
    else:
        raise ValueError("Unsupported data type. Use DataFrame or numpy array.")

    # Remove "/invocations" from model_uri
    response = requests.post(
        url=model_uri,  # Model URI without "/invocations"
        headers=headers,
        json=input_data
    )

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")

    predictions = response.json()["predictions"]
    y_proba_validation = np.array(predictions)[:]

    return y_proba_validation


def optimal_threshold(min_seuil_val):
    return min_seuil_val


def metier_cost(y_true, y_pred, cout_fn=10, cout_fp=1):
    vp = np.sum((y_true == 1) & (y_pred == 1))
    vn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cout = cout_fn * fn + cout_fp * fp
    return cout


def display_model_results(MLFLOW_URI, model, X_validation, y_validation,y_proba_validation, X_validation_df, y_validation_df, val_set_pred_proba, min_seuil_val, df_predictproba):
    
    #Titre du dashboard
    st.title("Tableau de bord intéractif d'évaluation de modèle")
    st.subheader("Résultats du Modèle")

    st.subheader("Seuil Optimal")
    st.markdown(f"Seuil optimal : {min_seuil_val}")
    st.info("Le seuil optimal est calculé pour minimiser les coûts métier. Plus le seuil est bas, plus le modèle est conservateur.")
   
    pourcentage_score = int(y_proba_validation[0] * 100)

    # Utiliser le widget progress pour afficher la jauge
    score_jauge = st.progress(pourcentage_score)
    score_jauge.progress(int(min_seuil_val * 100))

    explainer = shap.TreeExplainer(model)
    X_val_new_df = X_validation_df  # 
    
    selected_client = st.selectbox("Sélectionnez un client :", X_validation_df.index)
    
    st.subheader("Affichage du résultat de la prédiction")

    # Obtenir la probabilité de prédiction depuis l'API pour le client choisi
    selected_client_data = X_validation_df.loc[[selected_client]]
    api_prediction_proba = calculate_probabilities(MLFLOW_URI, selected_client_data)[0]

    # obtenir les probabilités depuis le fichier local
    prediction_value = int(val_set_pred_proba.loc[selected_client, 'pred_proba'] > min_seuil_val)
    prediction_proba = val_set_pred_proba.loc[selected_client, 'pred_proba']

    
    st.header('Résultat de la demande de prêt')
   
    if api_prediction_proba == 0:
        st.success(
            f"  \n __CREDIT ACCORDÉ__  \n  \nLa probabilité de défaut de remboursement pour le crédit demandé est de __{round(100*prediction_proba,1)}__% (inférieur aux {100*optimal_threshold(min_seuil_val)}% pour l'obtentiion d'un prêt).  \n "
        )
    else:
        st.error(
            f"__CREDIT REFUSÉ__  \nLa probabilité de défaut de remboursement pour le crédit demandé __{round(100*prediction_proba,1)}__% (supérieur aux {100*optimal_threshold(min_seuil_val)}% pour l'obtention d'un prêt).  \n "
        )
        
    st.subheader("Importance de variable locale")
    st.info("Importance des variables est une mesure qui permet de quantifier l'importance relative de chaque variable dans un modèle de prédiction. Cette mesure permet de comprendre quelles variables ont le plus grand impact sur les prédictions du modèle et donc de mieux comprendre les relations entre les variables et les prédictions.")
    

    sample_idx = selected_client
    predicted_class = int(model.predict(X_val_new_df.loc[[sample_idx]]))
    shap_values = explainer.shap_values(X_val_new_df.loc[[sample_idx]])[predicted_class]

    plot_shap_bar_plot(shap_values[0], X_val_new_df, X_val_new_df.columns, max_display=10)
    force_plot = shap.force_plot(explainer.expected_value[predicted_class], shap_values[0],
                                X_val_new_df.loc[[sample_idx]])
    st.components.v1.html(shap.getjs() + force_plot._repr_html_(), height=600, scrolling=True)

    st.subheader("Analyse des variables")

    selected_client_other = st.selectbox("Sélectionnez un autre client :", X_validation_df.index)
    st.subheader(f"Caractéristiques du Client {selected_client_other}")

    feature_options = X_validation_df.columns.tolist()
    selected_feature_1 = st.selectbox("Sélectionnez la première caractéristique :", feature_options)
    selected_feature_2 = st.selectbox("Sélectionnez la deuxième caractéristique :", feature_options)

    fig_dist = px.histogram(X_validation_df, x=selected_feature_1, color=y_validation_df["TARGET"],
                            marginal="rug", nbins=30, title=f"Distribution de {selected_feature_1}")
    st.plotly_chart(fig_dist)

    fig_dist_2 = px.histogram(X_validation_df, x=selected_feature_2, color=y_validation_df["TARGET"],
                              marginal="rug", nbins=30, title=f"Distribution de {selected_feature_2}")
    st.plotly_chart(fig_dist_2)

    fig_bivariate = px.scatter(X_validation_df, x=selected_feature_1, y=selected_feature_2,
                               color=y_proba_validation, color_continuous_scale="Viridis",
                               title=f"Analyse Bi-Variée ({selected_feature_1} vs {selected_feature_2})")
    st.plotly_chart(fig_bivariate)

    st.subheader("Importance des caractéristiques globales")
    fig_bar_plot, ax_bar_plot = plt.subplots()
    ax_bar_plot.barh(X_val_new_df.columns, shap_values[0], color='skyblue')
    ax_bar_plot.set_xlabel('Importance')
    ax_bar_plot.set_title('Importance des Caractéristiques (Local)')

    fig_summary_plot = plt.figure()
    shap.summary_plot(shap_values, features=X_val_new_df.loc[[sample_idx]], feature_names=X_val_new_df.columns)
    st.pyplot(fig_summary_plot)

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
    st.pyplot(fig)

# Positionner un client par rapport à d'autres clients
    


def plot_shap_bar_plot(shap_values, features, feature_names, max_display=10):
    plt.figure(edgecolor='black', linewidth=4)
    shap.bar_plot(shap_values, feature_names=feature_names, max_display=max_display)
    st.pyplot(plt.gcf())


def get_predictions(model_uri, data):
    pred = request_prediction(model_uri, data)
    return pred["predictions"]


def main():
    configure_page()

    model, X_validation, y_validation, X_validation_df, y_validation_df, val_set_pred_proba, importance_results, min_seuil_val, df_val_sample, df_predictproba = load_model_and_data()

    min_seuil_val = optimal_threshold(min_seuil_val)
    y_true = y_validation.flatten()

    print(val_set_pred_proba.head(10))
    # Vérifier si l'API REST locale est disponible
    # mlflow models serve -m mlflow_model --host 127.0.0.1 --port 8001
    local_mlflow_uri = 'http://127.0.0.1:8001/invocations'
    is_local_mlflow_available = check_api_availability(local_mlflow_uri)

    # Utiliser l'adresse locale si disponible, sinon utiliser l'adresse en ligne du serveur Azure
    #Note ce code ne marche pas avec le modèle serveur d'Azure studio
    #Le code spécial avec Azure studio a été effectué ailleurs.
    MLFLOW_URI = local_mlflow_uri if is_local_mlflow_available else 'https://projet-7-ocr-scoring-model.eastus2.inference.ml.azure.com/score'

    predictions = get_predictions(MLFLOW_URI, X_validation_df)


    # Calculer le coût métier
    cout = metier_cost(y_true, predictions > min_seuil_val)

    # Afficher les autres résultats du modèle
    display_model_results(MLFLOW_URI, model, X_validation, y_validation, predictions, X_validation_df, y_validation_df, val_set_pred_proba, min_seuil_val, df_predictproba)

    st.write(f"Coût métier : {cout}")

# Fonction pour vérifier la disponibilité de l'API locale
def check_api_availability(api_uri):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    # Essayez de se connecter à l'adresse locale
    result = sock.connect_ex(('127.0.0.1', 8001))
    sock.close()

    return result == 0

if __name__ == "__main__":
    main()