# Importation des bibliothèques nécessaires
import streamlit as st
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt


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


def load_model_and_data():
    lgbm_clf = pickle.load(open("C:/Users/jerom/projet_7/dashboard_streamlit/Archived/LGBMClassifier.pkl", "rb"))
    list_summary_plot_shap = joblib.load('C:/Users/jerom/projet_7/dashboard_streamlit/Archived/list_summary_plot_shap.joblib')
    X_validation = np.load("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/X_validation_np.npy")
    y_validation = np.load("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/y_validation_np.npy")
    X_validation_df = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/X_validation.csv")
    y_validation_df = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/y_validation.csv")
    df_val_sample = joblib.load("C:/Users/jerom/projet_7/dashboard_streamlit/src/df_val_sample.joblib")
    val_set_pred_proba = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/val_set_pred_proba.csv")
    train_set_pred_proba = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/train_set_pred_proba.csv")
    data_rfecv = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/data_rfecv.csv")
    importance_results = pickle.load(
        open('C:/Users/jerom/projet_7/dashboard_streamlit/Archived/results_permutation_importance.pkl', 'rb'))
    val_set_pred_proba.set_index("SK_ID_CURR", inplace=True)

    return lgbm_clf, list_summary_plot_shap, X_validation, y_validation, X_validation_df, y_validation_df, df_val_sample, val_set_pred_proba, train_set_pred_proba, data_rfecv, importance_results


def calculate_probabilities(lgbm_clf, X_validation):
    y_proba_validation = lgbm_clf.predict_proba(X_validation)[:, 1]
    return y_proba_validation


def optimal_threshold():
    min_seuil_val = joblib.load("C:/Users/jerom/projet_7/dashboard_streamlit/Api_ml/optimum_threshold.joblib")
    return min_seuil_val


def metier_cost(y_true, y_pred, cout_fn=10, cout_fp=1):
    vp = np.sum((y_true == 1) & (y_pred == 1))
    vn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cout = cout_fn * fn + cout_fp * fp
    return cout


def display_model_results(lgbm_clf, min_seuil_val, y_proba_validation, y_true, val_set_pred_proba, X_validation_df, y_validation_df):
 
    st.title("Dashboard d'Évaluation du Modèle")
    st.subheader("Résultats du Modèle")

    st.subheader("Seuil Optimal")
    st.markdown(f"Seuil optimal : {min_seuil_val}")

    st.subheader("Jauge Colorée pour le Score")
    pourcentage_score = int(y_proba_validation[0] * 100)
    score_jauge = st.progress(pourcentage_score)
    score_jauge.progress(int(min_seuil_val * 100))

    explainer = shap.TreeExplainer(lgbm_clf)
    X_val_new_df = X_validation_df  # Use X_validation_df directly
    sample_idx = X_val_new_df.sample(1).index[0]
    predicted_class = int(lgbm_clf.predict(X_val_new_df.loc[[sample_idx]]))
    shap_values = explainer.shap_values(X_val_new_df.loc[[sample_idx]])[predicted_class]

    selected_client = st.selectbox("Sélectionnez un client :", val_set_pred_proba.index)
    
    st.subheader("Affichage du résultat de la prédiction")
    prediction_value = int(val_set_pred_proba.loc[selected_client, 'pred_proba'] > min_seuil_val)

    if prediction_value == 1:
        st.write("**Crédit Refusé**", unsafe_allow_html=True, key="credit_refused")
        st.write("Nous sommes désolés, mais la prédiction du modèle indique que le crédit doit être refusé.")
    else:
        st.write("**Crédit Accordé**", key="credit_accepted")
        st.write("Félicitations! Le modèle prédit que le crédit peut être accordé.")
        
    st.subheader("Importance des variables pour un échantillon individuel")

    plot_shap_bar_plot(shap_values[0], X_val_new_df, X_val_new_df.columns, max_display=10)
    force_plot = shap.force_plot(explainer.expected_value[predicted_class], shap_values[0],
                                X_val_new_df.loc[[sample_idx]])
    st.components.v1.html(shap.getjs() + force_plot._repr_html_(), height=600, scrolling=True)

    st.subheader("Analyse des variables")

    selected_client_other = st.selectbox("Sélectionnez un autre client :", val_set_pred_proba.index)
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

    df_predictproba = pd.read_csv("C:/Users/jerom/projet_7/dashboard_streamlit/Cleaned/df_predictproba.csv")
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


def plot_shap_bar_plot(shap_values, features, feature_names, max_display=10):
    plt.figure(edgecolor='black', linewidth=4)
    shap.bar_plot(shap_values, feature_names=feature_names, max_display=max_display)
    st.pyplot(plt.gcf())


def main():
    configure_page()
    lgbm_clf, list_summary_plot_shap, X_validation, y_validation, X_validation_df, y_validation_df, df_val_sample, val_set_pred_proba, train_set_pred_proba, data_rfecv, importance_results = load_model_and_data()
    y_proba_validation = calculate_probabilities(lgbm_clf, X_validation)
    min_seuil_val = optimal_threshold()
    y_true = y_validation.flatten()
    cout = metier_cost(y_true, y_proba_validation > min_seuil_val)
    display_model_results(lgbm_clf, min_seuil_val, y_proba_validation, y_true, val_set_pred_proba, X_validation_df, y_validation_df)
    st.write(f"Coût métier : {cout}")



if __name__ == "__main__":
    main()
