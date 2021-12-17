import streamlit as st
import shap
import pickle
import pandas as pd
import numpy as np
import sqlalchemy
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import BytesIO
from src.meta import seed, bloods
from src import db
from src.dash import shap_plots

plt.rcParams['font.size'] = '8'

def get_xy(df_in, x_cols_in, y_col='y_label', seed=seed):
    df = df_in.copy()
    x_cols = x_cols_in.copy()

    for col in x_cols:
        df[col] = df[col].astype(float)

    df = df.dropna(axis=0, subset=x_cols)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    X = df[x_cols]
    y = df[y_col]

    return X, y


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def shap_page(dataset, iteration):
    model = db.from_db(f'{dataset}/ML/shap/{iteration}/model.pickle')
    clustering = db.from_db(f'{dataset}/ML/shap/{iteration}/clustering_df.csv').to_numpy()
    X_in = db.from_db(f'{dataset}/ML/shap/{iteration}/X.csv')
    X_ids = X_in['id']
    X = X_in.drop('id', axis=1)
    
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(X)
    
    col1, col2 = st.columns(2)
    
    col1.plotly_chart(shap_plots.shap_scatter(shap_values, X, color_type='shap'),use_container_width=True)
    col2.plotly_chart(shap_plots.shap_feature_importances(shap_values,X),use_container_width=True)
    
    st.write("The explanation of the prediction of a single individual")
    id_dict = {k: v for v, k in enumerate(X_ids)}
    id_val = st.selectbox('id', id_dict.keys())
    
    st_shap(shap.force_plot(explainer.expected_value, shap_values[id_dict[id_val],:], X.iloc[id_dict[id_val],:]))

    st.write("Combined explanation of the predictions")
    
    st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
    
def shap_page_old(dataset, iteration):

    model = db.from_db(f'{dataset}/ML/shap/{iteration}/model.pickle')
    clustering = db.from_db(f'{dataset}/ML/shap/{iteration}/clustering_df.csv').to_numpy()
    X = db.from_db(f'{dataset}/ML/shap/{iteration}/X.csv')
    
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer(X)
    shap_vals = explainer.shap_values(X)
    clustered = st.checkbox('Clustered')

    col1, col2 = st.columns(2)

    

    if not clustered:
        # shap bar
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.bar(shap_values, max_display=15, show=False)
        fig = plt.gcf() # gcf means "get current figure"
        fig.set_figheight(4)
        fig.set_figwidth(4)
        ax = plt.gca()
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        col1.image(buf)

    else:
        # shap bar cluster
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.bar(shap_values, max_display=15, clustering=clustering, show=False)
        fig = plt.gcf() # gcf means "get current figure"
        fig.set_figheight(4)
        fig.set_figwidth(4)
        ax = plt.gca()
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        col1.image(buf)

    # shap bee
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    fig = plt.gcf() # gcf means "get current figure"
    fig.set_figheight(4)
    fig.set_figwidth(4)
    ax = plt.gca()
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    col2.image(buf)

    # shap decision
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    shap_vals = explainer.shap_values(X)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.decision_plot(expected_value, shap_vals, X, show=False)
    fig = plt.gcf() # gcf means "get current figure"
    fig.set_figheight(4)
    fig.set_figwidth(4)
    ax = plt.gca()
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    col1.image(buf)

    # shap heatmap
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots.heatmap(shap_values, max_display=15, show=False)
    fig = plt.gcf() # gcf means "get current figure"
    fig.set_figheight(4)
    fig.set_figwidth(5)
    ax = plt.gca()
    ax.tick_params(labelsize=8)
    #plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    col2.image(buf)


