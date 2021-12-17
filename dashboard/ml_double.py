import streamlit as st
import pandas as pd
from src.dash.model_plots import *
from src import db, prep
from src.dash import prep as vizprep



def ml_double_page(dataset, iteration):
    scalers = ['raw', 'log', 'minMax']
    sfs_dirs = ['forward', 'backwards']
    scaler = st.sidebar.selectbox('Scaler', scalers)
    sfs_dir = st.sidebar.selectbox('SFS Direction', sfs_dirs)
    
    results_inner_tmp = db.from_db(f'{dataset}/ML/data/double/results_inner.csv')
    results_outer_tmp = db.from_db(f'{dataset}/ML/data/double/results_outer.csv')
    sfs_avg = db.from_db(f'{dataset}/ML/data/double/sfs_avg.csv')
    
    #db.to_db(sfs_avg, f'{dataset}/ML/data/double/sfs_avg.csv')
   

    sfs = sfs_avg[(sfs_avg["iteration"] == iteration) & (sfs_avg["scaler"] == scaler) & (sfs_avg["sfs_dir"] == sfs_dir)]
    results_outer = results_outer_tmp[(results_outer_tmp["iteration"] == iteration) & (results_outer_tmp["scaler"] == scaler) & (results_outer_tmp["sfs_dir"] == sfs_dir)]
    results_inner = results_inner_tmp[(results_inner_tmp["iteration"] == iteration) & (results_inner_tmp["scaler"] == scaler) & (results_inner_tmp["sfs_dir"] == sfs_dir)]

    df_scores_outer = vizprep.scores(results_outer)
    df_scores_inner = vizprep.scores(results_inner)
    
    results_all = results_outer_tmp[(results_outer_tmp["iteration"] == iteration)]
    
    models_list = ['LR', 'SVC', 'LDA', 'GB', 'RF', 'MLP']
    

    ## SFS DIR IS MISSING
    ## ONLY SET TO FORWARD
    df_scores_all = pd.DataFrame()

    for scaler in scalers:
        df_tmp = results_all[(results_all["scaler"] == scaler) & (results_all["sfs_dir"] == "forward")]
        top_scores_df = vizprep.scores(df_tmp)
        top_score = top_scores_df['Accuracy'].max()
        top_scores_df = top_scores_df[top_scores_df['Accuracy'] == top_score]
        top_scores_df['scaler'] = scaler

        df_scores_all = df_scores_all.append(top_scores_df)

    top_score = df_scores_all['Accuracy'].max()
    
    scores_type = st.sidebar.radio("Score View", ['Bar', 'Heatmap - Outer', 'Heatmap - Inner'])
    
    col1, col2 = st.columns(2)
    col2.write(df_scores_all[df_scores_all['Accuracy'] == top_score])
    if scores_type == 'Bar':
        col1.plotly_chart(scores_plt_double(df_scores_inner, df_scores_outer),use_container_width=True)
    elif scores_type == 'Heatmap - Outer':
        col1.plotly_chart(scores_plt(df_scores_outer),use_container_width=True)
    elif scores_type == 'Heatmap - Inner':
        col1.plotly_chart(scores_plt(df_scores_inner),use_container_width=True)
    
    
    col2.plotly_chart(rfe_plt(sfs),use_container_width=True)
    

    

    col4, col5, col6, col7 = st.columns([1,4,4,4])
    model_in = col4.radio('Model', ['MLP', 'GB', 'RF', 'LDA', 'SVC', 'LR', 'dummy'])
    col5.plotly_chart(roc_plot(results_outer, model_in),use_container_width=True)
    col6.plotly_chart(roc_hist(results_outer, model_in),use_container_width=True)
    col7.plotly_chart(roc_thresh(results_outer, model_in),use_container_width=True)