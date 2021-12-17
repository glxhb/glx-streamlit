import streamlit as st
import pandas as pd
from src.dash.model_plots import *
from src import db, prep
from src.dash import prep as vizprep



def ml_single_page(dataset, iteration):
    scalers = ['raw', 'log', 'minMax']
    sfs_dirs = ['forward', 'backwards']
    scaler = st.sidebar.selectbox('Scaler', scalers)
    sfs_dir = st.sidebar.selectbox('SFS Direction', sfs_dirs)
    
    sfs_check = st.sidebar.checkbox('Using best columns')
    
    if sfs_check:
        results_tmp = db.from_db(f'{dataset}/ML/data/single/results_sfs.csv')
    else:
        results_tmp = db.from_db(f'{dataset}/ML/data/single/results.csv')
        
    sfs_results_tmp = db.from_db(f'{dataset}/ML/data/single/sfs_results.csv')
    
    results_all = results_tmp[(results_tmp["iteration"] == iteration)]

    sfs = sfs_results_tmp[(sfs_results_tmp["iteration"] == iteration) & (sfs_results_tmp["scaler"] == scaler) & (sfs_results_tmp["sfs_dir"] == sfs_dir)]
    results = results_tmp[(results_tmp["iteration"] == iteration) & (results_tmp["scaler"] == scaler) & (results_tmp["sfs_dir"] == sfs_dir)]
    
    models_list = ['LR', 'SVC', 'LDA', 'GB', 'RF', 'MLP']
    
    scores_df = vizprep.scores(results)
    
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
    
    #col1, col2 = st.columns(2)
    #col1.plotly_chart(scores_plt(scores_df))
    
    #col2.plotly_chart(rfe_plt(sfs))
    
    col1, col2 = st.columns(2)
    col2.write(df_scores_all[df_scores_all['Accuracy'] == top_score])
    col1.plotly_chart(scores_plt(scores_df),use_container_width=True)
    
    
    col2.plotly_chart(rfe_plt(sfs),use_container_width=True)
    

    #st.plotly_chart(scores_plt_double(inner, outer, 900, 300))

    col4, col5, col6, col7 = st.columns([1,4,4,4])
    model_in = col4.radio('Model', ['MLP', 'GB', 'RF', 'LDA', 'SVC', 'LR', 'dummy'])
    col5.plotly_chart(roc_plot(results, model_in),use_container_width=True)
    col6.plotly_chart(roc_hist(results, model_in),use_container_width=True)
    col7.plotly_chart(roc_thresh(results, model_in),use_container_width=True)