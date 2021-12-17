import streamlit as st
import pandas as pd
from src import db
from src.dash import prep as vizprep
from src.prep import marker_ttest, get_z_table
from src.meta import bloods
import plotly.express as px


def t2dm_page():
    df_tmp = db.from_db(f't2dm/data/interim/df_z_corr.csv')

    konInj = df_tmp.groupby('iteration').get_group('konInj')
    t2dInj = df_tmp.groupby('iteration').get_group('t2dInj')

    kon_labs = ["kon_" + i for i in konInj['y_label']]
    konInj['y_label'] = kon_labs


    t2d_labs = ["t2d_" + i for i in t2dInj['y_label']]
    t2dInj['y_label'] = t2d_labs

    allInj = pd.concat([konInj, t2dInj], ignore_index=True)
    blood_paper = ['biglycan', 'cd44', 'syn1', 'syn3', 'syn4', 'perlecan', 'gpc1', 'mimecan', 'ks', 'hs', 'cs']
    df_plt = allInj.melt(id_vars=['id', 'y_label'], value_vars=blood_paper)
    fig = px.box(df_plt, x="variable", y="value", color='y_label')
    
    st.plotly_chart(fig)