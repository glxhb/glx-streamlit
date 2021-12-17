import streamlit as st
import pandas as pd
import plotly.express as px

from src import db, prep
from src.dash import prep as vizprep
from src.meta import bloods, datasets
from dashboard.data import data_page
from dashboard.ml_double import ml_double_page
from dashboard.ml_single import ml_single_page
from dashboard.t2dm import t2dm_page
from dashboard.shaps import shap_page

#from load_css import local_css
#local_css("style.css")

st.set_page_config(layout="wide", page_title="GLX Dashboard")

st.sidebar.markdown("**Dataset Options**")

extras = ['age', 'ethnicity', 'sex', 'sex_kat', 'education', 't2d_years', 'medi_t2d', 'medi_bp', 'medi_lipid', 'medi_anticoagulants', 
'medi_other', 'height', 'weight_b', 'bmi', 'bp_sys', 'bp_dia', 'hr', 'hb', 'alb', 'ca', 'k', 'crea', 'na', 'inr', 'alat', 'asat', 'bas_fos',
'bili', 'hba1c', 'tsh', 'glu', 'crp', 'chol', 'tg', 'hdl', 'ldl', 'dart']
metas = ['sv', 'co', 'tpr', 'resys', 'remap', 'redia', 'hr_ap']
hormon = ['cortisol', 'cpeptid', 'insulin', 'nefa', 'gh', 'adrenalin','noradrenalin', 'glukagon']
df_metas = ['merge_id', 'div_id', 'id','visit','time','disease','injection','temporal','div_id.1','sex_kat']

dataset = st.sidebar.selectbox('Dataset', ['t2dm', 't1dm'])
if dataset == 't2dm':
    df_tmp = db.from_db(f't2dm/data/interim/df_z_corr.csv')
else:
    df_tmp = db.from_db(f'{dataset}/data/processed/processed.csv')

page = st.sidebar.selectbox('Page', ["Data", 'ML - Single', 'ML - Double', 'Shap'])

iteration = st.sidebar.selectbox('Iteration', df_tmp['iteration'].unique())


df = df_tmp[(df_tmp["iteration"] == iteration)]
blood = [i for i in df.columns if i in bloods]

cols_all = [i for i in df.columns if i in extras + metas + hormon + blood]

st.sidebar.markdown("**Page Options**")

if page == "Data":
    data_page(dataset, iteration, cols_all)
        
elif page == "ML - Single":
    ml_single_page(dataset, iteration)
    
elif page == "ML - Double":
    ml_double_page(dataset, iteration)
    
elif page == 'allDiv':
    t2dm_page()
elif page == 'Shap':
    shap_page(dataset, iteration)