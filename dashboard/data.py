import streamlit as st
import pandas as pd
from src.dash.data_plots import *
from src import db
from src.dash import prep as vizprep
from src.prep import marker_ttest, get_z_table
from src.meta import bloods

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


def data_page(dataset, iteration, cols_all):
    #st.sidebar.markdown("**disease**: Healhty Baseline vs T2D Baseline")
    #st.sidebar.markdown("**T2dInj**: Diabetes: (Placebo C / Placebo B) vs (Ketones C / Ketones B)")
    #st.sidebar.markdown("**KonInj**: Control: (Placebo C / Placebo B) vs (Ketones C / Ketones B)")
    
    if dataset == 't2dm':
        df_tmp = db.from_db(f't2dm/data/interim/df_z_corr.csv')
    else:
        df_tmp = db.from_db(f'{dataset}/data/cleaned/cleaned.csv')
    
    df = df_tmp[(df_tmp["iteration"] == iteration)]
    df = df.dropna(axis=1, how='all')
    blood = [i for i in df.columns if i in bloods]
    
    
    with st.expander("Z table"):
        radio_col, table_col = st.columns([1,8])
        z_thresh = radio_col.radio("z-threshold", [4,3,2,1])
        z_table = get_z_table(df, blood, z_thresh=z_thresh)
        table_col.write("z table")
        table_col.write(z_table)
        
        #st.plotly_chart(cnt_plt(df, ))
        z_filter = radio_col.checkbox("Filter with z threshold")
        if z_filter:
            df = df[~df['id'].isin(z_table.index.to_list())]

    with st.expander("Normality Table"):
        ttest_df = marker_ttest(df, blood)
        ttest_df = ttest_df.T.reset_index().rename(columns={'index': 'marker'})
        st.dataframe(ttest_df, height=450)

    
    col1, col2 = st.columns(2)

    
    table_col.write("z table")
    pvals_di = {}
    
    for idx, row in ttest_df.iterrows():
        pvals_di[row['marker']] = row['ttest']

    log_scale = col1.checkbox('Log Scale')
    col1.plotly_chart(boxplt(df, x_cols=blood, y_col='y_label',p_val=pvals_di, logy=log_scale),use_container_width=True)
    
    y_lab = col2.selectbox('Y label', list(df['y_label'].unique()) + ['all'])
    if y_lab != 'all':
        df_hm = df[(df["y_label"] == y_lab)]
    else:
        df_hm = df.copy()
    df_hm = df_hm.dropna(axis=1, how="all")

    col2.plotly_chart(plot_dendro(df_hm),use_container_width=True)

    

    col2.plotly_chart(heatmap(df_hm))

    cols_scatter = [i for i in cols_all if i in df_hm.columns]
    x_val = col2.selectbox('X value', cols_scatter)
    y_val = col2.selectbox('Y Value', cols_scatter)
    color = col2.selectbox('Color', cols_scatter)
    col1.plotly_chart(scatter(df_hm, x_val, y_val, color),use_container_width=True)




    with st.expander("Input Data"):
        st.write(df)
        csv = convert_df(df)

        st.download_button(
           "Press to Download",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )


        st.write("z-score tables")
