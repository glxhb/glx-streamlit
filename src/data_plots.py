import pandas as pd
import numpy as np
from scipy.stats import kstest, mannwhitneyu
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
from collections import Counter
from sklearn.metrics import roc_curve, auc
from src.meta import bloods

customscale=[[0, "#ECECEC"],
           [1.0, "#00887A"]]

customscale2 =[[0, "#ECECEC"],
               [0.5, "#ECECEC"],
               [1.0, "#00887A"]]


customscale3 =[[0, "#ECECEC"],
               [0.5, "#00887A"],
               [1.0, "#00887A"]]

custom_div =[[0, "#77A6F7"],
               [0.5, "#E3E2DF"],
               [1.0, "#00887A"]]

colors = ["#77A6F7", "#00887A"]

many_colors = ["#f94144","#f3722c","#f8961e","#f9844a","#f9c74f","#90be6d","#43aa8b","#4d908e",
               "#577590","#277da1"]

def cnt_plt(df, h, margins=dict(l=0, r=0, t=0, b=0)):
    fig = go.Figure(data=[go.Bar(
                y=list(Counter(df['y_label']).keys()), x=list(Counter(df['y_label']).values()),
                text=list(Counter(df['y_label']).values()),
                textposition='auto', orientation='h', marker_color=colors
            )])

    fig.update_layout(boxmode='group', xaxis_tickangle=0, plot_bgcolor='rgba(0, 0, 0, 0)',
                      margin=margins)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
    fig.update_xaxes(showgrid=False)

    return fig


def boxplt(df,x_cols, y_col, p_val=None, logy=False, margins=dict(l=0, r=0, t=80, b=0)):
    def add_pvalue_annotation(df, marker, y_range, symbol='', p_val=None, add_lines=False):
        """
        arguments:
        days --- a list of two different days e.g. ['Thur','Sat']
        y_range --- a list of y_range in the form [y_min, y_max] in paper units
        """
        cols = df['y_label'].unique().tolist()
        markers = df['variable'].unique().tolist()
        
        if p_val:
            pvalue = p_val
        else:
            t1 = df[df[y_col]==cols[0]]
            t2 = df[df[y_col]==cols[1]]

            t1 = t1[t1['variable'] == marker]['value'].tolist()
            t2 = t2[t2['variable'] == marker]['value'].tolist()

            pvalue = stats.kruskal(t1, t2, nan_policy='omit')[1]
        if pvalue >= 0.05:
            symbol = 'ns'
        if pvalue < 0.05:
            symbol = '*'
        if pvalue < 0.01:
            symbol = '**'
        if pvalue < 0.001:
            symbol = '***'
        if pvalue < 0.0001:
            symbol = '****'

        if add_lines:
            fig.add_shape(type="line",
                xref="x", yref="paper",
                x0=marker[0], y0=y_range[0], x1=marker[0], y1=y_range[1],
                line=dict(
                    color="black",
                    width=2,
                )
            )
            fig.add_shape(type="line",
                xref="x", yref="paper",
                x0=marker[0], y0=y_range[1], x1=marker[1], y1=y_range[1],
                line=dict(
                    color="black",
                    width=2,
                )
            )
            fig.add_shape(type="line",
                xref="x", yref="paper",
                x0=marker[1], y0=y_range[1], x1=marker[1], y1=y_range[0],
                line=dict(
                    color="black",
                    width=2,
                )
            )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        bar_xcoord_map = {x: idx for idx, x in enumerate(markers)}
        fig.add_annotation(dict(font=dict(color="black",size=14),
            x=(bar_xcoord_map[marker]),
            y=y_range[1]*0.97,
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x",
            yref="paper", 
            hovertext=str(pvalue)
        ))
    
    df_test = df.melt(id_vars=[y_col], value_vars=x_cols)
    fig=go.Figure()
    for i, clas in enumerate(df_test[y_col].unique()):
        df_plot=df_test[df_test[y_col]==clas]
        #print(df_plot.head())
        fig.add_trace(go.Box(x=df_plot['variable'], y=df_plot['value'],
                 line=dict(color=colors[i]),
                 name=clas))

    fig.update_layout(boxmode='group', xaxis_tickangle=0, plot_bgcolor='rgba(0, 0, 0, 0)',
                      margin=margins)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
    fig.update_xaxes(showgrid=False)
    
    if logy:
        fig.update_yaxes(type="log")

    markers = df_test['variable'].unique().tolist()
    for m in markers:
        if p_val:
            p = p_val[m]
            add_pvalue_annotation(df_test, m,[1.01,1.05], p_val = p)
        else:
            add_pvalue_annotation(df_test, m,[1.01,1.2])

    return fig

def heatmap(df,margins=dict(l=0, r=0, t=0, b=0)):
    df = df.corr()
    cols = df.columns.values.tolist()
    z = df.to_numpy()
    z = np.triu(z)
    with np.nditer(z, op_flags=['readwrite']) as it:
        for x in it:
            if x == 0:
                x[...] = None
    hm_df = pd.DataFrame(data=z, columns=cols)

    z = hm_df.to_numpy()
    cols = hm_df.columns.values.tolist()

    l_0, l_1 = 5, 5
    fig = go.Figure(go.Heatmap(z=z, zmin=-1, zmax=1, x=cols, y=cols, colorscale=custom_div,
                            hoverongaps=False))
    fig.update_traces(showscale=False)
    fig.update_yaxes(side='right')
    fig.update_layout(plot_bgcolor='white', xaxis=dict(scaleanchor='y', constrain='domain'), margin=margins,
                      legend=dict(x= 1, y= 0.5))
    
    return fig

def scatter(df,x_val, y_val, color, margins=dict(l=0, r=0, t=0, b=0)):
    fig = px.scatter(df, x=x_val, y=y_val, color=color, 
                trendline="ols", trendline_scope="overall", trendline_color_override="black")
    fig.update_layout(plot_bgcolor='white', xaxis=dict(scaleanchor='y', constrain='domain'), margin=margins,
                      legend=dict(x= 1, y= 0.5))
    
    return fig

def plot_dendro(df):
    blood = [i for i in df.columns if i in bloods]
    test = df[blood].dropna(how="any", axis=0)
    fig = ff.create_dendrogram(test.transpose(), labels=test.columns.to_list())
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', height=350)

    return fig