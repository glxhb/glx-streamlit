import pandas as pd
import numpy as np
from scipy.stats import kstest, mannwhitneyu
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import Counter
from sklearn.metrics import roc_curve, auc

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




def scores_plt(score_in, margins=dict(l=0, r=0, t=0, b=0)):
    score = score_in.copy()
    score['model'] = pd.Categorical(score['model'], categories=['MLP', 'GB', 'RF', 'LDA', 'SVC', 'LR', 'dummy'], ordered=True)
    score.sort_values('model', inplace=True)
    models = score.pop('model').to_list()
    fig = ff.create_annotated_heatmap(score.round(decimals=3).to_numpy(), x=score.columns.to_list(), y=models, colorscale=customscale, zmin=0.5, zmax=1)
    fig['layout']['xaxis']['side'] = 'bottom'
    fig.update_layout(boxmode='group', xaxis_tickangle=0, plot_bgcolor='rgba(0, 0, 0, 0)')

    return fig


def rfe_plt(df_sfs, margins=dict(l=0, r=0, t=0, b=0)):
    rfecv_df = df_sfs[df_sfs['model'] != 'dummy']
    try:
        rfe_plt = px.line(rfecv_df, x='features', y='avg_score', color='model', hover_name='feature_names')
    except ValueError:
        rfe_plt = px.line(rfecv_df, x='features', y='avg_score', color='model')
    rfe_plt.update_yaxes(range=[0, 1])
    rfe_plt.update_layout(title_x=0.5, xaxis_tickangle=0, plot_bgcolor='rgba(0, 0, 0, 0)')
    rfe_plt.update_yaxes(showgrid=True, gridcolor='lightgrey')
    rfe_plt.update_xaxes(showgrid=False)

    return rfe_plt

def roc_hist(results_in, model):
    results = results_in[results_in['model'] == model]
    cls1 = [float(i) for i in results['cls1']]
    trues = [int(i) for i in results['y_label_n']]
    # The histogram of scores compared to true labels
    fig_hist = px.histogram(
        x=cls1, color=trues, nbins=50, color_discrete_sequence=colors,
        labels=dict(color='True Labels', x='Score')
    )

    fig_hist.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    #fig_hist.update_xaxes(range=[0, 1], constrain='domain')
    #fig_hist.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')

    return fig_hist

def roc_thresh(results_in, model):
    results = results_in[results_in['model'] == model]
    cls1 = [float(i) for i in results['cls1']]
    trues = [int(i) for i in results['y_label_n']]
    fpr, tpr, thresholds = roc_curve(trues, cls1, pos_label=1)
    # Evaluating model performance at various thresholds
    df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    }, index=thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"

    fig_thresh = px.line(df, title='TPR and FPR at every threshold', color_discrete_sequence=colors)

    #fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    #fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    fig_thresh.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')

    return fig_thresh

def roc_plot(results_in, model):
    results = results_in[results_in['model'] == model]
    cls1 = [float(i) for i in results['cls1']]
    trues = [int(i) for i in results['y_label_n']]
    fpr, tpr, thresholds = roc_curve(trues, cls1, pos_label=1)
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='FPR', y='TPR'), color_discrete_sequence=colors
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    #fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
    #fig_roc.update_xaxes(constrain='domain')
    #fig_roc.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    fig_roc.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')


    return fig_roc

def scores_plt_double(df_scores_inner, df_scores_outer, margins=dict(l=0, r=0, t=20, b=0)): 
    scores_outer = df_scores_outer[df_scores_outer['model'] != 'dummy']
    scores_inner = df_scores_inner[df_scores_inner['model'] != 'dummy']
    scorers = ["Accuracy", "ROC_AUC", "f1", "Sensitivity", "Specificity"]
    fig = make_subplots(1, 5, shared_yaxes=True, column_titles=scorers)
    for idx, col in enumerate(scorers):
        if idx == 0:
            showleg = True
        else:
            showleg = False
        fig.add_trace(go.Bar(x=scores_inner['model'].to_list(), y=scores_inner[col].to_list(),name='inner', marker_color='#1e88e5', showlegend=showleg), 1, idx+1)
        fig.add_trace(go.Bar(x=scores_outer['model'].to_list(), y=scores_outer[col].to_list(),name='outer', marker_color='#ff0d57', showlegend=showleg), 1, idx+1)


    fig.update_xaxes(matches='x')
    fig.update_yaxes(range=[0, 1.2], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0], showgrid=True, gridwidth=1, gridcolor='#d4d9df')
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    
    return fig