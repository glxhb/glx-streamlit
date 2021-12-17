import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap

cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
}  # #1E88E5 -> #ff0052
red_blue = LinearSegmentedColormap('RedBlue', cdict1)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

red_blue = matplotlib_to_plotly(red_blue, 255)

def shap_scatter(shap_values, X, colors=red_blue, color_type='mean', max_display=20, row_height=0.4):
    feature_names = X.columns.to_list()
    features = X.values
    
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    
    
    fig = go.Figure()
    annotations = []

    for pos, i in enumerate(feature_order):
        shaps = shap_values[:, i]
        values = features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True

        values = np.array(values, dtype=np.float64)  # make sure this can be numeric

        N = len(shaps)

        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))
        
        assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
        nan_mask = np.isnan(values)

        # trim the color range, but prevent the color range from collapsing
        if color_type == 'shap':
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            color_data = values[np.invert(nan_mask)]
            
        else:
            vmin = shap_values.min()
            vmax = shap_values.max()
            color_data = shaps
            



        fig.add_trace(
          go.Scatter(
            x = shaps[np.invert(nan_mask)],
            y = pos + ys[np.invert(nan_mask)],
            mode='markers',
            marker=dict(
                cmax=vmax,
                cmin=vmin,
                color=color_data,
                colorbar=dict(
                    title="",
                    showticklabels=False,
                ),
                colorscale=colors
            ),
            showlegend=False
          )
        )
        


        annotations.append(dict(xref='paper', x=-0.05, y=pos,
                                      xanchor='right', yanchor='middle',
                                      text=feature_names[i],
                                      font=dict(family='Arial',
                                                size=12),
                                      showarrow=False))

        fig.add_hline(y=pos, opacity=0.2)

    fig.update_layout(plot_bgcolor='white', annotations=annotations, width=500)

    fig.add_vline(x=0)
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})

    return fig

def shap_feature_importances(shap_values, X):
    df = pd.DataFrame(columns=X.columns, data=shap_values)
    fi_di = {}

    for col in df:
        fi_di[col] = df[col].abs().mean()

    df_fi = pd.DataFrame(fi_di, index=['shap']).T.sort_values(by='shap', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_fi['shap'].to_list(),
        y=df_fi.index.to_list(),
        name='Primary Product',
        marker_color='#ff0051',
        orientation='h'
    ))
    fig.update_layout(plot_bgcolor='white')
    
    return fig