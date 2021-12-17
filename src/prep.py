import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score, roc_curve, auc


def get_preds(df_in, thresh=0.5):
    df = df_in.copy()
    preds = []
    for idx, row in df_in.iterrows():
        if row['cls0'] >= thresh:
            preds.append(0)
        else:
            assert row['cls1'] > thresh
            preds.append(1)

    df['pred'] = preds

    return df

def get_score(df_in):
    df = get_preds(df_in)
    for m_name, m_data in df.groupby('model'):
        print(f"Model: {m_name}, Score: {balanced_accuracy_score(m_data['true'], m_data['pred'])}")
        
def scores(df):
    df_out = pd.DataFrame()
    for m_name, m_data in df.groupby('model'):
        trues = [int(i) for i in m_data['y_label_n']]
        preds = [int(i) for i in m_data['pred']]
        cls1 = [float(i) for i in m_data['cls1']]
        TN, FP, FN, TP = confusion_matrix(trues, preds).ravel()
        sens = TP/(TP+FN)
        spes = TN/(TN+FP)
        
        acc = balanced_accuracy_score(trues, preds)
        roc_auc = roc_auc_score(trues, cls1)
        f1 = f1_score(trues, preds)
        
        df_out = df_out.append({'Accuracy': acc, 'ROC_AUC': roc_auc, 'f1': f1, 'Sensitivity': sens,
                                'Specificity': spes, 'model': m_name}, ignore_index=True)
        
    return df_out

def get_scores(df):
    df_out = pd.DataFrame()
    for m_name, m_data in df.groupby('model'):
        trues = [int(i) for i in m_data['y_label_n']]
        preds = [int(i) for i in m_data['pred']]
        cls1 = [float(i) for i in m_data['cls1']]
        TN, FP, FN, TP = confusion_matrix(trues, preds).ravel()
        sens = TP/(TP+FN)
        spes = TN/(TN+FP)
        
        acc = balanced_accuracy_score(trues, preds)
        roc_auc = roc_auc_score(trues, cls1)
        f1 = f1_score(trues, preds)
        
        df_out = df_out.append({'Accuracy': acc, 'ROC_AUC': roc_auc, 'f1': f1, 'Sensitivity': sens,
                                'Specificity': spes, 'model': m_name}, ignore_index=True)
        
    return df_out

def get_z_table(df, cols_in):
    df_z = df.copy()
    df_z.index = df_z['merge_id']
    df_z_b = df_z[cols_in]
    for col in df_z_b:
        df_z_b[col] = stats.zscore(df_z_b[col], nan_policy='omit')

    df_z_table = df_z_b[(df_z_b > 3).any(1)]
    df_z_table = df_z_table[(df_z_table > 3)]
    df_z_table['disease'] = df_z['disease']

    return df_z_table

def scores(df):
    df_out = pd.DataFrame()
    for m_name, m_data in df.groupby('model'):
        trues = [int(i) for i in m_data['y_label_n']]
        preds = [int(i) for i in m_data['pred']]
        cls1 = [float(i) for i in m_data['cls1']]
        TN, FP, FN, TP = confusion_matrix(trues, preds).ravel()
        sens = TP/(TP+FN)
        spes = TN/(TN+FP)
        
        acc = balanced_accuracy_score(trues, preds)
        roc_auc = roc_auc_score(trues, cls1)
        f1 = f1_score(trues, preds)
        
        df_out = df_out.append({'Accuracy': acc, 'ROC_AUC': roc_auc, 'f1': f1, 'Sensitivity': sens,
                                'Specificity': spes, 'model': m_name}, ignore_index=True)
        
    return df_out