# +
import numpy as np
import sklearn
import scipy.stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from functools import partial


def make_models():
    res_models = {}
    prepro = {f'PCA_{k}': partial(PCA, n_components=k) for k in [2, 10, 20]}
    models = {f'RF_{k}': partial(RandomForestClassifier, n_estimators=k) for k in [50, 100, 500]}
    
    for kp, p in prepro.items():
        for km, m in models.items():
            res_models[f"{kp}-{km}"] = make_pipeline(p(), m())
    return res_models


def get_data_split(data, is_clf=True, random_state=42, val_size=0.1, test_size=0.2):
    xorig, yorig, xdata, ydata, baseline = data

    xtrain, xtest, ytrain, ytest, idxtrain, idxtest = train_test_split(xdata, ydata, xorig.index,
                                                                       random_state=random_state, test_size=test_size)
    
    test = xtest, ytest, idxtest
    splits = train_test_split(xtrain, ytrain, idxtrain, random_state=random_state, test_size=val_size)
    
    return {'data': splits, 'test': test, 'baseline': baseline}

    
def _eval_preds(model, ytest, pred, is_clf=True):
    if is_clf:
        res = {
            'auc': roc_auc_score(ytest, pred),
            'accuracy': accuracy_score(ytest, pred),
            'cm': confusion_matrix(ytest, pred, labels=model.classes_)
        }
    else:
        res = {
            'rmse': mean_squared_error(ytest, pred, squared=False),
            'pearson\'s_r': scipy.stats.pearsonr(ytest, pred)
        }
    return res


def eval_model_on_data(data, model, is_clf=True):
    xtrain, xtest, ytrain, ytest, idxtrain, idxtest = data['data']
    
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    
    metrics = _eval_preds(model, ytest, pred, is_clf=is_clf)
    return {'metrics': metrics, 'predictions': pred}


def eval_multiple_models(data, model_dict, is_clf=True):
    preds = {}
    df = []
    
    for key, model in model_dict.items():
        res = eval_model_on_data(data, model, is_clf=is_clf)
        
        preds[key] = res['predictions']
        res['metrics']['name'] = key
        df.append(res['metrics'])
        
    df = pd.DataFrame(df)
    return preds, df
# -


