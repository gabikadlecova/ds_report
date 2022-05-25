# +
import numpy as np
import sklearn
import scipy.stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from functools import partial


def make_models(is_clf=True, n_jobs=1):
    res_models = {}
    prepro = {f'PCA_{k}': partial(PCA, n_components=k) for k in [2, 10, 20]}
    rf_cls = RandomForestClassifier if is_clf else RandomForestRegressor
    models = {f'RF_{k}': partial(rf_cls, n_estimators=k, n_jobs=n_jobs) for k in [50, 100, 500]}
    
    for kp, p in prepro.items():
        for km, m in models.items():
            res_models[f"{kp}-{km}"] = make_pipeline(p(), m())
    return res_models


def get_data_split(data, is_clf=True, random_state=42, val_size=0.1, test_size=0.2):
    xorig, yorig, xdata, ydata, baseline = data
    stratify = None if not is_clf else ydata
    val_size = val_size / (1 - test_size)

    xtrain, xtest, ytrain, ytest, idxtrain, idxtest = train_test_split(xdata, ydata, xorig.index, stratify=stratify,
                                                                       random_state=random_state, test_size=test_size)
    
    test = xtest, ytest, idxtest
    stratify = None if not is_clf else ytrain
    splits = train_test_split(xtrain, ytrain, idxtrain, random_state=random_state, test_size=val_size, stratify=stratify)
    
    return {'data': splits, 'test': test, 'baseline': baseline}


def get_baseline(idx, baseline):
    return baseline[idx]

    
def eval_preds(ytest, pred, is_clf=True):
    if is_clf:
        res = {
            'auc': roc_auc_score(ytest, pred),
            'accuracy': accuracy_score(ytest, pred)
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
    
    metrics = eval_preds(ytest, pred, is_clf=is_clf)
    return {'metrics': metrics, 'predictions': pred}


def eval_multiple_models(data, model_dict, is_clf=True):
    preds = {}
    df = []
    
    for key, model in model_dict.items():
        print("Evaluating ", key)
        res = eval_model_on_data(data, model, is_clf=is_clf)
        
        preds[key] = res['predictions']
        res['metrics']['name'] = key
        df.append(res['metrics'])
        
    df = pd.DataFrame(df)
    return preds, df


def eval_baseline(d, is_clf=True):
    val_idx, yval = d['data'][-1], d['data'][3]
    test_idx, ytest = d['test'][-1], d['test'][1]
    vdf = eval_preds(yval, get_baseline(val_idx, d['baseline']), is_clf=is_clf)
    tdf = eval_preds(ytest, get_baseline(test_idx, d['baseline']), is_clf=is_clf)
    return pd.DataFrame([vdf]), pd.DataFrame([tdf])
# -


