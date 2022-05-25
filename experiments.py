import sklearn
import numpy as np
import pandas as pd

from spark_utils import get_dataset
from sklearn.compose import ColumnTransformer


def _parse_column(c):
    c_data = c.split('/')
    
    year = c_data[1]
    if 'yearly' in c:
        return int(year), None
    
    mon = c_data[2]
    return int(year), int(mon)


def _drop_conditions(c, last_year=2019, last_month=None):
    c = _parse_column(c)
    if last_month is None and c[0] >= last_year:
        return True
    
    if last_month is None:
        return False
    
    return c[0] >= last_year and c[1] >= last_month


def x_y_split(df, last_year=2019, last_month=None):
    drop_cols = [c for c in df.columns if '201' in c and _drop_conditions(c, last_year=last_year, last_month=last_month)]
    
    for c in drop_cols:
        c = _parse_column(c)
        
        if c[0] != last_year:
            continue
        if last_month is None or c[1] != last_month:
            continue

        y_col = c
        
    x_data = df.drop(columns=drop_cols)
    y_data = df[drop_cols]
    
    return x_data, y_data


def cut_age_groups(pdf, bins='equal'):
    if bins == 'equal':
        bins = 10
    elif bins == 'groups':
        bins = [5, 18, 25, 35, 50, 65, 75, 110]
    
    pdf['age'] = pd.cut(pdf['age'], bins)


def get_y_names(last_year, last_month, n_months=None):
    if last_month is None:
        return f'Rat:mean/{last_year}//yearly', f'Rat:mean/{last_year - 1}//yearly'
    
    prev_year = last_year if last_month > 1 else (last_year - 1)
    prev_month = (last_month - n_months) if last_month > 1 else 12
    return f'Rat:mean/{last_year}/{last_month}/{n_months}', f'Rat:mean/{prev_year}/{prev_month}/{n_months}'


def create_dataset(df, n_months=12, last_year=2019, last_month=None, bins='groups', y_discrete=True, diff=10, feature_cols=None,
            continuous_diff=False):

    if feature_cols is None:
        feature_cols = [
            ('Flag', 'str'), ('Gms', 'sum'), ('Gms', 'mean'), ('Rat', 'mean'), ('tit', 'max'), ('wtit', 'max')
        ]

    agg_df = get_dataset(df, feature_cols, n_months=n_months)
    agg_df = agg_df.toPandas()

    # filling newly created nans
    for c in agg_df.columns:
        if 'Flag' in c:
            agg_df[c] = agg_df[c].fillna('i')
        else:
            agg_df[c] = agg_df[c].fillna(0)
            
    # split the dataframe
    xorig, yorig = x_y_split(agg_df, last_year=last_year, last_month=last_month)
    cut_age_groups(xorig, bins=bins)
    
    # get dataset for fit
    xdata = xorig.drop(columns=['ID Number'])
    onehot_columns = ['age', 'Sex', 'K'] + [c for c in xdata.columns if 'Flag' in c]

    ct = sklearn.compose.ColumnTransformer([
        ('onehot', sklearn.preprocessing.OneHotEncoder(drop='if_binary'), onehot_columns)], remainder='passthrough')
    xdata = ct.fit_transform(xdata)
    
    # construct y column
    curr, prev = get_y_names(last_year, last_month, n_months)

    if y_discrete:
        ydata = yorig[curr] > (xorig[prev] + diff)
        ydata = ydata.astype(int)
        
        baseline = np.zeros(ydata.shape)
    else:
        ydata = yorig[curr] if not continuous_diff else (yorig[curr] - xorig[prev])
        baseline = xorig[prev] if not continuous_diff else np.zeros(yorig[curr].shape)
        
    return xorig, yorig, xdata, ydata, baseline
