import sklearn

from spark_utils import get_dataset
from model_utils import x_y_split    
from model_utils import cut_age_groups
from sklearn.compose import ColumnTransformer


def get_y_names(last_year, last_month, n_months=None):
    if last_month is None:
        return f'Rat:mean/{last_year}//yearly', f'Rat:mean/{last_year - 1}//yearly'
    
    prev_year = last_year if last_month > 1 else (last_year - 1)
    prev_month = (last_month - n_months) if last_month > 1 else 12
    return f'Rat:mean/{last_year}/{last_month}/{n_months}', f'Rat:mean/{prev_year}/{prev_month}/{n_months}'


def dataset(df, n_months=12, last_year='2019', last_month=None, bins='groups', y_discrete=True, diff=10):
    feature_cols = [('Flag', 'str'), ('Gms', 'sum'), ('Gms', 'mean'), ('Rat', 'mean'), ('tit', 'max'), ('wtit', 'max')]

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
    
    if y_discrete:
        curr, prev = get_y_names(last_year, last_month, n_months)
        ydata = yorig[curr] > (xorig[prev] + diff)
        ydata = ydata.astype(int)
    else:
        # todo continuous case