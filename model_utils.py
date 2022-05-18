# +
import pandas as pd


def _parse_column(c):
    c_data = c.split('/')
    
    year = c_data[1]
    if 'yearly' in c:
        return year, None
    
    mon = c[2]
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
