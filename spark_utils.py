# +
from pyspark.sql.functions import first


def get_mean(df):
    return df.mapValues(lambda v: (v, 1)) \
             .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
             .mapValues(lambda v: v[0] / v[1]) \


def get_sum(df):
    return df.reduceByKey(lambda a, b: a + b)


def get_max(df):
    return df.reduceByKey(lambda a, b: (a if a > b else b))


def get_string(df):    
    return df.reduceByKey(lambda a, b: (a if len(a) > len(b) else b))


def reduce_by_months(df, what, agg_func, agg_func_name='', n_months=1):
    idmap = {c: i for i, c in enumerate(df.columns)}
    name = f"{what}:{agg_func_name}"
        
    if n_months > 12:
        raise ValueError("Invalid months: ", n_months)
    elif n_months == 12:
        key_func = lambda x: (x[idmap['ID Number']], x[idmap['Year']])
        out_key_func = lambda x: (x[0][0], f"{name}/{x[0][1]}//yearly", x[1])
    else:
        key_func = lambda x: (x[idmap['ID Number']], x[idmap['Year']], (x[idmap['Mon']] - 1) // n_months)
        out_key_func = lambda x: (x[0][0], f"{name}/{x[0][1]}/{int(x[0][2] * n_months)}/{n_months}", x[1])
    
    if isinstance(what, list):
        res = df.rdd.map(lambda x: (key_func(x), tuple([x[idmap[w]] for w in what])))
    else:
        res = df.rdd.map(lambda x: (key_func(x), x[idmap[what]]))

    res = agg_func(res)
    res = res.map(out_key_func) \
             .toDF(['ID Number', 'Colname', name])
    
    return res


name_to_func = {
    'sum': get_sum,
    'mean': get_mean,
    'str': get_string,
    'max': get_max
}


def get_dataset(df, select_data, n_months=12):
    base_df = df.drop_duplicates(["ID Number"]).select('ID Number', 'Sex', 'K', 'age')
    
    for what, func in select_data:
        red = reduce_by_months(df, what, name_to_func[func], agg_func_name=func, n_months=n_months) \
            .groupBy('ID Number') \
            .pivot('Colname')
        
        red = red.agg(first(f"{what}:{func}")) if func == 'str' else red.max(f'{what}:{func}')
        base_df = base_df.join(red, "ID Number")
        
        red.unpersist()
        del red
        
    return base_df


def preprocess_data(data):
    # drop gender changes
    dupl_gender = data[['ID Number', 'Sex', 'Flag']].groupby(['ID Number', 'Sex']).count().reset_index()
    dupl_gender = dupl_gender[dupl_gender.duplicated('ID Number')]
    
    uniq_ids = data['ID Number'].drop_duplicates()
    uniq_ids = uniq_ids[uniq_ids.isin(dupl_gender['ID Number'])]

    drop_ids = data.index[data.index.isin(uniq_ids.index)]
    data.drop(index=data.index[drop_ids], inplace=True)
    
    # replace 0 birthyear if filled out later
    proper_bdays = data[['ID Number', 'B-day']].groupby('ID Number').max().to_dict()['B-day']
    data['B-day'] = [(d if (2019 - d) > 5 else proper_bdays[i]) for i, d in zip(data['ID Number'], data['B-day'])]
    
    # drop anything that couldn't be fixed
    data['age'] = (2019 - data['B-day']).astype('int')
    data = data[(data['age'] > 5) & (data['age'] < 110)].copy()
    
    # unify flag columns
    data.loc[data['Flag'] == 'wi', 'Flag'] = 'i'
    data.loc[data['Flag'] == 'w', 'Flag'] = ''
    
    string_cols = ['Name', 'Tit', 'WTit', 'OTit', 'FOA', 'Flag']
    data[string_cols] = data[string_cols].fillna('')
    
    # process titles
    female_tit = ["None", "WCM", "WFM", "WIM", "WGM"]
    tit = ["None", "CM", "FM", "IM", "GM"]

    female_tit = {t: i for i, t in enumerate(female_tit)}
    tit = {t: i for i, t in enumerate(tit)}

    data['tit'] = [(0 if t not in tit else tit[t]) for t in data['Tit']]
    data['wtit'] = [(0 if t not in female_tit else female_tit[t]) for t in data['Tit']]
    
    return data
