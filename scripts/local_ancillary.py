
import pandas as pd
import ujson as json

def add_site_covariates(args, X):
    """Add site covariates based on information gathered from all sites"""
    input_ = args["input"]
    all_sites = input_["covar_keys"]
    glob_uniq_ct = input_["global_unique_count"]

    reference_col_dict = input_["reference_columns"]

    all_sites = json.loads(all_sites)

    default_col_sortedval_dict = get_default_dummy_encoding_columns(X)

    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
            default_col_sortedval_dict.pop(key)
        else:
            default_col_sortedval_dict[key] = sorted(all_sites[key])[0]
            covar_dict = pd.get_dummies(all_sites[key], prefix=key, drop_first=False)
            X = merging_globals(args, X, covar_dict, all_sites, key)

    X = adjust_dummy_encoding_columns(X, reference_col_dict, default_col_sortedval_dict, glob_uniq_ct)
    X = X * 1
    X = X.apply( pd.to_numeric)

    #X = X.dropna(axis=0, how="any")

    return X


def merging_globals(args, X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix"""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X



def get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""

    default_col_sortedval_dict={}
    categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name]=sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict


def adjust_dummy_encoding_columns(df, ref_col_val_dict, data_def_col_val_dict, glob_uniq_ct):
    """ If a column is listed in reference_columns in the input for dummy encoding,
     then the values in this dict is used the reference column from the dataframe,
     otherwise the default sorted first value is used for a column."""

    for col_name in data_def_col_val_dict.keys():
        #Drop column only if it has two unique values (Dummy encoding). Otherwise use one-hot encoding
        if glob_uniq_ct[col_name] == 2 :
            ref_col_val =col_name +"_"+ ref_col_val_dict.get(col_name, data_def_col_val_dict[col_name])
            df.drop(ref_col_val, inplace=True, axis=1)

    return df


