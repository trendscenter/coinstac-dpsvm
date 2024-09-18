def return_uniques_and_counts(df):
    """Return unique-values of the categorical variables and their counts"""
    keys, count = dict(), dict()
    keys = (
        df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()
    )  # adding all columns
    count = {k: len(v) for k, v in keys.items()}

    return keys, count

def get_dummy_encoding_reference_dict(covar_keys):
    reference_dict = {}
    for k, v in covar_keys.items():
        reference_dict[k] = sorted(v)[0]
    return reference_dict