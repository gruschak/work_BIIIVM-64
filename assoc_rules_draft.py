import datetime
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

F_DATAS = r'data/dataset_2.csv'
F_RULES = r'data/assoc_rules.json'
F_LOG = r'log.txt'
F_RES = r'res.txt'
MIN_SUPP = 0.20
MIN_CONF = 0.60


def gen_binary(prefix, n):
    prefix = prefix or []
    if n == 0:
        yield prefix
    else:
        for digit in range(2):
            prefix.append(digit)
            yield from gen_binary(prefix, n - 1)
            prefix.pop()


def convert_to_set(value):
    try:
        return set(value)
    except ValueError:
        return value

"""
def gen_assoc_rules_from_csv(file_dataset=F_DATAS, file_rules=F_RULES, min_supp=0.2, min_conf=0.5):

    :param file_dataset: input file containing data
    :param file_rules: j-son file to generate
    :param min_supp:
    :param min_conf:
    :return: DataFrame

    Reads dataset from csv-file and writes association rules to json-file.


    ds = pd.read_csv(file_dataset, delimiter=',')
    del ds['Unnamed: 0']
    del ds['text']
    del ds['label']
    frequent_itemsets = fpgrowth(ds, min_support=min_supp, use_colnames=True)
    assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf, support_only=False)
    assoc_rules.to_json(file_rules)
"""


def get_assoc_rules(dataframe, min_supp=0.2, min_conf=0.5):
    """
    :param dataframe:
    :param min_supp:
    :param min_conf:
    :return:

    Generates association rules from data stored in the given DataFrame and returns the rules as a DataFrame
    """
    frequent_itemsets = fpgrowth(dataframe, min_support=min_supp, use_colnames=True)
    assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf, support_only=False)
    # Convert cells to type 'set':
    assoc_rules.loc[:, 'antecedents'] = assoc_rules['antecedents'].apply(convert_to_set)
    assoc_rules.loc[:, 'consequents'] = assoc_rules['consequents'].apply(convert_to_set)
    return assoc_rules


def load_data_from_csv(file_dataset=F_DATAS):

    dataset = pd.read_csv(file_dataset, delimiter=',')
    dataset.drop(labels=['Unnamed: 0', 'text', 'label'], axis=1, inplace=True)
    # Drop the lines that have 0 or 1 feature totally. Can't evaluate metrics with insufficient data.
    # this formula is complicated: dataset.drop((dataset[dataset.sum(axis=1) < 2]).index, inplace=True)
    # but is equivalent to:
    list_to_drop = dataset[dataset.sum(axis=1) < 2]
    dataset.drop(list_to_drop.index, inplace=True)

    return dataset


def load_assoc_rules_from_json(file_rules=F_RULES):
    assoc_rules = pd.read_json(file_rules)
    assoc_rules.loc[:, 'antecedents'] = assoc_rules['antecedents'].apply(convert_to_set)
    assoc_rules.loc[:, 'consequents'] = assoc_rules['consequents'].apply(convert_to_set)
    return assoc_rules


def print_as_vector(features_series):
    feature_counter = 0
    for feature_code in features_series:
        print(feature_code, end=' ')
        feature_counter += int(feature_code)
    print(f',F_SUM={feature_counter} ID={features_series.name}')


def print_as_matrix(dataset, first=0, last=0):

    if last > dataset.shape[0]-1:
        last = dataset.shape[0]-1
    record_number = first
    while record_number < last:
        line = dataset.iloc[record_number]
        feature_counter = 0
        for feature_code in line:
            print(feature_code, end=' ')
            feature_counter += int(feature_code)
        print(f',F_SUM={feature_counter}, ID={line.name}')
        record_number += 1


def data_split(dataset_in, line_id, dataset_out):

    # if line_id not in dataset_in.index:
    #    pass
    line_out = dataset_in.loc[line_id]  # dataset_in.index[line_id]
    dataset_out = dataset_in.drop(index=[0], inplace=False)
    return line_out


def features_from_series(row_as_series):
    """
    :param row_as_series: Pandas series
    :return: set
    """
    return set(row_as_series.index[row_as_series == 1].values)


def check_prediction(dframe_rules, set_features: set, the_feature: str, flog):
    """
    Провряет, попадает ли признак the_feature в правой части импликации ассоциативных правил,
    хранимых в строках датафрейма dframe_rules.

    :param dframe_rules: DataFrame of association rules
    :param set_features: Set of features for exploration
    :param the_feature: The goal feature
    :param flog: log file
    :return: True if the_feature appears in consequences of the rules
    that have antecedents as a subset of set_feature, otherwise False
    """
    print(f'searching for prediction of {the_feature} with antecedents \n{set_features}', file=flog)

    time_start = datetime.datetime.now()
    i_row = 0
    is_predicted = False  # flag that indicates that the_feature appeared in consequents

    while (i_row < dframe_rules.shape[0]) and not is_predicted:
        rule = dframe_rules.iloc[i_row]
        is_predicted = rule['antecedents'].issubset(set_features) and the_feature in rule['consequents']
        i_row += 1

    time_end = datetime.datetime.now()

    print(f'done in {int((time_end - time_start).total_seconds())} sec.', file=flog)
    print(f'{"OK" if is_predicted else "Not predicted"}', file=flog)

    return is_predicted


if __name__ == '__main__':
    """"
    with open(F_RES, 'w') as resfile:
        print(f'MIN-SUPPORT={MIN_SUPP}, MIN_CONFIDENCE={MIN_CONF}', file=resfile)

    with open(F_LOG, 'w') as logfile:
        print(f'MIN-SUPPORT={MIN_SUPP}, MIN_CONFIDENCE={MIN_CONF}\n{"*" * 40}', file=logfile)

        ds = load_data_from_csv()

        for line1_index in range(0, len(ds)):
            line1 = ds.loc[line1_index]  # line1 is a 'Series'
            print_as_vector(line1)

            # forming dataframe without line1
            ds1 = ds.drop(index=[line1_index], inplace=False)

            print('_____________________________', file=logfile)
            print(f'forming association rules with degraded dataframe ... ', end='', file=logfile)
            t0 = datetime.datetime.now()
            assoc_rules1 = get_assoc_rules(ds1, min_supp=MIN_SUPP, min_conf=MIN_CONF)     # type is DataFrame
            t1 = datetime.datetime.now()
            print(f'done in {int((t1 - t0).total_seconds())} sec.', file=logfile)

            features = features_from_series(line1)  # all features from line1 as a set
            print(f'features in the line ({len(features)} items):\n{features}', file=logfile)

            amount_features_predicted = 0
            i = 0
            for f in features:
                features1 = features.difference({f})
                print(f'[{line1_index}][{i}]**************************************', file=logfile)
                amount_features_predicted += check_prediction(assoc_rules1, features1, f, logfile)
                precision = round(amount_features_predicted / len(features), 2)
                i += 1
            print(f'for the line # {line1_index} predicted with precision {precision}', file=logfile)
            with open(F_RES, 'a') as resfile:
                print(f'"{line1_index}","{precision}","{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"', file=resfile)
    """
    ds = load_data_from_csv()
    print(ds.sum(axis=1)[ds.sum(axis=1) < 2])
    print(ds.shape[0])

