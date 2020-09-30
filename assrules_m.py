import datetime
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from multiprocessing import Pool
from itertools import repeat

F_DATAS = r'data/dataset_2.csv'


def convert_to_set(value):
    try:
        return set(value)
    except ValueError:
        return value


def get_assoc_rules(dataframe, min_supp=0.2, min_conf=0.5):
    """
    :param dataframe:
    :param min_supp:
    :param min_conf:
    :return: DataFrame

    Generates association rules from data stored in the given DataFrame and returns the rules as a DataFrame
    """
    frequent_itemsets = fpgrowth(dataframe, min_support=min_supp, use_colnames=True)
    assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf, support_only=False)
    # Convert cells in these two columns to type 'set':
    for col_name in ['antecedents', 'consequents']:
        assoc_rules.loc[:, col_name] = assoc_rules[col_name].apply(convert_to_set)
    return assoc_rules


def load_data_from_csv(file_dataset=F_DATAS):
    dataset = pd.read_csv(file_dataset, delimiter=',')
    dataset.drop(labels=['Unnamed: 0', 'text', 'label'], axis=1, inplace=True)
    # Drop the lines that have 0 or 1 feature totally. Can't evaluate metrics with insufficient data.
    # This formula is complicated: dataset.drop((dataset[dataset.sum(axis=1) < 2]).index, inplace=True)
    # but is equivalent to:
    list_to_drop = dataset[dataset.sum(axis=1) < 2]
    dataset.drop(list_to_drop.index, inplace=True)

    return dataset


"""
def data_split(dataset_in, line_id, dataset_out):

    # if line_id not in dataset_in.index:
    #    pass
    line_out = dataset_in.loc[line_id]  # dataset_in.index[line_id]
    dataset_out = dataset_in.drop(index=[0], inplace=False)
    return line_out
"""


def features_from_series(row_as_series):
    """
    :param row_as_series: Pandas series
    :return: set
    """
    return set(row_as_series.index[row_as_series == 1].values)


def check_prediction(dframe_rules, set_features: set, the_feature: str):
    """
    Провряет, попадает ли признак the_feature в правую часть импликации ассоциативных правил,
    хранимых в строках датафрейма dframe_rules и имющих в левой части состав признаков, каждый из которых
    входит в набор set_features.

    :param dframe_rules: DataFrame of association rules
    :param set_features: Set of features for exploration
    :param the_feature: The goal feature
    :return: True if the_feature appears in consequences of the rules
    that have antecedents as a subset of set_feature, otherwise False
    """
    # searching for prediction of ~the_feature~ with antecedents ~set_features~

    i_row = 0
    is_predicted = False  # flag that indicates that the_feature appeared in consequents

    while (i_row < dframe_rules.shape[0]) and not is_predicted:
        rule = dframe_rules.iloc[i_row]
        is_predicted = rule['antecedents'].issubset(set_features) and (the_feature in rule['consequents'])
        i_row += 1

    return is_predicted


def calc_for_portion(dataset: object, min_supp: float, min_conf: float, i_tuple: object):
    precision_sum_for_current_portion = 0
    i_start, i_finish = i_tuple

    with open(file_res, 'a') as resfile:
        print(f'calc_for_portion(i_start={i_start}, i_finish={i_finish}, {datetime.datetime.now()}', file=resfile)

    for ds_row_index in range(i_start, i_finish):

        # split ds into ds1 and line1
        line1 = dataset.iloc[ds_row_index]  # line1 is a 'Series'
        # forming dataframe without line1
        ds1 = dataset.drop(index=line1.name, inplace=False)

        # forming association rules with degraded dataframe
        assoc_rules1 = get_assoc_rules(ds1, min_supp=min_supp, min_conf=min_conf)  # type is DataFrame

        features = features_from_series(line1)  # all features from line1 as a set

        amount_features_predicted = 0

        for f in features:
            features1 = features.difference({f})
            amount_features_predicted += check_prediction(assoc_rules1, features1, f)

        precision = round(amount_features_predicted / len(features), 2)
        precision_sum_for_current_portion += precision
        # Debug ...
        # print(precision)
        # ... Debug

    return precision_sum_for_current_portion


def gen_tuples_for_loops(range_len: int, limit: int) -> list:
    """ generate a list of tuples of range borders like [(0,...,100), (100,...,200), ..., ( ...,limit)]
    It is supposed that the tuples will be used in a loop "for"
    """
    ranges = [(n * range_len, (n + 1) * range_len) for n in range(limit // range_len)]
    if limit % range_len > 0:
        ranges.append((range_len * (limit // range_len), limit))
    return ranges


if __name__ == '__main__':

    ds = load_data_from_csv()

    for p_min_conf in [50]:
        for p_min_supp in [15]:

            if (p_min_supp, p_min_conf) not in [(20, 50), (30, 50), (30, 60)]:  # these cases had been processed already

                file_res = 's{:0>2}'.format(p_min_supp) + '_c{:0>2}'.format(p_min_conf) + '.csv'

                portions = gen_tuples_for_loops(100, ds.shape[0])

                with open(file_res, 'w') as resfile:
                    print(f'MIN_SUPPORT={p_min_supp}, MIN_CONFIDENCE={p_min_conf}', file=resfile)

                print(f'writing to {resfile}, {datetime.datetime.now()}')

                with Pool() as p:
                    partial_precisions = p.starmap(calc_for_portion,
                                                   zip(repeat(ds), repeat(p_min_supp / 100), repeat(p_min_conf / 100),
                                                       portions))

                total_precision = sum(part for part in partial_precisions)

                avg_precision = total_precision / ds.shape[0]
                with open(file_res, 'a') as resfile:
                    print(f'avg.prec.={avg_precision}, {datetime.datetime.now()}', file=resfile)
