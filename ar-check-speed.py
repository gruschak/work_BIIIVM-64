import datetime
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

F_DATAS = r'data/dataset_2.csv'


def convert_to_set(value):
    try:
        return set(value)
    except ValueError:
        return value


def gen_assoc_rules(dataframe, min_supp=0.2, min_conf=0.5):
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


if __name__ == '__main__':

    ds = load_data_from_csv()

    p_min_conf = 50
    p_min_supp = 15

    t0 = datetime.datetime.now()
    print(f'started at {t0.isoformat()}')
    assoc_rules = gen_assoc_rules(ds, min_supp=p_min_supp, min_conf=p_min_conf)
    t1 = datetime.datetime.now()
    print(f'done in {(t1 - t0).seconds} sec.')

    print(assoc_rules.head(10))

