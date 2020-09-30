import os
import abc
from typing import List, Optional, Dict
from pandas import read_json
from interfaces import IModel, IModelRepo, IModelPreserver, IModelIdentifier
from datetime import datetime

DIR_MISS_HA_FEAT_MODELS = ""
F_ASSOC_RULES = "assoc_rules.json"


def convert_to_set(value):
    """"
    """
    try:
        return set(value)
    except ValueError:
        return value


class StoredModelNotFound(Exception):
    """
    """


class ModelBase(IModel, metaclass=abc.ABCMeta):
    """
    Базовый класс модели. Определяются getter и setter для атрибута model_id.
    """

    def __init__(self):
        self._model_id = None

    @property
    def model_id(self) -> IModelIdentifier:
        return self._model_id

    @model_id.setter
    def model_id(self, value: IModelIdentifier):
        self._model_id = value


class MissedHAFeaturesModelIdentifier(IModelIdentifier):
    """
    Идентификатор модели поиска ассоциированных признаков. Определяется инициализация атрибута model_type
    """
    def __init__(self, model_type: str):
        self._model_type = model_type

    @property
    def model_type(self) -> str:
        return self._model_type


class MissedHAFeaturesModelRepo(IModelRepo):

    def get_model(self, model_id: MissedHAFeaturesModelIdentifier) -> IModel:
        result = MissedHAFeaturesModel()

        if result is not None:
            result.model_id = model_id

        return result

    def get_preserver(self, model_id: MissedHAFeaturesModelIdentifier, **options) -> IModelPreserver:
        return MissedHAFeaturesModelPreserver(**options)


class MissedHAFeaturesModel(ModelBase):
    def __init__(self):
        super().__init__()
        self._assoc_rules = None  # data will be loaded from json-file as a Pandas Dataframe

    @property
    def assoc_rules(self):
        return self._assoc_rules

    @assoc_rules.setter
    def assoc_rules(self, value):
        self._assoc_rules = value

    def fit(self, X, y, **fit_options):
        raise NotImplementedError

    def predict(self, features, **apply_rules_options):

        # filter_series = self.assoc_rules['antecedents'] == set(features)

        t_start = datetime.now()
        aggregated_consequents = set()
        for i in range(self.assoc_rules.shape[0]):
            rule = self.assoc_rules.iloc[i]
            if rule['antecedents'].issubset(features):
                aggregated_consequents.update(rule['consequents'])
        aggregated_consequents.difference_update(features)
        t_finish = datetime.now()
        print((t_finish - t_start).seconds)

        return list(aggregated_consequents)


class MissedHAFeaturesModelPreserver(IModelPreserver):

    def __init__(self, storage_path: str):
        self._storage_path = storage_path

    @property
    def storage_path(self) -> str:
        return self._storage_path

    def load(self, model: MissedHAFeaturesModel) -> MissedHAFeaturesModel:
        if model.model_id.model_type == 'Apriori':
            model.assoc_rules = read_json(os.path.join(self.storage_path, 'apriori.json'))
        elif model.model_id.model_type == 'FPGrowth':
            model.assoc_rules = read_json(os.path.join(self.storage_path, 'fpgrowth.json'))
        model.assoc_rules.loc[:, 'antecedents'] = model.assoc_rules['antecedents'].apply(convert_to_set)
        model.assoc_rules.loc[:, 'consequents'] = model.assoc_rules['consequents'].apply(convert_to_set)

        return model

    def dump(self, model: MissedHAFeaturesModel) -> MissedHAFeaturesModel:
        raise NotImplementedError


class MissedHAFeaturesModelsCache(object):
    __shared_state = {
        "_repo_cached": None,
        "_model_cache": {}
    }

    def __init__(self):
        self.__dict__ = self.__shared_state

    @property
    def model_cache(self) -> Dict[str, Dict[str, MissedHAFeaturesModel]]:
        return self._model_cache

    @property
    def repo_cached(self) -> MissedHAFeaturesModelRepo:
        """ """
        if self._repo_cached is None:
            self._repo_cached = MissedHAFeaturesModelRepo()
        return self._repo_cached

    def get_model(self, model_type: str) -> Optional[IModel]:

        if model_type not in self.model_cache:
            self.model_cache[model_type] = {}

            model_id = MissedHAFeaturesModelIdentifier(model_type=model_type)

            model = self.repo_cached.get_model(model_id)
            if model is not None:
                preserver = self.repo_cached.get_preserver(
                    model_id=model_id,
                    # storage_path=get_env_value("PREDICT_HA_MODELS", None) or DIR_MISS_HA_FEAT_MODELS
                    storage_path=DIR_MISS_HA_FEAT_MODELS
                )
                try:
                    model = preserver.load(model)
                except StoredModelNotFound:
                    model = None
                    pass

            self.model_cache[model_type] = model

        return self.model_cache[model_type]


def missed_ha_features_by_features(features: List[str]):
    result = None

    model_cache = MissedHAFeaturesModelsCache()

    model = model_cache.get_model(model_type="FPGrowth")

    if model:
        result = model.predict(features)

    return result


# ________________________________________________________


if __name__ == '__main__':
    input_features = ["other_symptoms__sensorimotor_aphasia", "bad_habits__smoking", "kbs__localization__sternum"]
    res = missed_ha_features_by_features(input_features)
    print(res)
