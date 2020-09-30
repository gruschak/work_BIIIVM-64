import abc


class IModelIdentifier(metaclass=abc.ABCMeta):
    """
    def __init__(self):
        self.model_type = None

    """


class IModel(metaclass=abc.ABCMeta):
    """ Интерфейс модели """

    @property
    @abc.abstractmethod
    def model_id(self) -> IModelIdentifier:
        """ Возвращает идентификтор модели """

    @model_id.setter
    @abc.abstractmethod
    def model_id(self, value: IModelIdentifier):
        """ """

    @abc.abstractmethod
    def fit(self, X, y, **fit_options):
        """ """

    @abc.abstractmethod
    def predict(self, X, **predict_options):
        """ """


class IModelPreserver(metaclass=abc.ABCMeta):
    """ Интерфейс хранителя моделей """

    @abc.abstractmethod
    def load(self, model: IModel) -> IModel:
        """ Загружает модель """

    @abc.abstractmethod
    def dump(self, model: IModel) -> IModel:
        """ Сохранение модели """


class IModelRepo(metaclass=abc.ABCMeta):
    """ Репозиторий моделей """

    @abc.abstractmethod
    def get_model(self, model_id: IModelIdentifier) -> IModel:
        """ Возвращает объект модели """

    @abc.abstractmethod
    def get_preserver(self, model_id: IModelIdentifier, **options) -> IModelPreserver:
        """ Возвращает preserver для модели (указанной инстансом или id) """


