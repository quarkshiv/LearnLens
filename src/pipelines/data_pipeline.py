from src.common.data_loader import load_all_data
from src.common.merge_engine import MergeEngine


class DataPipeline:
    """
    Responsible for loading raw datasets
    and preparing merged datasets.
    """

    def run(self):

        datasets = load_all_data()

        engine = MergeEngine(datasets)

        registration = engine.merge_registration()

        return datasets, registration