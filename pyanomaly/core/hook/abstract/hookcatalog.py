'''
Refer to the detectron2's DatasetCatalog
'''
from typing import List
class HookCatalog(object):
    """
    A catalog that stores information about the datasets and how to obtain them.
    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.
    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.
    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    _REGISTERED = {}

    @staticmethod
    def register(name, func):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in HookCatalog._REGISTERED, "Dataset '{}' is already registered!".format(
            name
        )
        HookCatalog._REGISTERED[name] = func

    @staticmethod
    def get(name, hook_name):
        """
        Call the registered function and return its results.
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        Returns:
            list[dict]: dataset annotations.0
        """
        try:
            f = HookCatalog._REGISTERED[name]
        except KeyError:
            raise KeyError(
                "Hook '{}' is not registered! Available Hooks are: {}".format(
                    name, ", ".join(HookCatalog._REGISTERED.keys())
                )
            )
        return f(hook_name)

    @staticmethod
    def list() -> List[str]:
        """
        List all registered datasets.
        Returns:
            list[str]
        """
        return list(HookCatalog._REGISTERED.keys())

    @staticmethod
    def clear():
        """
        Remove all registered dataset.
        """
        HookCatalog._REGISTERED.clear()