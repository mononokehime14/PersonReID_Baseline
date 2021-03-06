# encoding: utf-8
from .cuhk03 import CUHK03
from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .msmt17_v2 import MSMT17_V2
from .msmt17_v1 import MSMT17_V1
from .ImageDataset import ImageDataset

__factory = {
    'CUHK03': CUHK03,
    'Market-1501': Market1501,
    'DukeMTMC-reID': DukeMTMC,
    'MSMT17_V2': MSMT17_V2,
    'MSMT17_V1': MSMT17_V1
}


def get_names():
    return __factory.keys()


def init_dataset(cfg,dataset_name, *args, **kwargs):
    if cfg.DATASETS.NAMES not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(dataset_name))
    return __factory[dataset_name](cfg,*args, **kwargs)
