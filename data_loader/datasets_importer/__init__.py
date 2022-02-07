# encoding: utf-8
from .cuhk03 import CUHK03
from .market1501 import Market1501
from .full_market1501 import Full_Market1501
from .dukemtmc import DukeMTMC
from .msmt17 import MSMT17
from .blackreid import BlackReID
from .blackreidv2 import BlackReID_FullBlack
from .ntuoutdoor_black import NTUOutdoor_black
from .ntuoutdoorv2_black import NTUOutdoorv2_black
from .ntuoutdoorv2_white import NTUOutdoorv2_white
from .ImageDataset import ImageDataset

__factory = {
    'CUHK03': CUHK03,
    'Market-1501': Market1501,
    'Full-Market-1501': Full_Market1501,
    'DukeMTMC-reID': DukeMTMC,
    'MSMT17_V2': MSMT17,
    'Black-reID': BlackReID,
    'Black-reID-FullBlack': BlackReID_FullBlack,
    'NTUV1_black': NTUOutdoor_black,
    'NTUV2_black': NTUOutdoorv2_black,
    'NTUV2_white': NTUOutdoorv2_white,
}


def get_names():
    return __factory.keys()


def init_dataset(cfg,dataset_name, *args, **kwargs):
    if cfg.DATASETS.NAMES not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(dataset_name))
    return __factory[dataset_name](cfg,*args, **kwargs)
