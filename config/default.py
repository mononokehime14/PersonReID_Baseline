from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'ResNet50'
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# Value of colorjitter brightness
_C.INPUT.BRIGHTNESS = 0.0
# Value of colorjitter contrast
_C.INPUT.CONTRAST = 0.0
# Value of colorjitter saturation
_C.INPUT.SATURATION = 0.0
# Value of colorjitter hue
_C.INPUT.HUE = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('Market1501')
# Setup storage directory for dataset
_C.DATASETS.STORE_DIR = ('/home/GTA')
#Choose colours for NTU_Outdoor_V2
_C.DATASETS.TEST_COLOUR = 'all'

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
_C.FILTERS = CN()

_C.FILTERS.LONGTERM = False
# Choose between long-term and short-term Re-ID
_C.FILTERS.LOCATION = []
# Choose from the 10 different locations, default is all
_C.FILTERS.WEATHER = []
# Choose different types of weather, default is all
_C.FILTERS.TIME = []
# Choose from different times, default is all. Pass the value as a string.
_C.FILTERS.ANIMATION = []
# Choose either walk or run animation
_C.FILTERS.CAM_HEIGHT = []
# Choose from person, surveillance and drone view, default is all
_C.FILTERS.PID_RANGE = [] #give a range i.e. [1, 500] gives the first 500 pid
# Choose the range of PIDs to be tested. Default is all
_C.FILTERS.YAW_ANGLE = [] #give a range i.e. [1,4] gives the first 4 yaw angles, max = 8
# Choose from the 8 different yaw angles
_C.FILTERS.QUERY_ANGLE = [1]
#Choose the angle for query, e.g. if you do query on angles 1 and 5, the rest of the images will be in the gallery.

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.STEP = 40

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.LOAD_EPOCH = 120


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.DEVICE = "cuda:0"
_C.OUTPUT_DIR = ""
_C.RE_RANKING = False