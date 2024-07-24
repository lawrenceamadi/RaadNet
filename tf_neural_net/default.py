'''
    Setup experiment configurations
'''

from __future__ import absolute_import, division, print_function
##print('\nDefault Script Called\n')
import os
import sys
import time
import argparse

from yacs.config import CfgNode as CN
from pathlib import Path

sys.path.append('../')
from tf_neural_net.commons import Logger, NETWORK_PREFIX


def default_experiment_configuration():
    # Default configurations
    # The default configurations is first changed by the config file
    # and finally by the specified runtime arguments

    _C = CN()

    _C.HOME_DIR = ''
    _C.DATA_DIR = ''
    _C.MODEL_DIR = '' # Path to home directory containing models
    _C.MODEL_PATH = ''
    _C.EXP_DIR = ''  # experiments' model directory, descendant of _C.MODEL_PATH
    _C.GPUS = (0,)

    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.GROUPS = ['Back']  # list of groups
    _C.MODEL.CONV_DIM = 3 # dimension of convolutions, 2->2d, 3->3d
    _C.MODEL.SUBNET_TYPE = 'combined' # [body_zones, body_groups, combined]
    _C.MODEL.IMAGES_PER_SEQ_GRP = 3 # Number of images per sequence group passed at a time
    _C.MODEL.TRANS_LEARN_PARAM_INIT = False # initialize with weights of pretrained model
    _C.MODEL.NETWORK_ARCHITECTURE = 'grp_model_v3_td' # model's network architecture
    _C.MODEL.PRETRAINED = '' # 'pretrained/Back_2019-11-08-16-54_opt_avg_f1.h5'
    _C.MODEL.IMAGE_SIZE = [80, 80, 1]  # [hgt, wdt, channels]
    _C.MODEL.REGION_DIM = [160, 160] # [wdt, hgt]
    _C.MODEL.ROI_POOL_DIM = [5, 5] # pooled roi fixed dimension for rois of all sizes
    _C.MODEL.FORCE_SQUARE = True # make sure a square image is passed to model
    _C.MODEL.DROPOUT_FC = 0.0
    _C.MODEL.DROPOUT_SP = 0.0
    _C.MODEL.WGT_REG_RESC = 0.01
    _C.MODEL.WGT_REG_DENSE = 0.01
    _C.MODEL.ENABLE_DENOISER = False # FB non-local max denoiser layer
    _C.MODEL.BATCH_NORMALIZE = False
    _C.MODEL.SEPARATE_FC_LAYERS = True
    _C.MODEL.EXTRA = CN(new_allowed=True)

    # Loss params
    _C.LOSS = CN()
    _C.LOSS.NUM_CLASSES = 2
    _C.LOSS.NET_OUTPUTS_ID = ['t', 'p'] # threat vs. body-part
    _C.LOSS.NET_OUTPUTS_LOSS = ['binary_crossentropy', 'categorical_crossentropy']
    _C.LOSS.NET_OUTPUTS_LOSS_COEF = [1.0, 0.2]
    _C.LOSS.DEFAULT_LOSS_BRANCH_WGT = [1.0, 0.0] # influence loss opt. on either output when not using threat bbox anot.
    _C.LOSS.NET_OUTPUTS_FCBLOCK_ACT = ['sigmoid', 'linear']
    _C.LOSS.NET_OUTPUTS_ACT = ['linear', 'softmax']
    _C.LOSS.SMOOTH_LABELS = 0.0 # smooth true labels (see tf.keras.losses.BinaryCrossentropy doc)
    _C.LOSS.PASS_SAMPLE_WGTS = False
    _C.LOSS.SEG_CONFIDENCE_CTR = False # whether to center around mean and variance before scaling
    _C.LOSS.SEG_CONFIDENCE_MIN = 0.0
    _C.LOSS.SEG_CONFIDENCE_MAX = 1.0
    _C.LOSS.BENIGN_CLASS_WGT = 1.0
    _C.LOSS.THREAT_CLASS_WGT = 1.0
    _C.LOSS.EXTRA = CN(new_allowed=True)

    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.BG_IMAGE = ''
    _C.DATASET.FORMAT = 'aps'
    _C.DATASET.W_IMG_DIR = ''
    _C.DATASET.IMAGE_DIR = ''
    _C.DATASET.EXTENSION = 'png'
    _C.DATASET.COLOR_RGB = True
    _C.DATASET.SCALE_PIXELS = False  # if false, use original pixel distribution as is
    _C.DATASET.PREPROCESS = False
    _C.DATASET.NORMALIZE = True
    _C.DATASET.ENHANCE_VISIBILITY = 0.0 # enhance image contrast when loading image. <= 0 : Don't
    _C.DATASET.FRESH_BUILD = False
    _C.DATASET.RETAIN_IN_MEMORY = True # numpy arrays of data (eg. X, y, w) are kept in memory
    _C.DATASET.PERSIST_ON_DISK = False # numpy arrays of data (eg. X, y, w) are write/read from disk
    _C.DATASET.RETAINED_DATA_DIR = '' # directory holding all persistent ndarrays of data type
    _C.DATASET.XY_BOUNDARY_ERROR = 0 # allowance for nci/roi coordinates to fall Out-Of-Bounds
    _C.DATASET.SUBSETS = CN()
    _C.DATASET.SUBSETS.SMALL_PORTION = False # use only a small portion of the dataset (set for debugging on laptop)
    _C.DATASET.SUBSETS.GROUPINGS = '../../Data/tsa_psc/dataSetDistribution.csv'
    _C.DATASET.SUBSETS.GROUP_TYPE = 'groupSet'
    _C.DATASET.SUBSETS.TRAIN_SETS = ['Train']
    _C.DATASET.SUBSETS.VALID_SETS = ['Validation', 'Evaluation']

    # IMAGE DATA AUGMENTATION
    _C.AUGMENT = CN()
    _C.AUGMENT.ODDS = 0.0 # odds of NOT augmenting a given sample (related to minority class size)
    _C.AUGMENT.X_SHIFT = 0.0 # max, (multiple of region window's width) shift center left/right
    _C.AUGMENT.Y_SHIFT = 0.0 # max, (multiple of region window's height) shift center up/down
    _C.AUGMENT.ROTATE = 0
    _C.AUGMENT.H_FLIP = False
    _C.AUGMENT.S_ZOOM = 0.0 # zoom in/out of ROI by this much
    #_C.AUGMENT.PER_IMAGE = True # different randomly generated configuration for each image
    _C.AUGMENT.N_CONTRAST = 0.0 # Advanced LAB histogram equalization contrast enhancement
    _C.AUGMENT.GRID_TILES = [5, 5] # number of rows and column in adaptive histogram equalization
    _C.AUGMENT.BRIGHTNESS = 0 # max: 127
    _C.AUGMENT.P_CONTRAST = 0 # max: 127 Per pixel contrast change for brightness
    _C.AUGMENT.WANDERING_ROI = False # whether or not roi (eg. 112x112) moves with x/y shift augment

    # Labels params
    _C.LABELS = CN()
    _C.LABELS.USE_THREAT_BBOX_ANOTS = False  # whether or not to use threat bounding box annotation
    _C.LABELS.THREAT_OVERLAP_THRESH = 0.0  # region contains threat if it significantly overlaps
    _C.LABELS.THREAT_ANOTS_MISMATCH = 0.0 # weight applied to sample if tsa and threat bbox disagree
    _C.LABELS.USE_PSEUDO_LABELS = False  # whether to learn with predicted labels of test set
    _C.LABELS.CSV = CN()
    _C.LABELS.CSV.GT_ZONE = '' #'Data/tsa_psc/stage1_labels_corrected.csv'
    _C.LABELS.CSV.KPTS_SET = ''
    _C.LABELS.CSV.FZK_MAP = '' #'fid_zones_kpts_map.csv'
    _C.LABELS.CSV.THREAT_OBJ_POS = '' # manually labelled threat object bounding-box coordinates
    _C.LABELS.CSV.PRED_PROB = '' # csv with predicted threat probability of test-set

    # train
    _C.TRAIN = CN()
    _C.TRAIN.ON_WHOLE_SET = False
    _C.TRAIN.INIT_LR = 0.001
    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.BETA_1 = 0.9
    _C.TRAIN.BETA_2 = 0.999
    _C.TRAIN.AMSGRAD = False
    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 150 # when <0, last epoch is computed from maximum step
    _C.TRAIN.WARMUP_EPOCHS = 5 # Initial training with fe frozen. Should be less than Patience
    _C.TRAIN.PATIENCE = 5 # when <0, ignore (same for _C.TRAIN.WARMUP_EPOCHS)
    _C.TRAIN.EOE_SHUFFLE = True # End of Epoch data shuffle
    _C.TRAIN.MINORITY_RESAMPLE = True
    _C.TRAIN.AUGMENTATION = True
    _C.TRAIN.WORKERS = 1 # used for generators or or keras.utils.Sequence
    _C.TRAIN.PRINT_FREQ = 1
    _C.TRAIN.CUSTOM_TRAIN_LOOP = False # if False use tf.keras.model.fit() to train network
    _C.TRAIN.BATCH_SIZE_PER_GPU = 32 # batch size per replica
    _C.TRAIN.BATCH_SIZE = 32 # effective batch size for distributed training
    _C.TRAIN.QUEUE_SIZE = 1 # max number of batches for which inputs (X, y, wgt) are prefetched
    _C.TRAIN.CACHE_TF_DATA = False # whether or not to cache tf.data input samples
    _C.TRAIN.TF_CHECKPOINTING = True # whether ot not to use ModelCheckpoint callback
    _C.TRAIN.TB = CN(new_allowed=True)
    _C.TRAIN.EXTRA = CN(new_allowed=True)

    # validation
    _C.VALID = CN()
    #_C.VALID.VALIDATE = True
    _C.VALID.CBK_VALIDATE = True
    _C.VALID.FIT_VALIDATE = True
    _C.VALID.EOE_SHUFFLE = False
    _C.VALID.MINORITY_RESAMPLE = False
    _C.VALID.AUGMENTATION = False
    _C.VALID.WORKERS = 1 # used for generators or or keras.utils.Sequence
    _C.VALID.PRINT_FREQ = 1
    _C.VALID.BATCH_SIZE_PER_GPU = 32
    _C.VALID.BATCH_SIZE = 32
    _C.VALID.QUEUE_SIZE = 1 # max number of batches for which inputs (X, y, wgt) are prefetched
    _C.VALID.N_DECIMALS = 7  # number of decimal places or precision of the predicted threat prob.
    _C.VALID.EOE_VALIDATE = False # do callback validation at the end of every epoch
    _C.VALID.VALIDATE_FREQ = 0 # number of steps between each validation operation
    _C.VALID.CACHE_TF_DATA = False  # whether or not to cache tf.data input samples
    _C.VALID.DELAY_CHECKPOINT = 0 # number of epochs that passes before optimal model saving

    # testing
    _C.TEST = CN()
    _C.TEST.MODEL_FILE = '' # some file at ../../Passenger-Screening-Challenge/models/perZone/
    _C.TEST.EOE_SHUFFLE = False # End of Epoch data shuffle
    _C.TEST.AUGMENTATION = False
    _C.TEST.WORKERS = -1  # used for generators or or keras.utils.Sequence
    _C.TEST.BATCH_SIZE_PER_GPU = 34 # multiple of 17
    _C.TEST.BATCH_SIZE = 34
    _C.TEST.QUEUE_SIZE = 8  # max number of batches for which inputs (X, y, wgt) are prefetched
    _C.TEST.N_DECIMALS = 7  # number of decimal places or precision of the predicted threat prob.
    _C.TEST.CACHE_TF_DATA = True  # whether or not to cache tf.data input samples

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.TURN_ON_DISPLAY = False
    _C.DEBUG.ONE_PER_STEP = False
    _C.DEBUG.PREDICTIONS = False
    _C.DEBUG.IMG_LOG_MAX = 15
    _C.DEBUG.TIMESTAMP = 'minute' # timestamp model with %Y-%m-%d-%H-%M

    return _C


def update_config(cfg, args, physical_gpus=1, logical_gpus=1, virtual_gpu_bsf=1.0, create_log=True):
    ##print('\nUpdate config called!!\n')
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    validate_configurations(cfg)

    # general runtime arguments
    if args.modelDir: cfg.MODEL_DIR = args.modelDir
    if args.dataDir: cfg.DATA_DIR = args.dataDir
    if args.group: cfg.MODEL.GROUPS = [args.group]

    # log experiment's configuration
    if cfg.DEBUG.TIMESTAMP=='second': timestamp = '%Y.%m.%d-%H.%M.%S'
    elif cfg.DEBUG.TIMESTAMP=='minute': timestamp = '%Y.%m.%d-%H.%M'
    else: timestamp = '%Y-%m'
    time_str = time.strftime(timestamp)
    exp_prefix = NETWORK_PREFIX[cfg.MODEL.SUBNET_TYPE]
    if len(cfg.MODEL.GROUPS) == 1:
        grp = cfg.MODEL.GROUPS[0]
    else: grp = exp_prefix.format(len(cfg.MODEL.GROUPS))
    exp_name = '{}_{}'.format(grp, time_str)

    # update some directory and file locations
    cfg.DATA_DIR = str(Path(os.path.abspath(cfg.DATA_DIR)))
    cfg.DATASET.ROOT = str(Path(cfg.DATA_DIR) / Path(cfg.DATASET.ROOT))
    cfg.DATASET.IMAGE_DIR = str(Path(cfg.DATASET.ROOT) / cfg.DATASET.IMAGE_DIR)
    cfg.DATASET.W_IMG_DIR = str(Path(cfg.DATASET.ROOT) / cfg.DATASET.W_IMG_DIR)
    cfg.DATASET.RETAINED_DATA_DIR = str(Path(cfg.DATASET.ROOT) / cfg.DATASET.RETAINED_DATA_DIR)

    cfg.HOME_DIR = str(Path(os.path.abspath(cfg.HOME_DIR)))
    cfg.MODEL_PATH = str(Path(cfg.HOME_DIR) / Path(cfg.MODEL_DIR) / cfg.DATASET.FORMAT)
    print('\ncfg.HOME_DIR: {}\ncfg.MODEL_PATH: {}\n'.format(cfg.HOME_DIR, cfg.MODEL_PATH))
    cfg.MODEL_DIR = str(Path(cfg.MODEL_PATH) / cfg.EXP_DIR / exp_name)
    cfg.LABELS.CSV.FZK_MAP = str(Path(cfg.HOME_DIR) / Path(cfg.LABELS.CSV.FZK_MAP))
    cfg.LABELS.CSV.KPTS_SET = str(Path(cfg.HOME_DIR) / Path(cfg.LABELS.CSV.KPTS_SET))
    cfg.LABELS.CSV.PRED_PROB = str(Path(cfg.MODEL_PATH) / Path(cfg.LABELS.CSV.PRED_PROB))

    cfg.LABELS.CSV.GT_ZONE = str(Path(cfg.HOME_DIR) / Path(cfg.LABELS.CSV.GT_ZONE))
    cfg.LABELS.CSV.THREAT_OBJ_POS = str(Path(cfg.HOME_DIR) / Path(cfg.LABELS.CSV.THREAT_OBJ_POS))
    cfg.DATASET.SUBSETS.GROUPINGS = str(Path(cfg.HOME_DIR) / Path(cfg.DATASET.SUBSETS.GROUPINGS))
    cfg.DATASET.BG_IMAGE = str(Path(cfg.HOME_DIR) / Path(cfg.DATASET.BG_IMAGE))
    if cfg.MODEL.PRETRAINED != '':
        cfg.MODEL.PRETRAINED = str(Path(cfg.MODEL_PATH) / cfg.MODEL.PRETRAINED)
    if cfg.TEST.MODEL_FILE != '':
        cfg.TEST.MODEL_FILE = str(Path(cfg.MODEL_DIR) / cfg.TEST.MODEL_FILE)

    # network variable runtime arguments
    if args.nEpochs: cfg.TRAIN.END_EPOCH = args.nEpochs
    if args.patience: cfg.TRAIN.PATIENCE = args.patience
    if args.learnRate: cfg.TRAIN.INIT_LR = args.learnRate
    if args.nGpus: cfg.GPUS = tuple(range(args.nGpus))
    if args.nWorkers: cfg.TRAIN.WORKERS = args.nWorkers
    if args.validate: cfg.VALID.CBK_VALIDATE = args.validate
    if args.batchSize: cfg.TRAIN.BATCH_SIZE_PER_GPU = args.batchSize
    if args.queueSize: cfg.TRAIN.QUEUE_SIZE = args.queueSize
    if args.verbose: cfg.TRAIN.PRINT_FREQ = args.verbose

    # derive effective batch size
    n_gpus = max(logical_gpus, min(physical_gpus, len(cfg.GPUS)))
    cfg.TRAIN.BATCH_SIZE = int(cfg.TRAIN.BATCH_SIZE_PER_GPU * n_gpus * virtual_gpu_bsf)
    cfg.VALID.BATCH_SIZE = int(cfg.VALID.BATCH_SIZE_PER_GPU * n_gpus * virtual_gpu_bsf)

    # handling over-fitting variables
    if args.batchNormalize: cfg.MODEL.BATCH_NORMALIZE = args.batchNormalize
    if args.regularize: cfg.MODEL.WGT_REG_DENSE = args.regularize
    if args.dropout: cfg.MODEL.DROPOUT = args.dropout
    if args.dataNormalize: cfg.DATASET.NORMALIZE = args.dataNormalize

    cfg.freeze()

    logger = None
    if create_log:
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        logfile = str(Path(cfg.MODEL_DIR) / '{}.log'.format(exp_name))
        logger = Logger(logfile)
    return exp_name, logger


def adapt_config(cfg, args, physical_gpus=1, logical_gpus=1):
    ##print('\nAdapt config called!!\n')
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    validate_configurations(cfg)
    #assert(cfg.DATASET.NORMALIZE != cfg.DATASET.PREPROCESS)

    # change some directory and file locations accordingly
    data_dir = cfg.DATA_DIR # eg. Linux data dir: '/media/sdgroup/lamadi/datasets/tsa'
    cfg.DATA_DIR = str(Path(os.path.abspath('../../../datasets/tsa')))
    cfg.DATASET.ROOT = cfg.DATA_DIR + str(Path(cfg.DATASET.ROOT.split(data_dir)[-1]))
    #cfg.LABELS.CSV.FZK_MAP = cfg.DATA_DIR + str(Path(cfg.LABELS.CSV.FZK_MAP.split(data_dir)[-1]))
    #cfg.LABELS.CSV.KPTS_SET = cfg.DATA_DIR + str(Path(cfg.LABELS.CSV.KPTS_SET.split(data_dir)[-1]))

    home_dir = cfg.HOME_DIR  # eg. Linux home dir: '/media/sdgroup/lamadi/cs691'
    cfg.HOME_DIR = str(Path(os.path.abspath('../')))
    cfg.DATASET.BG_IMAGE = cfg.HOME_DIR + str(Path(cfg.DATASET.BG_IMAGE.split(home_dir)[-1]))
    cfg.LABELS.CSV.GT_ZONE = cfg.HOME_DIR + str(Path(cfg.LABELS.CSV.GT_ZONE.split(home_dir)[-1]))
    cfg.LABELS.CSV.FZK_MAP = cfg.HOME_DIR + str(Path(cfg.LABELS.CSV.FZK_MAP.split(home_dir)[-1]))
    cfg.LABELS.CSV.KPTS_SET = cfg.HOME_DIR + str(Path(cfg.LABELS.CSV.KPTS_SET.split(home_dir)[-1]))
    cfg.LABELS.CSV.THREAT_OBJ_POS = cfg.HOME_DIR + \
                                    str(Path(cfg.LABELS.CSV.THREAT_OBJ_POS.split(home_dir)[-1]))

    if args.batchSize: cfg.TEST.BATCH_SIZE_PER_GPU = args.batchSize
    n_gpus = max(logical_gpus, min(physical_gpus, len(cfg.GPUS)))
    cfg.TEST.BATCH_SIZE = int(cfg.TEST.BATCH_SIZE_PER_GPU * n_gpus)

    cfg.freeze()


def validate_configurations(cfg):
    # confirm configuration expectations
    assert (cfg.DATASET.NORMALIZE != cfg.DATASET.PREPROCESS) # one or the other
    assert (cfg.DATASET.RETAIN_IN_MEMORY or cfg.DATASET.PERSIST_ON_DISK)  # !p or r == p -> r
    assert (cfg.DATASET.ENHANCE_VISIBILITY * cfg.AUGMENT.N_CONTRAST == 0)  # one or the other
    assert (cfg.MODEL.EXTRA.ROI_TYPE != 'oriented' or not cfg.AUGMENT.WANDERING_ROI) # p -> r
    # dataset subset groupings check
    if cfg.DATASET.SUBSETS.GROUP_TYPE != 'groupSet':
        k = int(cfg.DATASET.SUBSETS.GROUP_TYPE[1: cfg.DATASET.SUBSETS.GROUP_TYPE.find('_')])
        for t in cfg.DATASET.SUBSETS.TRAIN_SETS: assert (1<=t<=k), "t:{} > k:{}".format(t, k)
        for t in cfg.DATASET.SUBSETS.VALID_SETS: assert (1<=t<=k), "t:{} > k:{}".format(t, k)
    # image augmentation
    assert (0<=cfg.AUGMENT.S_ZOOM<1), "zoom scale must be in the interval [0,1)"
    assert (0<=cfg.AUGMENT.N_CONTRAST<=5), "recommended contrast enhancement is 3 or 4, max 5"
    # check model's network architecture and versions are compatible
    net_arch = cfg.MODEL.NETWORK_ARCHITECTURE
    version = cfg.MODEL.EXTRA.RES_CONV_VERSION
    assert (net_arch!='grp_model_v3_td' or version in ['v3'])
    assert (net_arch!='grp_model_v4_td' or version in ['v2','v3','v4','v5'])
    assert (net_arch!='grp_model_v5_td' or version in ['v5'])
    assert (net_arch!='grp_model_v6_td' or (version in ['v6','v7'] and
                                            len(cfg.LOSS.NET_OUTPUTS_ID)==1))
    # check specific model architecture modes
    assert (cfg.MODEL.EXTRA.POOL_FUNC in ['glob_avg', 'glob_max', 'max'])
    assert (cfg.MODEL.EXTRA.JOIN_GLOBREG_N_PREVZOI in ['no-join', 'add', 'concat'])
    assert (cfg.MODEL.EXTRA.JOIN_GLOBREG_N_PREVZOI!='no-join' or cfg.MODEL.EXTRA.CONCAT_ZOI_STG_BLOCKS)
    # check ground-truth annotations setup
    multi_outputs = len(cfg.LOSS.NET_OUTPUTS_ID) > 1
    assert (not cfg.LABELS.USE_THREAT_BBOX_ANOTS or multi_outputs)
    if multi_outputs:
        output2_id = cfg.LOSS.NET_OUTPUTS_ID[1]
        assert (not cfg.LABELS.USE_THREAT_BBOX_ANOTS or output2_id=='gt')
        # train with threat bbox vs. without threat bbox but warm-up bce then selective loss
        out2_loss = cfg.LOSS.NET_OUTPUTS_LOSS[1]
        assert (out2_loss!='multi_bce_loss' or cfg.LOSS.DEFAULT_LOSS_BRANCH_WGT==[1., 0.])
        select_bce_cond = cfg.LOSS.DEFAULT_LOSS_BRANCH_WGT==[0., 1.] \
                          or cfg.LOSS.EXTRA.SELECTIVE_BCE_EPOCH>0
        assert (out2_loss!='selective_loss' or select_bce_cond)


def runtime_args():
    '''
    runtime arguments for train_nn.py in training
    :return: runtime arguments dictionary
    '''
    ap = argparse.ArgumentParser(description='Train zone threat detection network')

    # general
    ap.add_argument('--cfg', required=True, type=str,
                    help='experiment configure file name')
    ap.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                    help='Modify config options using the command-line')

    # All arguments below only have an effect if specified during runtime
    ap.add_argument('--modelDir', type=str, help='model directory')
    ap.add_argument('--logDir', type=str, help='log directory')
    ap.add_argument('--dataDir', type=str, help='data directory')
    ap.add_argument('--group', type=str, help='one of 10 tsa body groups')
    ap.add_argument('--bodyZone', type=str, help='one of 17 tsa zones')

    # network variables
    ap.add_argument("-ep", "--nEpochs", type=int,
                    help="number of training epochs")
    ap.add_argument("-pt", "--patience", type=int,
                    help="max number of epochs to train after no significant improvement")
    ap.add_argument("-bs", "--batchSize", type=int,
                    help="batch size or number of samples in a mini-batch per gpu")
    ap.add_argument("-lr", "--learnRate", type=float,
                    help="learning rate of gradient decent optimizer algorithm")
    ap.add_argument("-vb", "--verbose", type=int,
                    help="keras verbose for print frequency")
    ap.add_argument("-ng", "--nGpus", type=int,
                    help="number of GPUs available for training")
    ap.add_argument("-nw", "--nWorkers", type=int,
                    help="number of CPU nodes available")
    ap.add_argument("-qs", "--queueSize", type=int,
                    help="number of batch/step calls of generator to be held at a time")

    # overfit variables
    ap.add_argument("-bn", "--batchNormalize", type=bool,
                    help="whether or not to apply batch normalization in network")
    ap.add_argument("-rg", "--regularize", type=float,
                    help="regularization coefficient for learned weights")
    ap.add_argument("-dp", "--dropout", type=float,
                    help="dropout rate for introducing randomness to network")
    ap.add_argument("-dn", "--dataNormalize", type=bool,
                    help="whether or not to normalize input images")

    # validation variablrs
    ap.add_argument("-vt", "--validate", type=bool,
                    help="whether or not to run validation during training")

    # test/evaluation variables
    ap.add_argument("-lf", "--logf1s", type=str,
                    help="list epochs of combined log-loss and f1-score optimal models to test")
    ap.add_argument("-af", "--avgf1s", type=str,
                    help="list epochs of average f1-score models to test, eg: 0,32,4500")
    ap.add_argument("-lg", "--loglos", type=str,
                    help="list epochs of log-loss models to test, eg: 0,32,4500")
    ap.add_argument("-nf", "--netfmt", type=str, default='h5',
                    help="format the model weight is saved in, eg: 'h5', 'tf")

    # args = vars(ap.parse_args()) # Converts to dictionary format
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        cfg = default_experiment_configuration()
        print(cfg, file=f)

