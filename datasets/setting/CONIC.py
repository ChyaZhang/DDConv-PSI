from easydict import EasyDict as edict


# init
__C_CELL = edict()

cfg_data = __C_CELL

__C_CELL.TRAIN_SIZE = (512,512)
__C_CELL.DATA_PATH = './CoNIC/'
__C_CELL.TRAIN_LST = 'train.txt'
__C_CELL.VAL_LST =  'val.txt'
__C_CELL.VAL4EVAL = 'val_gt_loc.txt'

__C_CELL.MEAN_STD = (
    [0.7645501890422113, 0.6048785551788937, 0.7896357515213314],
    [0.18823568040130273, 0.24407128665343705, 0.1471062947108759]
)

__C_CELL.LABEL_FACTOR = 1
__C_CELL.LOG_PARA = 1.

__C_CELL.RESUME_MODEL = ''#model path
__C_CELL.TRAIN_BATCH_SIZE = 4 #imgs

__C_CELL.VAL_BATCH_SIZE = 1 # must be 1


