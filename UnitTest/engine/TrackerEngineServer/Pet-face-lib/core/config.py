from yacs.config import CfgNode as CN


_C = CN()

_C.FACE_LIB = CN()
_C.FACE_LIB.DB_NAME = ''
_C.FACE_LIB.COL_NAME = ''
_C.FACE_LIB.MATCH_TH = 0.6

_C.PET_ENGINE = CN()
_C.PET_ENGINE.USE_ENGINE = False
_C.PET_ENGINE.PATH = ''
_C.PET_ENGINE.CFG = ''
_C.PET_ENGINE.FACE_DET_ID = 2
_C.PET_ENGINE.FACE_DET_TH = 0.6

_C.CROP_IMG = CN()
_C.CROP_IMG.SCALE = 1.5
_C.CROP_IMG.SIZE = 150


def get_cfg_defaults():
  """
    Get a yacs CfgNode object with default values for my_project.
  """
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()