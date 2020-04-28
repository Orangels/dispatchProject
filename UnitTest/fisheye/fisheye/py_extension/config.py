from ast import literal_eval
import copy
import os
import os.path as osp
import numpy as np
import yaml

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options. 
"""

cur_pth = os.getcwd()


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg_priv = __C
__C.LOG = 'logging.ERROR'

# loader for different framework pet_loader tensorrt_loader
__C.LOADER = 'pet_loader'
# ---------------------------------------------------------------------------- #
# Global options
# ---------------------------------------------------------------------------- #
__C.GLOBAL = AttrDict()

__C.GLOBAL.BANNER = 'PriVision System'

__C.GLOBAL.VISION_PROJECT_ROOT = '/home/user/workspace/PytorchEveryThing'

__C.GLOBAL.IM_SHOW_SIZE = (1296, 960)

__C.GLOBAL.SAVE_VIDEO_PATH = './files/save'

__C.GLOBAL.SAVE_VIDEO_MAX_SECOND = 1800 * 20

__C.GLOBAL.SAVE_VIDEO_FPS = 20

__C.GLOBAL.SAVE_VIDEO_SIZE = (1296, 960)

__C.MODULES = AttrDict()

# objcls: object_classification
__C.MODULES.OBJCLS = AttrDict()

# spfd: ssd_personface_det
__C.MODULES.SPFD = AttrDict()

__C.MODULES.SPFD.CFG = 'ckpts/ssd/wider_face/ssdlite/ssdlite_MV1-1.0-BN-PROB02_512x512_1x_COCO/' \
                       'ssdlite_MV1-1.0-BN-PROB02_512x512_1x_COCO.yaml'

# pfd: personface_det
__C.MODULES.PFD = AttrDict()

__C.MODULES.PFD.CFG = 'ckpts/wider_face/mscoco/e2e_faster_rcnn_R-50-FPN_2x_ms/' \
                      'e2e_faster_rcnn_R-50-FPN_2x_ms.yaml'

# fkp: personface_kpts
__C.MODULES.FKP = AttrDict()

__C.MODULES.FKP.CFG = 'ckpts/cls/face_keypoints/ldmk10_84/ldmk10_84.yaml'

# fkp3d: personface_kpts3d
__C.MODULES.FKP3D = AttrDict()

__C.MODULES.FKP3D.CFG = 'ckpts/cls/face_keypoints/ldmk10_84/face_kpt3d.yaml'

# fv: personface_verify
__C.MODULES.FV = AttrDict()

__C.MODULES.FV.CFG = 'ckpts/cls/face_verify/MobileFaceNetVerify/MobileFaceNetVerify.yaml'

__C.MODULES.FV.CFG2 = 'ckpts/cls/face_verify/MobileFaceNetVerify/MobileFaceNetVerify.yaml'

# age: age_gender
__C.MODULES.AGE = AttrDict()

__C.MODULES.AGE.CFG = 'ckpts/cls/face_verify/MobileFaceNetVerifyAgeGender/MobileFaceNetVerifyAgeGender.yaml'

# mtcnn: MTCNN
__C.MODULES.MTCNN = AttrDict()

__C.MODULES.MTCNN.CFG = 'ckpts/cls/MTCNN/Mtcnn_onet/Mtcnn_onet.yaml'

# ---------------------------------------------------------------------------- #
# FaceDemo options
# ---------------------------------------------------------------------------- #
__C.BUSS = AttrDict()

__C.BUSS.TRACK = AttrDict()

__C.BUSS.ASSESS = AttrDict()

__C.BUSS.VERIFY = AttrDict()

__C.BUSS.OTHER = AttrDict()

__C.BUSS.COUNT = AttrDict()
# -
__C.BUSS.TRACK.MAX_MISMATCH_TIMES = dict()
__C.BUSS.TRACK.MAX_MISMATCH_TIMES.update({"default": 30})
# -
__C.BUSS.ASSESS.MAX_SIZE = dict()
__C.BUSS.ASSESS.MAX_SIZE.update({"default": 120})
# -
__C.BUSS.ASSESS.MIN_SIZE = dict()
__C.BUSS.ASSESS.MIN_SIZE.update({"default": 60})
# -
__C.BUSS.ASSESS.MAX_ANGLE_YAW = dict()
__C.BUSS.ASSESS.MAX_ANGLE_YAW.update({"default": 20})
# -
__C.BUSS.ASSESS.MAX_ANGLE_PITCH = dict()
__C.BUSS.ASSESS.MAX_ANGLE_PITCH.update({"default": 30})
# -
__C.BUSS.ASSESS.MAX_ANGLE_ROLL = dict()
__C.BUSS.ASSESS.MAX_ANGLE_ROLL.update({"default": 180})
# -
__C.BUSS.ASSESS.MIN_BOX_SCORE = dict()
__C.BUSS.ASSESS.MIN_BOX_SCORE.update({"default": 0.6})
# -
__C.BUSS.ASSESS.MAX_BRIGHTNESS = dict()
__C.BUSS.ASSESS.MAX_BRIGHTNESS.update({"default": 200})
# -
__C.BUSS.ASSESS.MIN_BRIGHTNESS = dict()
__C.BUSS.ASSESS.MIN_BRIGHTNESS.update({"default": 10})
# -
__C.BUSS.VERIFY.THRESH = dict()
__C.BUSS.VERIFY.THRESH.update({"default": 0.5})
# -
__C.BUSS.OTHER.KPS_ON = dict()
__C.BUSS.OTHER.KPS_ON.update({"default": True})

__C.BUSS.COUNT.ROI_AREA = dict()
# __C.BUSS.COUNT.ROI_AREA.update({"default": [[0, 300], [1920, 300], [1920, 600], [0, 600]]})
__C.BUSS.COUNT.ROI_AREA.update({"default": None})

__C.BUSS.COUNT.ENTRANCE_LINE = dict()
__C.BUSS.COUNT.ENTRANCE_LINE.update({"default": None})

__C.BUSS.COUNT.ENTRANCE_DIRECTION = dict()
__C.BUSS.COUNT.ENTRANCE_DIRECTION.update({"default": None})

__C.BUSS.COUNT.ANGLE = dict()
__C.BUSS.COUNT.ANGLE.update({"default": 85})

__C.BUSS.COUNT.VECTOR_LEN = dict()
__C.BUSS.COUNT.VECTOR_LEN.update({"default": 0.5})
# ------------------------------------------------ #


__C.ALG = AttrDict()

__C.ALG.DET = AttrDict()

__C.ALG.TRACK = AttrDict()

__C.ALG.ASSESS = AttrDict()

__C.ALG.VERIFY = AttrDict()

__C.ALG.OTHER = AttrDict()

__C.ALG.DET.MODEL = 'ckpts/ssd/wider_face/ssdlite/ssdlite_MV1-1.0-BN-PROB02_512x512_1x_COCO/' \
                    'ssdlite_MV1-1.0-BN-PROB02_512x512_1x_COCO.yaml'

__C.ALG.ASSESS.MODEL = "ASSESS"

__C.ALG.ASSESS.MAX_SIZE = 500

__C.ALG.ASSESS.MIN_SIZE = 80

__C.ALG.ASSESS.MAX_ANGLE_YAW = 20

__C.ALG.ASSESS.MAX_ANGLE_PITCH = 30

__C.ALG.ASSESS.MAX_ANGLE_ROLL = 180

__C.ALG.ASSESS.L2_NORM = 0.3

__C.ALG.ASSESS.MIN_BOX_SCORE = 0.3

__C.ALG.ASSESS.MAX_BRIGHTNESS = 255

__C.ALG.ASSESS.MIN_BRIGHTNESS = 0

__C.ALG.VERIFY.MODEL = AttrDict()

__C.ALG.VERIFY.MODEL.CFG = 'ckpts/cls/face_verify/MobileFaceNetVerify/MobileFaceNetVerify.yaml'

__C.ALG.VERIFY.MODEL.NAME = 'wiwide_nature'

__C.ALG.VERIFY.MODEL.CFG2 = 'ckpts/cls/face_verify/MobileFaceNetVerify/MobileFaceNetVerify.yaml'

__C.ALG.VERIFY.MODEL.ENSUMBLE = True

__C.OTHER = AttrDict()

__C.OTHER.API_PLAN_A = True
# socket True

__C.OTHER.COUNT_PLAN_A = True
# true no entrance line
# false  yes entrance line

__C.OTHER.DBMODEL_TEST = True
# media_info_

__C.OTHER.COUNT_MODE = True
__C.OTHER.COUNT_DRAW = True
__C.OTHER.COUNT_DRAW_LESS = True
__C.OTHER.DRAW_TRACK = True
__C.OTHER.DRAW_HEAD = True
__C.OTHER.DRAW_ENTRA_LINE = True
__C.OTHER.DRAW_DIRECTION = True
__C.OTHER.DRAW_ROI = True
__C.OTHER.DRAW_TRACK_NUM = 25
# count_raw_


__C.FACEDEMO = AttrDict()

__C.FACEDEMO.MAC = '001F7A409C00'

__C.FACEDEMO.BOX_OUT_CONF = 0.3

__C.FACEDEMO.VERIFY_BATCH_SIZE = 256

__C.FACEDEMO.LDMK_BATCH_SIZE = 256

__C.FACEDEMO.CLUSTER_TOP_N = 3

__C.FACEDEMO.CLUSTER_TH = 0.7

__C.FACEDEMO.QUANTITY = 0.4

__C.FACEDEMO.SCREEN_STATE = False

__C.FACEDEMO.SCREEN_STATE_TH = 0.1

__C.FACEDEMO.USE_VERIFY = True

__C.FACEDEMO.IS_LOAD = True

__C.FACEDEMO.SHOW_PIC = True

__C.FACEDEMO.SAVE_PIC = True

__C.FACEDEMO.DRAW_IMG = True

__C.FACEDEMO.LOAD_VERIFY_MODEL = True

__C.FACEDEMO.FULLSCREEN = True

__C.FACEDEMO.USE_AGE = True

__C.FACEDEMO.DRAW_CHINESE = True

__C.FACEDEMO.JITTER_FACE = False

__C.FACEDEMO.USE_BRIGHTNESS = False

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def merge_priv_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.SafeLoader))
    _merge_a_into_b(yaml_cfg, __C)
    # update_cfg()


def merge_priv_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_priv_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, dict) and isinstance(value_b, dict):
        value_a = dict(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )
