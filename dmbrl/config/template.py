from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC


class EnvConfigModule:
    ENV_NAME           = None
    TASK_HORIZON       = None
    NTRAIN_ITERS       = None
    NROLLOUTS_PER_ITER = None
    PLAN_HOR           = None

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": None}
        self.OPT_CFG = {
            "Random": {
                "popsize": None
            },
            "CEM": {
                "popsize":    None,
                "num_elites": None,
                "max_iters":  None,
                "alpha":      None
            }
        }
        self.UPDATE_FNS = []

        # Fill in other things to be done here.

    @staticmethod
    def obs_preproc(obs):
        # Note: Must be able to process both NumPy and Tensorflow arrays.
        if isinstance(obs, np.ndarray):
            raise NotImplementedError()
        else:
            raise NotImplementedError

    @staticmethod
    def obs_postproc(obs, pred):
        # Note: Must be able to process both NumPy and Tensorflow arrays.
        if isinstance(obs, np.ndarray):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def targ_proc(obs, next_obs):
        # Note: Only needs to process NumPy arrays.
        raise NotImplementedError()

    @staticmethod
    def obs_cost_fn(obs):
        # Note: Must be able to process both NumPy and Tensorflow arrays.
        if isinstance(obs, np.ndarray):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def ac_cost_fn(acs):
        # Note: Must be able to process both NumPy and Tensorflow arrays.
        if isinstance(acs, np.ndarray):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS
        ))
        # Construct model below. For example:
        # model.add(FC(*args))
        # ...
        # model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model


CONFIG_MODULE = EnvConfigModule

