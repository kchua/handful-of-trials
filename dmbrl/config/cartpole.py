from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env


class CartpoleConfigModule:
    ENV_NAME = "MBRLCartpole-v0"
    TASK_HORIZON = 200
    NTRAIN_ITERS = 50
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        else:
            return tf.concat([tf.sin(obs[:, 1:2]), tf.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        if isinstance(obs, np.ndarray):
            return -np.exp(-np.sum(
                np.square(CartpoleConfigModule._get_ee_pos(obs, are_tensors=False) - np.array([0.0, 0.6])), axis=1
            ) / (0.6 ** 2))
        else:
            return -tf.exp(-tf.reduce_sum(
                tf.square(CartpoleConfigModule._get_ee_pos(obs, are_tensors=True) - np.array([0.0, 0.6])), axis=1
            ) / (0.6 ** 2))

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model

    @staticmethod
    def _get_ee_pos(obs, are_tensors=False):
        x0, theta = obs[:, :1], obs[:, 1:2]
        if are_tensors:
            return tf.concat([x0 - 0.6 * tf.sin(theta), -0.6 * tf.cos(theta)], axis=1)
        else:
            return np.concatenate([x0 - 0.6 * np.sin(theta), -0.6 * np.cos(theta)], axis=1)


CONFIG_MODULE = CartpoleConfigModule
