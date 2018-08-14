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


class PusherConfigModule:
    ENV_NAME = "MBRLPusher-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 27, 20
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(self, obs):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], self.ENV.ac_goal_pos

        if isinstance(obs, np.ndarray):
            tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=1)
            obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist
        else:
            tip_obj_dist = tf.reduce_sum(tf.abs(tip_pos - obj_pos), axis=1)
            obj_goal_dist = tf.reduce_sum(tf.abs(goal_pos - obj_pos), axis=1)
            return to_w * tip_obj_dist + og_w * obj_goal_dist

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.00025))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.00075))
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


CONFIG_MODULE = PusherConfigModule
