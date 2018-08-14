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


class ReacherConfigModule:
    ENV_NAME = "MBRLReacher3D-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 24, 17
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.reset()
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
        self.UPDATE_FNS = [self.update_goal]

        self.goal = tf.Variable(self.ENV.goal, dtype=tf.float32)
        self.SESS.run(self.goal.initializer)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self, sess=None):
        if sess is not None:
            self.goal.load(self.ENV.goal, sess)

    def obs_cost_fn(self, obs):
        if isinstance(obs, np.ndarray):
            return np.sum(np.square(ReacherConfigModule.get_ee_pos(obs, are_tensors=False) - self.ENV.goal), axis=1)
        else:
            return tf.reduce_sum(tf.square(ReacherConfigModule.get_ee_pos(obs, are_tensors=True) - self.goal), axis=1)

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
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.00025))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.00075))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.00075})
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
    def get_ee_pos(states, are_tensors=False):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]
        if are_tensors:
            rot_axis = tf.concat([tf.cos(theta2) * tf.cos(theta1), tf.cos(theta2) * tf.sin(theta1), -tf.sin(theta2)],
                                 axis=1)
            rot_perp_axis = tf.concat([-tf.sin(theta1), tf.cos(theta1), tf.zeros(tf.shape(theta1))], axis=1)
            cur_end = tf.concat([
                0.1 * tf.cos(theta1) + 0.4 * tf.cos(theta1) * tf.cos(theta2),
                0.1 * tf.sin(theta1) + 0.4 * tf.sin(theta1) * tf.cos(theta2) - 0.188,
                -0.4 * tf.sin(theta2)
            ], axis=1)

            for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
                perp_all_axis = tf.cross(rot_axis, rot_perp_axis)
                x = tf.cos(hinge) * rot_axis
                y = tf.sin(hinge) * tf.sin(roll) * rot_perp_axis
                z = -tf.sin(hinge) * tf.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = tf.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis = tf.where(tf.less(tf.norm(new_rot_perp_axis, axis=1), 1e-30),
                                             rot_perp_axis, new_rot_perp_axis)
                new_rot_perp_axis /= tf.norm(new_rot_perp_axis, axis=1, keepdims=True)
                rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis
        else:
            rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                      axis=1)
            rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
            cur_end = np.concatenate([
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2)
            ], axis=1)

            for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
                perp_all_axis = np.cross(rot_axis, rot_perp_axis)
                x = np.cos(hinge) * rot_axis
                y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
                z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                    rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
                new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
                rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end


CONFIG_MODULE = ReacherConfigModule
