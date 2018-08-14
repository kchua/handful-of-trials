from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import tensorflow as tf
import numpy as np
import gpflow

from dmbrl.misc.DotmapUtils import get_required_argument


class TFGP:
    def __init__(self, params):
        """Initializes class instance.

        Arguments:
            params
                .name (str): Model name
                .kernel_class (class): Kernel class
                .kernel_args (args): Kernel args
                .num_inducing_points (int): Number of inducing points
                .sess (tf.Session): Tensorflow session
        """
        self.name = params.get("name", "GP")
        self.kernel_class = get_required_argument(params, "kernel_class", "Must provide kernel class.")
        self.kernel_args = params.get("kernel_args", {})
        self.num_inducing_points = get_required_argument(
            params, "num_inducing_points", "Must provide number of inducing points."
        )

        if params.get("sess", None) is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        else:
            self._sess = params.get("sess")

        with self._sess.as_default():
            with tf.variable_scope(self.name):
                output_dim = self.kernel_args["output_dim"]
                del self.kernel_args["output_dim"]
                self.model = gpflow.models.SGPR(
                    np.zeros([1, self.kernel_args["input_dim"]]),
                    np.zeros([1, output_dim]),
                    kern=self.kernel_class(**self.kernel_args),
                    Z=np.zeros([self.num_inducing_points, self.kernel_args["input_dim"]])
                )
                self.model.initialize()

    @property
    def is_probabilistic(self):
        return True

    @property
    def sess(self):
        return self._sess

    @property
    def is_tf_model(self):
        return True

    def train(self, inputs, targets,
              *args, **kwargs):
        """Optimizes the parameters of the internal GP model.

        Arguments:
            inputs: (np.ndarray) An array of inputs.
            targets: (np.ndarray) An array of targets.
            num_restarts: (int) The number of times that the optimization of
                the GP will be restarted to obtain a good set of parameters.

        Returns: None.
        """
        perm = np.random.permutation(inputs.shape[0])
        inputs, targets = inputs[perm], targets[perm]
        Z = np.copy(inputs[:self.num_inducing_points])
        if Z.shape[0] < self.num_inducing_points:
            Z = np.concatenate([Z, np.zeros([self.num_inducing_points - Z.shape[0], Z.shape[1]])])
        self.model.X = inputs
        self.model.Y = targets
        self.model.feature.Z = Z
        with self.sess.as_default():
            self.model.compile()
            print("Optimizing model... ", end="")
            gpflow.train.ScipyOptimizer().minimize(self.model)
            print("Done.")

    def predict(self, inputs, *args, **kwargs):
        """Returns the predictions of this model on inputs.

        Arguments:
            inputs: (np.ndarray) The inputs on which predictions will be returned.
            ign_var: (bool) If True, only returns the mean prediction

        Returns: (np.ndarrays) The mean and variance of the model on the new points.
        """
        if self.model is None:
            raise RuntimeError("Cannot make predictions without initial batch of data.")

        with self.sess.as_default():
            mean, var = self.model.predict_y(inputs)
            return mean, var

    def create_prediction_tensors(self, inputs, *args, **kwargs):
        ""
        if self.model is None:
            raise RuntimeError("Cannot make predictions without initial batch of data.")

        inputs = tf.cast(inputs, tf.float64)
        mean, var = self.model._build_predict(inputs, full_cov=False)
        return tf.cast(mean, dtype=tf.float32), tf.cast(var, tf.float32)

    def save(self, *args, **kwargs):
        pass
