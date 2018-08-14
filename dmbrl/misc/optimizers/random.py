from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, sol_dim, popsize, tf_session,
                 upper_bound=None, lower_bound=None):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
        """
        super().__init__()
        self.sol_dim = sol_dim
        self.popsize = popsize
        self.ub, self.lb = upper_bound, lower_bound
        self.tf_sess = tf_session
        self.solution = None
        self.tf_compatible, self.cost_function = None, None

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        if not tf_compatible:
            self.tf_compatible = False
            self.cost_function = cost_function
        else:
            with self.tf_sess.graph.as_default():
                self.tf_compatible = True
                solutions = tf.random_uniform([self.popsize, self.sol_dim], self.ub, self.lb)
                costs = cost_function(solutions)
                self.solution = solutions[tf.cast(tf.argmin(costs), tf.int32)]

    def reset(self):
        pass

    def obtain_solution(self, *args, **kwargs):
        """Optimizes the cost function provided in setup().

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.tf_compatible:
            return self.tf_sess.run(self.solution)
        else:
            solutions = np.random.uniform(self.lb, self.ub, [self.popsize, self.sol_dim])
            costs = self.cost_function(solutions)
            return solutions[np.argmin(costs)]
