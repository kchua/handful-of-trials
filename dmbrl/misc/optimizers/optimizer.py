from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function, tf_compatible):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")
