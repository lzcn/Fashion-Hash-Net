"""Class for FashionNetSolver."""
import torch

from utils import get_named_class

from .basicsovler import BasicSolver


class FashionNetSolver(BasicSolver):
    """Base class for Model."""

    def adjust_before_epoch(self, epoch):
        if self.param.increase_hard:
            prob = pow(epoch / self.param.epochs, 0.5)
            self.loader["train"].set_prob_hard(prob)
        scale = pow(epoch * self.param.gamma + 1, 0.5)
        self.net.set_scale(scale)


class FinetuneSolver(BasicSolver):
    """Base class for Model."""

    def __init__(self, net, param):
        """Use param to initialize a Solver instance.

        Parameters
        ----------
        param: parameter to initialize the solver class
            param['net']: training net
            param['triplet']: whether to use triplet loss
            param['env']: the environment for visdom
            param['visdom_title']: title for figure

        """
        super(FinetuneSolver, self).__init__(net, param)

    def initOptim(self):
        """Init Optimizer."""
        # set the optimizer
        self.net.freeze_item_param()
        enum_optim = get_named_class(torch.optim)
        optimizer = enum_optim[self.param.optim]
        groups = self.param.optim_groups
        defaults = self.param.optim_defaults
        assert len(groups) == 1
        defaults.update(groups[0])
        param = self.net.user_embedding.parameters()
        self.optimizer = optimizer(param, **defaults)
