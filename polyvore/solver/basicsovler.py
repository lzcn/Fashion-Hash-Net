"""Class Solver."""
import logging
from time import time

import numpy as np
import torch
from torch.nn.parallel import data_parallel

import utils

LOGGER = logging.getLogger(__name__)


# TODO: Check BasicSolver
class BasicSolver(object):
    """Base class for Model."""

    def __init__(self, param, net, train_loader, test_loader):
        """Use param to initialize a Solver instance."""
        from torch.backends import cudnn

        cudnn.benchmark = True
        self.param = param
        self.net = net
        self.num_users = self.net.param.num_users
        self.best_acc = -np.inf
        self.best_loss = np.inf
        # data loader
        self.loader = dict(train=train_loader, test=test_loader)
        self.iter = 0
        self.last_epoch = -1
        self.parallel, self.device = utils.get_device(param.gpus)
        self.init_optimizer(param.optim_param)
        self.init_plot(param.visdom_env, param.visdom_title)

    def init_optimizer(self, optim_param):
        """Init Optimizer."""
        # set the optimizer
        optimizer = utils.get_named_class(torch.optim)[optim_param.name]
        groups = optim_param.groups
        num_child = self.net.num_gropus()
        if len(groups) == 1:
            groups = groups * num_child
        else:
            # num of groups should match the network
            assert num_child == len(
                groups
            ), """number of groups {},
            while size of children is {}""".format(
                len(groups), num_child
            )
        param_groups = []
        for child, param in zip(self.net.children(), groups):
            param_group = {"params": child.parameters()}
            param_group.update(param)
            param_groups.append(param_group)
        self.optimizer = optimizer(param_groups, **optim_param.grad_param)
        # set learning rate policy
        enum_lr_policy = utils.get_named_class(torch.optim.lr_scheduler)
        lr_policy = enum_lr_policy[optim_param.lr_scheduler]
        self.ReduceLROnPlateau = optim_param.lr_scheduler == "ReduceLROnPlateau"
        self.lr_scheduler = lr_policy(self.optimizer, **optim_param.scheduler_param)

    def init_plot(self, env, title):
        """Initialize plots for accuracy and loss."""
        self.tracer = utils.tracer.GroupPlotTracer(env, train=50, test=1)
        self.net.register_figure(self.tracer, title)
        self.tracer.logging()

    def gather_loss(self, losses, backward=False):
        """Gather all loss according to loss weight defined in self.net.

        1. Each loss in losses is shape of batch size
        2. Weight that is None will not be added into final loss

        Requires
        --------
        The loss return by self.net.forward() should match
        net.loss_weight in keys.
        """
        loss = 0.0
        results = {}
        for name, value in losses.items():
            weight = self.net.loss_weight[name]
            value = value.mean()
            # save the scale
            results[name] = value.item()
            if weight:
                loss += value * weight
        # save overall loss
        results["loss"] = loss.item()
        # wether to do back-propagation
        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return results

    @staticmethod
    def gather_accuracy(accuracy):
        """Gather all accuracy according in self.net.

        Each accuracy is shape of batch size. There no final accuracy.

        Return
        ------
        Each accuracy will be averaged and return.
        """
        return {k: v.sum().item() / v.numel() for k, v in accuracy.items()}

    def run(self):
        """Run solver from epoch [0, epochs - 1]."""
        while self.last_epoch < self.param.epochs - 1:
            self.step()

    def adjust_before_epoch(self, epoch):
        pass

    def step(self, epoch=None):
        """Step for train and evaluate.

        Re-implement `adjust_before_epoch` for extra setting.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.adjust_before_epoch(epoch)
        # train and test
        self.trainOneEpoch(epoch)
        result = self.testOneEpoch(epoch)
        loss, acc = result["loss"], result["accuracy"]
        # self.save(label=str(epoch))
        self.save(label="latest")
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_acc = acc
            self.save_net(label="best")
            LOGGER.info("Best model:")
            for k, v in result.items():
                LOGGER.info("-------- %s: %.3f", k, v)
        if self.ReduceLROnPlateau:
            self.lr_scheduler.step(-loss, epoch=epoch)
        else:
            self.lr_scheduler.step(epoch)
        self.last_epoch = epoch
        return result

    def step_batch(self, inputs):
        """Compute one batch."""
        if self.parallel:
            return data_parallel(self.net, inputs, self.param.gpus)
        return self.net(*inputs)

    def trainOneEpoch(self, epoch):
        """Run one epoch for training net."""
        # update the epoch
        self.net.train()
        phase = "train"
        # generate negative outfit before each epoch
        loader = self.loader[phase].make_nega()
        msg = "Train - Epoch[{}](%d): [%d]/[{}]:".format(epoch, loader.num_batch)
        latest_time = time()
        for idx, inputs in enumerate(loader):
            inputv = utils.to_device(inputs, self.device)
            data_time = time() - latest_time
            loss_, accuracy_ = self.step_batch(inputv)
            loss = self.gather_loss(loss_, backward=True)
            accuracy = self.gather_accuracy(accuracy_)
            batch_time = time() - latest_time
            results = dict(data_time=data_time, batch_time=batch_time)
            results.update(loss)
            results.update(accuracy)
            # update history and plotting
            self.tracer.update_history(group=phase, x=self.iter, data=results)
            if self.iter % self.param.display == 0:
                LOGGER.info(msg, self.iter, idx)
                self.tracer.update_trace(group=phase, x=self.iter, keys=results.keys())
                self.tracer.logging(phase)
                self.net.debug()
            self.iter += 1
            latest_time = time()
        LOGGER.info(msg, self.iter, loader.num_batch - 1)
        self.tracer.logging(phase)
        self.tracer.update_trace(group=phase, x=self.iter, keys=results.keys())

    def testOneEpoch(self, epoch=None):
        """Run test epoch.

        Parameter
        ---------
        epoch: current epoch. epoch = -1 means the test is
            done after net initialization.
        """
        if epoch is None:
            epoch = self.last_epoch
        # test only use pair outfits
        self.net.eval()
        phase = "test"
        latest_time = time()
        tracer = utils.tracer.Tracer(win_size=0)
        loader = self.loader[phase].make_nega()
        num_batch = loader.num_batch
        msg = "Epoch[{}]:Test [%d]/[{}]".format(epoch, num_batch)
        self.net.rank_metric.reset()
        for idx, inputs in enumerate(loader):
            # compute output and loss
            inputv = utils.to_device(inputs, self.device)
            uidx = inputs[-1].view(-1).tolist()
            batch_size = len(uidx)
            data_time = time() - latest_time
            with torch.no_grad():
                loss_, accuracy_ = self.step_batch(inputv)
                loss = self.gather_loss(loss_, backward=False)
                accuracy = self.gather_accuracy(accuracy_)
            # update time and history
            batch_time = time() - latest_time
            latest_time = time()
            data = dict(data_time=data_time, batch_time=batch_time)
            data.update(loss)
            data.update(accuracy)
            tracer.update_history(self.iter, data, weight=batch_size)
            LOGGER.info(msg, idx)
            tracer.logging()
        # compute average results
        rank_results = self.net.rank_metric.rank()
        results = {k: v.avg for k, v in tracer.get_history().items()}
        results.update(rank_results)
        self.tracer.update(phase, self.iter, results)
        LOGGER.info("Epoch[%d] Average:", epoch)
        self.tracer.logging(phase)
        return results

    def resume(self, file):
        """Resume from the latest state."""
        # resume net state
        LOGGER.info("Resuming training from %s", file)
        num_devices = torch.cuda.device_count()
        map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
        state_dict = torch.load(file, map_location=map_location)
        param = self.param
        self.__dict__.update(state_dict["state"])
        self.param = param
        self.net.load_state_dict(state_dict["net"])
        self.net.cuda(self.param.gpus[0])
        self.init_optimizer(param)
        self.init_schedular(param)
        self.optimizer.load_state_dict(state_dict["optim"])
        self.lr_scheduler.__dict__.update(state_dict["scheduler"])
        self.tracer.load_state_dict(state_dict["tracer"])

    # TODO:
    # []: save the history in a separated file
    def save(self, label=None):
        """Save network and optimizer."""
        if label is None:
            label = str(self.last_epoch)
        # self.save_solver(label)
        self.save_net(label)

    def save_solver(self, label):
        """Save the state of solver."""
        state_dict = {
            "net": self.net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "tracer": self.tracer.state_dict(),
            "state": dict(
                best_loss=self.best_loss,
                best_acc=self.best_acc,
                iter=self.iter,
                param=self.param,
                last_epoch=self.last_epoch,
            ),
            "scheduler": {
                k: v
                for k, v in self.lr_scheduler.__dict__.items()
                if k not in ["is_better", "optimizer"] and not callable(v)
            },
        }
        solver_path = self.format_filepath(label, "solver")
        LOGGER.info("Save solver state to %s", solver_path)
        torch.save(state_dict, solver_path)
        tracer_path = self.format_filepath(label, "tracer")
        LOGGER.info("Save tracer state to %s", tracer_path)
        torch.save(state_dict["tracer"], tracer_path)

    def save_net(self, label):
        """Save the net's state."""
        model_path = self.format_filepath(label, "net")
        LOGGER.info("Save net state to %s", model_path)
        torch.save(self.net.state_dict(), model_path)

    def format_filepath(self, label, suffix):
        """Return file-path."""
        filename = self.param.checkpoints + "_" + label + "." + suffix
        return filename
