import logging
import warnings

import numpy as np
from visdom import Visdom
from utils.meter import MeterFactory

LOGGER = logging.getLogger(__name__)


class Tracer(object):
    """Class for history tracer.

    Parameter
    ---------
    win_size: set the meter for tracer.
    """

    def __init__(self, win_size):
        self._history = dict()
        self._meter_factory = MeterFactory(win_size)

    def get_history(self):
        return self._history

    def get_meter(self, key):
        return self._history.setdefault(key, self._meter_factory())

    def load_state_dict(self, state_dict):
        self._history = state_dict["_history"]
        meter_type = type(self._meter_factory()).__name__
        for k, v in self._history.items():
            old_meter_type = type(v).__name__
            if not meter_type == old_meter_type:
                warn_info = (
                    "Previous meter factory ({}) of {} doesn't equal to the "
                    "new one ({})".format(old_meter_type, k, meter_type)
                )
                warnings.warn(warn_info)

    def state_dict(self):
        state = {"_history": self._history}
        return state

    def update_history(self, x: int, data: dict, **kwargs):
        """Update the history only."""
        for key, value in data.items():
            self.get_meter(key).update(x, value, **kwargs)

    def logging(self):
        for k, m in self._history.items():
            LOGGER.info("-------- %s: %s", k, m)

    def __repr__(self):
        result = ""
        for k, v in self._history.items():
            result += "{}:{}\n".format(k, v)
        return result


class GroupTracer(object):
    """Class for history tracer.

    This class trace multi-group histories, e.g train and test.
    Each group saves the key-meter pairs.


    Parameter
    ---------
    group_win_size: set the window size for meters in each group
    """

    def __init__(self, **group_win_size):
        # create meter factories for each group
        self._groups = {g: Tracer(s) for g, s in group_win_size.items()}

    def get_meter(self, group, key):
        try:
            return self._groups[group].get_meter(key)
        except KeyError:
            raise KeyError("%s not in " + "|".join(self._groups.keys()))

    def load_state_dict(self, state_dict):
        for g, state in state_dict.items():
            self._groups[g].load_state_dict(state)

    def state_dict(self):
        state_dict = {}
        for g, t in self._groups.items():
            state_dict[g] = t.state_dict()
        return state_dict

    def update_history(self, group, x: int, data: dict, **kwargs):
        """Update the one group's history."""
        for key, value in data.items():
            meter = self.get_meter(group, key)
            meter.update(x, value, **kwargs)

    def logging(self, group=None):
        if group:
            self._groups[group].logging()
        else:
            for group, tracer in self._groups.items():
                LOGGER.info("-------- Group: %s", group)
                tracer.logging()


class PlotTracer(Tracer):
    def __init__(self, win_size):
        super().__init__(win_size)


class GroupPlotTracer(GroupTracer):
    """Class for tracing training history.

    Parameter
    ---------
    env: environment for visdom plotting
    group_win_size: set the window size for meters in each group
    """

    def __init__(self, env="main", **group_win_size):
        super().__init__(**group_win_size)
        self.vis = Visdom(env=env)
        self._figure_cfg = dict()
        self._registered_figures = dict()
        self._registered_lines = dict()

    def register_figure(self, title, xlabel, ylabel, trace_dict):
        """Register a new figure for visdom.

        keys in trace_dict must be unique as a fingerprint to retrieval.
        The fingerprint is a concat of group name and the history name
        in this group. e.g 'train.loss' where 'train' is the group name and
        'loss' is the history name within 'train' group

        Parameters
        ----------
        title: figure title
        xlabel, ylabel: name for x-axis and y-axis
        trace_dict: key-value pairs for traces/lines in figure
            trace_dict.values() are legends for the traces
            trace_dict.keys() are names for the traces
        the name has format 'phase.key'
        Return
        ------
        win: window id in current visdom environment

        """
        # check validation
        group_in_trace = set([k.split(".")[0] for k in trace_dict.keys()])
        group_in_class = set(self._groups.keys())
        unrecognized = group_in_trace - group_in_class
        assert (
            not unrecognized
        ), "Unrecognized group names: {}. All groups are: {}".format(
            unrecognized, group_in_class
        )
        # register meters
        for key in trace_dict.keys():
            g, k = key.split(".")
            self.get_meter(g, k)
        # register lines
        num_trace = len(trace_dict)
        legend = list(trace_dict.values())
        x = np.zeros(1)
        y = np.ones((1, num_trace)) * np.nan
        opts = dict(title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
        win = self.vis.line(X=x, Y=y, opts=opts)
        self._figure_cfg[win] = dict(
            title=title, xlabel=xlabel, ylabel=ylabel, trace_dict=trace_dict
        )
        self._registered_figures[win] = trace_dict
        for line in trace_dict.keys():
            self._registered_lines[line] = win
        # logging
        LOGGER.info("Registered lines (#%s)", win)
        return win

    def state_dict(self):
        state_dict = dict()
        state_dict["_groups"] = super().state_dict()
        state_dict["_figure_cfg"] = self._figure_cfg
        state_dict["_registered_figures"] = self._registered_figures
        state_dict["_registered_lines"] = self._registered_lines
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["_groups"])
        LOGGER.info("Re-plotting lines.")
        old_cfg = state_dict["_figure_cfg"]
        old_figures = state_dict["_registered_figures"]
        old_lines = state_dict["_registered_lines"]
        for cfg in old_cfg.values():
            self.register_figure(**cfg)
        # re-plot lines
        for line, old_win in old_lines.items():
            win = self._registered_lines.get(line, None)
            if win:
                legend = old_figures[old_win][line]
                group, key = line.split(".")
                x, y = self.get_meter(group, key).numpy()
                self.vis.line(
                    X=x,
                    Y=y,
                    update="append",
                    name=legend,
                    win=win,
                    opts={"showlegend": True},
                )

    def update(self, group, x: int, data: dict, **kwargs):
        self.update_history(group, x, data, **kwargs)
        self.update_trace(group, x, data.keys())

    def update_trace(self, group, x: int, keys):
        """Update single trace only."""
        for key in keys:
            # update history
            meter = self.get_meter(group, key)
            line = "{}.{}".format(group, key)
            win = self._registered_lines.get(line, None)
            if win:
                legend = self._registered_figures[win][line]
                self.vis.line(
                    X=np.array([x]),
                    Y=np.array([meter.avg]),
                    name=legend,
                    win=win,
                    update="append",
                    opts={"showlegend": True},
                )
