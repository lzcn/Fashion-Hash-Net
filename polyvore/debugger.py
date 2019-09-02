import logging
import threading
from queue import Queue

import numpy as np

import utils

LOGGER = logging.getLogger(__name__)


# TODO: Check Debugger
class _Debugger(threading.Thread):
    message = "Basic Debugger"

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = Queue()
        self.daemon = True
        self.codes = None
        self.balance = 0.0
        self.ratio = 0.0

    def put(self, weights):
        if isinstance(weights, list):
            weights = [np.array(w.tolist()) for w in weights]
            weights = np.concatenate(weights)
        else:
            weights = np.array(weights.tolist())
        self.queue.put(weights)

    def run(self):
        print(threading.currentThread().getName(), self.message)
        while True:
            val = self.queue.get()
            if val is None:
                return
            if self.codes is None:
                self.balance, self.ratio = self.process(val)
            else:
                balance, ratio = self.process(val)
                self.balance = 0.8 * self.balance + 0.2 * balance
                self.ratio = 0.8 * self.ratio + 0.2 * ratio

    def process(self, weights):
        size = weights.size
        balance = np.sum(weights > 0) / size
        ratio = np.sum(np.abs(weights) > 0.95) / size
        self.codes = weights.reshape(-1).tolist()
        return balance, ratio

    def log(self):
        msg, arg = code_to_str(self.codes)
        LOGGER.debug(self.message)
        LOGGER.debug(msg, *arg)
        LOGGER.debug(
            "Codes balance: %s(p) vs %s(n).",
            utils.colour("%.3f" % self.balance),
            utils.colour("%.3f" % (1.0 - self.balance)),
        )
        LOGGER.debug(
            "Codes value: %s percent whose absolute value greater then 0.95",
            utils.colour("%.3f" % (self.ratio * 100)),
        )


@utils.singleton
class _ItemVisualDebugger(_Debugger):
    message = "Item Codes (Visual) Debugger"


@utils.singleton
class _ItemTextualDebugger(_Debugger):
    message = "Item Codes (Textual) Debugger"


@utils.singleton
class _UserDebugger(_Debugger):
    message = "User Codes Debugger"


_debugger_instance = {
    "user": _UserDebugger(),
    "item.v": _ItemVisualDebugger(),
    "item.s": _ItemTextualDebugger(),
}


def is_alive():
    alive = [v.is_live() for v in _debugger_instance.values()]
    return all(alive)


def start():
    for v in _debugger_instance.values():
        if not v.is_live():
            v.start()


def put(name, code):
    d = _debugger_instance[name]
    if not d.is_alive():
        d.start()
    d.put(code)


def log(name):
    d = _debugger_instance[name]
    if not d.is_alive():
        d.start()
    d.log()


def code_to_str(weight):
    size = len(weight)
    if size > 8:
        msg = "[" + "%1.3f," * 4 + "...," + "%1.3f," * 3 + "%1.3f]"
        args = weight[:4] + weight[-4:]
    else:
        msg = "[" + "%1.3f," * (size - 1) + "%1.3f]"
        args = weight
    return msg, args
