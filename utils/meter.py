"""Transfer utils."""
from collections import deque

import numpy as np

from utils.math import smooth

# TODO: merge two meters into one class


class AvgMeter(object):
    """History recorder with moving average."""

    def __init__(self, win_size=50):
        """Average meter for criterions."""
        self.x = []
        self.y = []
        self.val = np.nan
        self._queue = deque(maxlen=win_size)
        self.win_size = win_size

    def reset(self):
        """Reset all attribute."""
        self.val = np.nan
        self.x = []
        self.y = []
        self._queue.clear()

    @property
    def avg(self):
        """Return moving average."""
        try:
            return sum(self._queue) / len(self._queue)
        except ZeroDivisionError:
            return np.nan

    def update(self, x, y, **kwargs):
        """Update attributes."""
        self.x.append(x)
        self.y.append(y)
        self._queue.append(y)
        self.val = y

    def numpy(self):
        """Return smoothed numpy array history until now."""
        x = np.array(self.x)
        y = smooth(self.y, self.win_size)
        return x, y

    def __repr__(self):
        """Return val and avg."""
        msg = "{:.4f} ({:.4f})".format(self.val, self.avg)
        return msg


class GlobalMeter(object):
    """Compute global average with weights."""

    def __init__(self):
        """History without moving average."""
        self.x = []
        self.y = []
        self.weights = []
        self.val = np.NaN
        self._cum_val = 0.0
        self._cum_weight = 0.0

    def numpy(self):
        """Return history until now as numpy array."""
        x = np.array(self.x)
        v, w = np.array(self.y), np.array(self.weights)
        y = np.cumsum(v * w) / np.cumsum(w)
        return x, y

    @property
    def avg(self):
        """Return fake average."""
        try:
            return self._cum_val / self._cum_weight
        except ZeroDivisionError:
            return np.nan

    def update(self, x, val, weight=1):
        """Update attributes."""
        self.x.append(x)
        self.y.append(val)
        self.weights.append(weight)
        self.val = val
        self._cum_val += val * weight
        self._cum_weight += weight

    def __repr__(self):
        return "{:.4f} ({:.4f})".format(self.val, self.avg)


def MeterFactory(win_size=1):
    assert win_size >= 0, "win_size must be non-negative."

    def meter():
        return AvgMeter(win_size=win_size)

    if win_size == 0:
        return GlobalMeter
    return meter
