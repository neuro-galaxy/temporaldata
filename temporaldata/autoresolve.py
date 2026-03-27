from contextvars import ContextVar

_AUTORESOLVE = ContextVar("autoresolve", default=True)


def get_autoresolve() -> bool:
    return _AUTORESOLVE.get()


class autoresolve:
    """Context manager to control whether lazy attributes are auto-resolved.

    When autoresolve is disabled, accessing attributes on Lazy* objects
    returns the raw h5py.Dataset instead of loading into a numpy array.

    Args:
        enabled: Whether to auto-resolve or not

    Example::

        with temporaldata.autoresolve(False):
            ds = data.values  # h5py.Dataset, not np.ndarray
    """

    def __init__(self, enabled: bool):
        self._enabled = enabled

    def __enter__(self):
        self._token = _AUTORESOLVE.set(self._enabled)
        return self

    def __exit__(self, *exc):
        _AUTORESOLVE.reset(self._token)
        return False
