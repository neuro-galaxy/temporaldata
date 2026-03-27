from contextvars import ContextVar

_RESOLVE_ON_ACCESS = ContextVar("resolve_on_access", default=True)


def get_resolve_on_access() -> bool:
    return _RESOLVE_ON_ACCESS.get()


class resolve_on_access:
    """Context manager to control whether lazy attributes are resolved on access.

    When disabled, accessing attributes on Lazy* objects returns the raw
    h5py.Dataset instead of loading into a numpy array.

    Args:
        enabled: Whether to resolve on access or not

    Example::

        with temporaldata.resolve_on_access(False):
            ds = data.values  # h5py.Dataset, not np.ndarray
    """

    def __init__(self, enabled: bool):
        self._enabled = enabled

    def __enter__(self):
        self._token = _RESOLVE_ON_ACCESS.set(self._enabled)
        return self

    def __exit__(self, *exc):
        _RESOLVE_ON_ACCESS.reset(self._token)
        return False
