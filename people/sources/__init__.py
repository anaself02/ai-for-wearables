"""One adapter per device format. Look them up by `Person.adapter_key`."""

from .base import SourceError, SourceFile, SourceNotImplemented
from .emotibit import EmotiBitAdapter
from .fitbit_takeout import FitbitTakeoutAdapter

ADAPTERS = {
    adapter.key: adapter
    for adapter in (FitbitTakeoutAdapter(), EmotiBitAdapter())
}


def get_adapter(key):
    try:
        return ADAPTERS[key]
    except KeyError:
        raise SourceError("No reader exists for device format {!r}.".format(key))


__all__ = ["ADAPTERS", "SourceError", "SourceFile", "SourceNotImplemented",
           "get_adapter"]
