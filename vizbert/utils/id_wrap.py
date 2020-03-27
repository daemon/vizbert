from dataclasses import dataclass
from typing import Any


JSON_PRIMITIVES = {int, str, float, list, bool, type(None)}


@dataclass
class IdentityWrapper(object):
    obj: Any

    def __hash__(self):
        return id(self.obj)


def id_wrap(obj):
    return IdentityWrapper(obj)
