from dataclasses import asdict, is_dataclass
from typing import Any, ClassVar


class Configurable:
    __slots__ = ()

    registry_name: ClassVar[str | None] = None

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_params(self) -> dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)

        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and not callable(v)
        }

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
