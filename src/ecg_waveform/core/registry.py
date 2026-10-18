from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def _register(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"A {self.kind} named '{name}' is already registered.")
            self._items[name] = cls
            cls.registry_name = name
            return cls

        return _register

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}")
        return self._items[name]

    def build(self, name: str, params: dict[str, Any] | None = None) -> T:
        cls = self.get(name)
        try:
            return cls(**(params or {}))
        except TypeError as exc:
            raise TypeError(
                f"Invalid parameters for {self.kind} '{name}': {exc}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
