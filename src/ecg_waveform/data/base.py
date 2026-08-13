from abc import ABC, abstractmethod
from pathlib import Path

from ecg_waveform.core import ECGRecord


class BaseDataLoader(ABC):
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    @abstractmethod
    def __getitem__(self, record_name: str) -> ECGRecord: ...
