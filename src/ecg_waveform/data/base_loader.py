from abc import ABC, abstractmethod
from pathlib import Path

from ecg_waveform.config import DATA_DIR
from ecg_waveform.core import ECGRecord


class BaseDataLoader(ABC):
    def __init__(self, dataset_name: str, dataset_root: Path = DATA_DIR) -> None:
        self.dataset_name: str = dataset_name
        self.dataset_root: Path = dataset_root / dataset_name

    @abstractmethod
    def list_records(self) -> list[str]: ...

    @abstractmethod
    def __getitem__(self, record_name: str) -> ECGRecord: ...
