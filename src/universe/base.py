"""Common provider contract."""

from abc import ABC, abstractmethod
import pandas as pd

from src.universe.models import UniverseMember, UniverseSnapshot


class UniverseProvider(ABC):
    name: str
    index_name: str

    @abstractmethod
    def get_membership(
        self, as_of_date, *, require_point_in_time: bool = False
    ) -> tuple[UniverseMember, ...]:
        """Return members known for the date, never inferred from future membership."""

    @abstractmethod
    def get_snapshot(
        self, as_of_date=None, *, require_point_in_time: bool = False
    ) -> UniverseSnapshot:
        """Return an immutable snapshot with source provenance."""

    @staticmethod
    def normalize_date(value) -> pd.Timestamp:
        return pd.Timestamp(value).normalize()
