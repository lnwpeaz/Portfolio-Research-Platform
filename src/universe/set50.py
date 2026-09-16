"""SET50 provider contract; data remains deliberately unconfigured."""

from src.universe.models import POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE, PointInTimeMembershipUnavailable
from src.universe.sp500 import SP500UniverseProvider


class SET50UniverseProvider(SP500UniverseProvider):
    name = "set50"
    index_name = "SET50"

    def get_membership(self, as_of_date, *, require_point_in_time=False):
        try:
            return super().get_membership(
                as_of_date, require_point_in_time=require_point_in_time
            )
        except PointInTimeMembershipUnavailable as error:
            raise PointInTimeMembershipUnavailable(
                f"{POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE}: SET50 membership source "
                "is not configured. No history has been fabricated."
            ) from error
