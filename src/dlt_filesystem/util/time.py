"""UTC timestamp coercion across supported dlt versions."""

from datetime import timezone
from typing import TYPE_CHECKING, Any

from dlt.common.pendulum import pendulum

if TYPE_CHECKING:

    def ensure_datetime_utc(value: Any) -> pendulum.DateTime:
        """Coerce datetime, date, ISO strings, or epoch seconds to UTC."""
        ...

else:
    try:
        from dlt.common.time import ensure_pendulum_datetime_utc as ensure_datetime_utc
    except ImportError:
        from dlt.common.time import ensure_pendulum_datetime

        def ensure_datetime_utc(value: Any) -> pendulum.DateTime:
            """Coerce a timestamp to UTC independently of dlt's TimezoneContext."""
            return ensure_pendulum_datetime(value, tz=timezone.utc)
