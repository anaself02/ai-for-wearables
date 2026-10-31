"""EmotiBit adapter: a placeholder, on purpose.

Rahil's EmotiBit export has not been received, so its format is unknown and
nothing here pretends otherwise. No file name is accepted and parsing raises a
clear "not implemented" error. When a sample arrives this is replaced by a real
parser that satisfies the `DeviceAdapter` contract.

WHAT IS NEEDED FROM THE EXPORT BEFORE A PARSER CAN BE WRITTEN
-------------------------------------------------------------
1. The exact files or sheets produced (layout and names).
2. Which signals are present, and in particular whether HEART RATE is a signal
   in the export or whether only raw PPG (the optical pulse waveform) is.
3. The sampling rate of each signal.
4. The timestamp format, and its timezone or reference (UTC, local time, or
   time since the start of the recording).
5. Units for every column.
6. Any event-marker or annotation columns (for example a rest/task marker).
7. A recording of at least 10 minutes, ideally with a resting period and a
   task period.

If only raw PPG exists, deriving heart rate from it is a separate decision:
it would produce beat-level values (a different resolution from a consumer
device's smoothed readings) and the arousal model's caveats would change.
That choice is not made here.
"""

from .base import DeviceAdapter, SourceNotImplemented

MESSAGE = (
    "The EmotiBit reader is not implemented: awaiting a sample export from "
    "Rahil, expected after the mid-project presentation. Nothing was loaded. "
    "See people/sources/emotibit.py for exactly what is needed from the export."
)


class EmotiBitAdapter(DeviceAdapter):
    key = "emotibit"
    device_label = "EmotiBit"

    def accepts(self, name):
        return False

    def parse_file(self, source_file):
        raise SourceNotImplemented(MESSAGE)
