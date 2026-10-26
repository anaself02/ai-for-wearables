"""The post-login greeting, and the name it addresses.

Both are pure functions taking what they need, so the boundaries can be tested
without moving a clock or signing anybody in.

The hour is read from the server in `settings.TIME_ZONE` rather than from the
browser. That is the same clock `health.views._window` resolves "today" against,
and the page is server-rendered, so taking the time from JavaScript would mean
rendering one greeting and then correcting it in front of the reader.
"""

#: Hour boundaries, left-inclusive. Evening owns the small hours: somebody
#: awake at 02:00 is at the end of an evening, not the start of a morning.
MORNING_FROM = 5
AFTERNOON_FROM = 12
EVENING_FROM = 18


def greeting_for(moment) -> str:
    """"Good morning" / "Good afternoon" / "Good evening" for a datetime."""
    hour = moment.hour
    if MORNING_FROM <= hour < AFTERNOON_FROM:
        return "Good morning"
    if AFTERNOON_FROM <= hour < EVENING_FROM:
        return "Good afternoon"
    return "Good evening"


def display_name(user) -> str:
    """The first name to greet, or an empty string if there isn't one.

    Google stores whatever the account holder typed, which is very often all
    capitals. Shouting someone's name back at them is worse than not using it,
    so a name carrying no case information of its own is title-cased. A name
    that does carry case is left exactly as written: McDonald must not become
    Mcdonald because this function thought it knew better.

    Empty is a real answer. The template greets without a name rather than
    inventing one or falling back to a mangled email address.
    """
    name = (getattr(user, "first_name", "") or "").strip()
    if not name:
        return ""
    if name == name.upper() or name == name.lower():
        return name.title()
    return name
