"""The add / edit form for a labelled session.

Times are entered in the person's local time (America/Toronto by default) and
stored in UTC. The note is optional, short, and stays in the local database.
"""

from zoneinfo import ZoneInfo

from django import forms

from . import sessions
from .models import Session

RATINGS = [("", "No rating")] + [(str(n), str(n)) for n in range(1, 6)]


class SessionForm(forms.Form):
    date = forms.DateField(label="Date", widget=forms.DateInput(attrs={"type": "date"}))
    start_time = forms.TimeField(label="Start time",
                                 widget=forms.TimeInput(attrs={"type": "time"}))
    end_time = forms.TimeField(label="End time",
                               widget=forms.TimeInput(attrs={"type": "time"}))
    kind = forms.ChoiceField(label="Kind", choices=Session.Kind.choices)
    rating = forms.TypedChoiceField(label="How stressed did you feel? (1-5, optional)",
                                    choices=RATINGS, coerce=int, empty_value=None,
                                    required=False)
    note = forms.CharField(label="Note (optional, stays on this machine)",
                           max_length=200, required=False)

    def __init__(self, *args, person, instance=None, **kwargs):
        self.person = person
        self.instance = instance
        if instance is not None and not args and "initial" not in kwargs:
            start = instance.start_utc.astimezone(ZoneInfo(person.timezone))
            end = instance.end_utc.astimezone(ZoneInfo(person.timezone))
            kwargs["initial"] = {"date": start.date(), "start_time": start.time(),
                                 "end_time": end.time(), "kind": instance.kind,
                                 "rating": instance.rating or "", "note": instance.note}
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned = super().clean()
        day, start, end = (cleaned.get("date"), cleaned.get("start_time"),
                           cleaned.get("end_time"))
        if day is None or start is None or end is None:
            return cleaned
        try:
            cleaned["start_utc"] = sessions.local_to_utc(day, start, self.person.timezone)
            cleaned["end_utc"] = sessions.local_to_utc(day, end, self.person.timezone)
            sessions.check_span(self.person, cleaned["start_utc"], cleaned["end_utc"],
                                exclude_pk=getattr(self.instance, "pk", None))
        except sessions.SessionError as exc:
            raise forms.ValidationError(str(exc))
        return cleaned

    def save(self):
        session = self.instance or Session(person=self.person)
        session.kind = self.cleaned_data["kind"]
        session.start_utc = self.cleaned_data["start_utc"]
        session.end_utc = self.cleaned_data["end_utc"]
        session.rating = self.cleaned_data["rating"]
        session.note = self.cleaned_data["note"]
        session.save()
        return session
