"""The one form this project has: the two facts the fitness norms need."""

from datetime import date

from django import forms

from .models import Profile

#: Nobody using a wearable was born before this, and a typo in the year is the
#: most likely way a nonsense age gets in.
EARLIEST_YEAR = 1900


class ProfileForm(forms.ModelForm):
    date_of_birth = forms.DateField(
        required=True,
        label="Date of birth",
        widget=forms.DateInput(attrs={"type": "date"}),
        help_text="Stored encrypted. Kept as a date rather than an age so it stays correct.",
    )
    sex = forms.ChoiceField(
        required=True,
        choices=Profile.Sex.choices,
        label="Sex",
        help_text=(
            "The published VO2 max norms have two columns, so the comparison "
            "can only be made against one of them. That is a limit of the "
            "reference tables, not a claim about people."
        ),
    )

    class Meta:
        model = Profile
        fields = ["date_of_birth", "sex"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The model stores an ISO string, because the column is ciphertext.
        instance = kwargs.get("instance") or getattr(self, "instance", None)
        if instance is not None and instance.born:
            self.fields["date_of_birth"].initial = instance.born

    def clean_date_of_birth(self):
        born = self.cleaned_data["date_of_birth"]
        today = date.today()
        if born > today:
            raise forms.ValidationError("That date is in the future.")
        if born.year < EARLIEST_YEAR:
            raise forms.ValidationError("That date looks like a typo in the year.")
        return born

    def save(self, commit=True):
        profile = super().save(commit=False)
        profile.date_of_birth = self.cleaned_data["date_of_birth"].isoformat()
        if commit:
            profile.save()
        return profile
