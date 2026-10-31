"""Sign-up, login and claim forms.

Email is the login. It is stored lowercased in both `username` and `email`, and
uniqueness is checked case-insensitively, so "A@x.com" and "a@x.com" are one
account. Passwords go through Django's password validators.
"""

from django import forms
from django.contrib.auth import get_user_model, password_validation
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

from people import links
from people.models import Person

User = get_user_model()


def normalise_email(raw):
    return (raw or "").strip().lower()


class ProfileChoiceField(forms.ModelChoiceField):
    """The profiles with no linked account, shown by name only."""

    def label_from_instance(self, person):
        return person.display_name


class ClaimForm(forms.Form):
    person = ProfileChoiceField(
        queryset=Person.objects.none(), empty_label="Choose a profile",
        label="Which profile are you?",
        error_messages={"required": "Choose which profile you are.",
                        "invalid_choice": "That profile is not available."})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["person"].queryset = links.unlinked_people()


class SignupForm(ClaimForm):
    email = forms.CharField(label="Email", max_length=254)
    password1 = forms.CharField(label="Password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))
    password2 = forms.CharField(label="Confirm password", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))

    field_order = ["email", "password1", "password2", "person"]

    def clean_email(self):
        email = normalise_email(self.cleaned_data["email"])
        try:
            validate_email(email)
        except ValidationError:
            raise ValidationError("Enter a valid email address.")
        if (User.objects.filter(username__iexact=email).exists()
                or User.objects.filter(email__iexact=email).exists()):
            raise ValidationError("An account with this email already exists.")
        return email

    def clean(self):
        cleaned = super().clean()
        password1, password2 = cleaned.get("password1"), cleaned.get("password2")
        if password1 and password2 and password1 != password2:
            self.add_error("password2", "The two passwords do not match.")
        elif password1:
            email = cleaned.get("email", "")
            try:
                password_validation.validate_password(
                    password1, user=User(username=email, email=email))
            except ValidationError as exc:
                self.add_error("password1", exc)
        return cleaned

    def save(self):
        """Create the account and a PENDING claim. Never links anything."""
        email = self.cleaned_data["email"]
        user = User.objects.create_user(
            username=email, email=email, password=self.cleaned_data["password1"])
        claim = links.create_claim(user, self.cleaned_data["person"])
        return user, claim


class LoginForm(forms.Form):
    email = forms.CharField(label="Email", max_length=254)
    password = forms.CharField(label="Password", strip=False,
                               widget=forms.PasswordInput(attrs={"autocomplete": "current-password"}))
