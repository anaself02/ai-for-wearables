"""Model field that transparently encrypts its value at rest."""

from django.db import models

from .crypto import decrypt, encrypt


class EncryptedTextField(models.TextField):
    """A TextField whose value is Fernet-encrypted in the database.

    Python-side code reads and writes plain strings; only the column contents
    are ciphertext. Because the stored value is opaque, this field cannot be
    filtered on or indexed meaningfully - which is fine for OAuth tokens.
    """

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if not value:
            return value
        return encrypt(value)

    def from_db_value(self, value, expression, connection):
        if not value:
            return value
        return decrypt(value)
