"""Remove the user rows the retired Google sign-in created.

Signing in with Google created a Django user per Google account, with the Google
email as its username and an unusable password (nobody could ever log in with
it). Those rows hold a name and email and are no longer needed; left in place
the email would also block that person from creating an account of their own.

Only rows that are clearly leftovers are removed: an UNUSABLE password (stored
as a string starting with "!"), and neither staff nor superuser. Any account that
has a usable password, and every staff or superuser account, is left alone.
"""

from django.conf import settings
from django.db import migrations


def remove_google_users(apps, schema_editor):
    User = apps.get_model(*settings.AUTH_USER_MODEL.split("."))
    User.objects.filter(password__startswith="!", is_staff=False,
                        is_superuser=False).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("people", "0003_drop_oauth_tables"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [migrations.RunPython(remove_google_users, migrations.RunPython.noop)]
