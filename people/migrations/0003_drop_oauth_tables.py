"""Drop the tables left behind by the retired Google sign-in.

The app no longer signs in with Google, so the tables that held Google OAuth
tokens (encrypted) and the date of birth and sex used for fitness norms are
dead weight on a database that predates the change, and the tokens in
particular should not linger. `IF EXISTS` makes this a no-op on a database that
never had them, such as a fresh checkout.

This is raw SQL because the `accounts` app and its models no longer exist, so
there is nothing for a DeleteModel operation to refer to. Nothing else has a
foreign key to either table.
"""

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("people", "0002_seed_people")]

    operations = [
        migrations.RunSQL(
            sql=["DROP TABLE IF EXISTS accounts_googlecredential;",
                 "DROP TABLE IF EXISTS accounts_profile;"],
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
