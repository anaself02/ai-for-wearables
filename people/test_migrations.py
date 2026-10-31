"""Data migrations that clean up after features that were removed."""

from importlib import import_module

from django.apps import apps
from django.contrib.auth import get_user_model
from django.test import TestCase

User = get_user_model()


class RemoveGoogleSignInUsersTests(TestCase):
    def run_migration(self):
        module = import_module("people.migrations.0004_remove_google_sign_in_users")
        module.remove_google_users(apps, None)

    def test_only_unusable_password_non_staff_leftovers_are_removed(self):
        leftover = User.objects.create(username="old-google-user")          # unusable
        leftover.set_unusable_password()
        leftover.save()
        member = User.objects.create(username="member")
        member.set_password("a-usable-synthetic-passphrase-1")
        member.save()
        staff = User.objects.create(username="staff", is_staff=True)
        staff.set_unusable_password()
        staff.save()
        root = User.objects.create(username="root", is_superuser=True)
        root.set_unusable_password()
        root.save()

        self.run_migration()

        self.assertEqual(set(User.objects.values_list("username", flat=True)),
                         {"member", "staff", "root"})

    def test_it_is_safe_to_repeat_and_to_run_on_an_empty_table(self):
        self.run_migration()
        self.run_migration()
        self.assertEqual(User.objects.count(), 0)
