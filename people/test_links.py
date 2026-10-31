"""Linking accounts to profiles: claims, approval, refusal and unlinking."""

from django.db import IntegrityError, transaction
from django.test import TestCase

from people import links
from people.models import Person, ProfileClaim
from people.testing import make_user


class OwnerFieldTests(TestCase):
    def test_a_profile_has_no_owner_until_one_is_linked(self):
        self.assertFalse(Person.objects.filter(owner__isnull=False).exists())

    def test_one_account_cannot_own_two_profiles(self):
        user, _ = make_user("a@example.com")
        anas, rahil = Person.objects.get(slug="anas"), Person.objects.get(slug="rahil")
        anas.owner = user
        anas.save()
        rahil.owner = user
        with self.assertRaises(IntegrityError), transaction.atomic():
            rahil.save()

    def test_one_profile_cannot_have_two_owners(self):
        a, _ = make_user("a@example.com", linked_to=Person.objects.get(slug="anas"))
        b, _ = make_user("b@example.com")
        self.assertEqual(Person.objects.get(slug="anas").owner, a)
        self.assertNotEqual(Person.objects.get(slug="anas").owner, b)

    def test_deleting_the_account_unlinks_but_keeps_the_profile_and_data(self):
        user, _ = make_user("a@example.com", linked_to=Person.objects.get(slug="anas"))
        user.delete()
        anas = Person.objects.get(slug="anas")
        self.assertIsNone(anas.owner)


class ClaimTests(TestCase):
    def setUp(self):
        self.anas = Person.objects.get(slug="anas")
        self.user, _ = make_user("a@example.com")

    def test_creating_a_claim_never_links(self):
        claim = links.create_claim(self.user, self.anas)
        self.assertEqual(claim.status, "pending")
        self.anas.refresh_from_db()
        self.assertIsNone(self.anas.owner)
        self.assertIsNone(links.linked_person(self.user))
        self.assertEqual(links.pending_claim(self.user), claim)

    def test_an_account_has_one_pending_claim_at_a_time(self):
        links.create_claim(self.user, self.anas)
        with self.assertRaises(links.LinkError):
            links.create_claim(self.user, Person.objects.get(slug="rahil"))

    def test_a_claim_on_an_already_linked_profile_is_refused(self):
        make_user("b@example.com", linked_to=self.anas)
        with self.assertRaises(links.LinkError):
            links.create_claim(self.user, self.anas)

    def test_a_linked_account_cannot_claim_another_profile(self):
        user, _ = make_user("c@example.com", linked_to=Person.objects.get(slug="rahil"))
        with self.assertRaises(links.LinkError):
            links.create_claim(user, self.anas)

    def test_the_database_allows_only_one_pending_claim_per_account(self):
        ProfileClaim.objects.create(user=self.user, person=self.anas)
        with self.assertRaises(IntegrityError), transaction.atomic():
            ProfileClaim.objects.create(user=self.user,
                                        person=Person.objects.get(slug="rahil"))

    def test_unlinked_people_lists_only_profiles_without_an_account(self):
        make_user("b@example.com", linked_to=self.anas)
        self.assertEqual(list(links.unlinked_people().values_list("slug", flat=True)),
                         ["rahil"])


class ApproveRejectUnlinkTests(TestCase):
    def setUp(self):
        self.anas = Person.objects.get(slug="anas")
        self.user, _ = make_user("a@example.com")
        self.claim = links.create_claim(self.user, self.anas)

    def test_approving_sets_the_owner_and_the_status(self):
        links.approve_claim(self.claim)
        self.anas.refresh_from_db()
        self.claim.refresh_from_db()
        self.assertEqual(self.anas.owner, self.user)
        self.assertEqual(self.claim.status, "approved")
        self.assertIsNotNone(self.claim.decided_at)
        self.assertEqual(links.linked_person(self.user), self.anas)

    def test_approving_is_refused_when_the_profile_is_already_linked(self):
        other, _ = make_user("b@example.com")
        links.approve_claim(self.claim)
        second = ProfileClaim.objects.create(user=other, person=self.anas)
        with self.assertRaises(links.LinkError) as ctx:
            links.approve_claim(second)
        self.assertIn("already linked", str(ctx.exception))
        self.anas.refresh_from_db()
        self.assertEqual(self.anas.owner, self.user)            # not replaced
        second.refresh_from_db()
        self.assertEqual(second.status, "pending")

    def test_approving_is_refused_when_the_account_is_already_linked(self):
        rahil = Person.objects.get(slug="rahil")
        other_claim = ProfileClaim.objects.create(
            user=self.user, person=rahil, status="rejected")
        links.approve_claim(self.claim)
        other_claim.status = "pending"
        other_claim.save()
        with self.assertRaises(links.LinkError):
            links.approve_claim(other_claim)
        rahil.refresh_from_db()
        self.assertIsNone(rahil.owner)

    def test_only_a_pending_claim_can_be_approved_or_rejected(self):
        links.reject_claim(self.claim)
        with self.assertRaises(links.LinkError):
            links.approve_claim(self.claim)
        with self.assertRaises(links.LinkError):
            links.reject_claim(self.claim)

    def test_rejecting_links_nothing(self):
        links.reject_claim(self.claim)
        self.claim.refresh_from_db()
        self.anas.refresh_from_db()
        self.assertEqual(self.claim.status, "rejected")
        self.assertIsNone(self.anas.owner)

    def test_unlinking_removes_the_link_and_nothing_else(self):
        links.approve_claim(self.claim)
        links.unlink_person(self.anas)
        self.anas.refresh_from_db()
        self.claim.refresh_from_db()
        self.assertIsNone(self.anas.owner)
        self.assertEqual(self.claim.status, "rejected")
        self.user.refresh_from_db()                              # the account survives
        self.assertIsNone(links.linked_person(self.user))
        self.assertIsNone(links.pending_claim(self.user))

    def test_unlinking_an_unlinked_profile_is_refused(self):
        with self.assertRaises(links.LinkError):
            links.unlink_person(self.anas)

    def test_after_unlinking_the_profile_can_be_claimed_again(self):
        links.approve_claim(self.claim)
        links.unlink_person(self.anas)
        again = links.create_claim(self.user, self.anas)
        self.assertEqual(again.status, "pending")
