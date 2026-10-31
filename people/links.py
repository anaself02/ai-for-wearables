"""Linking an account to a profile: the only code that sets `Person.owner`.

A link is made by an administrator approving a claim, never by the account
itself. Each rule here exists to keep two people's data apart:

  * one account per profile, and one profile per account (also enforced by the
    database, since `Person.owner` is a one-to-one);
  * approving a claim for a profile that already has an owner is refused, not
    silently replaced;
  * unlinking removes the link and nothing else; it never touches readings.

Nothing here logs an email address.
"""

from django.db import transaction
from django.utils import timezone

from .models import Person, ProfileClaim


class LinkError(Exception):
    """A link change was refused; the message is shown to the administrator."""


def pending_claim(user):
    """The account's pending claim, or None."""
    if user is None or not user.is_authenticated:
        return None
    return user.claims.filter(status=ProfileClaim.Status.PENDING).first()


def linked_person(user):
    """The profile this account is linked to, or None."""
    if user is None or not user.is_authenticated:
        return None
    return Person.objects.filter(owner=user).first()


def unlinked_people():
    """Profiles with no linked account, for the sign-up choice (names only)."""
    return Person.objects.filter(owner__isnull=True)


def create_claim(user, person):
    """Record an account's request for a profile. Never links anything."""
    if person.owner_id is not None:
        raise LinkError("That profile already has a linked account.")
    if linked_person(user) is not None:
        raise LinkError("This account is already linked to a profile.")
    if pending_claim(user) is not None:
        raise LinkError("This account already has a pending request.")
    return ProfileClaim.objects.create(user=user, person=person)


@transaction.atomic
def approve_claim(claim):
    """Link the claim's account to its profile. Refuses rather than replaces."""
    claim = ProfileClaim.objects.select_for_update().get(pk=claim.pk)
    if claim.status != ProfileClaim.Status.PENDING:
        raise LinkError("Only a pending claim can be approved.")

    person = Person.objects.select_for_update().get(pk=claim.person_id)
    if person.owner_id is not None:
        raise LinkError("{} is already linked to an account.".format(person.display_name))
    if Person.objects.filter(owner=claim.user).exists():
        raise LinkError("This account is already linked to a profile.")

    person.owner = claim.user
    person.save(update_fields=["owner"])
    claim.status = ProfileClaim.Status.APPROVED
    claim.decided_at = timezone.now()
    claim.save(update_fields=["status", "decided_at"])
    return person


@transaction.atomic
def reject_claim(claim):
    claim = ProfileClaim.objects.select_for_update().get(pk=claim.pk)
    if claim.status != ProfileClaim.Status.PENDING:
        raise LinkError("Only a pending claim can be rejected.")
    claim.status = ProfileClaim.Status.REJECTED
    claim.decided_at = timezone.now()
    claim.save(update_fields=["status", "decided_at"])
    return claim


@transaction.atomic
def unlink_person(person):
    """Remove a profile's link. The account keeps existing but sees no data."""
    person = Person.objects.select_for_update().get(pk=person.pk)
    if person.owner_id is None:
        raise LinkError("{} is not linked to an account.".format(person.display_name))
    ProfileClaim.objects.filter(
        user_id=person.owner_id, person=person,
        status=ProfileClaim.Status.APPROVED,
    ).update(status=ProfileClaim.Status.REJECTED, decided_at=timezone.now())
    person.owner = None
    person.save(update_fields=["owner"])
    return person
