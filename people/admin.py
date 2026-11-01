"""Admin for profiles and claims.

Heart-rate readings and file records are deliberately NOT registered here, and
nothing on these pages shows a reading: only a count. The link between an account
and a profile can only be changed through the approve, reject and unlink actions,
which go through people.links and so keep its rules (one account per profile and
one profile per account, never replace an existing link).
"""

from django.contrib import admin, messages
from django.db.models import Sum

from . import links
from .models import Person, ProfileClaim

admin.site.site_header = "Health Dashboard administration"
admin.site.site_title = "Health Dashboard admin"


def _report(model_admin, request, done, problems, verb):
    if done:
        model_admin.message_user(request, "{} {}.".format(verb, ", ".join(done)),
                                 messages.SUCCESS)
    for problem in problems:
        model_admin.message_user(request, problem, messages.ERROR)


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("display_name", "device_label", "status_text", "readings",
                    "linked_account")
    fields = ("slug", "display_name", "device_label", "adapter_key", "timezone",
              "notes", "linked_account")
    readonly_fields = ("linked_account",)
    actions = ["unlink_account"]

    def get_queryset(self, request):
        return (super().get_queryset(request).select_related("owner")
                .annotate(_readings=Sum("heart_rate_days__n")))

    def has_delete_permission(self, request, obj=None):
        return False        # deleting a profile would delete its readings

    @admin.display(description="Status")
    def status_text(self, person):
        return person.status

    @admin.display(description="Readings", ordering="_readings")
    def readings(self, person):
        return getattr(person, "_readings", None) or 0

    @admin.display(description="Linked account")
    def linked_account(self, person):
        return person.owner.email if person.owner_id else "not linked"

    @admin.action(description="Unlink the account from the selected profiles")
    def unlink_account(self, request, queryset):
        done, problems = [], []
        for person in queryset:
            try:
                links.unlink_person(person)
                done.append(person.display_name)
            except links.LinkError as exc:
                problems.append(str(exc))
        _report(self, request, done, problems, "Unlinked")


@admin.register(ProfileClaim)
class ProfileClaimAdmin(admin.ModelAdmin):
    list_display = ("account", "person", "status", "created_at", "decided_at")
    list_filter = ("status",)
    list_select_related = ("user", "person")
    actions = ["approve", "reject"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False        # a claim changes only through approve and reject

    @admin.display(description="Account")
    def account(self, claim):
        return claim.user.email

    @admin.action(description="Approve the selected claims (link the account to the profile)")
    def approve(self, request, queryset):
        done, problems = [], []
        for claim in queryset:
            try:
                person = links.approve_claim(claim)
                done.append(person.display_name)
            except links.LinkError as exc:
                problems.append("{}: {}".format(claim.person.display_name, exc))
        _report(self, request, done, problems, "Linked")

    @admin.action(description="Reject the selected claims")
    def reject(self, request, queryset):
        done, problems = [], []
        for claim in queryset:
            try:
                links.reject_claim(claim)
                done.append(claim.person.display_name)
            except links.LinkError as exc:
                problems.append("{}: {}".format(claim.person.display_name, exc))
        _report(self, request, done, problems, "Rejected the claim for")
