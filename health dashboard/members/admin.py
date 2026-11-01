"""Admin for accounts: who is linked to which profile, and who is waiting.

Django's stock user admin is replaced on purpose. It has a password field that
displays part of the stored hash, and an "add user" form that sets passwords;
this one shows neither. Accounts are created only by signing up (or by
`createsuperuser`), never from here.
"""

from django.contrib import admin, messages
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group

from people import links
from people.models import ProfileClaim

User = get_user_model()

admin.site.unregister(User)
admin.site.unregister(Group)


class PendingClaimFilter(admin.SimpleListFilter):
    title = "pending claims"
    parameter_name = "pending"

    def lookups(self, request, model_admin):
        return [("yes", "Waiting for approval"), ("no", "No pending claim")]

    def queryset(self, request, queryset):
        pending = ProfileClaim.objects.filter(
            status=ProfileClaim.Status.PENDING).values("user_id")
        if self.value() == "yes":
            return queryset.filter(pk__in=pending)
        if self.value() == "no":
            return queryset.exclude(pk__in=pending)
        return queryset


@admin.register(User)
class AccountAdmin(admin.ModelAdmin):
    list_display = ("email", "linked_profile", "pending_claim_for", "is_active",
                    "last_login")
    list_filter = (PendingClaimFilter, "is_active", "is_staff")
    search_fields = ("email",)
    ordering = ("email",)
    fields = ("email", "linked_profile", "pending_claim_for", "is_active", "is_staff",
              "last_login", "date_joined")
    readonly_fields = ("email", "linked_profile", "pending_claim_for", "last_login",
                       "date_joined")
    actions = ["approve_claims", "reject_claims", "unlink_accounts"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("person")

    def get_readonly_fields(self, request, obj=None):
        fixed = list(super().get_readonly_fields(request, obj))
        if not request.user.is_superuser:       # only a superuser grants or revokes access
            fixed += ["is_active", "is_staff"]
        return fixed

    def has_add_permission(self, request):
        return False        # accounts are made by signing up, never from here

    @admin.display(description="Linked profile")
    def linked_profile(self, user):
        person = getattr(user, "person", None)
        return person.display_name if person else "not linked"

    @admin.display(description="Pending claim")
    def pending_claim_for(self, user):
        claim = links.pending_claim(user)
        return claim.person.display_name if claim else "none"

    def _each(self, request, queryset, action, verb):
        done, problems = [], []
        for user in queryset:
            try:
                done.append(action(user))
            except links.LinkError as exc:
                problems.append(str(exc))
        if done:
            self.message_user(request, "{} {}.".format(verb, ", ".join(done)),
                              messages.SUCCESS)
        for problem in problems:
            self.message_user(request, problem, messages.ERROR)

    @admin.action(description="Approve the selected accounts' pending claims")
    def approve_claims(self, request, queryset):
        def approve(user):
            claim = links.pending_claim(user)
            if claim is None:
                raise links.LinkError("An account has no pending claim to approve.")
            return links.approve_claim(claim).display_name
        self._each(request, queryset, approve, "Linked")

    @admin.action(description="Reject the selected accounts' pending claims")
    def reject_claims(self, request, queryset):
        def reject(user):
            claim = links.pending_claim(user)
            if claim is None:
                raise links.LinkError("An account has no pending claim to reject.")
            links.reject_claim(claim)
            return claim.person.display_name
        self._each(request, queryset, reject, "Rejected the claim for")

    @admin.action(description="Unlink the selected accounts from their profiles")
    def unlink_accounts(self, request, queryset):
        def unlink(user):
            person = links.linked_person(user)
            if person is None:
                raise links.LinkError("An account is not linked to a profile.")
            return links.unlink_person(person).display_name
        self._each(request, queryset, unlink, "Unlinked")
