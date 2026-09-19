from django.contrib import admin

from .models import GoogleCredential


@admin.register(GoogleCredential)
class GoogleCredentialAdmin(admin.ModelAdmin):
    list_display = ("user", "email", "expires_at", "needs_reconnect", "last_refreshed_at")
    list_filter = ("needs_reconnect",)
    search_fields = ("user__username", "email", "google_sub")
    # Never surface decrypted tokens in the admin UI.
    exclude = ("access_token", "refresh_token")
    readonly_fields = ("google_sub", "scopes", "created_at", "updated_at", "last_refreshed_at")
