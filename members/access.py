"""Who may open what.

  * Anonymous visitors are sent to /login/ for everything except /signup/ and
    /login/ (and the static files those pages need).
  * Staff may open the person chooser, any person, and /admin/.
  * A non-staff account with a linked profile is sent from / to its own profile
    and may open only /p/<own slug>/... ; any other slug is a 404, as if it did
    not exist. This is enforced here AND again where a person is resolved
    (people.views.person_page), so a new route cannot forget it.
  * A non-staff account with no linked profile sees only the waiting page and no
    data. A pending claim changes nothing until an administrator approves it.
  * Everything else is a 404 for a non-staff account.
"""

from django.conf import settings
from django.http import Http404, HttpResponseNotFound
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.http import urlencode

from people import links

ANONYMOUS_PATHS = ("/signup/", "/login/")
SIGNED_IN_PATHS = ("/logout/", "/waiting/", "/claim/")


def can_view(user, person):
    """Staff may view anyone; an account only the profile it is linked to."""
    if not user.is_authenticated:
        return False
    return bool(user.is_staff) or person.owner_id == user.pk


def person_or_404(user, person):
    if not can_view(user, person):
        raise Http404("No such profile.")
    return person


def _slug(path):
    parts = path.strip("/").split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "p" else None


class AccessControl:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        user = request.user

        if path in ANONYMOUS_PATHS or path.startswith(settings.STATIC_URL):
            return self.get_response(request)

        if not user.is_authenticated:
            target = reverse("members:login")
            if request.method == "GET" and path != "/":
                target += "?" + urlencode({"next": request.get_full_path()})
            return redirect(target)

        if user.is_staff:
            return self.get_response(request)

        # --- a signed-in account that is not staff ---
        if path.startswith("/admin"):
            return HttpResponseNotFound()
        if path in SIGNED_IN_PATHS:
            return self.get_response(request)

        own = links.linked_person(user)
        if own is None:
            if path == "/" or path.startswith("/p/"):
                return redirect("members:waiting")
            return HttpResponseNotFound()

        if path == "/":
            return redirect("people:person", slug=own.slug)
        if path.startswith("/p/") and _slug(path) == own.slug:
            return self.get_response(request)
        return HttpResponseNotFound()
