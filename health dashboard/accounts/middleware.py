"""Keep signed-in pages out of the browser's cache."""

#: no-store is the part that matters. no-cache alone permits the browser to
#: keep a copy as long as it revalidates, and the back/forward cache does not
#: revalidate; no-store forbids keeping it at all, which also opts the page out
#: of the bfcache.
NO_STORE = "no-store, no-cache, must-revalidate, max-age=0"


class NoStoreForSignedInPages:
    """Stop the back button showing a signed-out visitor the last page they saw.

    Signing out genuinely ends the session: afterwards every protected view
    redirects to the front door and serves no data. What the back button shows
    is not a live session, it is the browser replaying HTML it was allowed to
    keep, because nothing in the response told it not to.

    On a shared machine the distinction does not help anybody. Somebody's
    resting heart rate is on the screen after they signed out, and whether it
    arrived from the network or from a cache is not the point.

    Applied to every response for a signed-in request rather than by
    decorating the views, because the failure mode of the decorator approach is
    forgetting it on the one view added six months from now.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            response.headers["Cache-Control"] = NO_STORE
            # For anything old enough to ignore Cache-Control.
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response
