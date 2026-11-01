"""Keep signed-in pages out of the browser's cache.

After logging out, the back button replays whatever the browser kept. Every
response to a signed-in request is therefore marked no-store.
"""

from django.utils.cache import add_never_cache_headers

NO_STORE = "max-age=0, no-cache, no-store, must-revalidate, private"


class NoStoreForSignedIn:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if request.user.is_authenticated:
            add_never_cache_headers(response)
        return response
