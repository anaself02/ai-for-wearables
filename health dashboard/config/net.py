"""Outbound network policy.

This machine advertises IPv6 but has no working IPv6 route to the internet.
``health.googleapis.com`` resolves to eight AAAA records followed by eight A
records, and Python's urllib3 walks ``getaddrinfo`` results strictly in order
with no Happy Eyeballs. Every API call therefore spent ~21s timing out on each
dead IPv6 address before reaching IPv4 - about 169 seconds per request,
regardless of how much data was being fetched.

Browsers and curl hide this because they race the two families in parallel,
which is why the same requests are fast in the OAuth Playground.
"""

import logging

logger = logging.getLogger(__name__)


def force_ipv4():
    """Make every outbound requests/urllib3 call resolve to IPv4 only.

    urllib3 picks its address family from ``allowed_gai_family()``, which
    returns ``AF_INET`` when ``HAS_IPV6`` is false. Flipping that flag is the
    supported lever; there is no per-session address-family option.

    Process-wide by design: the Google Health calls and the OAuth token
    refreshes both need it.
    """
    import urllib3.util.connection

    urllib3.util.connection.HAS_IPV6 = False
    logger.info("Forcing IPv4 for outbound HTTP (FORCE_IPV4 is enabled).")
