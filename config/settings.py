"""Django settings for the personal health dashboard."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


SECRET_KEY = env("DJANGO_SECRET_KEY", required=True)
DEBUG = env("DJANGO_DEBUG", "False").lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "accounts",
    "health",
    "people",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # After authentication, because it asks whether the request is signed in.
    "accounts.middleware.NoStoreForSignedInPages",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"

# The API reports civil times in the device's own timezone (observed:
# startUtcOffset "-14400s", i.e. UTC-4 / America/New_York in DST). The day-window
# maths has to resolve "today" in that same zone, so this is set deliberately
# rather than left on UTC and happening to agree with the host clock.
TIME_ZONE = env("DJANGO_TIME_ZONE", "America/New_York")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

# ---------------------------------------------------------------------------
# Token encryption at rest
# ---------------------------------------------------------------------------
FERNET_KEY = env("FERNET_KEY", required=True)

# ---------------------------------------------------------------------------
# Google OAuth 2.0 / Google Health API
# ---------------------------------------------------------------------------
GOOGLE_OAUTH_CLIENT_ID = env("GOOGLE_OAUTH_CLIENT_ID", required=True)
GOOGLE_OAUTH_CLIENT_SECRET = env("GOOGLE_OAUTH_CLIENT_SECRET", required=True)
GOOGLE_OAUTH_REDIRECT_URI = env(
    "GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8000/oauth/callback/"
)

# Sign-in identity scopes + the three Google Health read scopes.
# Every health data type this app reads is covered by these three health scopes.
GOOGLE_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
]

GOOGLE_HEALTH_API_BASE = "https://health.googleapis.com/v4"

# ---------------------------------------------------------------------------
# API response cache
# ---------------------------------------------------------------------------
# The overview page needs one API call per metric, so an uncached browser
# refresh costs ten calls. Health readings do not change second to second.
# Set HEALTH_CACHE_SECONDS=0 to disable caching entirely.
HEALTH_CACHE_SECONDS = int(env("HEALTH_CACHE_SECONDS", "600"))

# A civil day that has already ended cannot gain new samples, so the intraday
# drill-down for a past day is held much longer.
HEALTH_CACHE_SECONDS_PAST_DAY = int(env("HEALTH_CACHE_SECONDS_PAST_DAY", "86400"))

# ---------------------------------------------------------------------------
# The WESAD-trained arousal model
# ---------------------------------------------------------------------------
# health/arousal.py runs a scikit-learn model that was trained in a SEPARATE
# project, "wearable-ml". This app reads four files from it, read-only, and
# never writes anything back:
#
#   models/svm_hr_only.joblib            the trained pipeline
#   models/feature_config_hr_only.json   which features, in which order
#   models/hr_training_distribution.json what the training data looked like
#   models/model_card.json               scores and limitations, rendered by
#                                        the "how this works" page
#
# The model is referenced in place rather than copied in here on purpose: one
# copy means retraining cannot leave the dashboard silently serving a stale
# model whose numbers no longer match that project's RESULTS.md.
#
# If the path does not exist the arousal card degrades to an explanation.
# Nothing else in the dashboard depends on it.
WEARABLE_ML_DIR = env("WEARABLE_ML_DIR", str(BASE_DIR.parent / "wearable-ml"))

# How long a computed arousal assessment is cached. It costs far more than a
# plain API read - several pages of samples, then a few thousand windows of
# feature extraction - so it is computed on demand rather than on page load.
AROUSAL_CACHE_SECONDS = int(env("AROUSAL_CACHE_SECONDS", "3600"))

# In-process cache: no extra service to run, and fast. It is per-process and
# does not survive a restart - move to the database or Redis backend if this is
# ever served by more than one worker.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "health-dashboard",
        "TIMEOUT": HEALTH_CACHE_SECONDS,
        "OPTIONS": {"MAX_ENTRIES": 2000},
    }
}

# Google returns the granted scope set in a different order than requested, and
# silently adds "openid" alongside the userinfo scopes. oauthlib treats that as a
# scope change and raises; relaxing the check is the documented workaround.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

if DEBUG:
    # oauthlib refuses to complete a token exchange over plain HTTP. The callback
    # runs on http://localhost during development, which Google explicitly allows.
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

# On a host with broken IPv6, every googleapis.com call stalls ~21s per dead
# AAAA record before falling back to IPv4. See config/net.py for the full story.
FORCE_IPV4 = env("FORCE_IPV4", "False").lower() in {"1", "true", "yes"}

if FORCE_IPV4:
    from .net import force_ipv4

    force_ipv4()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "accounts": {"level": "DEBUG" if DEBUG else "INFO"},
        "health": {"level": "DEBUG" if DEBUG else "INFO"},
    },
}
