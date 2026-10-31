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
    "health",
    "people",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
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

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "people": {"level": "DEBUG" if DEBUG else "INFO"},
        "health": {"level": "DEBUG" if DEBUG else "INFO"},
    },
}

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
# Files wait here, under a random token, between the validation report and the
# confirmation. Deleted on confirm, on cancel, or when an hour old. Gitignored.
UPLOAD_SPOOL_DIR = BASE_DIR / "uploads_tmp"

# Django's default of 100 files per request would refuse the Heart Rate folder of
# a real Takeout export, which holds about a hundred daily files.
DATA_UPLOAD_MAX_NUMBER_FILES = 1000

# Total size of one upload. The Heart Rate folder is ~110 MB; the whole export is
# over a gigabyte and is refused with a message asking for only the folder.
MAX_UPLOAD_BYTES = 600 * 1024 * 1024
