# Ethics and privacy

Covers both halves of this project: the Django dashboard in this repository,
which collects the data, and the `wearable-ml` project, which trains the
model and holds the derived results. Every claim below points at the code
that implements it.

## 1. What is collected, and why

Purpose: a graded course project for *AI for Wearable Health Applications*.
The data is my own, read from my own Google account, and is used to build and
evaluate a stress/affect classifier and to demonstrate it end to end. It is
not used for anything else.

Sixteen data types are registered in `health/datatypes.py`, all read-only:

| Scope | Data types |
|---|---|
| `…googlehealth.activity_and_fitness.readonly` | `exercise`, `steps`, `distance`, `active-minutes`, `daily-vo2-max`, `run-vo2-max`, `vo2-max` |
| `…googlehealth.health_metrics_and_measurements.readonly` | `heart-rate`, `daily-resting-heart-rate`, `heart-rate-variability`, `daily-heart-rate-variability`, `oxygen-saturation`, `daily-oxygen-saturation`, `daily-respiratory-rate`, `respiratory-rate-sleep-summary` |
| `…googlehealth.sleep.readonly` | `sleep` |

Plus `openid`, `userinfo.email` and `userinfo.profile` for sign-in. All three
identity scopes are actually consumed: `sub` and `email` identify the account
(`accounts/views.py`), and `given_name`/`family_name` populate the Django user
record. Nothing reads a profile photo.

Only `heart-rate` reaches the model. The arousal card
(`health/arousal.py`) uses wrist heart rate alone; the other fifteen types
exist for the dashboard's charts.

## 2. How it is secured

**Tokens encrypted at rest.** `GoogleCredential.access_token` and
`refresh_token` are `EncryptedTextField` (`accounts/fields.py`), which
Fernet-encrypts on write and decrypts on read via `accounts/crypto.py`
(AES-128-CBC with an HMAC). The key lives in `FERNET_KEY` in `.env`, never in
the database and never in the repository — so a stolen copy of `db.sqlite3`
on its own cannot call the API. Rotating the key makes stored tokens
undecryptable and forces a reconnect, which `decrypt()` handles deliberately
rather than crashing.

**No plaintext secrets in version control.** `.gitignore` excludes `.env`
(client secret, Django secret key, Fernet key) and `*.sqlite3` (the encrypted
tokens and cached readings). Verified: neither is tracked.

**Scoped, read-only API access.** Three health scopes, all `.readonly`, plus
sign-in. No write scopes, and nothing touching Gmail, Drive, Calendar or
Google account settings. "Connect Google" here does not mean full account
access.

**Local only.** The dashboard runs on `localhost` (`ALLOWED_HOSTS` is
`["localhost", "127.0.0.1"]`) and stores to local SQLite. Outbound traffic
goes to Google and nowhere else: `health.googleapis.com` for readings,
`oauth2.googleapis.com` for the token exchange and refresh,
`accounts.google.com` for the consent redirect, and Google's certificate
endpoint when an ID token is verified. No analytics, no third-party
services, no telemetry. The model runs in-process from a local file, so no
readings are sent anywhere for inference.

**Derived data stays local too.** In `wearable-ml`, everything computed from
my readings is written under `results/personal/` and excluded by that
project's `.gitignore`, so personal heart-rate data never enters a repository
that might be pushed. The tracked report `RESULTS.md` was deliberately split
to keep it free of personal readings.

## 3. Consent

Access is granted through Google's own OAuth 2.0 consent screen, which lists
each scope before anything is shared (`accounts/oauth.py`, using
`google-auth-oauthlib` with PKCE rather than a hand-rolled exchange). The
consent screen is in **Testing** mode, so Google expires the refresh token
roughly every seven days and consent must be given again — an unusually short
leash, and the app handles that expiry explicitly rather than hiding it
(`GoogleReconnectRequired` in `health/client.py`).

I can revoke access at any time at
[myaccount.google.com/permissions](https://myaccount.google.com/permissions),
which invalidates the refresh token immediately.

**Known gap, stated plainly:** there is no in-app "disconnect" button.
`accounts/views.py:sign_out` clears the Django session only — it does not
delete the stored credential. Revocation today means Google's settings page
plus deleting the row or the database file. A disconnect view that deletes
`GoogleCredential` and calls Google's revoke endpoint is the obvious next
addition.

## 4. PIPEDA and Quebec Law 25

Neither statute binds a personal, non-commercial project like this one —
PIPEDA governs commercial activity, and Law 25 governs enterprises. The
principles are still the right frame for the design, and the code follows
them:

- **Purpose limitation.** Data is used only to build and evaluate this course
  project's classifier. No secondary use, no sharing, no profiling of anyone.
- **Data minimization.** Three read-only health scopes, all exercised; no
  write access; no scope requested "just in case". Only heart rate is fed to
  the model. `health/cache.py` also limits repeat collection — a finished day
  is cached rather than re-fetched.
- **Consent.** Explicit, informed, scope-by-scope, and withdrawable at
  Google's side at any time.
- **Right of access and erasure.** The data subject is me, and the store is a
  local SQLite file. Access is direct (`/raw/` shows the API payloads
  verbatim); erasure is deleting `db.sqlite3` and `results/personal/`. There
  is no third party holding a copy to petition — a genuine privacy advantage
  of keeping this local rather than deploying it.
- **Safeguards proportionate to sensitivity.** Health data is sensitive
  information under both regimes, which is why tokens are encrypted at rest
  and secrets are kept out of version control even for a single-user local
  app.

## 5. What this is not

- **Not a commercial product.** No users, no customers, no revenue, no terms
  of service.
- **Not handling anyone else's data.** One account — mine. The schema allows
  multiple users (`OneToOneField` per user, per-user cache keys in
  `health/cache.py`) so that a collision could never expose one person's
  readings to another, but only one person's data exists here.
- **Not deployed.** `localhost` only; `ALLOWED_HOSTS` is `["localhost",
  "127.0.0.1"]`. Nothing is exposed to the internet.
- **Not a medical device, and not a validated stress detector.** The model was
  trained on laboratory-induced stress in 15 people, scores 0.499 macro-F1
  under leave-one-subject-out cross-validation on wrist heart rate alone, and
  is measurably out of distribution on my device's data. Its output is
  labelled low confidence everywhere it appears and cannot be constructed
  without that caveat attached (`health/arousal.py`). See
  `/arousal/how-this-works/` and `wearable-ml/RESULTS.md`.

If this ever became multi-user or publicly deployed, essentially all of the
above would have to be revisited: hosted storage, a real privacy policy, a
breach-response plan, and Law 25's transparency and consent obligations would
then genuinely apply.
