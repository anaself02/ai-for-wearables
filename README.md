# Personal Health Dashboard

Django app that reads personal fitness and health data live from the
**Google Health API v4** (`https://health.googleapis.com/v4`).

Server-rendered Django templates, SQLite, no JavaScript build step.

---

## Status

| Build step | State |
|---|---|
| 1. Django skeleton + Sign in with Google | done, needs your client ID/secret to run |
| 2. One view fetching one data type, raw JSON | done (`/raw/exercise/`) |
| 3. Remaining data types | done — 13 of 16 confirmed returning live data (see below) |
| 4. Dashboard UI with Chart.js | done |
| 5. Token auto-refresh | already wired via `AuthorizedSession`; hardening is step 5 |
| Historical VO2 max import | built; run the command to populate (see below) |

---

## One-time setup in Google Cloud Console

**1. Add the redirect URI.** On your existing OAuth 2.0 client
(Credentials → your "Web application" client → Authorized redirect URIs), add
exactly:

```
http://localhost:8000/oauth/callback/
```

Plain `http` is correct: Google exempts `localhost` from its HTTPS-only rule.
Keep the two entries you already have; this is an addition, not a replacement.

**2. Add the sign-in scopes.** On the OAuth consent screen's Data Access page,
add these three alongside the health scopes you already granted. They are
non-sensitive and are what lets the app create a Django `User` record:

```
openid
https://www.googleapis.com/auth/userinfo.email
https://www.googleapis.com/auth/userinfo.profile
```

No additional *health* scope is needed. The three you already have cover every
metric this app reads.

---

## Local setup

```
# from the project root
.venv\Scripts\python.exe -m pip install -r requirements.txt   # already installed
```

Open `.env` and paste your credentials into these two lines:

```
GOOGLE_OAUTH_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=...
```

`DJANGO_SECRET_KEY` and `FERNET_KEY` are already generated. `.env` is
gitignored; nothing secret is committed.

Then:

```
.venv\Scripts\python.exe manage.py migrate
.venv\Scripts\python.exe manage.py runserver
```

Open <http://localhost:8000/>. Use `localhost`, not `127.0.0.1` — the redirect
URI registered with Google is host-specific.

Run the tests with:

```
.venv\Scripts\python.exe manage.py test
```

---

## How the pieces fit

```
accounts/
  crypto.py      Fernet encrypt/decrypt, key from FERNET_KEY
  fields.py      EncryptedTextField - ciphertext in the column, plain str in Python
  models.py      GoogleCredential: one row per user, converts to/from google-auth
  oauth.py       Flow construction, authorization URL, ID-token verification
  views.py       landing / google_login / oauth_callback / sign_out
health/
  datatypes.py   registry of every data type: slug, filter field, scope, quirks
  client.py      GoogleHealthClient over AuthorizedSession (auto-refreshes)
  views.py       index + raw JSON dump per data type
```

### PKCE, and why the session carries two values

google-auth-oauthlib performs PKCE by default. `Flow.authorization_url()`
generates a `code_verifier` **on the Flow object** and sends only its SHA-256
challenge to Google; `Flow.fetch_token()` then has to send the original
verifier back.

The callback arrives as a separate HTTP request with a separate `Flow`
instance, so both the `state` *and* the `code_verifier` are stashed in the
Django session at `/oauth/login/` and popped at `/oauth/callback/`. Losing the
verifier produces exactly this error from Google:

```
(invalid_grant) Missing code verifier.
```

Restarting `runserver` mid-sign-in can also trigger it if the session is
cleared; the callback detects a missing verifier and asks the user to start
again rather than failing opaquely.

### Response caching

The overview needs one API call per metric, so an uncached browser refresh cost
ten calls to Google. `health/cache.py` puts a short-lived cache in front of every
fetch. Measured on this machine:

| page | cold | warm |
|---|---|---|
| `/dashboard/` | 2.26s | **0.24s** |
| `/metric/heart-rate/?days=90` | 4.61s | **0.21s** |
| `/metric/heart-rate/day/<date>/` | 3.11s | **0.30s** |

Every page shows when its data was fetched and carries a **Refresh now** link;
`?refresh=1` on any page deletes that entry and re-fetches.

Two properties are load-bearing and have tests:

- **Keys are per user.** These are personal health readings - a shared key would
  show one person another's data. The user id is part of every key, alongside the
  metric, its source endpoint and the exact window.
- **Failures are never cached.** The value is only written back after a
  successful return, so an error is retried on the next request instead of being
  remembered for the rest of the TTL.

A civil day that has already ended cannot gain new samples, so the intraday
drill-down for a past day is held for 24 hours rather than 10 minutes.

Tuning, via `.env`:

```
HEALTH_CACHE_SECONDS=600        # 0 disables caching entirely
HEALTH_CACHE_SECONDS_PAST_DAY=86400
```

The backend is `LocMemCache`: no extra service, but per-process and empty after a
restart. Switch to the database or Redis backend if this is ever served by more
than one worker.

### Checking the credentials

```
.venv\Scripts\python.exe manage.py check_google_auth
```

Forces a real token refresh against Google and then makes one live API call, so
a bad client secret is caught in seconds rather than during a browser sign-in.
All values are masked in the output. It distinguishes the two failure modes:

- `invalid_client` - the secret in `.env` does not match this OAuth client
- `invalid_grant` - the refresh token expired (see the 7-day note below)

### Rotating the client secret

Google lets one OAuth client hold two secrets at once, so this needs no downtime
and does **not** invalidate existing refresh tokens (they are bound to the client
id, not the secret).

1. Cloud Console -> Credentials -> the OAuth client -> **Add secret**
2. Copy the new value into `GOOGLE_OAUTH_CLIENT_SECRET` in `.env`
3. Restart the server - `runserver`'s autoreloader watches `.py` files but
   **not** `.env`, so an edit alone changes nothing:
   ```
   .venv\Scripts\python.exe manage.py runserver --noreload
   ```
4. `manage.py check_google_auth` - confirms the new secret before you touch the
   old one
5. Only then: Cloud Console -> **delete the old secret**
6. Re-run `check_google_auth` and do one browser sign-in to confirm the full flow

### Timezone

`TIME_ZONE` is `America/New_York`, set deliberately rather than left on UTC.

The API stamps civil times with the recording device's own offset - observed
`"startUtcOffset": "-14400s"`, i.e. UTC-4. Day windows therefore have to resolve
"today" in that same zone. All window maths goes through
`django.utils.timezone.localdate()`, never `date.today()`: the host clock and the
configured zone agree for most of the day but diverge by a full day between local
midnight and UTC midnight, which would silently shift every chart.

Override with `DJANGO_TIME_ZONE` in `.env` if the device reports a different zone.

`accounts.models` deliberately keeps using `timezone.now()` for token expiry -
that comparison is absolute and must not follow the display zone.

### Token storage

Access and refresh tokens are Fernet-encrypted before they reach SQLite. The
key lives only in `.env`, so a copy of `db.sqlite3` on its own is not enough to
call the API. `EncryptedTextField` does this in `get_prep_value` /
`from_db_value`, so application code never sees ciphertext.

Rotating `FERNET_KEY` makes existing tokens unreadable — the app logs a warning
and asks the affected user to reconnect rather than crashing.

### Token refresh

`google.auth.transport.requests.AuthorizedSession` renews an expired access
token from the stored refresh token before the request goes out.
`GoogleHealthClient` compares the token before and after each call and writes a
renewed one back to the database.

When Google *rejects* the refresh token, the client raises
`GoogleReconnectRequired`, flags the row with `needs_reconnect`, and the view
renders a "Reconnect with Google" page instead of a 500.

### The 7-day refresh token

While the OAuth consent screen is in **Testing** mode, Google expires refresh
tokens after **7 days**. Expect to hit the reconnect page about weekly. Moving
the consent screen to "In Production" removes this; refresh tokens then last
until revoked or unused for ~6 months.

---

## Google Health API notes

Facts below were read from the live discovery document
(`https://health.googleapis.com/$discovery/rest?version=v4`, revision
`20260916`), not from memory.

### Two spellings of every data type

The URL path uses kebab-case, filter expressions use snake_case:

```
GET /v4/users/me/dataTypes/daily-resting-heart-rate/dataPoints
  ?filter=daily_resting_heart_rate.date >= "2026-09-01"
```

`datatypes.py` keeps both and a test asserts they stay in sync.

### Which time field is filterable

| Kind | Filter field | Literal format |
|---|---|---|
| interval | `{t}.interval.civil_start_time` | `YYYY-MM-DD[THH:mm:ss]` |
| sample | `{t}.sample_time.civil_time` | `YYYY-MM-DD[THH:mm:ss]` |
| daily | `{t}.date` | `YYYY-MM-DD` |
| session | `{t}.interval.civil_start_time` | `YYYY-MM-DD[THH:mm:ss]` |
| **sleep** | `sleep.interval.civil_end_time` | sleep is filtered on END time only |

Only `>=` and `<` are supported. There is no `BETWEEN` and no `<=`.

### Data types used here

| Metric | dataType | Kind | Scope |
|---|---|---|---|
| Steps | `steps` | interval | activity_and_fitness |
| Distance | `distance` | interval | activity_and_fitness |
| Active minutes | `active-minutes` | interval | activity_and_fitness |
| Exercise sessions | `exercise` | session | activity_and_fitness |
| Heart rate | `heart-rate` | sample | health_metrics |
| Resting HR | `daily-resting-heart-rate` | daily | health_metrics |
| HRV | `heart-rate-variability` | sample | health_metrics |
| HRV daily | `daily-heart-rate-variability` | daily | health_metrics |
| SpO2 | `oxygen-saturation` | sample | health_metrics |
| SpO2 daily | `daily-oxygen-saturation` | daily | health_metrics |
| Respiratory rate | `daily-respiratory-rate` | daily | health_metrics |
| Respiratory rate by stage | `respiratory-rate-sleep-summary` | sample | health_metrics |
| Sleep | `sleep` | session | sleep |
| VO2 max (daily) | `daily-vo2-max` | daily | activity_and_fitness |
| VO2 max (runs) | `run-vo2-max` | sample | activity_and_fitness |
| VO2 max (measured) | `vo2-max` | sample | activity_and_fitness |

### Live confirmation run

30-day window, measured 2026-09-19, all HTTP 200:

| dataType | points | dataType | points |
|---|---|---|---|
| `exercise` | 24 | `daily-heart-rate-variability` | 24 |
| `steps` | 2918 | `oxygen-saturation` | 2678 |
| `distance` | 2078 | `daily-oxygen-saturation` | 24 |
| `active-minutes` | 1532 | `daily-respiratory-rate` | 24 |
| `heart-rate` | 5000 | `respiratory-rate-sleep-summary` | 59 |
| `daily-resting-heart-rate` | 28 | `sleep` | 12 |
| `heart-rate-variability` | 2265 | | |

The three VO2 max types (`daily-vo2-max`, `run-vo2-max`, `vo2-max`) return **zero
points even at 90 days**. That is expected, not a bug: Google Health only derives
VO2 max from outdoor runs recorded with GPS, and the exercise history over that
window contains `WALKING`, `TENNIS`, `SPORT`, `CARDIO_WORKOUT` and `BIKING` —
no running. These will start populating after a tracked outdoor run.

`heart-rate` returning exactly 5000 is a page boundary, not a total; raising
`max_pages` or switching to `dailyRollUp` gets the rest.

### dailyRollUp does not behave like list

Everything below was found by calling the live API, and several points contradict
the reference docs. `health/metrics.py` holds the extractors that deal with it.

**1. Four documented rollup types are actually rejected.** The docs page lists
`daily-resting-heart-rate` and `daily-heart-rate-variability` as rollup-capable.
The API disagrees:

```
HTTP 400  DailyRollup is not supported for data type daily-resting-heart-rate,
          but the following actions are supported: list, reconcile
```

Same for `daily-oxygen-saturation` and `daily-respiratory-rate`. All four are
already one point per day, so `list` is the right call and nothing is lost. The
registry flags were corrected to match the live API.

Verified rollup-capable, and all this dashboard uses: `steps`, `distance`,
`active-minutes`, `heart-rate`.

**2. The date lives somewhere else.** A rollup point carries
`civilStartTime.date`; a list point carries it inside the type's own payload at
`<payloadKey>.date`. No shared path.

**3. The value field is renamed, not reused.**

| | `list` | `dailyRollUp` |
|---|---|---|
| steps | `steps.count` | `steps.countSum` |
| distance | `distance.millimeters` | `distance.millimetersSum` |

**4. `active-minutes` does not roll up to a scalar.** It returns one entry per
activity level, which has to be summed:

```json
"activeMinutes": {"activeMinutesRollupByActivityLevel": [
  {"activityLevel": "LIGHT", "activeMinutesSum": "52"}
]}
```

**5. `heart-rate` rolls up to three values** — `beatsPerMinuteAvg`,
`beatsPerMinuteMin`, `beatsPerMinuteMax` — which is exactly what the overview
chart needs instead of 5000 raw samples.

**6. Types are inconsistent across the same metric.** `list` gives
`beatsPerMinute` as the **string** `"76"`; the rollup gives `beatsPerMinuteAvg`
as the **number** `57.93969891042169`. Ints are strings, floats are numbers, and
both appear in one response. Everything goes through a coercion helper.

**7. Other shape differences.** The response key is `rollupDataPoints`, not
`dataPoints`. Rollup points carry **no `dataSource`**, so the recording device is
unknown there. Civil times come back with `"time": {}` — an empty object rather
than an absent key.

**8. The 14-day cap is real and forces chunking.** A 90-day heart-rate chart is
seven separate requests. Run sequentially that took 19.7s; issued together it is
7.1s. See `metrics.fetch_points`.

### Dashboard structure

```
/dashboard/                       overview, one card per metric
/metric/<slug>/?days=7|30|90      detail chart, table for event metrics
/metric/<slug>/day/<YYYY-MM-DD>/  full-resolution intraday drill-down
/raw/ and /raw/<slug>/            unprocessed JSON, kept as a debugging aid
```

Metrics fall into four presentation groups, set in `health/metrics.py`:

- **daily** — one point per day. Steps, distance and active minutes come from
  `dailyRollUp`; resting HR, HRV, SpO2 and respiratory rate from `list`, because
  the API refuses to roll those up.
- **intraday** — heart rate is charted as a daily average with the day's min and
  max, with the full-resolution trace on its own single-day page. Steps, SpO2 and
  HRV have the same drill-down.
- **event** — exercise and sleep get tables. Sleep also gets a stacked bar per
  night broken down by stage.
- **unavailable** — the three VO2 max types render an empty-state card and make
  no API call at all.

The overview issues ten API calls; they are fanned out over a thread pool, which
takes the page from roughly ten seconds to about 1.7s.

### Historical metrics from Google Takeout

Some metrics cannot be obtained from the live API at all. Those - and only those
- are imported once by hand from a Takeout export. This is never a sync, and
nothing in `health/takeout.py` contacts Google.

```
manage.py import_takeout "<path to the Takeout folder>" [--only X] [--user Y] [--dry-run]
```

The path may be a single export file, its folder, or the Takeout root - the
command searches recursively. Re-running updates in place rather than
duplicating (unique on user + metric type + date), so it is safe after a newer
export. `manage.py import_takeout_vo2max` still works as an alias for
`--only vo2-max`.

**What is imported, and why it has to be**

| Metric | Rows | Why the live API cannot supply it |
|---|---|---|
| Sleep score | 45 | No sleep score endpoint exists. The API returns stages and minute counts only. |
| VO2 max | 57 | The passive `demographic_vo2_max` estimate was retired in the migration; live VO2 max needs a GPS-tracked run. |

**What is deliberately not imported.** Checked against a live 90-day query on
2026-09-19 - the API already serves all of these, so importing them would only
duplicate live data:

| Takeout folder | Rows | Live equivalent |
|---|---|---|
| Active Zone Minutes (AZM) | 1020 | `active-zone-minutes` - 913 points |
| Temperature | 48 | `daily-sleep-temperature-derivations` - 49 points |
| Heart Rate Variability | 4558 | `daily-heart-rate-variability` |
| Oxygen Saturation (SpO2) | 24869 | `daily-oxygen-saturation` |
| Stress Score | **0** | header-only file, nothing to import |
| Daily Readiness | **0** | header-only file, nothing to import |

**Sleep score format** - `Sleep Score/sleep_score.csv`:

```
sleep_log_entry_id,timestamp,overall_score,composition_score,revitalization_score,...
53548224804,2026-09-11T11:56:30Z,78,,79,,109,53,0.157...
```

`overall_score` is the headline 0-100 value; the sub-scores are frequently blank
in real exports and are kept in `extra` rather than charted. The `timestamp` is
UTC at wake time and is converted to the configured timezone before taking its
date, so a night is keyed the same way the live sleep data keys it - by the
civil date it *ends* on. Taking the UTC date directly happens to work for a
morning wake time but breaks across midnight UTC; there is a test for this.

**The file**

```
Takeout/Google Health/Global Export Data/demographic_vo2_max-<date>.json
```

```json
{"dateTime": "07/15/26 00:00:00",
 "value": {"demographicVO2Max": 53.14916,
           "demographicVO2MaxError": 3.0,
           "filteredDemographicVO2Max": 53.14916,
           "filteredDemographicVO2MaxError": 3.0}}
```

Two things about this format are easy to get wrong:

- **`dateTime` is `MM/DD/YY`,** not ISO and not day-first. Confirmed against a real
  export whose records run `07/15/26` to `09/11/26` exactly one day apart - read
  day-first that sequence is impossible. There is a test pinning this.
- **There are two values per record.** `demographicVO2Max` is the raw daily
  estimate; `filteredDemographicVO2Max` is the smoothed series whose error term
  shrinks from 3.0 to 0.36 as readings accumulate, and is what Fitbit displayed
  as the Cardio Fitness Score. The filtered value is stored; the raw one and the
  error term are kept in `HistoricalMetric.extra`.

**The model** - `health.models.HistoricalMetric(user, metric_type, date, value,
unit, source, extra)` - is generic; adding another legacy metric means writing a
parser and one `LegacyImport` entry in `health/takeout.py`.

**In the UI**, an imported reading is never presented as live. Two presentation
groups cover the difference:

- `UNAVAILABLE` - a live endpoint exists but returns nothing for this user (the
  three VO2 max cards). The imported value is shown *beside* the live empty
  state, which still says a GPS-tracked run is required.
- `LEGACY` - no live endpoint exists at all (sleep score). Rendered from stored
  history alone, with an explanation of why there is no live version.

Both get a dashed border, an amber `legacy` tag, and the value's date and source.
Neither makes an API call. All three VO2 max cards read the one `vo2-max` series.

### Performance: broken IPv6 makes every call take ~169 seconds

Symptom: every API-backed page took ~169s and browsers gave up with
`Broken pipe` in the server log. The time was identical regardless of response
size — 3.8 KB and 2.8 MB responses both took 169s — which ruled out pagination,
payload size and rate limiting.

Cause: this host advertises IPv6 but has no working IPv6 route.
`health.googleapis.com` resolves to 8 AAAA records **before** its 8 A records,
and urllib3 walks `getaddrinfo` results strictly in order with no Happy Eyeballs.
Each dead IPv6 address cost ~21s:

```
IPv6  2001:4860:4846:400::   FAILED  20.015s  TimeoutError
... x8 ...
IPv4  172.217.118.4          connect OK  0.046s
```

8 x 21s = ~169s. Browsers and curl hide this because they race both families in
parallel, which is why the same calls were fast in the OAuth Playground.

Fix: `FORCE_IPV4=True` in `.env`, applied by `config/net.py`. After it, the same
pages return in **under 1.3s**. Set it to `False` on a host with working IPv6.

Note that `runserver`'s autoreloader watches `.py` files but **not `.env`** —
after changing `.env` you must restart the server by hand.

### Gotchas worth knowing before step 4

- **`exercise` and `sleep` cap `pageSize` at 25** — both default *and* maximum.
  Everything else defaults to 1440, max 10000.
- **Daily step totals should use `dailyRollUp`, not `list`.** `list` on `steps`
  returns intraday buckets. `POST .../dataPoints:dailyRollUp` aggregates per
  civil day server-side and returns `rollupDataPoints[].steps.countSum`.
- **`dailyRollUp`'s `range` takes `CivilDateTime`, not a bare date:**
  `{"range": {"start": {"date": {"year": 2026, "month": 9, "day": 1}}, ...}}`.
  Several doc pages show the flat form; the discovery document does not.
- **`dailyRollUp` range caps:** 14 days for `heart-rate`, `active-minutes`,
  `total-calories` and `calories-in-heart-rate-zone`; 90 days otherwise.
- **Rolling up `daily-resting-heart-rate` returns a *personal range*** under
  `restingHeartRatePersonalRange`, not a plain daily value — for a resting-HR
  trend chart, plain `list` is the right call. Same for
  `daily-heart-rate-variability`.
- **There is no sleep score.** The API exposes `sleep.summary.minutesAsleep`,
  `minutesInSleepPeriod`, `minutesAwake`, `minutesToFallAsleep`,
  `minutesAfterWakeUp`, a `stages[]` array (LIGHT / DEEP / REM / AWAKE) and
  `stagesSummary[]`. Sleep efficiency must be computed:
  `round(minutesAsleep / minutesInSleepPeriod * 100)`.
- **VO2 max has three data types.** `daily-vo2-max` is the passive daily
  estimate, carrying `cardioFitnessLevel` and an `estimated` flag for
  low-confidence values — this is the one to show on a dashboard.
  `run-vo2-max` only populates from outdoor runs recorded with GPS.
- **Int64 fields arrive as JSON strings** (`steps.count`, `beatsPerMinute`,
  `minutesAsleep`). Cast before doing arithmetic.
