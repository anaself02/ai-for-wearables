# Health Dashboard — project context

A handoff document. Read this before changing anything; it explains not just
what the project is, but the decisions it has already made and why, so that
they do not get quietly undone.

---

## What this is

A personal health dashboard: a Django web app that signs in to Google with
OAuth and reads **one person's own fitness and health data live from the Google
Health API**, then presents it. Steps, heart rate, resting heart rate, HRV,
blood oxygen, respiratory rate, distance, active minutes, exercise sessions and
sleep — including per-night sleep stages with real timestamps.

It is not a product. One user, no customers, no multi-tenancy, no deployment.
It runs locally against `db.sqlite3` with a `.env` beside it, and it is somebody's
own dashboard over their own body's data.

**Two sources of data, and the difference matters throughout the UI:**

- **Live**, fetched from the API on page load through a short cache. Ten of the
  fourteen registered metrics.
- **Imported**, a one-time load from a Google Takeout export of Fitbit-era data,
  stored in `HistoricalMetric`. These can never update, so the interface labels
  them `legacy` and styles them with a dashed border. Dashed means exactly one
  thing across the whole app: *not live measured data.*

Some cards are permanently empty and say why. VO2 max is the clearest case: the
live API only derives it from GPS-tracked outdoor runs, and this account has
none, so all three VO2 max metrics sit in an `unavailable` group that renders an
explanation instead of a number. The only VO2 max figure that exists is the
Fitbit demographic estimate in the Takeout import — which is what makes the
fitness age partly circular (see Known gaps).

### What each page does

| page | what it is for |
|---|---|
| `/` | Logged-out front door. Mark, one sentence, Sign in with Google. |
| `/dashboard/` | The overview. One card per metric, current value and a week-long sparkline. **Measurements only** — nothing inferred. |
| `/insights/` | Everything the app *derives*: trends, correlations with significance testing and multiple-comparison correction, the sleep-debt arithmetic, the arousal signal, and methodology notes. Kept off the overview on purpose. |
| `/fitness/` | Where one VO2 max figure sits in published population norms for age and sex, plus a fitness age beside the calendar age. |
| `/sleep-calculator/` | Given a wake time, bedtimes counted in whole sleep cycles — using a cycle length measured from your own stage data when it clears the gates, beside the textbook 90 minutes. |
| `/metric/<slug>/` | One metric over a window: chart, and a session or night table where that applies. |
| `/metric/sleep/night/<date>/` | The hypnogram. Per-stage bands built from real elements, not a canvas, so hover text and screen-reader output come from the markup. |
| `/raw/<slug>/` | The unprocessed API response, to prove the pipeline is not inventing anything. |
| `/settings/` | Date of birth and sex — needed only because the fitness norms are age- and sex-specific. |

### How the data flows

```
Google OAuth  ->  GoogleCredential (Fernet-encrypted tokens in SQLite)
                        |
                  health/client.py          token refresh, reconnect handling
                        |
                  health/cache.py           read-through, in-memory, ~10 min
                        |
                  health/metrics.py         one extractor per metric, raw JSON
                        |                   -> {labels, datasets} for charts
          +-------------+-------------+
          |                           |
   views.index                  views.insights
   (cards)                      health/insights.py   trends, correlations
                                health/arousal.py    WESAD model on heart rate
                                health/fitness.py    norms, percentile
                                health/sleep_cycles.py  REM-to-REM intervals
```

The overview and the insights page call the same `_load_window()` and share a
cache key, so whichever is opened second costs no API calls.

The **arousal signal** is worth flagging as unusual: it runs a support-vector
model trained on the WESAD research dataset (laboratory-induced stress, 28
subjects) against the day's wrist heart rate. The model is trained in a separate
project; this app reads its `model_card.json` so the figures on the methodology
page cannot drift from the training project's own results. It runs on request,
not on page load, because it scores a few thousand windows.

---

## The one thing that defines this project

**It refuses to overclaim.** Every derived number on screen is shown with what
it rests on, and the project repeatedly chooses to say "this cannot be known"
rather than print a plausible figure. This is not decoration — it is the
organising principle, and several features exist only in the shape they do
because of it:

- The **arousal card** runs a WESAD-trained model against wrist heart rate. Its
  headline and its caveat are rendered by the same template branch, deliberately,
  so no code path can print the label without the caveat. A test enforces that.
- The **insights page** shows statistically weak results rather than hiding
  them, labelled "not distinguishable from chance", because a tested-and-weak
  result is more defensible than silence. It states its own multiple-comparison
  correction and notes where two metrics are definitionally entangled.
- The **fitness page** prints a percentile from a published norm table and says
  on the page that the table was transcribed from a secondary source and
  spot-checked on two of 84 values — because that is what actually happened.
- The **sleep calculator** measures a personal cycle length but also reports
  the evidence that the number is biased upward, and refuses to prefer it to
  the textbook figure until it clears explicit gates.

If you add a feature, match this standard. Do not add composite scores. The
dashboard already inherits one (Fitbit's sleep score) that it cannot explain,
and the fitness page needed three cards to make one number honest.

---

## Stack and hard constraints

- **Django**, server-rendered templates. Python 3.13 in `.venv/`.
- **No npm, no build step, no bundler.** Deliberate, longstanding decision.
- **No webfonts.** System font stacks only (`--sans`, `--mono`). This is why
  the Google sign-in button deviates from Google's reference CSS on font.
- **Chart.js from CDN, pinned**, loaded in `templates/health/_chart_scripts.html`.
- **SQLite**, one user, local only. No remote git.
- Windows host; use `.venv\Scripts\python.exe`.

---

## Layout

```
accounts/     auth, OAuth, profile, middleware
health/       everything about the data
config/       settings, urls
templates/    all HTML
static/       app.css, charts.js, theme.js
docs/         ETHICS_AND_PRIVACY.md
```

### Key modules

| file | what it holds |
|---|---|
| `health/metrics.py` | The metric registry. Every metric, its slug, its extractor, its chart type. Also the hypnogram builder. Start here to understand the data. |
| `health/client.py` | Google Health API client, token refresh, `GoogleReconnectRequired`. |
| `health/cache.py` | Short-lived read-through cache. **In-memory only** (locmem) — see Known gaps. |
| `health/insights.py` | The derived-statistics layer: trends, correlations, significance, methodology notes. |
| `health/arousal.py` | The WESAD model runner and its out-of-distribution check. |
| `health/fitness.py` | VO2 max percentile and fitness age, against one published norm table. |
| `health/sleep_cycles.py` | Cycle length measured from REM-onset to REM-onset, plus bedtime arithmetic. |
| `health/views.py` | All health views. `_load_window()` is the shared fetch both the overview and insights use. |
| `accounts/models.py` | `GoogleCredential` (Fernet-encrypted tokens) and `Profile` (encrypted DOB, sex). |
| `accounts/middleware.py` | `no-store` on signed-in responses. |

### Routes

```
/                          landing (logged out) / redirect to dashboard
/dashboard/                overview — measurements only
/insights/                 derived layer + arousal signal + methodology
/fitness/                  VO2 max percentile and fitness age
/sleep-calculator/         bedtimes by whole sleep cycles
/metric/<slug>/            one metric, chart + table
/metric/<slug>/day/<date>/ intraday detail
/metric/sleep/night/<date>/ the hypnogram
/raw/  /raw/<slug>/        unprocessed API responses
/settings/                 date of birth + sex
/oauth/login|callback|logout/
```

---

## The design system

Rebuilt from scratch in `d263a56`. Three rules, stated at the top of
`static/app.css`:

1. **Colour means data.** The interface is monochrome — ink on ground for
   text, rules, fills, the pressed state, the current tab. The only coloured
   things on a page carry information: a sleep stage, the direction an insight
   moved, a failed request. **There is no `--accent` token.** It was deleted.
   Do not reintroduce one.
2. **Numbers are mono, prose is not.** Every figure uses `--mono` with tabular
   figures so a column can be compared without being read.
3. **Structure is hairlines, not boxes.** The cards are one ruled grid: a 1px
   gap with each cell painting its own hairline into it via `box-shadow`, so no
   cell needs to know whether it is on an edge and a half-empty last row leaves
   no slab of divider colour behind.

**Tokens** live in one `:root` block: `--s1..--s7` spacing (4px scale),
`--r-0/1/2/3/pill` radius, an integer type scale, `--ink/--ink-2/--ink-3`,
`--ground/--surface/--surface-2/--rule/--rule-strong`, and the data colours
`--pos/--warn/--neg/--data-4/--data-5`.

**Day and Night are both real.** Tokens are defined three times: light on bare
`:root`, dark under `@media (prefers-color-scheme: dark)` guarded with
`:root:not([data-theme="light"])`, and dark again under `:root[data-theme="dark"]`
so the toggle wins in both directions. A pinned choice is applied by an inline
script in `<head>` before first paint, so the page never flashes the wrong theme.
`static/theme.js` drives the Day/Night/Auto control and fires a `themechange`
event; `charts.js` listens and recolours in place.

**The mark** (`templates/_mark.html`) is three offset bars — the sleep timeline
reduced until it reads at 16px. Monochrome `currentColor`. One SVG serves the
top bar (18px), the landing page (56px) and the favicon (24px on a tile).

**The Google sign-in button** follows Google's published branding guidelines,
not this app's button style, because that is a condition of using the mark. The
"G" is the official asset pasted in unmodified; `accounts/test_landing.py` pins
all five paths and colours character-for-character so a reformatter cannot
silently alter someone else's trademark. Two documented deviations: font (no
webfonts) and corner radius (one radius scale).

---

## Known gaps and open threads

**1. Nothing persists.** This is the biggest structural gap. Only the Takeout
imports live in the database (`HistoricalMetric`). Everything else is fetched
live into an in-memory cache that dies with the process. Consequences: no trend
longer than the API window, no page renders when the token has expired, analysis
is not reproducible without a live call, and the overview fans out ~10 API calls.
A `sync_daily` command writing daily rollups to the DB would be the highest-value
addition and would improve nearly everything else.

**2. The ACSM norm table is unverified.** `health/fitness.py` `NORMS` was
transcribed from a secondary web rendering of The Cooper Institute ACLS table as
published in ACSM's Guidelines for Exercise Testing and Prescription, 11th ed.,
Wolters Kluwer, 2021, Table 4.7. **Two of 84 values** were spot-checked against a
different secondary rendering. Nobody has opened the book. The page says this in
those words and a test pins the wording. When the 11th edition is available:
check `NORMS` against Table 4.7, correct any drift, then update the wording — the
test will fail and force the update to be deliberate.

**3. The measured sleep cycle is biased upward by an unknown amount.** Measured
over 60 real nights: median 110 min, 119 cycles across 42 nights, split-half
agreement within 2 min. But 10 of 134 raw intervals exceeded 150 minutes (up to
296), which proves REM periods are sometimes missed — and a miss landing under
the 150 ceiling is counted as one long cycle. Also the first cycle of a night is
not shorter than later ones (110 vs 109) when clinically it should be. Both facts
are computed and shown on the page rather than living in a comment. Re-run the
audit with `manage.py estimate_sleep_cycles --days 60`.

**4. Fitness age is partly circular, by construction.** The only VO2 max
available is Google's `demographicVO2Max`, which Google derives from resting
heart rate, age, sex and weight. Age goes in, so age comes out. Both numbers are
still shown, because the comparison is the point, but the page says to read the
**gap** rather than the absolute number — age largely cancels in the subtraction,
leaving resting heart rate and weight. A genuinely non-circular fitness age needs
a VO2 max measured from a GPS-tracked outdoor run; the code flips automatically
(`measured=True`) if one ever appears.

**5. Small ones.** No 404 template. Logged-out visitors have no theme control
(it lives in the authenticated top bar). The arousal model runs on one day, on
request, rather than over a window.

---

## Working on it

```
.venv\Scripts\python.exe manage.py runserver     # .env already sets DJANGO_DEBUG=True
.venv\Scripts\python.exe manage.py test          # 321 tests, all passing
.venv\Scripts\python.exe manage.py estimate_sleep_cycles --days 60
```

Use **`localhost:8000`, not `127.0.0.1:8000`** — the Google redirect URI is
registered against `localhost` and sign-in fails otherwise.

`runserver` reloads on `.py` changes and templates need only a browser refresh,
but **CSS changes need a hard refresh** (Ctrl+F5) or you will be served the
cached stylesheet and think your change did not land.

### Conventions

- Conventional Commits. Messages explain *why*, not what — read the last dozen
  for the register. They are long on purpose.
- Every commit must leave the suite green on its own, not just at the tip.
- Comments explain decisions and trade-offs, not mechanics.
- Tests pin **behaviour that reviews cannot catch**: that a caveat renders in the
  same branch as its headline, that a trademark is unmodified, that a citation
  keeps saying how far it was checked, that `{# #}` comments do not span lines
  (that one shipped a bug — Django's `{# #}` is single-line only and the text
  rendered onto the page).
- Do not weaken a test to make a change pass. If the behaviour moved, move the
  assertion and say so.

### Screenshot harness

There is a throwaway harness pattern used to review design changes: render the
real views with the API faked exactly as `health/test_dashboard.py` fakes it,
against a Django test database (so `db.sqlite3` is never touched), serve the
output with `python -m http.server`, and screenshot with headless Edge
(`--headless=new --screenshot=... --window-size=...`). No Playwright or
chromium-cli is installed. Pin `data-theme` on `<html>` to shoot both themes.

---

## Where it stands

321 tests passing. Recent history, newest first:

```
1201667 fix(auth): stop the back button showing pages after sign-out
3193ca0 feat(landing): build a real front door, with one mark and Google's own button
df82cfc feat(sleep): report the evidence behind the measured cycle length
0195a37 fix(fitness): read both numbers off one table, and say what was checked
3216203 feat(fitness): show the fitness age beside the calendar age
b62bfe7 feat: add a fitness percentile page and a sleep cycle calculator
4719504 style(nav): make the section tabs buttons
d263a56 feat(ui): rebuild the interface as an instrument panel, in two themes
60619fc feat(insights): give the derived layer its own page
```

`0195a37` is worth reading as an example of the project's standard: the
percentile and the fitness age had been read off two different cohorts whose
means differ by ~6 ml/kg/min, so the page could call one reading above average
for its age and older than its age in the same panel. The fix was to read both
off one table, and a test now pins the invariant that makes the contradiction
impossible.
