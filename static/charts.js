/* Builds every Chart.js chart on the page from JSON embedded by Django's
 * json_script filter. Each canvas declares what it needs via data- attributes,
 * so no per-chart inline script is needed.
 *
 *   <canvas data-chart="chart-steps" data-kind="bar" data-unit="steps"></canvas>
 *   {{ series|json_script:"chart-steps" }}
 */
(function () {
  "use strict";

  /* Chart colours come from app.css. Chart.js paints to a canvas and cannot
   * resolve var() itself, so the tokens are read once at startup instead of
   * being copied here - which is what they were, and a second copy of a
   * palette drifts from the first the moment either is touched.
   *
   * The fallbacks are the same values app.css holds, and only apply if the
   * stylesheet has not loaded by the time this runs.
   */
  function token(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return value.trim() || fallback;
  }

  var PALETTE = ["#6ea8fe", "#45d9c0", "#f0b849", "#f2777a", "#b58cf0"];
  var GRID = "rgba(255,255,255,0.06)";
  var TICK = "#949db4";

  function readTheme() {
    PALETTE = [
      token("--accent", PALETTE[0]),
      token("--pos", PALETTE[1]),
      token("--warn", PALETTE[2]),
      token("--neg", PALETTE[3]),
      token("--series-5", PALETTE[4]),
    ];
    GRID = token("--chart-grid", GRID);
    TICK = token("--muted", TICK);
  }

  function readSeries(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (err) {
      return null;
    }
  }

  function shortLabel(value) {
    // "2026-09-19" -> "09-19"; clock labels pass through untouched.
    return /^\d{4}-\d{2}-\d{2}$/.test(value) ? value.slice(5) : value;
  }

  function buildDatasets(series, kind, spark) {
    return series.datasets.map(function (dataset, index) {
      var colour = PALETTE[index % PALETTE.length];
      var faint = dataset.faint === true;
      var base = {
        label: dataset.label,
        data: dataset.data,
        borderColor: colour,
        backgroundColor: colour,
        borderWidth: faint ? 1.5 : 2,
        // Gaps in the data are real (a day with no reading), so leave them open.
        spanGaps: false,
      };

      if (kind === "line") {
        base.tension = 0.32;
        base.pointRadius = spark ? 0 : 2.5;
        base.pointHoverRadius = 4;
        base.borderDash = faint ? [4, 4] : undefined;
        base.fill = false;
      } else {
        base.borderRadius = 4;
        base.borderWidth = 0;
        base.maxBarThickness = spark ? 10 : 34;
      }
      return base;
    });
  }

  function build(canvas) {
    var series = readSeries(canvas.dataset.chart);
    if (!series || !series.labels || !series.labels.length) return;

    var declared = canvas.dataset.kind || "line";
    var spark = canvas.dataset.spark === "true";
    var stacked = declared === "stacked-bar";
    var kind = declared === "line" ? "line" : "bar";
    var unit = canvas.dataset.unit || "";

    new Chart(canvas.getContext("2d"), {
      type: kind,
      data: {
        labels: series.labels.map(shortLabel),
        datasets: buildDatasets(series, kind, spark),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: spark ? false : { duration: 300 },
        interaction: { mode: "index", intersect: false },
        layout: { padding: spark ? 0 : 4 },
        plugins: {
          legend: {
            display: !spark && series.datasets.length > 1,
            labels: { color: TICK, boxWidth: 10, boxHeight: 10, usePointStyle: true },
          },
          tooltip: {
            enabled: !spark,
            callbacks: {
              label: function (ctx) {
                var value = ctx.parsed.y;
                if (value === null || value === undefined) return ctx.dataset.label + ": no data";
                return ctx.dataset.label + ": " + value + (unit ? " " + unit : "");
              },
            },
          },
        },
        scales: spark
          ? { x: { display: false }, y: { display: false } }
          : {
              x: {
                stacked: stacked,
                grid: { display: false },
                ticks: { color: TICK, maxRotation: 0, autoSkipPadding: 14 },
              },
              y: {
                stacked: stacked,
                beginAtZero: kind === "bar",
                grid: { color: GRID },
                ticks: { color: TICK },
              },
            },
      },
    });
  }

  function init() {
    if (typeof Chart === "undefined") return;
    readTheme();
    Chart.defaults.color = TICK;
    Chart.defaults.font.family =
      'ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif';
    document.querySelectorAll("canvas[data-chart]").forEach(build);

    // Day picker on the metric detail page.
    var picker = document.querySelector("[data-day-picker]");
    if (picker) {
      picker.addEventListener("change", function () {
        if (picker.value) window.location = picker.dataset.dayPicker + picker.value + "/";
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
