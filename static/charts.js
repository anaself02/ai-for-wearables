/* Builds every Chart.js chart on the page from JSON embedded by Django's
 * json_script filter. Each canvas declares what it needs via data- attributes,
 * so no per-chart inline script is needed.
 *
 *   <canvas data-chart="chart-steps" data-kind="bar" data-unit="steps"></canvas>
 *   {{ series|json_script:"chart-steps" }}
 *
 * Colour comes from app.css and nothing is hardcoded here. A canvas paints
 * pixels rather than inheriting them, so the values are read out of the
 * stylesheet at startup and read again whenever the theme changes.
 */
(function () {
  "use strict";

  var charts = [];

  function token(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return value.trim() || fallback;
  }

  /* The sleep stage colours live on the .seg-* classes rather than in the
   * token block, because they are a data ramp rather than interface colour.
   * Reading them off a probe element keeps the stacked sleep chart and the
   * hypnogram on one set of values instead of two that drift. */
  function probe(className) {
    var el = document.createElement("span");
    el.className = className;
    el.style.display = "none";
    document.body.appendChild(el);
    var colour = getComputedStyle(el).backgroundColor;
    document.body.removeChild(el);
    return colour;
  }

  /* Series colours. The first is ink: a chart with one series is not using
   * colour to say anything, so it does not use any. */
  var PALETTE = [];
  var STAGES = {};
  var GRID = "rgba(128,128,128,0.15)";
  var TICK = "#8d887f";

  function readTheme() {
    PALETTE = [
      token("--ink", "#1b1a18"),
      token("--data-4", "#4550c2"),
      token("--pos", "#0d8a7c"),
      token("--data-5", "#7d45b0"),
      token("--warn", "#a97a12"),
    ];
    STAGES = {
      awake: probe("seg-awake"),
      restless: probe("seg-restless"),
      rem: probe("seg-rem"),
      light: probe("seg-light"),
      deep: probe("seg-deep"),
    };
    GRID = token("--chart-grid", GRID);
    TICK = token("--ink-3", TICK);
  }

  function colourFor(label, index) {
    var stage = String(label || "").toLowerCase();
    if (STAGES[stage]) return STAGES[stage];
    return PALETTE[index % PALETTE.length];
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
      var colour = colourFor(dataset.label, index);
      var faint = dataset.faint === true;
      var base = {
        label: dataset.label,
        data: dataset.data,
        borderColor: colour,
        backgroundColor: colour,
        borderWidth: faint ? 1 : 1.75,
        // Gaps in the data are real (a day with no reading), so leave them open.
        spanGaps: false,
      };

      if (kind === "line") {
        base.tension = 0.25;
        base.pointRadius = 0;
        base.pointHoverRadius = spark ? 0 : 4;
        base.pointHitRadius = spark ? 0 : 12;
        base.borderDash = faint ? [3, 3] : undefined;
        base.fill = false;
      } else {
        base.borderRadius = 2;
        base.borderWidth = 0;
        base.maxBarThickness = spark ? 7 : 26;
      }
      return base;
    });
  }

  function axisFont() {
    return {
      family: token("--mono", "ui-monospace, monospace"),
      size: 11,
    };
  }

  function build(canvas) {
    var series = readSeries(canvas.dataset.chart);
    if (!series || !series.labels || !series.labels.length) return;

    var declared = canvas.dataset.kind || "line";
    var spark = canvas.dataset.spark === "true";
    var stacked = declared === "stacked-bar";
    var kind = declared === "line" ? "line" : "bar";
    var unit = canvas.dataset.unit || "";

    var chart = new Chart(canvas.getContext("2d"), {
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
        layout: { padding: spark ? 0 : 2 },
        plugins: {
          legend: {
            display: !spark && series.datasets.length > 1,
            labels: {
              color: TICK,
              boxWidth: 8,
              boxHeight: 8,
              usePointStyle: true,
              font: axisFont(),
            },
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
                border: { color: GRID },
                grid: { display: false },
                ticks: { color: TICK, font: axisFont(), maxRotation: 0, autoSkipPadding: 16 },
              },
              y: {
                stacked: stacked,
                beginAtZero: kind === "bar",
                border: { display: false },
                grid: { color: GRID, drawTicks: false },
                ticks: { color: TICK, font: axisFont(), padding: 8 },
              },
            },
      },
    });

    charts.push({ chart: chart, spark: spark });
  }

  /* Day to Night, or the system flipping under an Auto setting. Nothing about
   * the data changed, so the charts are recoloured in place rather than
   * rebuilt. */
  function restyle() {
    readTheme();
    charts.forEach(function (entry) {
      var chart = entry.chart;
      chart.data.datasets.forEach(function (dataset, index) {
        var colour = colourFor(dataset.label, index);
        dataset.borderColor = colour;
        dataset.backgroundColor = colour;
      });

      if (!entry.spark) {
        var scales = chart.options.scales;
        scales.x.ticks.color = TICK;
        scales.x.border.color = GRID;
        scales.y.ticks.color = TICK;
        scales.y.grid.color = GRID;
        chart.options.plugins.legend.labels.color = TICK;
      }
      chart.update("none");
    });
  }

  function init() {
    if (typeof Chart === "undefined") return;
    readTheme();
    Chart.defaults.color = TICK;
    Chart.defaults.font.family = token("--sans", "system-ui, sans-serif");
    document.querySelectorAll("canvas[data-chart]").forEach(build);
    window.addEventListener("themechange", restyle);

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
