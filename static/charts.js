/* Builds every Chart.js chart on the page from JSON embedded by Django's
 * json_script filter. Each canvas declares what it needs via data- attributes,
 * so no per-chart inline script is needed.
 *
 *   <canvas data-chart="chart-steps" data-kind="bar" data-unit="steps"></canvas>
 *   {{ series|json_script:"chart-steps" }}
 */
(function () {
  "use strict";

  var PALETTE = ["#6ea8fe", "#45d9c0", "#f0b849", "#f2777a", "#b58cf0"];
  var GRID = "rgba(255,255,255,0.06)";
  var TICK = "#949db4";

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

  /* ---- hypnogram -------------------------------------------------------
   * One horizontal band per sleep stage, each stage's segments drawn across a
   * shared time axis.
   *
   * Every segment is its own dataset with `grouped: false`, so they all draw at
   * full band height in their own row instead of being bunched side by side.
   * The x axis is minutes from bedtime (linear) rather than a time scale: that
   * needs no date adapter, and the tick callback prints real clock times from
   * the night's local start minute.
   */

  var STAGE_COLOURS = {
    Awake: "#f0b849",
    REM: "#b58cf0",
    Light: "#6ea8fe",
    Deep: "#3f6fd8",
  };

  function clockAt(minuteOfDay) {
    var total = ((Math.round(minuteOfDay) % 1440) + 1440) % 1440;
    var h = Math.floor(total / 60);
    var m = total % 60;
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m;
  }

  function durationLabel(minutes) {
    var whole = Math.round(minutes);
    var h = Math.floor(whole / 60);
    var m = whole % 60;
    return h ? h + "h " + (m < 10 ? "0" : "") + m + "m" : m + "m";
  }

  function buildHypnogram(canvas) {
    var data = readSeries(canvas.dataset.hypnogram);
    if (!data || !data.segments || !data.segments.length) return;

    var rowLabels = data.rows.map(function (row) {
      return row.label + " — " + row.duration;
    });

    var datasets = data.segments.map(function (segment) {
      var cells = data.rows.map(function (_, index) {
        return index === segment.row ? [segment.start, segment.end] : null;
      });
      return {
        label: segment.stage,
        data: cells,
        backgroundColor: STAGE_COLOURS[segment.stage] || "#6ea8fe",
        borderWidth: 0,
        borderSkipped: false,
        borderRadius: 2,
        grouped: false,
        barPercentage: 1,
        categoryPercentage: 0.72,
      };
    });

    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: { labels: rowLabels, datasets: datasets },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            callbacks: {
              title: function (items) {
                return items[0].dataset.label;
              },
              label: function (ctx) {
                var pair = ctx.raw;
                if (!pair) return "";
                return (
                  clockAt(data.start_minute_of_day + pair[0]) +
                  " – " +
                  clockAt(data.start_minute_of_day + pair[1]) +
                  "  (" +
                  durationLabel(pair[1] - pair[0]) +
                  ")"
                );
              },
            },
          },
        },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: data.total_minutes,
            grid: { color: GRID },
            ticks: {
              color: TICK,
              maxRotation: 0,
              autoSkipPadding: 18,
              // Hourly marks read far better than arbitrary minute counts.
              stepSize: data.total_minutes > 480 ? 120 : 60,
              callback: function (value) {
                return clockAt(data.start_minute_of_day + value);
              },
            },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#e8ebf3", font: { size: 12.5 } },
          },
        },
      },
    });
  }

  function init() {
    if (typeof Chart === "undefined") return;
    Chart.defaults.color = TICK;
    Chart.defaults.font.family =
      'ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif';
    document.querySelectorAll("canvas[data-chart]").forEach(build);
    document.querySelectorAll("canvas[data-hypnogram]").forEach(buildHypnogram);

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
