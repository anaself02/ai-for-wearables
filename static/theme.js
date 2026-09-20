/* Day / Night / Auto.
 *
 * Auto is the default and means "whatever the system says", which is the only
 * setting that can be right at both ends of a day. Day and Night pin the
 * choice and survive a reload; base.html applies a pinned choice inline,
 * before the first paint, so the page never flips theme in front of you.
 *
 * Anything that paints its own colours rather than inheriting them - the
 * charts - listens for the themechange event this fires.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "theme";
  var root = document.documentElement;

  function stored() {
    try {
      var value = localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : "auto";
    } catch (err) {
      return "auto";
    }
  }

  function remember(choice) {
    try {
      if (choice === "auto") {
        localStorage.removeItem(STORAGE_KEY);
      } else {
        localStorage.setItem(STORAGE_KEY, choice);
      }
    } catch (err) {
      /* The choice still applies to this page; it just will not outlive it. */
    }
  }

  function apply(choice) {
    if (choice === "auto") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", choice);
    }
    markCurrent(choice);
    window.dispatchEvent(new CustomEvent("themechange"));
  }

  function markCurrent(choice) {
    var buttons = document.querySelectorAll("[data-theme-set]");
    Array.prototype.forEach.call(buttons, function (button) {
      var isCurrent = button.dataset.themeSet === choice;
      button.setAttribute("aria-pressed", isCurrent ? "true" : "false");
    });
  }

  function init() {
    markCurrent(stored());

    document.addEventListener("click", function (event) {
      var button = event.target.closest("[data-theme-set]");
      if (!button) return;
      var choice = button.dataset.themeSet;
      remember(choice);
      apply(choice);
    });

    // On Auto, the system can change under us - at sunset, or on a schedule.
    var media = window.matchMedia("(prefers-color-scheme: dark)");
    var onSystemChange = function () {
      if (stored() === "auto") window.dispatchEvent(new CustomEvent("themechange"));
    };
    if (media.addEventListener) {
      media.addEventListener("change", onSystemChange);
    } else if (media.addListener) {
      media.addListener(onSystemChange);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
