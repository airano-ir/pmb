/* PMB docs — scroll-reveal.
   Tags structural content blocks with .pmb-reveal, then reveals each as it
   enters the viewport (IntersectionObserver). Re-runs on Material's instant
   navigation via the document$ observable. Reduced-motion is handled in CSS. */
(function () {
  var SEL = [
    "h2", "h3",
    ".grid", ".pmb-cards", ".pmb-feature-grid", ".pmb-signal-grid",
    ".pmb-widget-grid", ".pmb-note-grid", ".pmb-mermaid",
    "table", ".admonition", "details", "blockquote", ".highlight"
  ].join(", ");

  function setup() {
    var root = document.querySelector(".md-content__inner");
    if (!root) return;

    var els = Array.prototype.slice.call(root.querySelectorAll(SEL))
      .filter(function (el) { return !el.closest(".pmb-reveal"); });
    if (!els.length) return;

    els.forEach(function (el) { el.classList.add("pmb-reveal"); });

    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("pmb-in"); });
      return;
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("pmb-in");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });

    els.forEach(function (el) { io.observe(el); });

    // Safety net: never leave content invisible if the observer misfires.
    setTimeout(function () {
      els.forEach(function (el) { el.classList.add("pmb-in"); });
    }, 1400);
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(setup);
  } else if (document.readyState !== "loading") {
    setup();
  } else {
    document.addEventListener("DOMContentLoaded", setup);
  }
})();
