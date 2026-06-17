(function () {
  var THEMES = {
    light: {
      background: "transparent",
      primaryColor: "#ffffff",
      primaryBorderColor: "#4f46e5",
      primaryTextColor: "#131a2a",
      secondaryColor: "#eef2ff",
      secondaryBorderColor: "#4f46e5",
      tertiaryColor: "#f6f8fc",
      tertiaryBorderColor: "#e3e8ef",
      lineColor: "#8a94a6",
      textColor: "#131a2a",
      mainBkg: "#ffffff",
      nodeBorder: "#4f46e5",
      clusterBkg: "#f6f8fc",
      clusterBorder: "#e3e8ef",
      edgeLabelBackground: "#f6f8fc",
      actorBkg: "#ffffff",
      actorBorder: "#4f46e5",
      actorTextColor: "#131a2a",
      signalColor: "#8a94a6",
      signalTextColor: "#131a2a",
      labelBoxBkgColor: "#eef2ff",
      labelBoxBorderColor: "#4f46e5",
      noteBkgColor: "#f6f8fc",
      noteBorderColor: "#e3e8ef",
    },
    dark: {
      background: "transparent",
      primaryColor: "#1a2233",
      primaryBorderColor: "#8b8cf7",
      primaryTextColor: "#e7edf6",
      secondaryColor: "#1c2740",
      secondaryBorderColor: "#8b8cf7",
      tertiaryColor: "#10151f",
      tertiaryBorderColor: "#232c3d",
      lineColor: "#6b7789",
      textColor: "#e7edf6",
      mainBkg: "#1a2233",
      nodeBorder: "#8b8cf7",
      clusterBkg: "#10151f",
      clusterBorder: "#232c3d",
      edgeLabelBackground: "#10151f",
      actorBkg: "#1a2233",
      actorBorder: "#8b8cf7",
      actorTextColor: "#e7edf6",
      signalColor: "#6b7789",
      signalTextColor: "#e7edf6",
      labelBoxBkgColor: "#1c2740",
      labelBoxBorderColor: "#8b8cf7",
      noteBkgColor: "#10151f",
      noteBorderColor: "#232c3d",
    },
  };

  function currentScheme() {
    var scheme =
      document.body.getAttribute("data-md-color-scheme") ||
      document.documentElement.getAttribute("data-md-color-scheme");
    return scheme === "slate" ? "dark" : "light";
  }

  function configureMermaid() {
    if (!window.mermaid) {
      return false;
    }

    var variables = THEMES[currentScheme()];
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: Object.assign(
        { fontFamily: "Inter, sans-serif", fontSize: "14px" },
        variables
      ),
    });

    return true;
  }

  function rememberSource(diagram) {
    if (!diagram.dataset.pmbMermaidSource) {
      diagram.dataset.pmbMermaidSource = diagram.textContent.trim();
    }
  }

  function resetDiagram(diagram) {
    rememberSource(diagram);
    diagram.textContent = diagram.dataset.pmbMermaidSource;
    diagram.removeAttribute("data-processed");
  }

  async function renderMermaid() {
    if (!configureMermaid()) {
      return;
    }

    var diagrams = Array.from(document.querySelectorAll(".pmb-mermaid"));
    diagrams.forEach(rememberSource);

    for (var index = 0; index < diagrams.length; index += 1) {
      var diagram = diagrams[index];

      if (diagram.getAttribute("data-processed") === "true") {
        continue;
      }

      var source = diagram.dataset.pmbMermaidSource;
      var target = diagram;

      if (diagram.tagName !== "DIV") {
        target = document.createElement("div");
        target.className = "pmb-mermaid";
        target.dataset.pmbMermaidSource = source;
        diagram.replaceWith(target);
      }

      try {
        var id = "pmb-mermaid-" + Date.now() + "-" + index;
        var result = await window.mermaid.render(id, source);
        target.innerHTML = result.svg;
        target.setAttribute("data-processed", "true");

        if (result.bindFunctions) {
          result.bindFunctions(target);
        }
      } catch (error) {
        target.textContent = source;
        target.classList.add("pmb-mermaid--error");
        console.warn("Mermaid render failed", error);
      }
    }
  }

  function rerenderMermaid() {
    Array.from(document.querySelectorAll(".pmb-mermaid")).forEach(resetDiagram);
    renderMermaid();
  }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(renderMermaid);
  } else {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  }

  // Re-render when the palette toggle (a radio input) flips the color scheme.
  document.addEventListener("change", function () {
    window.setTimeout(rerenderMermaid, 80);
  });
})();
