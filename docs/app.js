/*
 * Renders the demo from measured JSON only.
 *
 * Every figure shown comes from data/probe.json and data/sweep.json, which are
 * written by the tool itself. Nothing is hard-coded here, so the page cannot
 * drift away from what was actually measured: if the data changes, the page
 * changes with it.
 */

const MISSING = -1;

const fmtInt = (n) => n.toLocaleString("en-US");

function fmtSelectivity(s) {
  const pct = s * 100;
  if (pct >= 1) return `${pct.toFixed(0)}%`;
  if (pct >= 0.01) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(4)}%`;
}

async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

/* ---------- section 1: one query, slot by slot ---------- */

function renderProbe(probe) {
  document.getElementById("m-vectors").textContent = fmtInt(probe.n_vectors);
  document.getElementById("m-allowed").textContent = fmtInt(probe.n_allowed);
  document.getElementById("m-sel").textContent = fmtSelectivity(probe.selectivity);
  document.getElementById("m-k").textContent = probe.k;

  const machine = probe.machine || {};
  document.getElementById("m-machine").textContent =
    [machine.os, machine.arch, `${machine.cpu_count} cores`, `Python ${machine.python}`]
      .filter(Boolean)
      .join(" · ");

  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  // Worst first: the failure is the point of the page.
  const ordered = [...probe.probes].sort((a, b) => a.n_correct - b.n_correct);

  for (const entry of ordered) {
    const row = document.createElement("div");
    row.className = "row" + (entry.n_returned < probe.k ? " broken" : "");

    const label = document.createElement("div");
    label.innerHTML =
      `<div class="name">${entry.backend}</div>` +
      `<div class="kind">${entry.index_type}${entry.is_graph_based ? " · graph" : " · exhaustive"}</div>`;

    const slots = document.createElement("div");
    slots.className = "slots";
    for (let i = 0; i < probe.k; i++) {
      const id = entry.returned[i];
      const slot = document.createElement("div");
      if (id === MISSING || id === undefined) {
        slot.className = "slot empty";
        slot.textContent = "—";
        slot.title = `Slot ${i + 1}: nothing returned`;
      } else {
        const hit = entry.correct[i];
        slot.className = "slot " + (hit ? "hit" : "miss");
        slot.textContent = id;
        slot.title = `Slot ${i + 1}: id ${id} — ${hit ? "a true top-10 neighbour" : "not in the true top-10"}`;
      }
      slots.appendChild(slot);
    }

    const tally = document.createElement("div");
    tally.className = "tally";
    const cls = entry.n_correct === probe.k ? "good" : "bad";
    tally.innerHTML =
      `<b class="${cls}">${entry.n_correct}</b> / ${probe.k} correct<br>` +
      `<span class="kind">${entry.n_returned} returned</span>`;

    row.append(label, slots, tally);
    grid.appendChild(row);
  }

  renderVerdict(probe, ordered);
}

function renderVerdict(probe, ordered) {
  const worst = ordered[0];
  const best = ordered[ordered.length - 1];
  const node = document.getElementById("verdict");

  node.innerHTML =
    `<b>${worst.backend}</b> returned <b>${worst.n_returned}</b> of ` +
    `${probe.k} requested results — ${worst.n_correct} of them correct. ` +
    `Given the identical query and the identical ${fmtInt(probe.n_allowed)} allowed vectors, ` +
    `<b>${best.backend}</b> found <b>${best.n_correct}</b>. ` +
    `The neighbours were there. No error was raised.`;
}

/* ---------- section 2: the sweep slider ---------- */

let sweepState = null;

function renderSweep(sweep) {
  const selectivities = [...new Set(sweep.points.map((p) => p.selectivity))].sort((a, b) => b - a);
  sweepState = { sweep, selectivities };

  const slider = document.getElementById("sel");
  slider.max = String(selectivities.length - 1);
  slider.value = "0";
  slider.addEventListener("input", () => drawSweepRow(Number(slider.value)));

  document.getElementById("s-widest").textContent = fmtSelectivity(selectivities[0]);
  document.getElementById("s-narrowest").textContent =
    fmtSelectivity(selectivities[selectivities.length - 1]);

  drawSweepRow(0);
}

function drawSweepRow(index) {
  const { sweep, selectivities } = sweepState;
  const selectivity = selectivities[index];
  const points = sweep.points.filter((p) => p.selectivity === selectivity);

  const allowed = points.length ? points[0].n_allowed : 0;
  document.getElementById("sel-label").textContent =
    `${fmtSelectivity(selectivity)} (${fmtInt(allowed)})`;

  const body = document.getElementById("sweep-body");
  body.innerHTML = "";

  for (const point of [...points].sort((a, b) => a.recall_at_k - b.recall_at_k)) {
    const short = point.mean_returned < sweep.k - 0.05;
    const tr = document.createElement("tr");
    if (short) tr.className = "broken";
    tr.innerHTML =
      `<td>${point.backend}</td>` +
      `<td class="kind">${point.index_type}</td>` +
      `<td class="num">${point.mean_returned.toFixed(1)} / ${sweep.k}</td>` +
      `<td class="num">${point.recall_at_k.toFixed(3)}</td>` +
      `<td class="num">${point.latency_ms.toFixed(2)} ms</td>`;
    body.appendChild(tr);
  }

  const shortfalls = points.filter((p) => p.mean_returned < sweep.k - 0.05);
  const degraded = points.filter(
    (p) => p.mean_returned >= sweep.k - 0.05 && p.recall_at_k < 0.9
  );

  const note = document.getElementById("sweep-note");
  const parts = [];
  if (shortfalls.length) {
    parts.push(
      `${shortfalls.map((p) => p.backend).join(", ")} returned fewer than ${sweep.k} results.`
    );
  }
  if (degraded.length) {
    parts.push(
      `${degraded.map((p) => p.backend).join(", ")} returned a full set, but ` +
        `${degraded.map((p) => `${((1 - p.recall_at_k) * 100).toFixed(0)}%`).join(", ")} of it was wrong — ` +
        `the failure a count-only check misses.`
    );
  }
  note.textContent = parts.length ? parts.join(" ") : "Every index returned a full, correct result set at this width.";
}

/* ---------- boot ---------- */

(async function main() {
  try {
    const [probe, sweep] = await Promise.all([
      loadJSON("data/probe.json"),
      loadJSON("data/sweep.json"),
    ]);
    renderProbe(probe);
    renderSweep(sweep);
  } catch (error) {
    // Say what broke rather than leaving em-dash placeholders on screen.
    document.getElementById("verdict").textContent =
      `Could not load measurement data: ${error.message}`;
    console.error(error);
  }
})();
