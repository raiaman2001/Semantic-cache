const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const LBL = {
  L1_exact: "L1 Exact Hit",
  L2_semantic: "L2 Semantic Hit",
  L3_llm: "L3 LLM Call"
};

let lastInspection = null;

async function api(path, method = "GET", body) {
  const h = { "Content-Type": "application/json" };
  const key = $("#apikey").value.trim();
  if (key) h["x-api-key"] = key;

  const r = await fetch("/api" + path, {
    method,
    headers: h,
    body: body ? JSON.stringify(body) : undefined
  });

  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    const msg = e.detail || `Request failed (${r.status})`;
    toast(msg, true);
    throw new Error(msg);
  }
  return r.json();
}

function toast(m, isError = false) {
  const t = $("#toast");
  t.textContent = m;
  t.style.background = isError ? "var(--bad)" : "var(--ink)";
  t.style.color = isError ? "#fff" : "var(--bg)";
  t.classList.add("on");
  setTimeout(() => t.classList.remove("on"), 3000);
}

// Accessible tab navigation
document.querySelectorAll("#tabs button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(e => {
      e.classList.remove("on");
      e.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll("main .tab-pane").forEach(e => e.classList.remove("on"));

    b.classList.add("on");
    b.setAttribute("aria-selected", "true");
    const target = $("#" + b.dataset.t);
    if (target) target.classList.add("on");

    ({ stats: loadStats, entries: loadEntries })[b.dataset.t]?.();
  };
});

function updateInspector(r, query) {
  lastInspection = { r, query };
  const badge = $("#inspect-badge");
  badge.className = `badge ${r.level}`;
  badge.textContent = LBL[r.level] || r.level;

  let reason = "";
  if (r.level === "L1_exact") {
    reason = "Exact normalized hash match found in Tier 1 cache (Redis / SQLite). Instant sub-millisecond retrieval with 0 LLM tokens spent.";
  } else if (r.level === "L2_semantic") {
    reason = `Vector cosine similarity (${r.score}) exceeded category threshold (${r.threshold}). Lexical antonym guard and Cross-Encoder reranker approved the match.`;
  } else {
    reason = "No exact match or semantic match above threshold. LLM generated response and populated L1/L2 caches for future queries.";
    if (r.coalesced) {
      reason += " [Coalesced: Shared execution with an identical in-flight request].";
    }
  }

  $("#inspector-content").innerHTML = `
    <div class="inspector-grid">
      <div class="inspector-item"><b>Resolved Level</b><span>${LBL[r.level]}</span></div>
      <div class="inspector-item"><b>Latency</b><span>${r.ms} ms</span></div>
      <div class="inspector-item"><b>Cosine Similarity</b><span>${r.score > 0 ? r.score : "N/A (Miss)"}</span></div>
      <div class="inspector-item"><b>Active Threshold</b><span>${r.threshold} (${r.category})</span></div>
      <div class="inspector-item"><b>Coalesced</b><span>${r.coalesced ? "Yes (Shared LLM)" : "No"}</span></div>
      <div class="inspector-item"><b>Log ID</b><span>${r.log_id || "N/A"}</span></div>
    </div>
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--line);">
      <b>Routing Explanation:</b>
      <p style="margin:4px 0 0;color:var(--ink);">${reason}</p>
    </div>
  `;
}

function setMeta(box, r, query) {
  box.className = `msg ${r.level}`;
  box.querySelector(".meta").innerHTML = `
    <span class="badge ${r.level}">${LBL[r.level]}</span> · <strong>${r.ms} ms</strong>
    ${r.level === "L2_semantic" ? ` · similarity <strong>${r.score}</strong> (thr ${r.threshold}, ${r.category})` : ""}
    ${r.coalesced ? " · <em>coalesced with in-flight request</em>" : ""}
  `;

  updateInspector(r, query);

  // Feedback buttons for L2 matches
  if (r.level === "L2_semantic" && r.log_id) {
    const f = document.createElement("div");
    f.className = "row";
    f.style.marginTop = "10px";
    f.innerHTML = `
      <button class="btn ghost" style="padding:6px 12px;font-size:13px;" data-g="1">👍 Good match</button>
      <button class="btn ghost" style="padding:6px 12px;font-size:13px;" data-g="0">👎 False match</button>
    `;
    f.querySelectorAll("button").forEach(x => {
      x.onclick = async () => {
        const isGood = x.dataset.g === "1";
        await api("/feedback", "POST", { log_id: r.log_id, good: isGood });
        toast(isGood ? "Feedback recorded: Threshold relaxed slightly." : "Feedback recorded: Threshold tightened.");
        f.remove();
      };
    });
    box.append(f);
  }
}

async function ask() {
  const qInput = $("#q");
  const q = qInput.value.trim();
  if (!q) return;

  const btn = $("#go");
  btn.disabled = true;
  btn.classList.add("loading");

  const body = {
    query: q,
    ns: $("#ns").value.trim() || "default",
    doc_ver: $("#ver").value.trim() || "v1",
    tag: $("#tag").value.trim(),
    use_cache: $("#usec").checked
  };

  const box = document.createElement("div");
  box.className = "msg";
  box.innerHTML = `
    <b>${esc(q)}</b>
    <div class="meta mute">Resolving query...</div>
    <div class="ans"></div>
  `;

  const chat = $("#chat");
  const empty = chat.querySelector(".empty-state");
  if (empty) empty.remove();

  chat.prepend(box);
  qInput.value = "";
  const ans = box.querySelector(".ans");

  try {
    if ($("#stream").checked) {
      const h = { "Content-Type": "application/json" };
      const key = $("#apikey").value.trim();
      if (key) h["x-api-key"] = key;

      const resp = await fetch("/api/ask/stream", {
        method: "POST",
        headers: h,
        body: JSON.stringify(body)
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Stream request failed");
      }

      const rd = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const p of parts) {
          if (!p.startsWith("data: ")) continue;
          const o = JSON.parse(p.slice(6));
          if (o.meta) setMeta(box, o.meta, q);
          if (o.t) ans.textContent += o.t;
        }
      }
    } else {
      const r = await api("/ask", "POST", body);
      setMeta(box, r, q);
      ans.textContent = r.answer;
    }
  } catch (e) {
    box.className = "msg";
    box.style.borderLeftColor = "var(--bad)";
    box.querySelector(".meta").innerHTML = `<span style="color:var(--bad);font-weight:600;">Error: ${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    qInput.focus();
  }
}

$("#go").onclick = ask;
$("#q").onkeydown = e => {
  if (e.key === "Enter") {
    e.preventDefault();
    ask();
  }
};

const tbl = (el, head, rows) => {
  $(el).innerHTML = rows.length
    ? `<thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`
    : `<tbody><tr><td colspan="${head.length}" class="empty-state" style="text-align:center;padding:24px;">No records available.</td></tr></tbody>`;
};

async function loadStats() {
  try {
    const s = await api("/stats");
    const L = s.levels;
    const ms = k => (L[k] ? Math.round(L[k].avg_ms) + " ms" : "-");

    $("#kpis").innerHTML = [
      ["Cache Hit Rate", (s.hit_rate * 100).toFixed(1) + "%"],
      ["Total Requests", s.total],
      ["Cached Entries", s.entries],
      ["Tokens Saved", s.tokens_saved.toLocaleString()],
      ["Cost Saved", "$" + s.cost_saved.toFixed(4)],
      ["Avg L1 / L2 / LLM", `${ms("L1_exact")} / ${ms("L2_semantic")} / ${ms("L3_llm")}`]
    ].map(([l, v]) => `
      <div class="kpi">
        <b>${v}</b>
        <span>${l}</span>
      </div>
    `).join("");

    $("#thr").innerHTML = Object.entries(s.thresholds).map(([k, v]) => `
      <label>${k}: <input type="number" step="0.01" min="0.5" max="0.99" id="t_${k}" value="${v}" style="width:85px"></label>
    `).join("") + `<button class="btn" id="savethr">Save Thresholds</button>`;

    $("#savethr").onclick = async () => {
      await api("/threshold", "POST", { code: +$("#t_code").value, general: +$("#t_general").value });
      toast("Thresholds updated successfully");
    };

    tbl(
      "#recent",
      ["Query", "Routing Result", "Cosine Score", "Latency"],
      s.recent.map(r => [
        esc(r.query),
        `<span class="badge ${r.level}">${LBL[r.level]}</span>`,
        r.score > 0 ? r.score.toFixed(3) : "-",
        Math.round(r.ms) + " ms"
      ])
    );

    tbl(
      "#top",
      ["Query", "Reuses (Hits)"],
      s.top.filter(t => t.hits).map(t => [esc(t.query), `<strong>${t.hits}</strong>`])
    );
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

async function loadEntries() {
  try {
    const e = await api("/entries");
    tbl(
      "#etable",
      ["Query", "Answer Preview", "Namespace", "Tag", "Reuses", "Action"],
      e.map(x => [
        `<strong>${esc(x.query)}</strong>`,
        esc(x.answer.slice(0, 85)) + (x.answer.length > 85 ? "…" : ""),
        esc(x.ns),
        esc(x.tag || "-"),
        x.hits,
        `<button class="btn ghost" style="padding:4px 10px;font-size:12px;" data-d="${x.id}">Delete</button>`
      ])
    );

    document.querySelectorAll("[data-d]").forEach(b => {
      b.onclick = async () => {
        await api("/entries/" + b.dataset.d, "DELETE");
        toast("Entry deleted");
        loadEntries();
      };
    });
  } catch (err) {
    console.error("Failed to load entries:", err);
  }
}

$("#clr").onclick = async () => {
  if (confirm("Are you sure you want to purge all cached vectors and metadata?")) {
    await api("/clear", "POST");
    loadEntries();
    toast("Cache purged completely");
  }
};

$("#inv").onclick = async () => {
  const tag = $("#invtag").value.trim();
  if (!tag) {
    toast("Please enter a tag to invalidate", true);
    return;
  }
  const r = await api("/invalidate", "POST", { tag });
  toast(`${r.deleted} entries invalidated`);
  $("#invtag").value = "";
  loadEntries();
};

$("#dowarm").onclick = async () => {
  const q = $("#warm").value.split("\n").filter(x => x.trim());
  if (!q.length) {
    toast("Please enter queries to warm", true);
    return;
  }
  toast("Warming cache in background...");
  const r = await api("/warm", "POST", { queries: q, ns: $("#ns").value.trim() || "default" });
  toast(`Warmed ${r.warmed} queries into semantic cache`);
  $("#warm").value = "";
};

$("#doeval").onclick = async () => {
  const btn = $("#doeval");
  btn.disabled = true;
  toast("Running evaluation sweep...");

  try {
    const d = await api("/evaluate");
    const c = d.curve;
    const W = 620, H = 220;
    const x = t => 45 + ((t - 0.5) / 0.48) * (W - 70);
    const y = v => H - 25 - v * (H - 50);

    const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="2.5" points="${c.map(p => x(p.t) + "," + y(p[k])).join(" ")}"/>`;

    $("#evalout").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision & Recall by Threshold">
        ${[0, 0.5, 1].map(v => `<line x1="45" x2="${W - 20}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="8" y="${y(v) + 4}" fill="var(--mute)" font-size="11">${v}</text>`).join("")}
        ${c.filter((_, i) => i % 3 === 0).map(p => `<text x="${x(p.t)}" y="${H - 6}" fill="var(--mute)" font-size="11" text-anchor="middle">${p.t}</text>`).join("")}
        ${line("precision", "var(--acc)")}
        ${line("recall", "var(--hit)")}
        ${line("precision_rr", "var(--miss)")}
        <text x="${W - 130}" y="16" fill="var(--acc)" font-size="11" font-weight="600">■ Raw Precision</text>
        <text x="${W - 250}" y="16" fill="var(--miss)" font-size="11" font-weight="600">■ Precision + Reranker</text>
        <text x="${W - 65}" y="16" fill="var(--hit)" font-size="11" font-weight="600">■ Recall</text>
      </svg>
      <div class="scroll" style="margin-top:14px;">
        <table>
          <thead><tr><th>Test Pair</th><th>Paraphrase</th><th>Cosine Sim</th><th>Reranker</th><th>Cache Hit</th></tr></thead>
          <tbody>
            ${d.pairs.map(p => `
              <tr>
                <td>${esc(p.a)}<br><span class="mute">${esc(p.b)}</span></td>
                <td>${p.same ? '<strong style="color:var(--hit)">Yes</strong>' : '<span class="mute">No</span>'}</td>
                <td>${p.sim}</td>
                <td>${p.rerank}</td>
                <td>${p.pass ? '<span class="badge L2_semantic">Accepted</span>' : '<span class="badge L3_llm">Rejected</span>'}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
    toast("Evaluation complete");
  } catch (err) {
    toast("Evaluation failed: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
};

$("#dosim").onclick = async () => {
  const r = await api(`/simulate?queries=${$("#sq").value}&hit_rate=${$("#sh").value}&llm_ms=${$("#sl").value}`);
  $("#simout").innerHTML = [
    ["LLM Calls Saved", r.llm_calls_saved.toLocaleString()],
    ["Time Saved", r.time_saved_min + " min"],
    ["Cost Saved", "$" + r.cost_saved.toFixed(2)],
    ["Blended Latency", r.avg_ms + " ms"]
  ].map(([l, v]) => `
    <div class="kpi">
      <b>${v}</b>
      <span>${l}</span>
    </div>
  `).join("");
};

api("/config").then(c => {
  $("#cfg").textContent = `L1: ${c.l1} · Qdrant: ${c.vectors} · Reranker: ${c.reranker ? "Enabled" : "Disabled"}`;
}).catch(() => {
  $("#cfg").textContent = "API Offline";
});
