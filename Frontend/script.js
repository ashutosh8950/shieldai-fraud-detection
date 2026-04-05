/* ═══════════════════════════════════════════════════════
   ShieldAI — Frontend Application Logic
   API_BASE: point to your Flask backend
═══════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:5000/api";

// ── State ─────────────────────────────────────────────
let hourlyChart = null;
let txnPage     = 0;
const TXN_LIMIT = 15;

// ── Navigation ────────────────────────────────────────
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const view = item.dataset.view;
    switchView(view);
  });
});

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));

  document.querySelector(`[data-view="${view}"]`)?.classList.add("active");
  document.getElementById(`view-${view}`)?.classList.add("active");

  const titles = {
    dashboard:    "Dashboard",
    predict:      "Scan Transaction",
    transactions: "Transactions",
    alerts:       "Alerts",
  };
  document.getElementById("pageTitle").textContent = titles[view] || view;

  if (view === "dashboard")    refreshDashboard();
  if (view === "transactions") { txnPage = 0; loadTransactions(); }
  if (view === "alerts")       loadAlerts();
}

// ── Clock ─────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.toTimeString().slice(0, 8) + " UTC";
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════
async function refreshDashboard() {
  try {
    const stats = await apiFetch("/stats");

    setText("kpi-total",      stats.total_transactions.toLocaleString());
    setText("kpi-fraud",      stats.total_fraud.toLocaleString());
    setText("kpi-fraud-rate", `${stats.fraud_rate}% rate`);
    setText("kpi-alerts",     stats.open_alerts.toLocaleString());
    setText("kpi-avg",        `$${stats.avg_amount.toLocaleString()}`);
    setText("kpi-acc",        stats.model_accuracy
      ? (stats.model_accuracy * 100).toFixed(1) + "%" : "—");
    setText("kpi-auc", stats.model_auc
      ? `AUC: ${(stats.model_auc * 100).toFixed(1)}%` : "AUC: —");

    renderHourlyChart(stats.hourly_distribution);
    renderRecentFraud(stats.recent_frauds);
  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

function renderHourlyChart(data) {
  const ctx = document.getElementById("hourlyChart").getContext("2d");

  const hours   = Array.from({ length: 24 }, (_, i) => i);
  const counts  = new Array(24).fill(0);
  const frauds  = new Array(24).fill(0);

  data.forEach(row => {
    counts[row.hour]  = row.cnt || 0;
    frauds[row.hour]  = row.fraud_cnt || 0;
  });

  if (hourlyChart) hourlyChart.destroy();

  Chart.defaults.color = "#6b7085";
  Chart.defaults.font.family = "'Space Mono', monospace";
  Chart.defaults.font.size   = 10;

  hourlyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hours.map(h => h + ":00"),
      datasets: [
        {
          label: "Total",
          data: counts,
          backgroundColor: "rgba(124,106,255,0.3)",
          borderColor: "#7c6aff",
          borderWidth: 1,
          borderRadius: 3,
        },
        {
          label: "Fraud",
          data: frauds,
          backgroundColor: "rgba(255,76,106,0.5)",
          borderColor: "#ff4c6a",
          borderWidth: 1,
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { boxWidth: 10, padding: 14 },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.03)" },
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.05)" },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderRecentFraud(rows) {
  const tbody = document.getElementById("recentFraudTable");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty">No fraud detected yet</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><span style="color:var(--accent);font-size:11px">${r.txn_id}</span></td>
      <td>$${Number(r.amount).toFixed(2)}</td>
      <td>${r.merchant_name}</td>
      <td><span class="status-badge badge-blocked">${(r.fraud_probability * 100).toFixed(0)}%</span></td>
    </tr>
  `).join("");
}

// ═══════════════════════════════════════════════════════
// PREDICT
// ═══════════════════════════════════════════════════════
const SAMPLES = {
  legit: {
    card: "4111 1111 1111 1234",
    merchant: "Starbucks Coffee",
    amount: 6.75,
    hour: 8,
    distance: 2,
    mrisk: 0.05,
    gap: 18,
    daily: 2,
    velocity: 0.08,
    online: false,
  },
  fraud: {
    card: "5500 0000 0000 0004",
    merchant: "CryptoExchange XZ",
    amount: 4850.00,
    hour: 3,
    distance: 320,
    mrisk: 0.92,
    gap: 0.2,
    daily: 14,
    velocity: 0.95,
    online: true,
  },
};

function fillSample(type) {
  const s = SAMPLES[type];
  setValue("f-card",     s.card);
  setValue("f-merchant", s.merchant);
  setValue("f-amount",   s.amount);
  setValue("f-hour",     s.hour);
  setValue("f-distance", s.distance);
  setValue("f-mrisk",    s.mrisk);
  setValue("f-gap",      s.gap);
  setValue("f-daily",    s.daily);
  setValue("f-velocity", s.velocity);
  document.getElementById("f-online").checked = s.online;
}

async function runPrediction(e) {
  e.preventDefault();

  const card     = document.getElementById("f-card").value.trim();
  const merchant = document.getElementById("f-merchant").value.trim();
  const amount   = parseFloat(document.getElementById("f-amount").value);

  if (!card || !merchant || isNaN(amount) || amount <= 0) {
    alert("Please fill in Card Number, Merchant, and Amount.");
    return;
  }

  const btn = document.getElementById("scanBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Analysing…`;

  const payload = {
    card_number:        card,
    merchant_name:      merchant,
    amount:             amount,
    hour:               parseInt(document.getElementById("f-hour").value)     || new Date().getHours(),
    distance_from_home: parseFloat(document.getElementById("f-distance").value) || 5,
    merchant_risk:      parseFloat(document.getElementById("f-mrisk").value)   || 0.2,
    prev_txn_gap_hrs:   parseFloat(document.getElementById("f-gap").value)     || 24,
    num_daily_txns:     parseInt(document.getElementById("f-daily").value)     || 3,
    velocity_score:     parseFloat(document.getElementById("f-velocity").value) || 0.1,
    is_online:          document.getElementById("f-online").checked ? 1 : 0,
  };

  try {
    const result = await apiFetch("/predict", { method: "POST", body: JSON.stringify(payload) });
    showResult(result);
  } catch (err) {
    alert("API error: " + err.message + "\n\nMake sure the Flask backend is running.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⬡ Scan Transaction</span>`;
  }
}

function showResult(r) {
  const placeholder = document.querySelector(".result-placeholder");
  const content     = document.getElementById("resultContent");
  placeholder.classList.add("hidden");
  content.classList.remove("hidden");

  // Verdict
  const verdict = document.getElementById("resultVerdict");
  if (r.is_fraud) {
    verdict.innerHTML = `<span class="verdict-fraud">🚨 FRAUD DETECTED</span>`;
  } else {
    verdict.innerHTML = `<span class="verdict-safe">✓ TRANSACTION APPROVED</span>`;
  }

  // Gauge animation
  const prob = r.fraud_probability;
  const pct  = Math.round(prob * 100);
  const arc  = document.getElementById("gaugeArc");
  const totalLen = 251;
  const offset   = totalLen - (totalLen * prob);

  let color = "var(--accent-green)";
  if (prob >= 0.75) color = "var(--danger)";
  else if (prob >= 0.40) color = "var(--warn)";

  arc.style.transition   = "stroke-dashoffset 1s ease, stroke 0.3s";
  arc.style.stroke       = color;
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 50);
  document.getElementById("gaugePct").textContent = pct + "%";

  // Meta
  document.getElementById("resultMeta").innerHTML = `
    <strong>TXN ID:</strong> ${r.txn_id}<br>
    <strong>Card:</strong> ${r.card_masked}<br>
    <strong>Merchant:</strong> ${r.merchant}<br>
    <strong>Amount:</strong> $${r.amount.toFixed(2)}<br>
    <strong>Status:</strong>
      <span class="status-badge ${r.is_fraud ? "badge-blocked" : "badge-approved"}">${r.status}</span>
    &nbsp;
    <span class="status-badge badge-${r.risk_level.toLowerCase()}">${r.risk_level} RISK</span><br>
    <strong>Timestamp:</strong> ${new Date(r.timestamp).toLocaleString()}
  `;

  // Reasons
  const reasonsWrap = document.getElementById("reasonsWrap");
  const reasonsList = document.getElementById("reasonsList");
  if (r.fraud_reasons && r.fraud_reasons.length) {
    reasonsWrap.classList.remove("hidden");
    reasonsList.innerHTML = r.fraud_reasons.map(rs => `<li>${rs}</li>`).join("");
  } else {
    reasonsWrap.classList.add("hidden");
  }
}

// ═══════════════════════════════════════════════════════
// TRANSACTIONS
// ═══════════════════════════════════════════════════════
async function loadTransactions() {
  const filter = document.getElementById("txnFilter").value;
  const offset = txnPage * TXN_LIMIT;
  const url = `${API_BASE}/transactions?limit=${TXN_LIMIT}&offset=${offset}${filter !== "" ? "&fraud=" + filter : ""}`;

  const tbody = document.getElementById("txnTableBody");
  tbody.innerHTML = `<tr><td colspan="7" class="empty"><span class="spinner"></span> Loading…</td></tr>`;

  try {
    const data = await apiFetch(url);
    renderTxnTable(data.rows);
    renderPagination(data.total, offset);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty">Error loading data</td></tr>`;
  }
}

function renderTxnTable(rows) {
  const tbody = document.getElementById("txnTableBody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty">No transactions found</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const risk = probToRisk(r.fraud_probability);
    return `
      <tr>
        <td style="color:var(--accent);font-size:11px">${r.txn_id}</td>
        <td style="color:var(--text-muted)">${new Date(r.timestamp).toLocaleString()}</td>
        <td>${r.card_number}</td>
        <td>${r.merchant_name}</td>
        <td>$${Number(r.amount).toFixed(2)}</td>
        <td><span class="status-badge badge-${risk.toLowerCase()}">${(r.fraud_probability * 100).toFixed(1)}%</span></td>
        <td><span class="status-badge ${r.is_fraud ? "badge-blocked" : "badge-approved"}">${r.status}</span></td>
      </tr>
    `;
  }).join("");
}

function renderPagination(total, offset) {
  const totalPages = Math.ceil(total / TXN_LIMIT);
  const pag = document.getElementById("txnPagination");

  if (totalPages <= 1) { pag.innerHTML = ""; return; }

  let html = "";
  const cur = Math.floor(offset / TXN_LIMIT);
  for (let i = 0; i < Math.min(totalPages, 8); i++) {
    html += `<button class="page-btn ${i === cur ? "active" : ""}" onclick="goPage(${i})">${i + 1}</button>`;
  }
  if (totalPages > 8) html += `<span style="color:var(--text-muted);font-size:11px">… ${totalPages} pages</span>`;
  pag.innerHTML = html;
}

function goPage(n) {
  txnPage = n;
  loadTransactions();
}

// ═══════════════════════════════════════════════════════
// ALERTS
// ═══════════════════════════════════════════════════════
async function loadAlerts() {
  const list = document.getElementById("alertsList");
  list.innerHTML = `<div class="empty"><span class="spinner"></span> Loading…</div>`;

  try {
    const alerts = await apiFetch("/alerts");
    if (!alerts.length) {
      list.innerHTML = `<div class="empty">No alerts found</div>`;
      return;
    }

    list.innerHTML = alerts.map(a => `
      <div class="alert-item alert-${a.severity.toLowerCase()} ${a.resolved ? "alert-resolved" : ""}" id="alert-${a.id}">
        <div class="alert-body">
          <div class="alert-top">
            <span class="alert-sev alert-sev-${a.severity}">${a.severity}</span>
            <span class="alert-txn">${a.txn_id}</span>
          </div>
          <div class="alert-msg">${a.message}</div>
          <div class="alert-time">${new Date(a.timestamp).toLocaleString()}</div>
        </div>
        ${!a.resolved
          ? `<button class="resolve-btn" onclick="resolveAlert(${a.id})">✓ Resolve</button>`
          : `<span style="color:var(--text-muted);font-size:10px">RESOLVED</span>`
        }
      </div>
    `).join("");
  } catch (err) {
    list.innerHTML = `<div class="empty">Error loading alerts</div>`;
  }
}

async function resolveAlert(id) {
  try {
    await apiFetch(`/alerts/${id}/resolve`, { method: "POST" });
    const el = document.getElementById(`alert-${id}`);
    if (el) el.classList.add("alert-resolved");
    loadAlerts();
  } catch (err) {
    alert("Failed to resolve alert");
  }
}

// ═══════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════
async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function probToRisk(p) {
  if (p >= 0.75) return "HIGH";
  if (p >= 0.40) return "MEDIUM";
  return "LOW";
}

// ── Boot ──────────────────────────────────────────────
refreshDashboard();
