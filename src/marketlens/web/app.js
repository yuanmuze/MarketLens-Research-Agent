const byId = (id) => document.getElementById(id);
const state = { searchId: 0, agentId: 0 };

function money(value) {
  return value == null ? "Price unavailable" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function setSearchLoading(active) {
  const button = byId("search-form").querySelector("button");
  button.disabled = active;
  button.textContent = active ? "Searching…" : "Search";
}

function showProducts(results) {
  const target = byId("results");
  target.replaceChildren();
  const template = byId("product-template");
  results.forEach((product) => {
    const card = template.content.cloneNode(true);
    card.querySelector(".product-rank").textContent = `#${product.rank}`;
    card.querySelector(".source-tag").textContent = product.source;
    card.querySelector("h3").textContent = product.title;
    card.querySelector(".brand-name").textContent = product.brand || "Independent brand";
    card.querySelector(".price").textContent = money(product.price);
    card.querySelector(".rating").textContent = product.rating == null ? "Unrated" : `★ ${product.rating.toFixed(1)}`;
    card.querySelector(".score-row strong").textContent = Number(product.score).toFixed(4);
    target.appendChild(card);
  });
}

async function searchCatalog(event) {
  event?.preventDefault();
  const query = byId("search-query").value.trim();
  if (!query) return;
  const currentId = ++state.searchId;
  const params = new URLSearchParams({ q: query, strategy: byId("strategy").value, top_k: "6" });
  const budget = byId("max-budget").value;
  const rating = byId("min-rating").value;
  if (budget) params.set("max_budget", budget);
  if (rating) params.set("min_rating", rating);
  byId("search-meta").textContent = "Searching hybrid catalog…";
  setSearchLoading(true);
  try {
    const response = await fetch(`/search?${params}`, { headers: { "X-Request-ID": `web-search-${Date.now()}` } });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "Search failed");
    if (currentId !== state.searchId) return;
    showProducts(payload.results);
    byId("search-meta").textContent = `${payload.total_results} results · ${payload.duration_ms.toFixed(1)} ms · ${payload.query}`;
  } catch (error) {
    if (currentId === state.searchId) byId("search-meta").textContent = `Unable to search: ${error.message}`;
  } finally { if (currentId === state.searchId) setSearchLoading(false); }
}

async function getHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    if (!response.ok) throw new Error("not ready");
    const count = data.product_count ?? data.catalog_size ?? "—";
    byId("catalog-count").textContent = Number(count).toLocaleString();
    byId("api-status").classList.add("ready");
    byId("api-status").innerHTML = "<span></span> API ready";
    byId("footer-backend").textContent = `${data.embedding_backend || "memory"} semantic backend`;
  } catch (_) {
    byId("api-status").textContent = "API unavailable";
    byId("footer-backend").textContent = "Backend unavailable";
  }
}

async function runAgent(event) {
  event.preventDefault();
  const button = byId("agent-form").querySelector("button");
  const target = byId("agent-result");
  const message = byId("agent-message").value.trim();
  if (!message) return;
  const currentId = ++state.agentId;
  button.disabled = true; button.textContent = "Researching…";
  target.className = "agent-result";
  target.innerHTML = "<div class=\"spark\">✦</div><p>Searching the catalog and validating product evidence…</p>";
  try {
    const response = await fetch("/agent/recommend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, mode: byId("agent-mode").value, max_results: 3, request_id: `web-agent-${Date.now()}` }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "Recommendation failed");
    if (currentId !== state.agentId) return;
    const recommendations = data.recommendations.map((item) => `<div class="recommendation"><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.brand || "")}${item.price != null ? ` · ${money(item.price)}` : ""}</span></div>`).join("");
    target.innerHTML = `<h3>${escapeHtml(data.answer)}</h3>${recommendations || "<p>No matching products found.</p>"}<div class="agent-meta">${data.tool_calls} tool call${data.tool_calls === 1 ? "" : "s"} · ${data.latency_ms.toFixed(0)} ms · ${escapeHtml(data.mode_used)}</div>`;
  } catch (error) {
    if (currentId === state.agentId) target.innerHTML = `<h3>Recommendation unavailable</h3><p>${escapeHtml(error.message)}</p>`;
  } finally { if (currentId === state.agentId) { button.disabled = false; button.textContent = "Generate recommendation"; } }
}

byId("search-form").addEventListener("submit", searchCatalog);
byId("agent-form").addEventListener("submit", runAgent);
getHealth();
searchCatalog();
