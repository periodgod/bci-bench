(function () {
  "use strict";
  const COLORS = { ingest: "#3d6be0", signal: "#2f8f83", features: "#d98a1c", classic: "#7a5af0", deep: "#d0457a",
    foundation: "#16a1c4", protocol: "#5f8f2a", stats: "#c0392b", results: "#8a6d3b", interface: "#6b7486", other: "#999" };
  const VIEWS = [["overview", "总览"], ["modules", "全部模块"], ["capabilities", "能力标签"], ["graph", "依赖关系"], ["migration", "旧代码 → bench"], ["tests", "测试与核验"]];
  // claim levels: what a tag does / does not assert. "design_intent" vs "evaluation" is the key distinction.
  const CLAIM_STYLE = { mechanism: ["机制", "c-mech"], design_intent: ["设计意图", "c-intent"], evaluation: ["评估用途", "c-eval"],
    measurement: ["度量", "c-meas"], artifact: ["产物管理", "c-art"] };
  const DIMS = [["stage", "流程阶段"], ["method", "方法机制"], ["goal", "处理目标"]];
  const capFilter = { stage: new Set(), method: new Set(), goal: new Set(), claim: new Set() };
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let CAT = null, state = { view: "overview", module: null, layer: null, query: "" };

  // ---- backend bridge: pywebview js_api, or the local HTTP fallback ----
  const api = {
    async call(name, ...args) {
      if (window.pywebview && window.pywebview.api) return window.pywebview.api[name](...args);
      const r = await fetch(`/api/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(args) });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    },
  };
  function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); }

  // ---- tiny Python highlighter ----
  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  const BI = new Set("self cls print len range zip enumerate sorted set dict list tuple int float str bool isinstance super np torch nn F".split(" "));
  const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfu]{0,2}"(?:\\.|[^"\\\n])*"|[rbfu]{0,2}'(?:\\.|[^'\\\n])*')|(@[\w.]+)|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)\b/g;
  function highlight(src) {
    let out = "", last = 0, prev = "";
    src.replace(TOKEN, (m, com, str, dec, num, word, idx) => {
      out += esc(src.slice(last, idx)); last = idx + m.length;
      if (com) out += `<span class="tk-com">${esc(m)}</span>`;
      else if (str) out += `<span class="tk-str">${esc(m)}</span>`;
      else if (dec) out += `<span class="tk-dec">${esc(m)}</span>`;
      else if (num) out += `<span class="tk-num">${m}</span>`;
      else if (KW.has(word)) out += `<span class="tk-kw">${m}</span>`;
      else if (prev === "def" || prev === "class") out += `<span class="tk-def">${m}</span>`;
      else if (BI.has(word)) out += `<span class="tk-bi">${m}</span>`;
      else out += m;
      if (word) prev = word;
      return m;
    });
    out += esc(src.slice(last));
    return out.split("\n").map((l, i) => `<span class="ln">${i + 1}</span>${l}`).join("\n");
  }
  function codeBlock(src, startLine = 1) {
    const html = highlight(src).replace(/<span class="ln">(\d+)<\/span>/g, (m, n) => `<span class="ln">${+n + startLine - 1}</span>`);
    return `<pre class="code" tabindex="0">${html}</pre>`;
  }
  // ---- helpers ----
  const layerOf = (id) => CAT.layers.find((l) => l.id === id) || { id, name: "其他", desc: "" };
  const modsIn = (id) => CAT.modules.filter((m) => m.layer === id);
  const verBadge = (v) => `<span class="badge ${v === "v0.2" ? "v02" : v === "v0.1" ? "v01" : "vboth"}">${esc(v)}</span>`;
  const reqBadges = (m) => [...m.requires.filter((r) => !m.optional.includes(r)).map((r) => `<span class="badge b-req">${esc(r)}</span>`),
    ...m.optional.map((r) => `<span class="badge b-opt" title="try/except 可选导入">${esc(r)} 可选</span>`)].join(" ") || `<span class="badge b-ok">仅 numpy</span>`;
  const modLink = (n) => `<button class="link" data-mod="${esc(n)}">${esc(n)}</button>`;
  const publicSyms = (m) => m.symbols.filter((s) => s.public);
  const symbolSummary = (s) => s.summary_zh || s.summary || "（暂无说明）";
  function bilingualSummary(s) {
    const zh = symbolSummary(s), en = s.summary && s.summary !== zh ? s.summary : "";
    return `<small>${esc(zh)}</small>${en ? `<small class="english-note" title="English source docstring">${esc(en)}</small>` : ""}`;
  }

  function modChip(m) {
    return `<button class="mod" data-mod="${esc(m.name)}"><b>${esc(m.name)} ${verBadge(m.version)}</b>
      <small>${esc(m.desc)}</small><small>${m.loc} 行 · ${publicSyms(m).length} 个公开接口${(m.capabilities || []).length ? ` · ${m.capabilities.length} 个能力标签` : ""}</small></button>`;
  }
  function header(title, meta) {
    const s = CAT.stats, v = CAT.verify;
    const verify = v ? `<span class="badge ${v.passed === v.total ? "b-ok" : "b-bad"}">核验 ${v.passed}/${v.total}</span>` : `<span class="badge b-warn">未找到核验记录</span>`;
    return `<div class="top"><div><h1>${esc(title)}</h1><div class="meta">${meta || ""}</div></div><div>${verify}</div></div>
      <div class="stats">${[["模块", s.modules], ["代码行", s.loc.toLocaleString()], ["公开函数", s.functions], ["公开类", s.classes],
        ["测试用例", s.tests], ["v0.2 新模块", s.v02_modules]].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("")}</div>`;
  }

  // ---- views ----
  function viewOverview() {
    const g = CAT.git;
    let html = header("模块总览", `v${esc(CAT.version)} · 分支 <code>${esc(g.branch || "?")}</code> · ${esc(g.commit || "")}${g.dirty_files ? ` · 未提交改动 ${g.dirty_files} 个文件` : ""} · 扫描于 ${esc(CAT.generated)}`);
    html += `<div class="card"><h2>怎么读这张图</h2><p class="desc" style="margin:0">
      <span class="badge v01">v0.1</span> 机械手 BCI 原有模块；<span class="badge v02">v0.2</span> 为毕业论文、NeurIPS 赛题、OmniEEG 赛题新增的分组协议层；
      <span class="badge b-req">依赖</span> 为必需的重型依赖，<span class="badge b-opt">可选</span> 为缺失时不影响其他功能的依赖。点击模块查看函数、类和源码。</p></div>`;
    html += `<div class="layers">${CAT.layers.filter((l) => modsIn(l.id).length).map((l) => `
      <section class="card layer" style="--c:${COLORS[l.id]}"><h2><span class="dot" style="background:${COLORS[l.id]}"></span>${esc(l.name)}
        <span class="badge v01" style="margin-left:auto">${modsIn(l.id).length} 个模块</span></h2>
        <p class="desc">${esc(l.desc)}</p><div class="chips">${modsIn(l.id).map(modChip).join("")}</div></section>`).join("")}</div>`;
    return html;
  }

  function viewLayer(id) {
    const l = layerOf(id);
    return header(l.name, esc(l.desc)) + modsIn(id).map((m) => `<div class="card">${moduleHead(m)}
      <div class="chips" style="margin-top:10px">${publicSyms(m).map((s) => `<button class="mod" data-mod="${esc(m.name)}" data-sym="${esc(s.name)}">
        <b>${esc(s.name)}</b><small>${s.kind === "class" ? "类" : "函数"} · ${esc(symbolSummary(s))}</small>${s.summary && s.summary !== symbolSummary(s) ? `<small class="english-note">${esc(s.summary)}</small>` : ""}</button>`).join("")}</div></div>`).join("");
  }

  function moduleHead(m) {
    return `<h2><span class="dot" style="background:${COLORS[m.layer]}"></span><span style="font-family:var(--mono)">wyu_bci_bench.${esc(m.name)}</span> ${verBadge(m.version)}</h2>
      <p class="desc">${esc(m.desc)}</p>
      <dl class="kv"><dt>文件</dt><dd><button class="link" data-open="${esc(m.file)}" title="用默认编辑器打开">${esc(m.file)}</button>（${m.loc} 行）</dd>
        <dt>依赖</dt><dd>${reqBadges(m)}</dd>
        <dt>引用模块</dt><dd>${m.imports_internal.map(modLink).join(" ") || "—"}</dd>
        <dt>被谁引用</dt><dd>${m.used_by.map(modLink).join(" ") || "—"}</dd></dl>`;
  }

  function symCard(s, open) {
    const methods = s.methods && s.methods.length ? `<div class="methods">方法：${s.methods.map((m) => `<code title="${esc(m.summary)}">${esc(m.name)}${esc(m.signature)}</code>`).join("")}</div>` : "";
    return `<div class="sym" id="sym-${esc(s.name)}"><header><span class="kind k-${s.kind}">${s.kind === "class" ? "类" : "函数"}</span>
      <span class="name">${esc(s.name)}</span><code class="sig">${esc(s.signature)}</code>
      ${s.torch_guarded ? `<span class="badge b-opt">需 PyTorch</span>` : ""}
      <span class="badge v01">L${s.line}–${s.end}</span>
      <button class="toggle" data-toggle="${esc(s.name)}" aria-expanded="${open}">${open ? "收起源码" : "查看源码"}</button></header>
      ${s.summary_zh ? `<div class="sum sum-zh">中文说明：${esc(s.summary_zh)}</div>` : ""}
      ${s.summary ? `<div class="sum english-note">English: ${esc(s.summary)}</div>` : ""}${methods}${capTags(s.capabilities)}
      <div class="src" ${open ? "" : "hidden"}>${open ? codeBlock(s.source, s.line) : ""}</div></div>`;
  }

  // ---- capability tags ----
  const capData = () => CAT.capabilities || { dimensions: { stage: [], method: [], goal: [] }, claim_levels: {}, entries: [] };
  const capLabel = (dim, id) => ((capData().dimensions[dim] || []).find((x) => x.id === id) || { label: id }).label;
  function claimBadge(level) {
    const [name, cls] = CLAIM_STYLE[level] || [level, "c-art"];
    return `<span class="claim ${cls}" title="${esc(capData().claim_levels[level] || "")}">${esc(name)}</span>`;
  }
  function capTags(caps) {
    if (!caps || !caps.length) return "";
    return caps.map((c) => `<div class="captags">${claimBadge(c.claim_level)}
      ${DIMS.map(([d]) => (c[d] || []).map((id) => `<button class="tag t-${d}" data-cap="${d}:${esc(id)}" title="筛选：${esc(capLabel(d, id))}">${esc(capLabel(d, id))}</button>`).join("")).join("")}</div>`).join("");
  }
  function capMatches(e) {
    return DIMS.every(([d]) => !capFilter[d].size || (e[d] || []).some((id) => capFilter[d].has(id))) &&
      (!capFilter.claim.size || capFilter.claim.has(e.claim_level));
  }
  function viewCapabilities() {
    const cap = capData(), all = cap.entries, hits = all.filter(capMatches);
    if (!all.length) return header("能力标签", "") + `<div class="empty">catalog 中没有能力标签（需要 explorer/capabilities.json）。</div>`;
    const count = (d, id) => all.filter((e) => (d === "claim" ? e.claim_level === id : (e[d] || []).includes(id))).length;
    const chip = (d, id, label) => `<button class="fchip t-${d}" data-capf="${d}:${esc(id)}" aria-pressed="${capFilter[d].has(id)}">${esc(label)}<span>${count(d, id)}</span></button>`;
    const active = Object.values(capFilter).reduce((n, s) => n + s.size, 0);
    const legend = Object.entries(cap.claim_levels).map(([k, v]) => `<div>${claimBadge(k)} <span>${esc(v)}</span></div>`).join("");
    const filters = [...DIMS.map(([d, name]) => `<div class="frow"><b>${name}</b><div>${(cap.dimensions[d] || []).filter((x) => count(d, x.id))
      .map((x) => chip(d, x.id, x.label)).join("")}</div></div>`),
      `<div class="frow"><b>声明等级</b><div>${Object.keys(cap.claim_levels).map((k) => chip("claim", k, (CLAIM_STYLE[k] || [k])[0])).join("")}</div></div>`].join("");
    const rows = hits.map((e) => {
      const m = CAT.modules.find((x) => x.name === e.module), s = m && m.symbols.find((x) => x.name === e.symbol);
      return `<tr><td><button class="link" data-mod="${esc(e.module)}" data-sym="${esc(e.symbol)}">${esc(e.module)}.${esc(e.symbol)}</button>
        <div class="desc" style="margin:2px 0 0">${esc((s && s.summary) || "").slice(0, 110)}</div></td>
        <td>${claimBadge(e.claim_level)}</td>
        ${DIMS.map(([d]) => `<td>${(e[d] || []).map((id) => `<button class="tag t-${d}" data-cap="${d}:${esc(id)}">${esc(capLabel(d, id))}</button>`).join("") || "—"}</td>`).join("")}</tr>`;
    }).join("");
    return header("能力标签", "每个关键接口按流程阶段、方法机制、处理目标标注；“声明等级”说明该标签断言到什么程度。") +
      `<div class="card"><h2>声明等级怎么读</h2><p class="desc">重点区分<b>设计意图</b>（希望改善某问题，效果取决于数据与协议）与<b>评估用途</b>（只用于测量该问题，本身不解决它）。
        例如跨被试 SupCon 的“缓解域偏移”是设计意图；跨会话划分只是评估域偏移下的泛化。</p><div class="legend">${legend}</div></div>
       <div class="card"><h2>筛选 <span class="badge v01">同一行内任一匹配，不同行同时满足</span>
         ${active ? `<button class="toggle" data-capclear="1">清除 ${active} 个条件</button>` : ""}</h2><div class="filters">${filters}</div></div>
       <div class="card"><h2>匹配接口 <span class="badge v02">${hits.length} / ${all.length}</span></h2>
         ${hits.length ? `<div class="tablewrap"><table class="captable"><thead><tr><th>接口</th><th>声明等级</th>${DIMS.map(([, n]) => `<th>${n}</th>`).join("")}</tr></thead>
         <tbody>${rows}</tbody></table></div>` : `<div class="empty">没有同时满足这些条件的接口</div>`}</div>`;
  }
  function toggleCap(spec, only) {
    const [d, id] = spec.split(/:(.*)/s);
    if (!capFilter[d]) return;
    if (only) { Object.values(capFilter).forEach((s) => s.clear()); capFilter[d].add(id); }
    else capFilter[d].has(id) ? capFilter[d].delete(id) : capFilter[d].add(id);
  }

  function viewModule(name, focus) {
    const m = CAT.modules.find((x) => x.name === name);
    if (!m) return `<div class="empty">找不到模块 ${esc(name)}</div>`;
    const pub = publicSyms(m), priv = m.symbols.filter((s) => !s.public);
    return header(`wyu_bci_bench.${m.name}`, `${esc(layerOf(m.layer).name)} · ${pub.length} 个公开接口，${priv.length} 个内部函数`) +
      `<div class="card">${moduleHead(m)}${m.doc ? `<div class="doc" style="margin-top:12px">${esc(m.doc)}</div>` : ""}</div>
      <div class="card"><h2>公开接口</h2><p class="desc">点击“查看源码”展开实现；行号与文件一致。</p>${pub.map((s) => symCard(s, s.name === focus)).join("") || `<div class="empty">无</div>`}</div>
      ${priv.length ? `<div class="card"><h2>内部函数</h2>${priv.map((s) => symCard(s, s.name === focus)).join("")}</div>` : ""}
      <div class="card"><h2>完整源码 <button class="toggle" data-full="${esc(m.name)}" aria-expanded="false">展开</button></h2><div class="full" hidden></div></div>`;
  }
  function viewGraph() {
    const layers = CAT.layers.filter((l) => modsIn(l.id).length);
    const colW = 156, rowH = 50, top = 52, W = 60 + layers.length * colW;
    const pos = {};
    let maxRows = 0;
    layers.forEach((l, ci) => modsIn(l.id).forEach((m, ri) => { pos[m.name] = { x: 24 + ci * colW, y: top + ri * rowH, m }; maxRows = Math.max(maxRows, ri + 1); }));
    const H = top + maxRows * rowH + 24, nodeW = 116;
    let edges = "", nodes = "", heads = "";
    layers.forEach((l, ci) => { heads += `<text x="${24 + ci * colW + nodeW / 2}" y="22" text-anchor="middle" fill="${COLORS[l.id]}" font-weight="700" style="font-family:'Microsoft YaHei'">${esc(l.name)}</text>`; });
    for (const m of CAT.modules) for (const t of m.imports_internal) {
      const a = pos[m.name], b = pos[t]; if (!a || !b) continue;
      const x1 = a.x, y1 = a.y + 11, x2 = b.x + nodeW, y2 = b.y + 11, dx = Math.max(30, Math.abs(x1 - x2) / 2);
      edges += `<path class="edge" marker-end="url(#arrow)" data-a="${esc(m.name)}" data-b="${esc(t)}" d="M${x1},${y1} C${x1 - dx},${y1} ${x2 + dx},${y2} ${x2 + 3},${y2}"/>`;
    }
    for (const [n, p] of Object.entries(pos)) {
      nodes += `<g class="node" data-mod="${esc(n)}" data-node="${esc(n)}" tabindex="0" role="button" aria-label="${esc(n)}">
        <rect x="${p.x}" y="${p.y}" width="${nodeW}" height="22" rx="6" fill="#fff" stroke="${COLORS[p.m.layer]}" stroke-width="1.6"/>
        <text x="${p.x + nodeW / 2}" y="${p.y + 15}" text-anchor="middle">${esc(n.length > 15 ? n.replace("foundation.", "fdn.") : n)}</text>
        <title>${esc(n)}：${esc(p.m.desc)}\n引用 ${p.m.imports_internal.length} · 被引用 ${p.m.used_by.length}</title></g>`;
    }
    return header("依赖关系", "箭头从使用方指向被引用模块；悬停高亮一条模块的全部连线，点击进入模块。") +
      `<div class="graph-wrap"><svg class="graph" viewBox="0 0 ${W} ${H}" role="img" aria-label="模块依赖图"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a96aa"/></marker></defs>${heads}${edges}${nodes}</svg></div>
       <div class="card" style="margin-top:14px"><h2>引用最多的模块</h2><p class="desc">改动这些模块影响面最大，改后务必跑单元测试和核验脚本。</p>
       <table><thead><tr><th>模块</th><th>被引用次数</th><th>被谁引用</th></tr></thead><tbody>${[...CAT.modules].sort((a, b) => b.used_by.length - a.used_by.length)
        .filter((m) => m.used_by.length).slice(0, 8).map((m) => `<tr><td>${modLink(m.name)}</td><td>${m.used_by.length}</td><td>${m.used_by.map(modLink).join(" ")}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function viewMigration() {
    const rows = CAT.migration.map((r) => `<tr><td>${inline(r.old)}</td><td>${inline(r.new)}</td></tr>`).join("");
    return header("旧代码 → bench 对照", "旧脚本保持原样（结果报告里记录了它们的 sha256）；新实验只从 bench 导入。来源：docs/EEG_FOUNDATION_PROTOCOLS.md") +
      `<div class="card"><input class="filter" id="migFilter" type="search" placeholder="筛选，例如 cell_score / bootstrap / CBraMod" aria-label="筛选对照表">
       <table id="migTable"><thead><tr><th style="width:52%">旧实现（challenge2026_track02/proteus_update1/ 等）</th><th>bench 中的位置</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");

  function viewTests() {
    const v = CAT.verify;
    const checks = v ? `<table><thead><tr><th>核验项</th><th>结果</th><th>细节</th></tr></thead><tbody>${v.checks.map((c) =>
      `<tr><td>${esc(c.name)}</td><td><span class="badge ${c.ok ? "b-ok" : "b-bad"}">${c.ok ? "通过" : "失败"}</span></td><td><code>${esc(c.detail)}</code></td></tr>`).join("")}</tbody></table>`
      : `<div class="empty">还没有核验记录。运行 <code>python scripts/verify_proteus_port.py --with-bids</code> 后点“重新扫描源码”。</div>`;
    const scripts = CAT.scripts.map((s) => `<div class="sym"><header><span class="kind k-method">${esc(s.kind)}</span><span class="name">${esc(s.file)}</span>
      <span class="badge v01">${s.loc} 行</span>${s.tests.length ? `<span class="badge b-ok">${s.tests.length} 个测试</span>` : ""}
      <button class="toggle" data-script="${esc(s.file)}" aria-expanded="false">查看源码</button></header>
      ${s.summary ? `<div class="sum">${esc(s.summary)}</div>` : ""}${s.tests.length ? `<div class="methods">${s.tests.map((t) => `<code>${esc(t)}</code>`).join("")}</div>` : ""}
      <div class="src" hidden></div></div>`).join("");
    return header("测试与核验", v ? `最近核验 ${esc(v.time)}${v.with_bids ? "（含原始 EDF 重提取）" : ""}` : "") +
      `<div class="card"><h2>与历史结果 / 原实现的等价核验 ${v ? `<span class="badge ${v.passed === v.total ? "b-ok" : "b-bad"}">${v.passed}/${v.total}</span>` : ""}</h2>${checks}</div>
       <div class="card"><h2>测试、脚本与示例</h2>${scripts}</div>`;
  }

  function viewSearch(q) {
    const t = q.toLowerCase(), hits = [];
    for (const m of CAT.modules) {
      if ((m.name + " " + m.desc + " " + m.summary).toLowerCase().includes(t)) hits.push({ m, s: null });
      for (const s of m.symbols) {
        const tags = (s.capabilities || []).flatMap((c) => DIMS.flatMap(([d]) => (c[d] || []).map((id) => capLabel(d, id)))).join(" ");
        if ((s.name + " " + s.summary + " " + s.summary_zh + " " + tags).toLowerCase().includes(t)) hits.push({ m, s });
      }
    }
    return header(`搜索：${q}`, `${hits.length} 个结果`) + (hits.length ? hits.slice(0, 200).map(({ m, s }) =>
      `<button class="result" data-mod="${esc(m.name)}" ${s ? `data-sym="${esc(s.name)}"` : ""}>${s ? `<code>${esc(s.name)}</code> <span class="badge v01">${s.kind === "class" ? "类" : "函数"}</span> · ${esc(m.name)}`
        : `<code>${esc(m.name)}</code> <span class="badge v02">模块</span>`}<div class="desc" style="margin:2px 0 0">${esc((s ? symbolSummary(s) : m.desc) || "").slice(0, 140)}</div>${s && s.summary && s.summary !== symbolSummary(s) ? `<div class="english-note">${esc(s.summary)}</div>` : ""}</button>`).join("")
      : `<div class="empty">没有匹配项</div>`);
  }
  // ---- rendering & navigation ----
  function renderNav() {
    $("#nav").innerHTML = VIEWS.map(([id, label]) => {
      const n = id === "modules" ? CAT.modules.length : id === "migration" ? CAT.migration.length : id === "tests" ? CAT.stats.tests
        : id === "capabilities" ? capData().entries.length : "";
      return `<button data-view="${id}" ${state.view === id && !state.layer && !state.query && !state.module ? 'aria-current="page"' : ""}>${label}<span class="count">${n}</span></button>`;
    }).join("");
    $("#layerNav").innerHTML = CAT.layers.filter((l) => modsIn(l.id).length).map((l) =>
      `<button class="layer-link" data-layer="${l.id}" ${state.layer === l.id ? 'aria-current="page"' : ""}><span class="dot" style="background:${COLORS[l.id]}"></span>${esc(l.name)}<span class="count">${modsIn(l.id).length}</span></button>`).join("");
    $("#brandMeta").textContent = `v${CAT.version} · ${CAT.modules.length} 个模块`;
    $("#footMeta").innerHTML = `源码：${esc(CAT.root)}<br>扫描：${esc(CAT.generated)}${CAT.snapshot ? "<br>（离线快照）" : ""}`;
  }

  function render(focusSym) {
    renderNav();
    let html;
    if (state.query) html = viewSearch(state.query);
    else if (state.module) html = viewModule(state.module, focusSym);
    else if (state.layer) html = viewLayer(state.layer);
    else html = { overview: viewOverview, graph: viewGraph, migration: viewMigration, tests: viewTests, capabilities: viewCapabilities,
      modules: () => header("全部模块", "按分层排列") + `<div class="card"><div class="chips">${CAT.layers.flatMap((l) => modsIn(l.id)).map(modChip).join("")}</div></div>` }[state.view]();
    const main = $("#main");
    main.innerHTML = html;
    if (focusSym) { const el = document.getElementById("sym-" + focusSym); if (el) el.scrollIntoView({ block: "start" }); }
    else main.scrollTop = 0;
  }

  function go(next, focusSym) {
    state = { view: "overview", module: null, layer: null, query: "", ...next };
    const hash = state.module ? `module/${state.module}` : state.layer ? `layer/${state.layer}` : state.view === "capabilities" ? capHash() : state.view;
    if (location.hash.slice(1) !== hash) history.replaceState(null, "", "#" + hash);
    render(focusSym);
  }
  function capHash() {
    const q = Object.entries(capFilter).filter(([, s]) => s.size).map(([d, s]) => `${d}=${[...s].join(",")}`).join("&");
    return "capabilities" + (q ? "?" + q : "");
  }
  function fromHash() {
    const raw = decodeURIComponent(location.hash.slice(1));
    if (raw.startsWith("capabilities")) {
      Object.values(capFilter).forEach((s) => s.clear());
      for (const part of (raw.split("?")[1] || "").split("&").filter(Boolean)) {
        const [d, v] = part.split("=");
        if (capFilter[d]) v.split(",").filter(Boolean).forEach((id) => capFilter[d].add(id));
      }
      return { view: "capabilities" };
    }
    const [kind, arg] = raw.split("/");
    if (kind === "module" && arg) return { module: arg };
    if (kind === "layer" && arg) return { layer: arg };
    return VIEWS.some(([id]) => id === kind) ? { view: kind } : {};
  }

  document.addEventListener("click", async (e) => {
    const t = e.target.closest("[data-view],[data-layer],[data-mod],[data-toggle],[data-full],[data-script],[data-open],[data-cap],[data-capf],[data-capclear],[data-menu]");
    if (!t) return;
    if (t.dataset.menu) { document.body.classList.toggle("nav-open"); return; }
    if (t.dataset.cap) { toggleCap(t.dataset.cap, true); return go({ view: "capabilities" }); }
    if (t.dataset.capf) {
      toggleCap(t.dataset.capf, false);
      const y = $("#main").scrollTop; history.replaceState(null, "", "#" + capHash()); render(); $("#main").scrollTop = y;
      return;
    }
    if (t.dataset.capclear) { Object.values(capFilter).forEach((s) => s.clear()); return go({ view: "capabilities" }); }
    document.body.classList.remove("nav-open");
    if (t.dataset.open) { try { await api.call("open_file", t.dataset.open); } catch (err) { toast("无法打开：" + err.message); } return; }
    if (t.dataset.view) return go({ view: t.dataset.view });
    if (t.dataset.layer) return go({ layer: t.dataset.layer });
    if (t.dataset.toggle) {
      const card = t.closest(".sym"), box = card.querySelector(".src"), open = box.hidden;
      if (open && !box.innerHTML) {
        const m = CAT.modules.find((x) => x.name === state.module), s = m && m.symbols.find((x) => x.name === t.dataset.toggle);
        if (s) box.innerHTML = codeBlock(s.source, s.line);
      }
      box.hidden = !open; t.textContent = open ? "收起源码" : "查看源码"; t.setAttribute("aria-expanded", open);
      return;
    }
    if (t.dataset.full) {
      const box = t.closest(".card").querySelector(".full"), open = box.hidden;
      if (open && !box.innerHTML) box.innerHTML = codeBlock(CAT.modules.find((x) => x.name === t.dataset.full).source);
      box.hidden = !open; t.textContent = open ? "收起" : "展开"; t.setAttribute("aria-expanded", open);
      return;
    }
    if (t.dataset.script) {
      const box = t.closest(".sym").querySelector(".src"), open = box.hidden;
      if (open && !box.innerHTML) box.innerHTML = codeBlock(CAT.scripts.find((x) => x.file === t.dataset.script).source);
      box.hidden = !open; t.textContent = open ? "收起源码" : "查看源码"; t.setAttribute("aria-expanded", open);
      return;
    }
    if (t.dataset.mod) { $("#search").value = ""; return go({ module: t.dataset.mod }, t.dataset.sym); }
  });
  document.addEventListener("keydown", (e) => {
    const node = e.target.closest && e.target.closest("g.node");
    if (node && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); go({ module: node.dataset.mod }); }
    if (e.key === "/" && document.activeElement !== $("#search")) { e.preventDefault(); $("#search").focus(); }
  });
  document.addEventListener("mouseover", (e) => {
    const node = e.target.closest && e.target.closest("g.node");
    const svg = document.querySelector("svg.graph"); if (!svg) return;
    const name = node ? node.dataset.node : null;
    svg.querySelectorAll(".edge").forEach((p) => p.classList.toggle("hot", !!name && (p.dataset.a === name || p.dataset.b === name)));
    const linked = new Set([name]);
    if (name) svg.querySelectorAll(".edge.hot").forEach((p) => { linked.add(p.dataset.a); linked.add(p.dataset.b); });
    svg.querySelectorAll("g.node").forEach((g) => g.classList.toggle("dim", !!name && !linked.has(g.dataset.node)));
  });
  document.addEventListener("input", (e) => {
    if (e.target.id === "search") { const q = e.target.value.trim(); q ? (state = { ...state, query: q }, render()) : go({}); }
    if (e.target.id === "migFilter") {
      const q = e.target.value.toLowerCase();
      document.querySelectorAll("#migTable tbody tr").forEach((tr) => { tr.hidden = q && !tr.textContent.toLowerCase().includes(q); });
    }
  });
  $("#rescan").addEventListener("click", async () => {
    try { CAT = await api.call("rescan"); toast(`已重新扫描：${CAT.modules.length} 个模块`); render(); }
    catch (err) { toast("扫描失败：" + err.message); }
  });

  let booted = false;
  async function boot() {
    if (booted) return;
    booted = true;
    try { CAT = await api.call("get_catalog"); go(fromHash()); }
    catch (err) { $("#main").innerHTML = `<div class="empty">读取失败：${esc(err.message)}</div>`; }
  }
  if (window.pywebview && window.pywebview.api) boot();
  else window.addEventListener("pywebviewready", boot, { once: true });
  setTimeout(() => { if (!CAT && !(window.pywebview && window.pywebview.api)) boot(); }, 600);
})();
