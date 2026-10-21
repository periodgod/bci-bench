(function () {
  "use strict";
  // metro line colours, one per layer; each holds >= 3:1 against white so rings and line bars read on the light ground
  const COLORS = { ingest: "#1f5fc4", signal: "#0a8a7a", features: "#d0661a", classic: "#6c3fc4", deep: "#c4307a",
    foundation: "#0e86b8", protocol: "#3f8a1f", stats: "#c42b2b", results: "#9a6b12", interface: "#5f6873", other: "#7a828c" };
  const VIEWS = [["overview", "线路图"], ["modules", "全部模块"], ["animations", "模型视频"], ["capabilities", "能力标签"], ["graph", "依赖关系"], ["migration", "旧代码 → bench"], ["tests", "测试与核验"]];
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
    return `<small>${esc(zh)}</small>${en ? `<small class="english-note" title="源码 docstring 原文">${esc(en)}</small>` : ""}`;
  }

  const lineChip = (id) => `<button class="linechip" data-layer="${esc(id)}" title="查看这条线路"><span class="swatch" style="background:${COLORS[id]}"></span>${esc(layerOf(id).name)}</button>`;
  // opts: { mono: identifier title, line: layer id shown as a line chip beside the title }
  function header(title, meta, opts = {}) {
    return `<div class="top"><div class="title"><h1 class="${opts.mono ? "mono" : ""}">${esc(title)}</h1>${opts.line ? lineChip(opts.line) : ""}</div>
      ${meta ? `<div class="meta">${meta}</div>` : ""}${opts.after || ""}</div>`;
  }
  function figures() {
    const s = CAT.stats;
    return `<div class="figures">${[["个模块", s.modules], ["行代码", s.loc.toLocaleString()], ["个公开函数", s.functions], ["个公开类", s.classes],
      ["个测试", s.tests], ["个 v0.2 新模块", s.v02_modules]].map(([k, v]) => `<span><b>${v}</b>${k}</span>`).join("")}</div>`;
  }

  // ---- views ----
  function viewOverview() {
    const g = CAT.git;
    const meta = `v${esc(CAT.version)} · 分支 <code>${esc(g.branch || "?")}</code> · ${esc(g.commit || "")}${g.dirty_files ? ` · 未提交改动 ${g.dirty_files} 个文件` : ""} · 扫描于 ${esc(CAT.generated)}`;
    const lines = CAT.layers.filter((l) => modsIn(l.id).length).map((l) => {
      const mods = modsIn(l.id);
      return `<div class="mline" style="--c:${COLORS[l.id]}">
        <button class="lbadge" data-layer="${esc(l.id)}" title="${esc(l.desc)}"><span class="swatch"></span><b>${esc(l.name)}</b><small>${mods.length} 站</small></button>
        <ul class="stations" style="--n:${mods.length}">${mods.map((m) => {
          const xl = transferLines(m), xfer = xl.length >= 2;
          // interchange: like station signage, the connecting lines' bullets sit under the station name
          const bullets = xfer ? `<span class="xlines" aria-label="换乘：${xl.map((id) => esc(layerOf(id).name)).join("、")}">${xl.map((id) =>
            `<i style="background:${COLORS[id]}" title="${esc(layerOf(id).name)}"></i>`).join("")}</span>` : "";
          return `<li><button class="stn${xfer ? " xfer" : ""}" data-mod="${esc(m.name)}" data-stn="${esc(m.name)}"
            title="${esc(m.desc)}${xl.length ? `\n被另外 ${xl.length} 条线路引用` : ""}"><span class="ring"></span><b>${esc(m.name).replace(/\./g, ".<wbr>")}</b>
            <small>${m.loc} 行${m.version === "v0.1" ? "" : ` ${verBadge(m.version)}`}</small>${bullets}</button></li>`;
        }).join("")}</ul></div>`;
    }).join("");
    return header("线路图", meta, { after: figures() }) +
      `<div class="metro" id="metro">${lines}</div>
       <p class="readout" id="readout" aria-live="polite">把指针移到站点上，会用黄色标出它和哪些站相连；点击进入该模块。</p>
       <div class="card"><h2>怎么读</h2><div class="mlegend">
         <span><i class="r"></i>一个站 = 一个模块，环的颜色 = 所在线路（功能分层）</span>
         <span><i class="x"></i>换乘站：被另外两条及以上线路引用，站名下的色点是换乘的线路</span>
         <span><i class="t"></i>黄色 = 当前站及与它有引用关系的站</span>
         <span>${verBadge("v0.2")} 为毕业论文、NeurIPS 与 OmniEEG 赛题新增；未标注的是 v0.1 机械手 BCI 原有模块</span>
       </div><p class="desc" style="margin:12px 0 0">线路从上到下大致按数据流排列：接入 → 信号 → 特征 → 模型 → 协议 → 统计 → 结果。线路长度与站数成正比。每个模块的依赖、接口和源码见“全部模块”。</p></div>`;
  }
  // other layers whose modules import this one, in line order
  function transferLines(m) {
    const ids = new Set(m.used_by.map((u) => (CAT.modules.find((x) => x.name === u) || {}).layer).filter((id) => id && id !== m.layer));
    return CAT.layers.map((l) => l.id).filter((id) => ids.has(id));
  }

  function viewModules() {
    const rows = CAT.layers.filter((l) => modsIn(l.id).length).map((l) => `<tr class="grp" style="--c:${COLORS[l.id]}"><td colspan="6">
      <span class="swatch"></span><button class="link" data-layer="${esc(l.id)}" style="font-family:var(--sans);font-weight:700">${esc(l.name)}</button>
      <span style="font-weight:400;color:var(--dim)">${esc(l.desc)}</span></td></tr>` + modsIn(l.id).map((m) => `<tr>
        <td>${modLink(m.name)}</td><td>${esc(m.desc)}</td><td>${verBadge(m.version)}</td>
        <td class="num">${m.loc}</td><td class="num">${publicSyms(m).length}</td><td>${reqBadges(m)}</td></tr>`).join("")).join("");
    return header("全部模块", `${CAT.modules.length} 个模块，按线路（功能分层）分组`) +
      `<div class="tablewrap"><table><thead><tr><th>模块</th><th>做什么</th><th>版本</th><th class="num">行数</th><th class="num">公开接口</th><th>依赖</th></tr></thead>
       <tbody>${rows}</tbody></table></div>
       <p class="desc" style="margin-top:12px;color:var(--dim)"><span class="badge b-req">torch</span> 必需的重型依赖；<span class="badge b-opt">可选</span> 用 try/except 导入，缺失时不影响其他功能。</p>`;
  }

  function viewLayer(id) {
    const l = layerOf(id);
    return header(l.name, `${esc(l.desc)} · ${modsIn(id).length} 站`, { line: id }) + modsIn(id).map((m) => `<div class="card">${moduleHead(m, true)}
      <div class="chips" style="margin-top:10px">${publicSyms(m).map((s) => `<button class="mod" data-mod="${esc(m.name)}" data-sym="${esc(s.name)}">
        <b>${esc(s.name)}</b><small>${s.kind === "class" ? "类" : "函数"} · ${esc(symbolSummary(s))}</small>${s.summary && s.summary !== symbolSummary(s) ? `<small class="english-note">${esc(s.summary)}</small>` : ""}</button>`).join("")}</div></div>`).join("");
  }

  // withTitle: the layer page lists several modules, so each needs its own heading; the module page already has one
  function moduleHead(m, withTitle) {
    return `${withTitle ? `<h2><button class="link" data-mod="${esc(m.name)}" style="font-size:15px;font-weight:700">wyu_bci_bench.${esc(m.name)}</button> ${verBadge(m.version)}</h2>` : ""}
      <p class="desc">${esc(m.desc)}</p>
      <dl class="kv"><dt>文件</dt><dd><button class="link" data-open="${esc(m.file)}" title="用默认编辑器打开">${esc(m.file)}</button>（${m.loc} 行）</dd>
        <dt>依赖</dt><dd>${reqBadges(m)}</dd>
        <dt>引用模块</dt><dd>${m.imports_internal.map(modLink).join(" ") || "—"}</dd>
        <dt>被谁引用</dt><dd>${m.used_by.map(modLink).join(" ") || "—"}</dd></dl>`;
  }

  function symCard(s, open) {
    const methods = s.methods && s.methods.length ? `<div class="methods">方法：${s.methods.map((m) => `<code title="${esc(m.summary)}">${esc(m.name)}${esc(m.signature)}</code>`).join("")}</div>` : "";
    // fixed slots: kind · name · line range · torch on the first row, the signature always on its own row, broken only after ", "
    return `<div class="sym" id="sym-${esc(s.name)}"><header><span class="kind k-${s.kind}">${s.kind === "class" ? "类" : "函数"}</span>
      <span class="name">${esc(s.name)}</span><span class="badge v01">L${s.line}–${s.end}</span>
      ${s.torch_guarded ? `<span class="badge b-opt">需 PyTorch</span>` : ""}
      <button class="toggle" data-toggle="${esc(s.name)}" aria-expanded="${open}">${open ? "收起源码" : "查看源码"}</button></header>
      ${s.signature ? `<code class="sig">${esc(s.signature).split(", ").map((x) => `<span class="arg">${x}</span>`).join(", ")}</code>` : ""}
      ${s.summary_zh || s.summary ? `<div class="sum sum-zh">${esc(symbolSummary(s))}</div>` : ""}
      ${s.summary_zh && s.summary && s.summary !== s.summary_zh ? `<div class="sum english-note">docstring 原文：${esc(s.summary)}</div>` : ""}${methods}${capTags(s.capabilities)}
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
        <div class="desc" style="margin:2px 0 0">${esc(s ? symbolSummary(s) : "").slice(0, 110)}</div></td>
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
  function viewAnimations() {
    const animations = CAT.animations || [];
    const ready = animations.filter((animation) => animation.available).length;
    return header("模型与方法视频", `${ready} / ${animations.length} 个视频可播放 · 搜索模型名也会显示对应视频`) +
      `<section class="card animations" aria-label="模型与方法讲解视频">
        ${animations.length ? `<div class="animation-list">${animations.map((animation) => `<button class="animation-link" data-animation="${esc(animation.id)}" ${animation.available ? "" : "disabled"}>
          <b>${esc(animation.title)}</b><small>${animation.available ? "播放视频" : "视频尚未渲染"} · ${esc(animation.group)} · ${esc(animation.module)}</small></button>`).join("")}</div>` : `<div class="empty">没有登记视频</div>`}
        <div class="animation-player" hidden><h3></h3><video controls preload="metadata" playsinline></video><p class="desc"></p></div>
      </section>`;
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
    const animations = (CAT.animations || []).filter((animation) => animation.module === m.name);
    const animationSection = animations.length ? `<section class="card animations" aria-labelledby="animation-heading">
      <h2 id="animation-heading">模型与方法动画 <span class="badge v01">${animations.filter((animation) => animation.available).length} / ${animations.length} 个视频已渲染</span></h2>
      <p class="desc">每段动画对应本模块中的模型或方法；说明文字标出实现状态和证据边界。</p>
      <div class="animation-list">${animations.map((animation) => `<button class="animation-link" data-animation="${esc(animation.id)}" ${animation.available ? "" : "disabled"}>
        <b>${esc(animation.title)}</b><small>${animation.available ? "播放视频" : "视频尚未渲染"} · ${esc(animation.group)}</small></button>`).join("")}</div>
      <div class="animation-player" hidden><h3></h3><video controls preload="metadata" playsinline></video><p class="desc"></p></div>
    </section>` : "";
    return header(`wyu_bci_bench.${m.name}`, `${verBadge(m.version)} · ${pub.length} 个公开接口，${priv.length} 个内部函数`, { mono: true, line: m.layer }) +
      `<div class="card">${stationStrip(m)}${moduleHead(m)}${m.doc ? `<div class="doc" style="margin-top:14px">${docHtml(m.doc)}</div>` : ""}</div>
      ${animationSection}
      <div class="card"><h2>公开接口</h2><p class="desc">点击“查看源码”展开实现；行号与文件一致。</p>${pub.map((s) => symCard(s, s.name === focus)).join("") || `<div class="empty">无</div>`}</div>
      ${priv.length ? `<div class="card"><h2>内部函数</h2>${priv.map((s) => symCard(s, s.name === focus)).join("")}</div>` : ""}
      <div class="card"><h2>完整源码 <button class="toggle" data-full="${esc(m.name)}" aria-expanded="false">展开</button></h2><div class="full" hidden></div></div>`;
  }
  // the station page's own line: every station of this layer, this one marked "you are here"
  function stationStrip(m) {
    const mods = modsIn(m.layer);
    return `<nav class="strip" style="--c:${COLORS[m.layer]};--n:${mods.length}" aria-label="${esc(layerOf(m.layer).name)} 线路上的站">
      <ol>${mods.map((x) => `<li><button class="stn sm${x.name === m.name ? " self" : ""}" data-mod="${esc(x.name)}" ${x.name === m.name ? 'aria-current="page"' : ""}>
        <span class="ring"></span><b>${esc(x.name).replace(/\./g, ".<wbr>")}</b></button></li>`).join("")}</ol></nav>`;
  }
  // docstrings: hard-wrapped source lines rejoin into paragraphs; lists, indented blocks and >>> examples keep their lines
  function docHtml(doc) {
    return String(doc).split(/\n\s*\n/).map((p) => {
      const lines = p.split("\n");
      const keep = lines.some((l) => /^\s*([-*•]|\d+[.)]|>>>|\s{4})/.test(l) || /^\s{2,}\S/.test(l));
      const body = keep ? lines.map(esc).join("<br>")
        : lines.map((l) => l.trim()).reduce((a, l) => !a ? l : /[\u3000-\u9fff]$/.test(a) && /^[\u3000-\u9fff]/.test(l) ? a + l : a + " " + l, "");
      return `<p${keep ? ' class="pre"' : ""}>${keep ? body : esc(body)}</p>`;
    }).join("");
  }
  function viewGraph() {
    // same orientation as the line map: one row per layer (a faint track in the line colour), modules as pills along it
    const layers = CAT.layers.filter((l) => modsIn(l.id).length);
    const labelW = 150, colW = 172, nodeW = 156, nodeH = 26, rowH = 64, top = 44;
    const maxN = Math.max(...layers.map((l) => modsIn(l.id).length)), W = labelW + maxN * colW + 20, H = top + layers.length * rowH + 4;
    const pos = {};
    layers.forEach((l, ri) => modsIn(l.id).forEach((m, ci) => { pos[m.name] = { x: labelW + ci * colW, y: top + ri * rowH, m }; }));
    let edges = "", nodes = "", heads = "";
    layers.forEach((l, ri) => { const y = top + ri * rowH, n = modsIn(l.id).length;
      heads += `<rect x="0" y="${y + nodeH / 2 - 5}" width="22" height="10" rx="5" fill="${COLORS[l.id]}"/>
        <text class="head" x="30" y="${y + nodeH / 2 + 5}">${esc(l.name)}</text>
        <rect x="${labelW + nodeW / 2}" y="${y + nodeH / 2 - 3}" width="${Math.max(0, (n - 1) * colW)}" height="6" rx="3" fill="${COLORS[l.id]}" opacity=".22"/>`; });
    for (const m of CAT.modules) for (const t of m.imports_internal) {
      const a = pos[m.name], b = pos[t]; if (!a || !b) continue;
      const ax = a.x + nodeW / 2, bx = b.x + nodeW / 2;
      let d;
      if (a.y === b.y) { const lift = 18 + Math.abs(ax - bx) / 10; d = `M${ax},${a.y} C${ax},${a.y - lift} ${bx},${b.y - lift} ${bx},${b.y - 2}`; }
      else { const down = b.y > a.y, y1 = down ? a.y + nodeH : a.y, y2 = down ? b.y - 2 : b.y + nodeH + 2, k = (y2 - y1) / 2;
        d = `M${ax},${y1} C${ax},${y1 + k} ${bx},${y2 - k} ${bx},${y2}`; }
      edges += `<path class="edge" marker-end="url(#arrow)" data-a="${esc(m.name)}" data-b="${esc(t)}" d="${d}"/>`;
    }
    for (const [n, p] of Object.entries(pos)) {
      nodes += `<g class="node" data-mod="${esc(n)}" data-node="${esc(n)}" tabindex="0" role="button" aria-label="${esc(n)}，引用 ${p.m.imports_internal.length}，被引用 ${p.m.used_by.length}">
        <rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="${nodeH / 2}" stroke="${COLORS[p.m.layer]}" stroke-width="3"/>
        <text x="${p.x + nodeW / 2}" y="${p.y + 17.5}" text-anchor="middle">${esc(n)}</text>
        <title>${esc(n)}：${esc(p.m.desc)}\n引用 ${p.m.imports_internal.length} · 被引用 ${p.m.used_by.length}</title></g>`;
    }
    return header("依赖关系", "箭头从使用方指向被引用模块；指针或键盘焦点停在模块上，会标出它的全部连线；点击进入模块。") +
      `<div class="graph-wrap"><svg class="graph" viewBox="0 0 ${W} ${H}" style="max-width:${W * 1.15}px" role="img" aria-label="模块依赖图"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a929c"/></marker></defs>${heads}${edges}${nodes}</svg></div>
       <div class="card" style="margin-top:14px"><h2>引用最多的模块</h2><p class="desc">改动这些模块影响面最大，改后务必跑单元测试和核验脚本。</p>
       <table><thead><tr><th>模块</th><th>被引用次数</th><th>被谁引用</th></tr></thead><tbody>${[...CAT.modules].sort((a, b) => b.used_by.length - a.used_by.length)
        .filter((m) => m.used_by.length).slice(0, 8).map((m) => `<tr><td>${modLink(m.name)}</td><td>${m.used_by.length}</td><td>${m.used_by.map(modLink).join(" ")}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function viewMigration() {
    const rows = CAT.migration.map((r) => `<tr><td>${inline(r.old)}</td><td>${inline(r.new)}</td></tr>`).join("");
    return header("旧代码 → bench 对照", "旧脚本保持原样（结果报告里记录了它们的 sha256）；新实验只从 bench 导入。来源：docs/EEG_FOUNDATION_PROTOCOLS.md") +
      `<div class="card"><input class="filter" id="migFilter" type="search" placeholder="筛选，例如 cell_score / bootstrap / CBraMod" aria-label="筛选对照表" aria-describedby="migCount">
       <span class="filter-count" id="migCount" role="status"></span>
       <table id="migTable"><thead><tr><th style="width:52%">旧实现（challenge2026_track02/proteus_update1/ 等）</th><th>bench 中的位置</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");

  function viewTests() {
    const v = CAT.verify;
    // "一致" only means the port reproduces the historical number; an interval that spans 0 is flagged so it is not read as an effect
    const spansZero = (d) => { const m = /\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]/.exec(d || ""); return m && +m[1] < 0 && +m[2] > 0; };
    const checks = v ? `<table><thead><tr><th>核验项</th><th>复现</th><th>细节</th></tr></thead><tbody>${v.checks.map((c) =>
      `<tr><td>${esc(c.name)}</td><td><span class="badge ${c.ok ? "b-ok" : "b-bad"}">${c.ok ? "一致" : "不一致"}</span></td><td><code>${esc(c.detail)}</code>${spansZero(c.detail) ? ` <span class="badge v01" title="置信区间跨 0：差异不显著">区间含 0</span>` : ""}</td></tr>`).join("")}</tbody></table>`
      : `<div class="empty">还没有核验记录。运行 <code>python scripts/verify_proteus_port.py --with-bids</code> 后点“重新扫描源码”。</div>`;
    const scripts = CAT.scripts.map((s) => `<div class="sym"><header><span class="kind k-method">${esc(s.kind)}</span><span class="name">${esc(s.file)}</span>
      <span class="badge v01">${s.loc} 行</span>${s.tests.length ? `<span class="badge b-ok">${s.tests.length} 个测试</span>` : ""}
      <button class="toggle" data-script="${esc(s.file)}" aria-expanded="false">查看源码</button></header>
      ${s.summary ? `<div class="sum">${esc(s.summary)}</div>` : ""}${s.tests.length ? `<div class="methods">${s.tests.map((t) => `<code>${esc(t)}</code>`).join("")}</div>` : ""}
      <div class="src" hidden></div></div>`).join("");
    return header("测试与核验", v ? `最近核验 ${esc(v.time)}${v.with_bids ? "（含原始 EDF 重提取）" : ""}` : "") +
      `<div class="card"><h2>与历史结果 / 原实现的等价核验 ${v ? `<span class="badge ${v.passed === v.total ? "b-ok" : "b-bad"}">${v.passed}/${v.total}</span>` : ""}</h2>
       <p class="desc">“一致”只说明 bench 移植后能复现报告里的历史数字，不说明效应成立。基于 PROTEUS 十名公开参与者的数字都只算开发证据。</p>${checks}</div>
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
    for (const animation of CAT.animations || []) {
      if ((animation.title + " " + animation.group + " " + animation.bench + " " + animation.claim).toLowerCase().includes(t)) hits.push({ animation });
    }
    return header(`搜索：${q}`, `${hits.length} 个结果`) + (hits.length ? `<section class="animations">${hits.slice(0, 200).map(({ m, s, animation }) => animation
      ? `<button class="animation-link result-video" data-animation="${esc(animation.id)}" ${animation.available ? "" : "disabled"}><b>${esc(animation.title)}</b><small>${animation.available ? "播放对应视频" : "视频尚未渲染"} · ${esc(animation.group)} · ${esc(animation.module)}</small></button>`
      :
      `<button class="result" data-mod="${esc(m.name)}" ${s ? `data-sym="${esc(s.name)}"` : ""}>${s ? `<code>${esc(s.name)}</code> <span class="badge v01">${s.kind === "class" ? "类" : "函数"}</span> · ${esc(m.name)}`
        : `<code>${esc(m.name)}</code> <span class="badge v02">模块</span>`}<div class="desc" style="margin:2px 0 0">${esc((s ? symbolSummary(s) : m.desc) || "").slice(0, 140)}</div>${s && s.summary && s.summary !== symbolSummary(s) ? `<div class="english-note">${esc(s.summary)}</div>` : ""}</button>`).join("")}
        <div class="animation-player" hidden><h3></h3><video controls preload="metadata" playsinline></video><p class="desc"></p></div></section>`
      : `<div class="empty">没有匹配项</div>`);
  }
  // ---- rendering & navigation ----
  function renderNav() {
    $("#nav").innerHTML = VIEWS.map(([id, label]) => {
      const n = id === "modules" ? CAT.modules.length : id === "animations" ? CAT.stats.animation_videos : id === "migration" ? CAT.migration.length : id === "tests" ? CAT.stats.tests
        : id === "capabilities" ? capData().entries.length : "";
      return `<button data-view="${id}" ${state.view === id && !state.layer && !state.query && !state.module ? 'aria-current="page"' : ""}>${label}<span class="count">${n}</span></button>`;
    }).join("");
    $("#layerNav").innerHTML = CAT.layers.filter((l) => modsIn(l.id).length).map((l) =>
      `<button class="layer-link" data-layer="${l.id}" ${state.layer === l.id ? 'aria-current="page"' : ""}><span class="swatch" style="background:${COLORS[l.id]}"></span>${esc(l.name)}<span class="count">${modsIn(l.id).length}</span></button>`).join("");
    $("#brandMeta").textContent = `v${CAT.version} · ${CAT.modules.length} 个模块`;
    $("#footMeta").innerHTML = `源码：${esc(CAT.root)}<br>扫描：${esc(CAT.generated)}${CAT.snapshot ? "<br>（离线快照）" : ""}`;
    // the verification stamp stays on every view; a failure is stamped, never hidden
    const v = CAT.verify, st = $("#stamp");
    st.className = "stamp " + (!v ? "none" : v.passed === v.total ? "ok" : "bad");
    st.innerHTML = v ? `<b>${v.passed === v.total ? "复现核验一致" : "复现核验未通过"} ${v.passed}/${v.total}</b><small>${esc(v.time)}${v.with_bids ? " · 含 EDF 重提取" : ""}</small>`
      : `<b>没有核验记录</b><small>运行 verify_proteus_port.py 后重新扫描</small>`;
  }

  // station trace on the line map: the hovered/focused station and every station it imports or is imported by turn yellow
  function trace(name) {
    const metro = $("#metro"), out = $("#readout");
    if (!metro) return;
    const m = name && CAT.modules.find((x) => x.name === name);
    metro.classList.toggle("tracing", !!m);
    const linked = new Set(m ? [...m.imports_internal, ...m.used_by] : []);
    metro.querySelectorAll(".stn").forEach((b) => { b.classList.toggle("self", !!m && b.dataset.stn === name); b.classList.toggle("lit", linked.has(b.dataset.stn)); });
    if (!out) return;
    out.innerHTML = m ? `<code>${esc(m.name)}</code>（${esc(layerOf(m.layer).name)}）引用 ${m.imports_internal.length} 个模块${m.imports_internal.length ? `：${m.imports_internal.map(esc).join("、")}` : ""}；被 ${m.used_by.length} 个模块引用${m.used_by.length ? `：${m.used_by.map(esc).join("、")}` : ""}。`
      : "把指针移到站点上，会用黄色标出它和哪些站相连；点击进入该模块。";
  }

  function render(focusSym) {
    renderNav();
    let html;
    if (state.query) html = viewSearch(state.query);
    else if (state.module) html = viewModule(state.module, focusSym);
    else if (state.layer) html = viewLayer(state.layer);
    else html = { overview: viewOverview, animations: viewAnimations, graph: viewGraph, migration: viewMigration, tests: viewTests, capabilities: viewCapabilities,
      modules: viewModules }[state.view]();
    const main = $("#main");
    main.innerHTML = html;
    if (focusSym) { const el = document.getElementById("sym-" + focusSym); if (el) el.scrollIntoView({ block: "start" }); }
    else main.scrollTop = 0;
  }

  // push = a user navigation (adds a history entry so Back works); boot and Back/Forward themselves replace
  function go(next, focusSym, push = true) {
    state = { view: "overview", module: null, layer: null, query: "", ...next };
    const hash = state.module ? `module/${state.module}` : state.layer ? `layer/${state.layer}` : state.view === "capabilities" ? capHash() : state.view;
    if (location.hash.slice(1) !== hash) history[push ? "pushState" : "replaceState"](null, "", "#" + hash);
    render(focusSym);
  }
  window.addEventListener("popstate", () => { if (CAT) { $("#search").value = ""; go(fromHash(), null, false); } });
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
    const t = e.target.closest("[data-view],[data-layer],[data-mod],[data-toggle],[data-full],[data-script],[data-open],[data-cap],[data-capf],[data-capclear],[data-menu],[data-animation]");
    if (!t) return;
    if (t.dataset.menu) { document.body.classList.toggle("nav-open"); return; }
    if (t.dataset.animation) {
      const animation = (CAT.animations || []).find((item) => item.id === t.dataset.animation);
      if (!animation || !animation.available) return;
      const player = t.closest(".animations")?.querySelector(".animation-player");
      if (!player) return;
      const video = player.querySelector("video");
      player.querySelector("h3").textContent = animation.title;
      player.querySelector(".desc").textContent = `${animation.bench}。${animation.claim}`;
      video.src = `/animation/${encodeURIComponent(animation.id)}`;
      player.hidden = false;
      video.load();
      video.play().catch(() => {});
      player.scrollIntoView({ block: "nearest", behavior: "smooth" });
      return;
    }
    if (t.dataset.cap) { toggleCap(t.dataset.cap, true); return go({ view: "capabilities" }); }
    if (t.dataset.capf) {
      toggleCap(t.dataset.capf, false);
      const y = $("#main").scrollTop; history.replaceState(null, "", "#" + capHash()); render(); $("#main").scrollTop = y;
      return;
    }
    if (t.dataset.capclear) { Object.values(capFilter).forEach((s) => s.clear()); return go({ view: "capabilities" }); }
    document.body.classList.remove("nav-open");
    if (t.dataset.open) { try { await api.call("open_file", t.dataset.open); } catch (err) { toast("无法打开：" + err.message); } return; }
    if (t.dataset.view) { $("#search").value = ""; return go({ view: t.dataset.view }); }
    if (t.dataset.layer) { $("#search").value = ""; return go({ layer: t.dataset.layer }); }
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
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
    if (e.key === "/" && !typing && !e.ctrlKey && !e.altKey && !e.metaKey) { e.preventDefault(); $("#search").focus(); }
  });
  function graphTrace(e) {
    const node = e.target.closest && e.target.closest("g.node");
    const svg = document.querySelector("svg.graph"); if (!svg) return;
    const name = node ? node.dataset.node : null;
    svg.querySelectorAll(".edge").forEach((p) => p.classList.toggle("hot", !!name && (p.dataset.a === name || p.dataset.b === name)));
    const linked = new Set([name]);
    if (name) svg.querySelectorAll(".edge.hot").forEach((p) => { linked.add(p.dataset.a); linked.add(p.dataset.b); });
    svg.querySelectorAll("g.node").forEach((g) => { g.classList.toggle("dim", !!name && !linked.has(g.dataset.node)); g.classList.toggle("lit", !!name && linked.has(g.dataset.node)); });
  }
  document.addEventListener("mouseover", graphTrace);
  document.addEventListener("focusin", graphTrace);
  document.addEventListener("mouseover", (e) => { if ($("#metro")) { const s = e.target.closest && e.target.closest(".stn"); trace(s ? s.dataset.stn : null); } });
  document.addEventListener("focusin", (e) => { if ($("#metro")) { const s = e.target.closest && e.target.closest(".stn"); trace(s ? s.dataset.stn : null); } });
  document.addEventListener("input", (e) => {
    if (e.target.id === "search") { const q = e.target.value.trim(); q ? (state = { ...state, query: q }, render()) : go({}); }
    if (e.target.id === "migFilter") {
      const q = e.target.value.toLowerCase();
      let n = 0;
      document.querySelectorAll("#migTable tbody tr").forEach((tr) => { tr.hidden = q && !tr.textContent.toLowerCase().includes(q); if (!tr.hidden) n++; });
      $("#migCount").textContent = q ? (n ? `${n} / ${CAT.migration.length} 条匹配` : `没有匹配“${e.target.value}”的条目，换一个函数名或文件名试试`) : "";
    }
  });
  $("#rescan").addEventListener("click", async (e) => {
    const b = e.currentTarget;
    b.disabled = true; b.textContent = "正在扫描…";
    try { CAT = await api.call("rescan"); toast(`已重新扫描：${CAT.modules.length} 个模块`); render(); }
    catch (err) { toast("扫描失败：" + err.message + "。源码目录可用后再试一次"); }
    finally { b.disabled = false; b.textContent = "重新扫描源码"; }
  });

  let booted = false;
  async function boot() {
    if (booted) return;
    booted = true;
    try { CAT = await api.call("get_catalog"); go(fromHash(), null, false); }
    catch (err) { $("#main").innerHTML = `<div class="empty">读取失败：${esc(err.message)}</div>`; }
  }
  if (window.pywebview && window.pywebview.api) boot();
  else window.addEventListener("pywebviewready", boot, { once: true });
  setTimeout(() => { if (!CAT && !(window.pywebview && window.pywebview.api)) boot(); }, 600);
})();
