---
name: WYU-BCI Bench Explorer
description: 把 wyu-bci-bench 画成白底地铁线路图的本地源码目录：十个分层是线路，模块是站，跨层引用是换乘。
colors:
  line-ingest: "#1f5fc4"
  line-signal: "#0a8a7a"
  line-features: "#d0661a"
  line-classic: "#6c3fc4"
  line-deep: "#c4307a"
  line-foundation: "#0e86b8"
  line-protocol: "#3f8a1f"
  line-stats: "#c42b2b"
  line-results: "#9a6b12"
  line-interface: "#5f6873"
  line-other: "#7a828c"
  trace: "#ffd319"
  trace-soft: "#fff3b8"
  ok: "#136a2f"
  ok-soft: "#e3f3e8"
  bad: "#c42b2b"
  bad-soft: "#fbe5e5"
  stamp-ok: "#6fd38f"
  stamp-bad: "#ff8b8b"
  stamp-none: "#e7b45a"
  ground: "#ffffff"
  ground-2: "#f4f5f7"
  ground-3: "#eceef1"
  ink: "#16191d"
  ink-2: "#454c55"
  dim: "#636c78"
  rule: "#dde0e5"
  rule-2: "#c3c8cf"
  neutral-ring: "#8a929c"
  rail: "#1f2329"
  rail-2: "#2b3038"
  rail-ink: "#f2f4f6"
  rail-sub: "#a7aeb8"
  rail-nav-ink: "#d5d9df"
  rail-edge: "#3a414b"
  rail-well: "#15181d"
  syntax-keyword: "#a01c6d"
  syntax-string: "#1e6b2a"
  syntax-comment: "#5f6873"
  syntax-number: "#a14a00"
  syntax-decorator: "#7c4a00"
  syntax-definition: "#0b4fa8"
  syntax-builtin: "#005f73"
typography:
  headline:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.2px"
  headline-mono:
    fontFamily: "Cascadia Mono, Cascadia Code, Consolas, Microsoft YaHei UI, monospace"
    fontSize: "22px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0"
  title:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.6
  body:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.6
  body-dense:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.6
    fontFeature: "tnum"
  badge:
    fontFamily: "Segoe UI Variable Text, Segoe UI, Microsoft YaHei UI, Microsoft YaHei, PingFang SC, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: "18px"
    fontFeature: "tnum"
  station:
    fontFamily: "Cascadia Mono, Cascadia Code, Consolas, Microsoft YaHei UI, monospace"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.35
  figure:
    fontFamily: "Cascadia Mono, Cascadia Code, Consolas, Microsoft YaHei UI, monospace"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1
    fontFeature: "tnum"
  code:
    fontFamily: "Cascadia Mono, Cascadia Code, Consolas, Microsoft YaHei UI, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  hair: "2px"
  stamp: "3px"
  badge: "4px"
  control: "6px"
  pill: "999px"
  ring: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  section: "18px"
  page-top: "26px"
  page-x: "36px"
  rail-width: "248px"
  station-cell: "clamp(124px, calc((100vw - 248px - 72px - 224px) / 5), 280px)"
components:
  button-rescan:
    backgroundColor: "{colors.rail-ink}"
    textColor: "{colors.ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.control}"
    padding: "8px"
    width: "100%"
  button-rescan-hover:
    backgroundColor: "{colors.ground}"
  button-toggle:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "2px 12px"
  button-toggle-expanded:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  rail-nav-item:
    backgroundColor: "{colors.rail}"
    textColor: "{colors.rail-nav-ink}"
    rounded: "{rounded.control}"
    padding: "7px 10px"
  rail-nav-item-current:
    backgroundColor: "{colors.rail-2}"
    textColor: "{colors.ground}"
  input-rail-search:
    backgroundColor: "{colors.rail-well}"
    textColor: "{colors.rail-ink}"
    rounded: "{rounded.control}"
    padding: "8px 34px 8px 12px"
  input-filter:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "7px 12px"
    width: "320px"
  line-chip:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "2px 12px 2px 8px"
  filter-chip:
    backgroundColor: "{colors.ground-3}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "3px 12px"
  filter-chip-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  badge-v02:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    typography: "{typography.badge}"
    rounded: "{rounded.badge}"
    padding: "0 7px"
  badge-ok:
    backgroundColor: "{colors.ok-soft}"
    textColor: "{colors.ok}"
    rounded: "{rounded.badge}"
    padding: "0 7px"
  badge-bad:
    backgroundColor: "{colors.bad-soft}"
    textColor: "{colors.bad}"
    rounded: "{rounded.badge}"
    padding: "0 7px"
  claim-design-intent:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    rounded: "{rounded.stamp}"
    padding: "0 7px"
  claim-evaluation:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.stamp}"
    padding: "0 7px"
  claim-measurement:
    backgroundColor: "{colors.ground-3}"
    textColor: "{colors.ink}"
    rounded: "{rounded.stamp}"
    padding: "0 7px"
  symbol-tile:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  symbol-tile-hover:
    backgroundColor: "{colors.ground-2}"
  doc-panel:
    backgroundColor: "{colors.ground-2}"
    textColor: "{colors.ink-2}"
    typography: "{typography.body-dense}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
  code-block:
    backgroundColor: "{colors.ground-2}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
  station-ring:
    backgroundColor: "{colors.ground}"
    rounded: "{rounded.ring}"
    size: "20px"
  station-ring-transfer:
    size: "26px"
  verification-stamp:
    backgroundColor: "{colors.rail}"
    textColor: "{colors.stamp-ok}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
  toast:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    rounded: "{rounded.control}"
    padding: "9px 16px"
---

# Design System: WYU-BCI Bench Explorer

## Overview

**Creative North Star: "白底地铁导视"**

整套界面是一张地铁线路图加一套车站导视。bench 的十个功能分层是十条线路，每条线一种饱和线路色；模块是站，跨层被引用的模块是换乘站；模块详情页是站牌页。底图是白纸与墨色正文，左侧是一条深炭色导视侧栏（248px），侧栏里白字、线路色条、核验印记。颜色的工作只有三种：线路色标“这是哪条线”，荧光黄标“你在这里 / 与它相连”，状态色标“核验结果”。其余一切是墨、灰与细线。

密度偏高，适合与 VS Code 并排在 1707×960 左右的 Windows 窗口里工作：正文 14px，表格与说明 13.5px，标识符、站名、签名、数字一律 Cascadia 等宽。层级只靠字号、字重与等宽/无衬线的切换建立，不靠装饰色块。界面平铺无投影，分区靠 1px 细线和浅灰底板（`ground-2`）。

已在构建中落实的拒绝项：不用暗底卡片网格加 KPI 统计条（统计只在总览以一行文字出现），不用渐变、辉光、渐变文字，不加载外部字体或任何网络资源。

**Key Characteristics:**
- 线路色只出现在线条、站点圆环、换乘色点、线路标签与色条上。
- 荧光黄是叠加层：悬停追踪、当前站、选区、焦点，从不改写线路色。
- 声明等级与核验结果用墨色印记的“形状”区分，不借线路色。
- 系统无衬线中文字体 + Cascadia 等宽；等宽只服务于标识符、代码和数字。
- 平铺无阴影；细线分区；圆角克制（2–6px），胶囊与圆环是线路图自带的形。
- 唯一的实心填充按钮是侧栏底部的“重新扫描源码”。

## Colors

一套高饱和的线路色家族，压在白纸、墨色正文和一条深炭色侧栏之上，外加一种只做叠加的荧光黄。

### Primary
线路色，每层一色，全部对白底 ≥ 3:1（实测 3.75–6.68:1，`line-features` 最低 3.75:1，`line-other` 3.89:1），因为它们画的是 4–6px 的线和 4px 的环，不是正文。
- **数据接入蓝**（`line-ingest`）、**信号青绿**（`line-signal`）、**特征橙**（`line-features`）、**经典机器学习紫**（`line-classic`）、**深度模型品红**（`line-deep`）、**基础模型湖蓝**（`line-foundation`）、**划分与评估草绿**（`line-protocol`）、**指标与统计红**（`line-stats`）、**结果与提交赭金**（`line-results`）、**可视化与命令行石板灰**（`line-interface`）、**其他浅石板灰**（`line-other`）。
- 值的唯一来源是 `ui/app.js` 顶部的 `COLORS` 对象，通过 `--c` 自定义属性或内联 `background`/`fill`/`stroke` 下发；`style.css` 不重复定义。

### Secondary
- **导视荧光黄**（`trace`）：线路图上悬停/聚焦站及其引用站的圆环填充与站名底色、站牌页条带上的“本站”、依赖图节点悬停/选中底色、文本选区、侧栏当前视图的圆点、侧栏内的焦点框。墨色压在上面 12.25:1。
- **淡荧光黄**（`trace-soft`）：主区筛选输入框聚焦时的 3px 光圈。

### Tertiary
核验与状态色，只在徽章、核验表和侧栏印记上出现。
- **通过绿 / 通过绿底**（`ok` / `ok-soft`）：“一致”、“仅 numpy”、测试计数徽章，5.83:1。
- **未通过红 / 未通过红底**（`bad` / `bad-soft`）：“不一致”、核验未全过的计数；徽章另加 1px 同色描边，4.68:1。
- **侧栏印记三色**（`stamp-ok` / `stamp-bad` / `stamp-none`）：深炭侧栏上的核验印记文字与 2px 边框，分别为“复现核验一致”、“复现核验未通过”、“无核验记录”，对 `rail` 7.0–8.6:1。
- 注意：`bad` 与 `line-stats` 当前是同一个值，`ok` 与 `line-protocol` 色相相近；这是已知撞色，不是规则，新界面不要依赖“红 = 指标与统计线”来传达失败。

### Neutral
- **白纸**（`ground`）：主区底色、输入框、轮廓按钮与站点圆环的内芯。
- **浅灰底板**（`ground-2`）：代码块、docstring 面板、行内代码、悬停行与悬停瓦片。
- **深一级底板**（`ground-3`）：`v0.1+v0.2` 徽章、函数类型标记、流程阶段标签、“度量”印记。
- **墨**（`ink`）：正文、标题、实心徽章（`v0.2`、“类”、“设计意图”）、按下态的筛选 chip 与展开态按钮、表头下的 1px 粗线、换乘站外圈。
- **次墨**（`ink-2`）：说明文字、签名、次要正文，8.68:1。
- **暗注**（`dim`）：元信息、行号、计数、表头、英文 docstring 注释，对白 5.32:1、对 `ground-2` 4.88:1。
- **细线 / 次细线**（`rule` / `rule-2`）：分区线、表格行线；轮廓控件与徽章描边、依赖图边线。
- **中性环灰**（`neutral-ring`）：“机制”印记描边、“处理目标”虚线标签、图例里的示意圆环、依赖图箭头，3.15:1，仅作非文字图形。
- **导视炭 / 炭二**（`rail` / `rail-2`）：侧栏底色；侧栏悬停与当前项底色。
- **侧栏白字 / 侧栏灰字 / 侧栏导航字**（`rail-ink` / `rail-sub` / `rail-nav-ink`）：品牌与印记计数；元信息与分组名（7.06:1）；导航项默认字色。
- **侧栏边线 / 侧栏井**（`rail-edge` / `rail-well`）：搜索框与 `/` 键帽的描边；搜索框内凹的底色。
- **语法色**（`syntax-*`）：仅限源码块内的 Python 高亮。

### Named Rules
**The Line Colour Rule.** 线路色只画线路：线条、站环、换乘色点、线路标签色条、依赖图节点描边、模块表分组下划线。它从不用作文字色、按钮底色、状态色或装饰底。

**The Overlay Rule.** 黄色只叠加，不替换。被追踪的站，圆环内芯和站名底变黄，环本身仍是线路色；其余站降到 0.3 透明度，线条降到 0.45。

**The Stamp-Not-Hue Rule.** 五种声明等级（机制 / 设计意图 / 评估用途 / 度量 / 产物管理）只靠印记形状区分：灰描边、墨实心、墨描边白底、灰底无边、点线描边。任何声明等级都不借线路色或状态色。

## Typography

**Display Font:** 无独立展示字体；标题沿用正文栈
**Body Font:** Segoe UI Variable Text（回退 Segoe UI、Microsoft YaHei UI、Microsoft YaHei、PingFang SC、system-ui）
**Label/Mono Font:** Cascadia Mono（回退 Cascadia Code、Consolas、Microsoft YaHei UI）

**Character:** 系统中文无衬线负责说话，Cascadia 负责“原样照抄”：模块名、函数名、签名、文件路径、行数和统计数字都用等宽，读者一眼分清哪些是解释、哪些是能复制进代码的东西。离线运行，不加载任何外部字体。

### Hierarchy
- **Headline**（700，24px，1.3，-0.2px）：视图标题，如“线路图”“全部模块”。窄窗口降到 21px。
- **Headline Mono**（600，22px，1.3）：模块页标题 `wyu_bci_bench.xxx`，窄窗口 18px 并允许断词。
- **Title**（700，16px）：分区标题（“怎么读”“公开接口”“完整源码”）。
- **Body**（400，14px，1.6）：全局正文。
- **Body Dense**（400，13.5px，1.6）：表格、说明段、键值表、docstring 面板；说明段限宽 78ch，docstring 76ch，符号摘要 90ch，签名 120ch。
- **Station**（600，13px，1.35，等宽）：线路图站名，`.` 后允许换行；站牌条带里降为 500/12.5px，本站 700。
- **Figure**（600，15px，等宽，等宽数字）：总览标题下那一行统计里的数字。
- **Code**（400，12.5px，1.6，等宽）：源码块、签名、方法列表、行内代码。
- **Label / Badge**（12px；徽章 600、行高 18px）：计数、元信息、版本与依赖徽章；声明印记 700/11.5px、行高 17px。全部计数开 `tabular-nums`。

### Named Rules
**The Mono-Means-Literal Rule.** 等宽字体只给会被原样复制或比对的东西：标识符、路径、签名、源码、行号、数字。中文解释、按钮文字、标题永远用无衬线。

**The Weight-Not-Paint Rule.** 层级靠字号与字重（400 / 600 / 700）和细线建立，不靠彩色底块或彩色标题。

## Layout

两栏固定框架：左侧 248px 深炭侧栏（品牌、搜索、视图导航、线路列表、底部核验印记与“重新扫描源码”），右侧主区独立滚动，内边距 26px 36px 64px。主区顶部永远是标题行（标题 + 可选线路 chip）加一行元信息；总览额外多一行文字统计。

线路图是一个两列网格：左列 200px 线路标签（色条 + 层名 + “N 站”），右列是站点轨道。所有线路共用一个站距 `station-cell`，因此线条长度与站数成正比，站点在各线之间纵向对齐。每条线最小高 72px。

内容区不用卡片网格分块，而是“细线分区”：每个分区上方 1px `rule`，上下内边距 18px / 22px。只有符号瓦片列表用 `auto-fill, minmax(260px, 1fr)` 网格，间距 8px。键值表为 88px 标签列 + 内容列，行距 6px、列距 16px。

间距值集中在 4 / 8 / 12 / 16px，分区与页边用 18 / 22 / 26 / 36px；并非严格 4px 基线（构建中仍有 7、9、10、14px 的内边距）。

响应式两档：
- **≤ 1180px**：线路标签移到轨道上方，站距改为 `max(100px, calc((100vw - 340px) / 5))`。
- **≤ 820px**：侧栏变为 264px 抽屉（左上 40px 菜单按钮呼出，背后 45% 墨色遮罩）；主区内边距 64px 16px 40px；线路改为竖排，站点横排成行；网格全部单列。

## Elevation & Depth

平面系统，无阴影。深度只来自三种手段：侧栏与主区的明暗对比、`ground` / `ground-2` / `ground-3` 三级底板、1px 细线（表头下改为 1px 墨线）。悬停只换描边色或底色，不位移、不抬升。

### Shadow Vocabulary
- **抽屉遮罩**（`box-shadow: 0 0 0 100vmax rgba(22, 25, 29, .45)`）：仅窄窗口下侧栏抽屉打开时，遮住主区。
- **输入聚焦光圈**（`box-shadow: 0 0 0 3px var(--trace-soft)`）：仅主区筛选输入框聚焦时。

### Named Rules
**The Flat Station Rule.** 任何面板、瓦片、按钮在任何状态下都不投影。需要强调时换墨色描边或换底板一级。

## Shapes

形状语言来自线路图本身：圆环、胶囊、圆角线条是“原生形”，矩形控件只用很小的圆角。

- **圆环**（50%）：站点（20px，4px 线路色边；换乘站 26px，外加 2px 墨色外圈、偏移 2px）、换乘色点（10px）、侧栏当前视图圆点（6px）。
- **胶囊**（999px）：线路 chip、能力标签、筛选 chip。
- **线条**：总览线路 6px 高、3px 圆角；站牌条带 4px；线路标签色条 26×10px、侧栏色条 14×6px。
- **矩形圆角**：站名高亮底 2px；声明印记与焦点框 3px；徽章与行内代码 4px；按钮、输入框、瓦片、代码块、docstring 面板、依赖图外框、核验印记 6px。没有大于 6px 的矩形圆角。
- **描边**：1px 为常态；未通过徽章 1px 同色；声明印记 1.5px；核验印记 2px `currentColor`。虚线描边专指“可选 / 方法 / 处理目标”，点线描边专指“产物管理”。

## Components

### Buttons
- **Shape:** 小圆角矩形（6px）。
- **Primary（唯一实心）：** “重新扫描源码”，侧栏底部通栏，`rail-ink` 底、`ink` 字、600/13px、内边距 8px；悬停变纯白；扫描中禁用为 0.6 透明度、`progress` 光标并改字“正在扫描…”。
- **Toggle（轮廓）：** “查看源码 / 展开”，白底、1px `rule-2` 描边、2px 12px、12.5px；悬停描边变墨；展开态（`aria-expanded="true"`）墨底白字。符号行里固定靠右上。
- **Link（文字）：** 等宽 13px、墨色、下划线用 `rule-2`，偏移 3px；悬停下划线变墨并加粗到 2px。用于模块名、文件路径、引用列表。
- **Focus：** 全局 2px 墨色外框、偏移 2px、3px 圆角；侧栏内改为 `trace` 黄框。

### Chips
- **线路 chip：** 白底胶囊、1px `rule-2` 描边、600/13px，左侧 18px 线路色条；悬停描边变墨。出现在线路页与模块页标题旁。
- **能力标签：** 12.5px 胶囊，三个维度靠形区分：流程阶段 = `ground-3` 实底；方法机制 = 白底 `rule-2` 实线；处理目标 = 白底 `neutral-ring` 虚线。
- **筛选 chip：** 同上胶囊加计数（11.5px `dim`）；按下态（`aria-pressed="true"`）墨底白字，计数变浅灰。

### Badges & Claim Stamps
- **版本徽章：** `v0.1` 白底灰描边；`v0.2` 墨实心白字；`v0.1+v0.2` `ground-3` 底。行号范围 `L20–47` 复用 `v0.1` 样式。
- **依赖徽章：** “torch”`ground-3` 实底；“可选 / 需 PyTorch”虚线描边；“仅 numpy”通过绿。
- **核验徽章：** 一致 = 通过绿底绿字；不一致 = 红底红字加红描边；置信区间跨 0 时追加中性 `v0.1` 样式说明徽章。
- **声明印记：** 700/11.5px、3px 圆角、1.5px 描边，形状见 The Stamp-Not-Hue Rule。

### Cards / Containers
- **Corner Style:** 分区不画框；仅符号瓦片、docstring 面板、代码块、依赖图外框用 6px。
- **Background:** 分区透明；docstring 面板与代码块 `ground-2`；瓦片白底，悬停 `ground-2`。
- **Shadow Strategy:** 无，见 Elevation & Depth。
- **Border:** 分区上方 1px `rule`；瓦片 1px `rule`，悬停变墨。
- **Internal Padding:** 分区 18px 0 22px；瓦片 10px 12px；面板与代码块 12px 16px。

### Inputs / Fields
- **侧栏搜索：** 内凹在 `rail-well` 底、1px `rail-edge` 描边、6px 圆角、13px；右侧 `/` 键帽提示快捷键；聚焦时描边变 `trace` 黄，无外光。
- **主区筛选：** 白底、1px `rule-2`、6px、宽 320px；聚焦描边变墨并加 3px `trace-soft` 光圈；右侧跟一个 13px `dim` 的命中计数。

### Navigation
- **侧栏视图导航：** 13.5px、`rail-nav-ink` 字、7px 10px、6px 圆角；悬停 `rail-2` 底白字；当前项 `rail-2` 底、白字、600，左侧 6px 黄圆点；计数右对齐 12px `rail-sub`。
- **侧栏线路列表：** 13px，左侧 14×6px 线路色条，右侧站数。分组名“线路（功能分层）”为 12px `rail-sub`。
- **窄窗口：** 侧栏变为抽屉，见 Layout。

### Metro Line Map（签名组件）
总览的整张线路图。每条线：左侧线路标签（26×10px 色条 + 13.5px 层名 + “N 站”），右侧一条 6px 线路色轨道，轨道终止于最后一站中心。每站是一个无边框按钮：线路色圆环、等宽站名、`dim` 色“N 行”与版本徽章；换乘站环更大、加墨色外圈，站名下排出引用它的其他线路的色点。悬停或聚焦一站时进入追踪态（The Overlay Rule），下方 `aria-live` 读出行说明连通关系。站点悬停另有 `ground-2` 底。

### Station Strip（站牌条带）
模块页顶部的所在线路缩略：4px 线路色条，132px 一站，16px 小环；本站环与站名用黄色标出，站名加粗。告诉读者“你在这条线的第几站”。

### Verification Stamp（核验印记）
侧栏底部常驻、点击进入“测试与核验”。透明底、2px `currentColor` 边、6px 圆角；第一行 700/14px 结论加计数（如“复现核验一致 29/29”），第二行 11.5px `rail-sub` 时间与“含 EDF 重提取”。颜色随状态取 `stamp-ok` / `stamp-bad` / `stamp-none`。未通过时照样盖在原位，不隐藏。

### Symbol Row（接口行）
模块页每个公开函数/类一行：类型标记（类 = 墨实心，函数 = `ground-3`，方法 = 虚线）、等宽 700 名称、行号徽章、可选依赖徽章、右上“查看源码”；签名单独一行，只在 “, ” 处断行；然后中文摘要、可选 docstring 原文（`dim` 12px）、方法列表、能力标签。字段位置固定。

### Code Block
`ground-2` 底、6px、12.5px 等宽、最高 520px 滚动；3.4em 宽的 `dim` 行号不可选中；Python 高亮仅用 `syntax-*` 七色。

### Toast
右下角 22px 处墨底白字、6px 圆角；0.2s 淡入并上移 6px（`cubic-bezier(.16, 1, .3, 1)`），2.2s 后消失。

## Do's and Don'ts

### Do:
- **Do** 从 `ui/app.js` 的 `COLORS` 取线路色，用 `--c` 下发给线条、圆环、色条；新视图要标层就复用线路 chip 或 26×10px 色条。
- **Do** 用黄色 `trace` 表示“当前 / 与它相连 / 被选中”，并保持被叠加元素的线路色描边不变。
- **Do** 标识符、路径、签名、源码、行数和计数一律用 Cascadia 等宽，计数开 `tabular-nums`。
- **Do** 用 1px `rule` 细线和 `ground-2` 底板分区；悬停只换描边或底色。
- **Do** 核验失败照样显示在原位，用 `bad` 徽章或 `stamp-bad` 印记加文字结论。
- **Do** 区分声明等级只用五种印记形状，并在旁边保留中文名称。
- **Do** 正文对比度保持 ≥ 4.5:1（`dim` 对 `ground-2` 已是下限 4.88:1，不要放在更深的底上）。

### Don't:
- **Don't** 把线路色用作文字色、按钮底色、状态色或装饰底；也不要给声明等级套线路色。
- **Don't** 让黄色替换线路色（例如把站环描边改黄）。
- **Don't** 增加第二个实心填充按钮；展开态 toggle 与按下态 chip 的墨底只是状态，不是默认样式。
- **Don't** 给面板、瓦片或按钮加投影、渐变、辉光或渐变文字；阴影只限抽屉遮罩与筛选框聚焦光圈。
- **Don't** 使用大于 6px 的矩形圆角或大圆角卡片；圆形与胶囊只留给站环、色点和 chip。
- **Don't** 在总览以外重复全库统计，也不要把统计做成数字卡片条。
- **Don't** 引入外部字体、CDN 或任何网络资源。
