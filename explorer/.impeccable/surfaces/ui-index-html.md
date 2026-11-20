---
version: 1
slug: "ui-index-html"
primary_target: "ui/index.html"
related_targets: ["ui/app.js","ui/style.css"]
---

# Surface: WYU-BCI Bench Explorer（ui/index.html 及其全部 hash 视图）

## Scope and mode

整个 Explorer 窗口：总览、全部模块、分层、模块详情、能力标签、依赖关系、旧代码对照、测试与核验、搜索。Mode: Operate。

## Audience, job, constraints

实验室研究生/新生在 Windows 工作站（白天日光灯、系统浅色主题、与 VS Code 并排、1707×960 逻辑分辨率或半屏）查 bench：找接口、读源码、分清声明等级与核验结果。离线，无外部字体或请求；中文界面，标识符原样；保持全部现有交互、hash 路由与 data-* 钩子。

## Direction contract

THESIS: bench 是一张线路图：十个功能分层是十条线，模块是站，跨层引用是换乘。拒绝“暗底卡片网格 + 统计数字条”的 SaaS 仪表盘套路。

OWN-WORLD: 白底地铁导视。深炭色导视侧栏、白字；每层一种饱和线路色，只用于线、站点圆环、层标签；墨色正文；荧光黄只做高亮叠加（悬停路径、搜索命中），从不改动线路色；核验印记用实心状态色加文字。系统无衬线中文字体 + Cascadia 等宽仅用于标识符/代码/数字。无渐变、无辉光、无圆角大卡。

STORY: 新生先看到整张线路图，知道 bench 分几层、每层几站；点站进入站牌页，看到接口、源码、谁引用它、它声明做到什么程度；侧栏底部始终挂着核验印记。

FIRST VIEWPORT: 左 248px 炭色侧栏（品牌、搜索、视图、线路列表、核验印记）。主区顶部一行标题 + 版本/分支/扫描时间的元信息；其下即线路图：每层一条水平线路，左端层名标签，线上按顺序排站点（圆环 + 等宽站名 + 行数），换乘站加粗环；占满首屏宽度。统计只以一行文字出现。

FORM: 地铁线路图 / 车站导视（自有清单第 6 位）。seed:cedc1cef6e5cacfa

Raises:
- 来自 variable-font-specimen：层级只靠字号与字重对比，严格 4px 基线节奏，不靠装饰色块。
- 来自 jet-age ticket wallet：失败状态不隐藏，盖“未通过”印记留在原位。
- 来自 warm consumer app：唯一实心填充按钮是“重新扫描源码”；线路色永不用于可点击外观之外的装饰。
- 来自 Saville catalog sleeve：每个站牌页字段位置固定，同一信息永远在同一位置。
- 来自 orienteering map：高亮层（黄色）只叠加，不改写线路色与底图。

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

- 用户是否要求保留暗色（10:54 未记录的暗色改版来源未确认）；暗色版备份在 explorer/ui_backup_20261021_dark/。
