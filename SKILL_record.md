---
name: record
description: "Turn a finished piece of research -- or anything new the user brings in -- into a verdict record, put it in the prior library, and let the library re-summarise itself. Checks that the cause of death keys to the failure-mode taxonomy, and when it does not, proposes the new mode rather than forcing the record into the nearest fit; reports how the shape of the record changed and whether what a new study should check first has moved. Use after any study reaches a verdict, including one that survived, and whenever a new finding, paper, incident or correction arrives that the record does not yet know. Chinese requests: 记到判负库里、这个结果入库、更新判负库、把这条结论沉淀下来、重新归纳一下。"
---

# Record it, and let the library re-summarise itself

This is the step that makes the rest of it compound. A referee that judges and
forgets is a calculator. The library is what turns one post-mortem into
something the next study can retrieve, and it only exists if this happens every
time.

**Including the ones that survived.** A record of nothing but rejections
teaches only pessimism, and eleven of the verdicts here are survivals — they
are what tells you the machine is not simply saying no.

## When to run this

- a study reached a verdict — any verdict, `SURVIVES` included
- a conclusion you already held turned out to be wrong
- a paper, report or incident taught something the record does not know
- a project produced a trap: a data convention, an alignment rule, a silent
  failure

## Step 1 — write the claim as it was tested, not as it was hoped

One sentence, in the past tense, specific enough that retrieval can match it
later: the construction, the universe, the horizon, the cost. "动量有效" is not
a claim. "12-1 动量 ETF 轮动有选股 alpha" is.

## Step 2 — the cause of death, keyed to the taxonomy

```python
from falsifier import taxonomy as T
[m.id for m in T.MODES if "null" in m.family]
```

**If nothing fits, do not key it to the nearest thing that does.** An
approximate cause of death teaches the wrong lesson to whoever retrieves it
later. What it means is one of two things, and both are useful:

- the record is vague — go back and say what actually decided it, or
- **the taxonomy is short a mode** — write the mode first, then the record

Backfilling this library has twice opened the taxonomy back up, five modes the
first time and three the second. That is the intended direction of traffic, not
a failure.

## Step 3 — evidence is the number, not a summary

The field is called `evidence` because it holds the figure that decided it.
"零基准表现更好" is not evidence. "换手匹配后从 100 分位塌到 21.4 分位" is.

## Step 4 — the lesson is the part that transfers

The claim is about one study. The lesson has to be usable by a study that has
nothing else in common with it. Write what the next person should do
differently, not what happened.

Weak: "这个因子没有增量。"
Strong: "轮动/TopN 类策略一律先做换手匹配零基准,否则比的是手续费不是选股。"

## Step 5 — put it in and read what changed

```python
import falsifier as F
r = dict(id="...", claim="...", verdict="REJECTED", killed_by=["..."],
         evidence="...", lesson="...", applies_to=["..."],
         project="...", date="2026-09-15", source="...")
print(F.render_append(F.append_prior(r)))        # dry_run=True to rehearse
```

It refuses the record if the id collides, the schema does not fit, the cause of
death keys to nothing, or the lesson is too thin to teach. **Each refusal is
the check working.**

Then read the summary. It prints what a new study should check first, and says
so when that has changed — which it has twice: `null-random-structure-wins`,
then `signal-duplicates-existing`, now `cost-eats-it`. That number is the
library re-summarising itself, and it is the closest thing here to the machine
learning something.

## Step 6 — if it overturned something, retract that something

A library that only grows teaches its own mistakes forever. When this finding
overturns an earlier record, mark the earlier one:

```python
# in priors/verdicts.jsonl, on the record being retired:
"withdrawn": "为什么它不再成立, 以及新的证据是什么",
"superseded_by": "<新记录的 id>",
"withdrawn_date": "2026-09-15"
```

A withdrawn verdict **stays searchable** — it is exactly what you want surfaced
when you are about to re-derive it — but leaves the checklist of traps, because
a trap that turned out not to be one is not a trap.

**Both retractions here had a `lesson` that had become bad advice**, not just a
stale number. That is the case to watch for: the number being wrong costs one
study, and the lesson being wrong costs every study that retrieves it.

## Step 7 — check it did not land on top of something

```python
print(F.validate_priors() or "clean")
[x.id for x in F.load_priors() if x.project == "<项目名>"]
```

And before the next study on the same project, read that list. Two records
written on one day here carried conclusions a robustness review had overturned
a month earlier; they were caught only because someone happened to open the
right memory.

## 定期补库:怎么找还没入库的东西

每做完一个研究就追加一条,是理想状态;实际会漏,而漏掉的往往是**方法论**而不是结论——
因为结论有人会记得,方法论以为"写在项目文档里了"。

2026-09 做过三轮,合计补 26 条。**三个矿脉,按产出排序**,每一个查的是不同的东西:

### 矿脉一:哪些**记忆**从没产出过记录(产出 5 条)

每条记录都带 `source`,所以这个查得很干净:

```python
import json, os
used = {json.loads(l).get('source','') for l in open('priors/verdicts.jsonl',encoding='utf-8')}
never = [f for f in sorted(os.listdir('memory'))
         if f.endswith('.md') and f != 'MEMORY.md' and f not in used]
```

★大部分命中**不是研究主张**(数据盘点、环境、图表约定、课程作业),要先剔。
剩下的才是真缺口。

### 矿脉二:哪些**项目目录**在库里零记录(产出 8 条)

判负库是从记忆蒸馏的,而记忆是会话写的——**项目自己的 `docs/` 里可能有从没进过记忆的东西**。

```bash
# 扫目录 → 对照 {x.project for x in F.load_priors()} → 按 docs 里的判决行数排序
grep -cE "判负|不成立|未复现|全灭|归零|反而|相反" $d/**/*.md
```

★**多数"缺口"是命名误报**(`esg` = `gws-factor`、`神奇九转` = `td9` 都早已入库),
按判决行数排序能直接把真候选顶上来。
★这条矿脉挖到了库里**最缺的一类:选题阶段的判负**——在写第一行研究代码之前用探针
实测把论文前提逐条兑现,四条全部成立。**可行性判负和结果判负一样值钱,而且便宜得多。**

### 矿脉三:归档仓 / 文档有而记忆没有的(产出 13 条)

对**已经入过库的项目**,再扫一遍它的文档,找只写在文档里的那些。

```bash
grep -rlE "自我推翻|我一度|读错|撤回|原以为|意外|反而|推翻" --include="*.md" <项目根>
```

★**专找"意外"**:反号、翻转、自我推翻、与预期相反。那一类最容易只留在文档里,
而它恰恰是转移价值最高的。这一轮里"七个美国版复现有六个与论文方向相反"就是这么挖出来的。

### 矿脉四:**机器自己**(2026-09-19 新开,产出 10 条)

★这条是最晚发现的,也是最该早做的。判负机器的校准过程本身就是一批对**研究实践**的判负——
不是这个包的 bug 报告,是"你一直在用的那条审计规则在 h>1 时无效"这种。
它们写在 `docs/calibration.md` 和项目记忆里,而 **`/prior-check` 搜的是判负库,不是文档**。

> **沉淀在文档里 = 检索不到 = 下次还会犯。**

判据:一条校准发现如果**换个人、换个项目也会踩**,它就是判负库的内容;只对这份代码成立的
(某个函数签名、某个默认值)留在 calibration.md。

### 矿脉五:**成对 / 多版本项目只入了一半**(产出 3 条)

按项目名去重会让这类缺口完全看不出来,因为那个名字"已经在库里了"。

```python
ps = {x.project for x in F.load_priors()}
# 中美版、v1/v2/v3、公开树/私有树:逐对检查两边都在不在
[d for d in os.listdir('.') if d not in ps]
```

★这轮命中三对:`Bakshi2019_CN` 在库而 `_US` 不在、`GoodBadVol2025_US` 在而 `_CN` 不在、
`Investable2021_US` 在而 `_CN` 不在。而 `Investable2021_CN` 里恰好放着**库里一条已有结论的源头**
——"动态换月推翻我们自己四处中国结论"早就入库了,**触发它的那次诊断**(固定日历档位有
21~22% 的月末选中当日零成交合约)却没有。

★★推广成一条查法:**对库里每条"后来被推翻/被修正"的记录,问一句「推翻它的那个东西入库了吗」**。
结论比诊断更容易被记住,而下次能救人的是诊断。

### 矿脉六:**记录只覆盖了一角**的项目(2026-09-19 新开,产出 37 条,本条最肥)

前五条矿脉查的都是"有没有",这一条查"够不够"。做了很久的主力项目,**早期结论入了库,
后面十轮的方法论没有** —— 因为那时候已经"记过了"。

查法是一个比值:**记忆里的判负行数 ÷ 该项目已有记录数**,按比值排序。

```python
v = len(re.findall(r'判负|证伪|不成立|推翻|全灭|归零|反号', memory_text))
n = sum(1 for r in P if 项目名 in r.project)
# 比值 > 6 的逐个打开
```

★实测比值 >6 的**全部有真缺口**:comovement-network-anchor 13.5(27 行 / 2 条)、
commodity-equity-linkage 13.0、l2-kline-breadth-factor 12.0、方正 12.0、
style-contamination 11.0、falsifier 自己 9.1。
★**比值低不代表没有**:afternoon-breakdown 比值只有 2.4,但绝对量 34 行,仍补出 10 条 ——
所以**绝对行数 ≥20 的也要打开看一眼**,不管比值。

★★这道缝比"零记录"那道产出高得多。零记录的往往是小项目;记录只覆盖一角的往往是主力项目,
而主力项目的后期方法论正是最贵的那部分。

## 补库时的五条纪律

1. **读实际数字,不要照索引那一行写。** 索引是压缩过的,复述它等于把压缩损失固化进库。
2. **尊重原作者的判断。** 有一条位次实测原作者自己写了"样本仅 2 票 × 3 天,是描述统计不是
   结论,未入判负库"——那个判断是对的。但其中**不依赖样本量**的那半(论文的前提在这个市场
   是假的,这是关于数据格式的事实)可以入,写进教训里而不单独立条。
3. **别人的数字不进库。** 外部主张只有在这里被测过之后才进,带我们自己的证据。
4. **死因 key 不上就停下来。** 那是最有用的那次失败:要么记录含糊,要么分类法短一个模式。
   四轮里这样补出了 7 个新模式。
5. **表观矛盾要当场解开,写成第三条记录,两条都留着。** 库里"把截面砍到 1/20,判别力不降
   ⇒ 观测同源"这个手法是对的(它量的是 **bp**),但写成检查时用 **IC** 就会全误杀
   (IC 是相关系数,**与 N 无关**)。不要撤回任何一条——新写的那条说明前一条在什么条件下成立。
   ★**不要上自动矛盾检测**:试过三种判据,全在健康的库上狼来了(22/22/185 对)。

## 什么时候停

产出会掉。前三条矿脉挖完之后剩下的多是命名误报和非研究目录,**再往下会开始重复,或者挖到
不值得入库的描述统计**。停在那里,下次有新项目了再扫。

★但**矿脉四、五、六不会因为"挖过了"而枯竭**:每次改机器都会产生新的校准发现,每次做中美版
或 v2 都会产生新的半边,每一轮新实验都会让主力项目的"记录/记忆"比值重新掉下来。
把它们放进收尾清单,而不是补库清单。

## What this is not

It is not a lab notebook. A record is one claim, one verdict, one number, one
transferable lesson. Everything else belongs in the project's own documents —
and if a record needs a paragraph to explain, the claim was not stated sharply
enough in step 1.
