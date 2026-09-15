---
name: mechanism
description: "Work out the economic account of a quantitative result that looks like it works -- why this edge should exist, who is on the other side, and what else would have to be true if the story were right. Produces a falsifiable mechanism and the implications that can come back false, which is what Prereg.mechanism and Prereg.implications require and what P1/P6 hold the study to. Use when a factor, signal or strategy has produced a promising number and the economic explanation has not been written down yet, and before the number is believed or reported. Chinese requests: 这个因子的经济含义是什么、为什么会有这个效应、谁在对面、这个解释站得住吗、机制怎么写。"
---

# Find the mechanism, or establish that there isn't one

A number that works and a story that explains it are two separate claims. This
skill produces the second one in a form that can be tested, because a mechanism
that cannot fail is not an explanation — it is a caption.

The point is not to make the result more believable. **The point is that a real
mechanism tells you where the result will stop working**, and that is the part
that transfers to the next study. A factor with a number and no mechanism is
worth exactly one backtest; a factor with a mechanism is worth a research line.

## Rules of engagement

1. **A mechanism written after seeing the result is a description of the
   result.** If the number came first — and it usually did — say so plainly.
   Then only the implications count as evidence, because they are the only part
   that was not fitted to what you already saw.
2. **A story that explains everything explains nothing.** If it is equally
   comfortable with the opposite sign, it is not a mechanism. Ask yourself what
   result would have made you tell a different story; if there isn't one, stop.
3. **Every mechanism owes a differential prediction.** Not "this should work",
   but *where* it should be stronger and *where* weaker. A mechanism with no
   cross-sectional and no time-series prediction has not been stated yet.
4. **Name who is on the other side.** Someone is taking the losing end of this
   trade, repeatedly, and not stopping. If you cannot say who they are and why
   they keep doing it, you have not found the mechanism — you have found a
   correlation with a caption.
5. **Do not let the mechanism rescue a dead number.** If the claim failed the
   gauntlet, a good story does not revive it. The order is: number survives,
   then mechanism. Never the reverse.

## Step 1 — say what the position actually is

Not the formula. The position. *When* does it go on, *what* does it hold,
*against whom*, and *how long*. Write one sentence a trader would recognise.
Most "mechanisms" dissolve here, because the position turns out to be something
already named — short-term reversal, a size tilt, a liquidity premium wearing a
new label.

## Step 2 — which family is it?

A persistent edge has to come from one of these. Pick one, and say why the
others are not it. "Several at once" is usually a sign that none has been
identified.

| family | you are paid for | the prediction that would falsify it |
|---|---|---|
| **风险溢价** risk premium | bearing something that hurts | it should be **worst exactly in the bad states**. An edge that is positive in every regime and every year is not compensation for risk — that is the reason "all years positive" is a warning, not a feature |
| **行为偏差** behavioural | someone systematically mis-reacting | the error should be **largest where the bias is strongest** (retail-held, low attention, hard to value) and should attenuate where arbitrage is cheap |
| **摩擦与约束** frictions | someone who *cannot* trade | it should **scale with how tight the constraint is** and disappear when it relaxes. Mandates, index rules, T+1, price limits, capital, borrow |
| **信息扩散** diffusion | being earlier than others | it should have a **decay clock**, and be stronger where attention is scarcer. If it does not decay, it is not diffusion |
| **市场结构** structure | a mechanical feature of the venue | it should key to **that mechanism's calendar** — rebalance dates, expiry, settlement — not to the calendar in general |
| **流动性提供** liquidity | absorbing an imbalance | the payoff should **track the imbalance you absorbed** and the inventory risk you carried, and should vanish when there is nothing to absorb |

## Step 3 — write the implications that could come back false

At least one of each, and they go in `Prereg.implications` before anything is
run:

- **横截面**: which names should show it more, which less, and why.
- **时序**: when should it be stronger, when weaker.
- **一个反向预测**: something the mechanism says should *not* work, or should
  have the opposite sign. This is the most informative one and the one people
  skip.

If an implication is one you already know holds, it is not evidence. Pick ones
you do not know the answer to.

## Step 4 — discriminate

If two families predict the same thing here, you have not chosen between them —
you have two captions. Name the single observation that separates them, and get
it. Common pairs that need separating:

- diffusion vs behavioural: both predict drift. **Does it decay on a clock?**
- risk premium vs mispricing: both predict a positive average. **Is the payoff
  concentrated in bad states, or does it avoid them?**
- friction vs behavioural: both predict a cross-sectional tilt. **Does the
  effect track the constraint's tightness over time?**

## Step 5 — set the bar before you look

Write down what result would make you abandon this mechanism. If nothing would,
go back to rule 2.

## Step 6 — check the record

Run `prior-check` on the **mechanism**, not only on the claim. The record
carries cases where the number replicated cleanly and the stated channel did
not, which is the most common way a replication goes wrong, and cases where a
mechanical-sounding argument was used to choose a search direction and turned
out to be flat when finally measured.

## Handing it to the referee

The output of this skill is two fields:

```python
prereg = falsifier.Prereg(
    claim="...",
    mechanism="<one sentence, from step 2>",
    implications=["<cross-sectional>", "<time-series>", "<the reverse prediction>"],
    primary_metric="...", horizon=...)
prereg.freeze()
```

Then `P1` requires the mechanism to exist and `P6` requires each implication to
come back `held` / `failed` / `untested` — and **`untested` reads as
INCONCLUSIVE, not as a pass**, because a claim that includes a mechanism is a
claim about the mechanism.

## 三档处置

这一步的产出不是「有机制」或「没机制」两档,是三档,而**中间那档最危险**:

| | 意味着什么 | 下游怎么处置 |
|---|---|---|
| **找到了,推论也测过** | 机制成立 | 正常进入组合层判断;机制告诉你它在哪里会失效 |
| **找不到经济解释** | 这是个**异象**,不是被理解的东西 | **必须人工筛查**。候选数按"本来可以搜的整个空间"算而不是"实际试了几个";按异象报告;**绝不能按已经懂了的尺寸下注** |
| **说了但没测(似懂非懂)** | **最危险的一档** | `P6` 读作 INCONCLUSIVE 而不是通过。**半懂不能被记成懂** —— 一旦写进记录, 下一轮就会有人把它当成已经成立的前提 |

「找不到解释」**不是判负**。一个没有解释的异象可以是真的,`P1` 因此报 INCONCLUSIVE 而不是
FAIL —— REJECTED 会关掉一条研究线, INCONCLUSIVE 把它升给人。两者都不是通过。

## What this skill does not do

It does not decide whether the number is real — that is `/falsify`. It does not
search for factors. And it cannot manufacture a mechanism where there isn't one:
**"no economic account survives step 2" is a legitimate and useful output.**

The failure this skill exists to prevent is not "no mechanism". It is **a
half-understood mechanism recorded as an understood one** — because once that
enters the record, the next study inherits it as a settled premise and stops
asking.
