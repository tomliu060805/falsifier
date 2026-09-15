---
name: read-report
description: "Triage a sell-side research report, a paper, or a WeChat article: separate what is already refuted from what is worth testing from what is reusable regardless. Extracts the claim the evidence actually supports rather than the one in the title, names the checks the authors did not run, cross-references the record of past verdicts, and ends with one of three decisions -- refuted on its face, worth testing (and the single check that would decide it), or true-but-not-worth-testing here. Use whenever a report, paper or article arrives and the question is what to do with it. Chinese requests: 看一下这篇研报、这篇文章有什么用、哪些是判负的、这个值不值得做、帮我读一下这个。"
---

# Read a report: what is dead, what is worth testing, what is reusable

The output is not a summary. A summary is what the document already is. The
output is a decision about where to spend the next week, and the evidence for
it.

## What you are actually doing

Three separations, in this order:

1. **The claim the evidence supports vs the claim in the title.** These are
   routinely different, and the gap is usually where the whole finding lives.
2. **Refuted / worth testing / not worth testing.** Most things are the first
   or the third. Very few are the second, and saying so is the value.
3. **The conclusion vs the method.** A dead conclusion often leaves a live
   method — a null construction, a data trap, an alignment rule. That residue
   is frequently the only thing in the document worth keeping, and it survives
   the conclusion being wrong.

## The base rate, from this record

Of the replications in `priors/verdicts.jsonl`: **15 rejected, 3 partial, 1
survived.** Of thirteen brokerage timing strategies replicated under one
convention, three survived costs. Start from that prior. A report that looks
strong is, before you read it, most likely to be one of the fifteen.

The most frequent causes of death for this genre, in order:
`cost-eats-it`, `mechanism-stated-but-not-the-one-working`,
`null-random-structure-wins`, `signal-duplicates-existing`.

## Step 1 — strip the narrative and write the claim

One sentence, in the form "X predicts Y over horizon Z, in universe U, net of
C". If the document does not let you fill in every slot, that absence is your
first finding and usually the most important one: **an unstated universe or an
unstated cost is not an oversight, it is where the result is.**

Then ask what the *evidence* supports, which is often narrower: a shorter
period, one index, gross, before turnover.

The clearest case in this record: a paper's headline was 13% a year. Decomposed,
it was **12.1pp survivorship bias + 4.1pp from equal weighting + −0.3pp from
the signal itself.** Every number in the paper was correct.

## Step 2 — the checklist of what they did not run

Go through the axes and mark each one `done` / `not done` / `claimed but not
shown`. **The "not done" column is the answer**, because it bounds what can be
believed no matter how good the numbers look.

| ask | what to look for | if absent |
|---|---|---|
| how many candidates were tried | a search whose size is disclosed | assume it is large; the reported result is the survivor of an undisclosed search |
| a null baseline | random/matched comparison, not just a t-stat | the result is compatible with something a coin produces at the same turnover |
| cost, and at what rate | basis points, both legs, stated convention | gross. Almost every A-share intraday result dies here |
| the universe and when it was fixed | point-in-time constituents | survivorship |
| out of sample, and whether it was consumed | a period untouched during development | the "out of sample" is a second in-sample |
| the mechanism, and whether it was tested | implications that could have failed | a caption, not a mechanism |
| execution | fills, limits, delay | a book that could not have been entered |

Two specific traps in this genre:

- **"我们做了稳健性检验" usually means different parameters, not a different
  null.** Changing a lookback and getting the same answer tests nothing that
  matters. Ask what would have had to come out differently.
- **The start of the backtest is a choice.** If it begins in 2016 or 2019, ask
  what happens if it begins two years earlier, and whether the data even exists
  that far back.

## Step 3 — cross-reference the record

Run `prior-check` on the claim before deciding anything. This is cheap and it
is the point of keeping the record. Look for two things: the same claim tested
before, and the same *shape* of claim — the traps are shared by shape, not by
topic.

**Do not write the report's numbers into the prior record.** That library is a
record of verdicts *we reached*, on data we ran. An external claim goes in only
after it has been tested here, with our own evidence. A library that accumulates
other people's conclusions stops being evidence and becomes a reading list.

## Step 4 — the decision

End with exactly one of three, and the reason:

| | means | what to do |
|---|---|---|
| **判负** | it fails on its face, or this record already killed it | write down which check decides it and why, and stop. Note it in the session, not in the prior library |
| **值得测** | it could be real and the deciding test is affordable | name the **single** check that would settle it, the data needed, and the cost. If that is more than a few days, it is probably the third row |
| **不值得测** | it may well be true and still not worth it here | say why: cost, capacity, the universe is untradable, the edge is smaller than the friction. **True and useless is the most common outcome and the least often stated** |

## Step 5 — harvest the method

Regardless of the decision, ask what is reusable:

- a null construction or a placebo design
- a data trap, a field convention, an alignment rule
- a diagnostic that separates two explanations
- a negative result on a standard construction — which is publishable
  methodology and belongs in the public half

A report whose conclusion is dead and whose method is good is a **good report**.
Say so; that is a different verdict from a bad one.

## Rules of engagement

1. **Do not summarise.** If the reply could have been produced by reading only
   the abstract, it has no value.
2. **Quote the number that decides it.** "The cost assumption is missing" is
   weak; "it reports 7.4% long-short and neutralising industry and float market
   cap takes the Sharpe to 0.002" is a finding.
3. **Separate what the document shows from what it asserts.** Authors are
   usually honest about the first and expansive about the second.
4. **Be as hard on a negative as on a positive.** A report that claims
   something does not work can be wrong in exactly the same ways, and closing a
   line of work on weak evidence costs more than opening one.
5. **Say what would change your mind.** If nothing would, you are not reading,
   you are reacting.
