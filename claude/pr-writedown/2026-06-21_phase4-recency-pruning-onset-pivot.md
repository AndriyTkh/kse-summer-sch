# PR Write-down — Phase 4: recency curves, feature pruning, region exclusion, onset pivot

**Date:** 2026-06-21 (Day-2 evening session)
**Scope:** Started as "wire Phase-3 into viz + UCDP (2a)"; turned into a deep
measure-everything pass on feature value, training depth, region quality, and the
walk-forward's worth — ending in a decision to reframe the whole target from *whether* an
alert fires to *when* (onset). This is a retrospective for the next session.

---

## What got APPLIED to code this session (locked)

| Change | File(s) | State |
|---|---|---|
| **UCDP regional prior wired** (2a / #17) | `loaders.load_ucdp`, `features.add_ucdp_features` | done — leak-safe per-oblast cumulative `ucdp_deaths_prior`/`ucdp_events_prior` |
| **Dead-region exclusion** | `config.MODEL_OBLASTS`, `index.build_master_index` | done — drop crimea/sevastopol/luhanska (0–0.2% base rate, no siren coverage) |
| **Grid clamp `GRID_START = 2023-07`** (B + Bq) | `config.py`, `index.py` | done — drops off-distribution 2022 ground-war regime |
| **Threat features DEPRECATED** | `config.THREAT_CHANNELS = ()`, `features.add_threat_features` | done — disabled for the whether-model; lean 6-col allowlist kept for onset revival |
| **Walk-forward DROPPED from run** | `run_phase3.py` | done — removed `compare_policies` block; Bq intervals only |
| Phase-3 viz wiring (intervals/drift panels) | `viz/...`, `run_phase3.py` | intervals.json live; **drift.json now inactive** (walk-forward gone) |

---

## GOOD decisions

1. **Measured before every cut.** Each call (prune, clamp, exclude, drop walk-forward) was
   backed by a throwaway probe/experiment, not intuition. Repeatedly overturned priors.
2. **Single-fit-vs-walk-forward experiment.** Showed the walk-forward retrain harness — the
   #1 compute cost — buys **≤0.005 PR-AUC** at the operating point (recency weighting already
   absorbs drift). Justified dropping it. Biggest ROI of the session.
3. **Dead-region exclusion + per-oblast lift reporting** over blunt deletion. Kept the
   operationally-critical frontline oblasts, fixed the metric honestly (0/24 below skill).
4. **UCDP as the regional location signal.** Principled, leak-safe (annual, prior-years-only);
   the only per-oblast discriminator a pooled model otherwise lacks. Modest but real at 6h.
5. **Learning curve by training depth.** Revealed depth needs are **horizon-dependent**
   (1h plateaus ~120d; 6h wants ~2.8yr) — a clamp can't be one-size.
6. **Threat prune validated free** (36→6, PR-AUC unchanged/slightly up before deprecation).
7. **Caught the hollow headline.** Recognized "whether an alert fires in next H" is trivially
   monotone in H (6h base rate 0.51) and the high PR-AUC is mostly persistence/autocorrelation
   — *before* shipping it as the product. Led to the onset reframe.

## BAD / churny / risky decisions (lessons)

1. **Jumped to a date cut (2022-10-10) before measuring** — proposed + edited config/STRUCTURE,
   user rejected ("too far"), reverted. Should have run the learning curve FIRST. (Later the
   curve landed us at 2023-07 anyway — measurement would have skipped the round-trip.)
2. **Built the Phase-3 drift viz panel, then dropped walk-forward** — the drift.json export +
   React panel are now dead weight (hide gracefully, but wasted effort). Ordering miss.
3. **Wired UCDP before knowing it'd help** — turned out marginal (+0.0045 @6h, ~0 @1h). Cheap
   (2 cols) and it's the location signal, so net-keep — but built on spec, not evidence.
4. **Pruned threat to 6, then deprecated threat entirely** — churn. The prune work isn't wasted
   (it's the revival allowlist for onset) but two passes where one would do.
5. **Clamp conclusions rest on a single 8-week holdout.** The 2023-01 win and the 2024 dip were
   non-monotone — possibly fold-specific. Not multi-fold confirmed. Treat 2023-07 as good but
   not bulletproof.

---

## KEY FINDINGS (numbers worth keeping)

- **Feature gain is dominated by alert-history** (74–81%). `alert_roll_168h` alone = 53% @6h.
- **Threat features add only ~+1% PR-AUC** — because the target is persistence (autocorrelation),
  not onset. 94% of waves are national-broadcast (no per-oblast targeting). Threat are LEADING
  indicators of a *new* strike → belong to the onset model, not this one.
- **UCDP:** +0.0045 PR-AUC @6h, ~0 @1h. Strong per-oblast separation (donetska prior 10.9 vs
  lvivska 5.0).
- **Walk-forward retrain adaptive gain ≤0.005** vs a recent single fit.
- **Learning curve:** 1h plateaus ~120d; 6h peaks ~2.8yr (0.930) ≈ full; only 2022 droppable.
  Clamp sweet spot 2023; **2024-01 is a trap** (6h 0.90, non-monotone).
- **Region base rates:** crimea/sevastopol 0.0%, luhanska 0.2% (dead); dnipropetrovska 82%,
  kharkivska/donetska 67% (near-permanent → persistence inflation).
- **Bq quantile:** halves pinball vs naive baseline (0.039 vs 0.091 @6h = real skill) BUT
  **bands under-cover: 0.64 vs 0.80 nominal** — too tight, needs conformal calibration.
- **Bq ≈ 3× B compute** (3 quantile fits per horizon).

---

## THE PIVOT (next session) — *whether* → *when*

The product critique that reframes everything: predicting *whether* an alert fires in H is
near-useless (monotone in H, persistence-driven). Switch the target to **alert ONSET**, chosen
path **A then B**:

- **A. Onset-in-window** — retarget B to "a NEW alert starts in (t,t+H], evaluated from a quiet
  state." Reuses the whole pipeline; multi-horizon = a timing profile. **Revives threat
  features** (leading indicators of new strikes). Numbers will look worse but mean more.
- **B. Time-to-next-onset** — survival/hazard for hours-until-next-strike. Retarget
  `survival.py` (currently models alert *duration*) to next-onset. True "when."

Do A first (de-risk: confirm onset is predictable + threat revives), then B.

---

## OPEN LOOSE ENDS (carry to PLAN.md)

- [ ] Onset reframe A→B (the headline next work)
- [ ] Bq under-coverage conformal fix (bands promise 80%, deliver 64%)
- [ ] Per-oblast lift reporting in `evaluate` + viz (decided, not yet built)
- [ ] Gain-share-by-group viz panel (user flagged "very useful")
- [ ] Present Bq to end users as **expected alert-hours + range**, never pinball
- [ ] Walk-forward → write up as run-once study ("drift slow, recency suffices"); drift viz now dead
- [ ] Threat revival (lean 6-col allowlist) for the onset model
- [ ] Multi-fold confirm of the 2023-07 clamp (optional, single-holdout caveat)
- [ ] Delete throwaway probes: `probe_features.py`, `exp_recency.py`, `exp_curve.py`, `exp_curve_bq.py` (+ `*_out.txt`)
