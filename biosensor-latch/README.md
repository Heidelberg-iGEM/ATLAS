# Biosensing Design Pipeline

A computational pipeline that designs **protein switches for an ultrasound-readable biosensor**. You give it a protein you want to detect; it returns single-chain constructs that are predicted to change conformation when that protein binds.

This folder holds the orchestrator, the four scoring/building scripts it drives, four filter profiles, two worked RFdiffusion3 input configs, a GPU-free wiring self-check, and this document. It is **not** a turnkey install: three heavy third-party components and one patched team script are *not* in here and cannot be. Read [§5 Prerequisites](#5-prerequisites--what-you-must-install-before-anything-runs) **before** planning a run — on a fresh machine that setup, not the design work, is the bulk of the effort.

---

## 1. What the sensor does, and why the pipeline looks like this

**The readout.** Gas vesicles (GVs) are hollow protein nanostructures that scatter ultrasound. The protein **GvpC** binds the GV shell and mechanically stiffens it, which changes how the GVs behave under acoustic pressure. Anything that changes GvpC's availability therefore changes an ultrasound signal — that is the sensor's output channel.

**The switch.** The design keeps a **GvpC repeat region as a fixed structural motif** and builds two things onto it:

- a **blocking latch** — a de novo peptide designed to fold back onto the construct and hold it in a defined "closed" state;
- a **target-binding module** — a known binder against the protein you want to detect, fused directly to the latch's free end (no linker).

The intended logic is **mutually exclusive binding**: the latch and the target compete for the same physical space. With no target around, the latch sits closed. When the target protein appears, it binds the target-binding module — and because the target's own bulk overlaps where the latch sits, binding it *forces the latch open*. That conformational change propagates to GvpC availability, and from there to the ultrasound readout.

**Why that premise dictates the whole scoring strategy.** Everything downstream of sequence design is testing one geometric question: *does target binding actually collide with the latch?* A design where the target can bind without disturbing the latch is a mechanically silent sensor — correctly folded, confidently predicted, and useless. This is why the last stage inverts its success criterion and *rewards* steric clashes (§3, stage 5). It is the single most counter-intuitive thing about this pipeline.

---

## 2. Scope and honesty notes — read before you trust any output

This document describes the pipeline **as actually implemented in the scripts in this folder**, not an idealized version. Five things you should know up front:

1. **Complex prediction is opt-in, and nothing here uses AlphaFold-Multimer.** Two distinct points, often conflated:
   - *The co-folded stage is off by default.* Stage 6 (`score_gfp_fusion_partners.py`, Test 2/3/4) predicts the target-bound state, but only with `--run_partners`; without that flag the pipeline is the 5-stage version and the bound state is never modelled. Even with it, stage 6 is **report-only** unless you also pass `--partners_required`. Stage 5 remains the cheap geometric proxy: no GPU, no folding assumption, but no induced fit either.
   - *No multimer weights anywhere.* Every AF2 call — stage 4 and stage 6 alike — runs the **monomer pTM** models (`model_config("model_1_ptm")`, params `model_<N>_ptm`), not AlphaFold-Multimer. Stage 6 co-folds by appending each partner as an additional chain built from an idealized placeholder backbone and predicting with those same monomer weights. So "AF2 multimer" in loose usage here means *multi-chain co-folding with monomer-pTM weights*. How the chain separation is encoded happens inside FlexCraft's AF input builder, which is not in this folder — *bitte gegenchecken* if it matters to you.
2. **The "GFP" naming is historical, not conceptual.** Every script says `--gfp_binder_pdb`, `GFPSegment_pLDDT`, and so on, because GFP was the first target used as a testbed. The *role* is "target-binding module". For your own target you swap the structure files and the chain IDs; you do not rename anything. See §5.
3. **The filter thresholds are not calibrated.** They were carried over from an earlier architecture and, on real data, the monomer gate's `pTM ≥ 0.45` rejected **175/175** constructs in one Repeat3-4 run. The rigid-docking default (`≥ 1` clash) is explicitly labelled a placeholder in its own source. Treat both profiles as starting points to be re-derived from your own score distributions (§7).
4. **End-to-end cluster validation is an open point.** As of 2026-09-21 it had been written and tested locally but had **not yet run end-to-end on a GPU cluster**. *Bitte gegenchecken* before relying on it — and run `--dry_run` plus a deliberately tiny first batch regardless.
5. **Stage 2 needs a *patched* `batch_partial_mpnn.py`.** The stock script is hardwired to N-terminal redesign. C-terminal latches (`--redesign_terminus c`) require a patch that adds `--redesign-terminus {n,c}` / `--fixed-end-sequence` **and** writes the `boundary_residue_id` + `redesign_terminus` CSV columns the bridge reads. Copies of the script from before the 2026-09-26 patch silently produced `redesign_terminus=n` / `boundary_residue_id=A1` for genuinely C-terminal designs and 640 CSV rows had to be corrected by hand. The orchestrator detects the stale version and falls back to the CLI value plus the anchor sequence — **with a warning you have to actually read in the log.** Details in §5.

---

## 3. The pipeline, stage by stage

One **batch** runs stages 1–5, plus stage 6 if you enabled it. A **construct** is one (backbone × MPNN sequence × register offset) triple, and the orchestrator loops over fresh batches until enough constructs pass.

```
                            ┌──────────────── foundry env (GPU) ────────────────┐
  RFD3 YAML  ──▶  [1] rfd3 design  ──▶  backbones (.cif.gz)
                                                │
                                   [2] batch_partial_mpnn.py  ──▶  sequences + redesign_summary.csv
                            └───────────────────────┬───────────────────────────┘
                                                    │   bridge (inside the orchestrator)
                            ┌───────────────────────▼──────── FlexCraft env ────┐
   target+binder   ──▶  [3] build_gfp_fusion_csv.py   ──▶  fusion constructs CSV
   complex PDB                                             (× register offsets)
                                                    │
                                   [4] score_gfp_fusion_monomer.py  (AlphaFold2)
                                                    │   ──▶ monomer gate: does it fold?
                                                    │
                                   [5] score_gfp_rigid_docking.py   (no model)
                                                    │   ──▶ does the target collide with the latch?
                                                    │
                                   [6] score_gfp_fusion_partners.py (AlphaFold2)
                                                    │   OPT-IN: --run_partners
                                                    │   only the winners of 4+5 are co-folded
                                                    │   ──▶ does the target actually bind + displace?
                            └───────────────────────┼───────────────────────────┘
                                                    ▼
                                      final_passing_constructs.csv
```

### Stage 1 — Latch backbone generation · RFdiffusion3

| | |
|---|---|
| **Tool** | `rfd3 design` (RFdiffusion3), foundry env, GPU |
| **Model type** | Generative diffusion model over protein backbone coordinates |
| **Input** | A YAML holding the contig definition, fixed atoms, and hotspots |
| **Output** | Backbone structures (`.cif.gz`) — coordinates only, no sequence |
| **Orchestrator flags** | `--rfd3_yaml`, `--rfd3_n_batches`, `--rfd3_diffusion_batch_size` (default 8), `--rfd3_seed_base`, `--rfd3_extra_args` |

**What it does.** Diffusion generates new backbone geometry for the latch while the GvpC region is **held fixed** as a motif. The YAML is where that constraint lives: the contig says which residues are fixed scaffold and which are to be invented, and the hotspots bias the generated latch toward the surface patch it is supposed to cover.

**Why a diffusion model.** The latch is not a natural protein and there is nothing to retrieve from a database. It has to satisfy a purely geometric brief — fold back onto a specific face of a specific motif, from a specific attachment point — which is exactly what backbone diffusion conditioned on a fixed motif is for.

**Known gap.** Whether `rfd3` accepts a `seed=` argument is **unverified**. If it does not, `--rfd3_seed_base` silently does nothing and every batch may regenerate *identical* backbones — which would quietly defeat the whole batching loop. Confirm by diffing two batches' outputs before trusting a long run; pass the real parameter via `--rfd3_extra_args` if the name differs.

### Stage 2 — Sequence design · ProteinMPNN (partial)

| | |
|---|---|
| **Tool** | `batch_partial_mpnn.py`, foundry env; checkpoint `proteinmpnn_v_48_020.pt` |
| **Model type** | Inverse-folding GNN — structure in, sequence out |
| **Input** | Stage 1 backbones |
| **Output** | Sequences + `redesign_summary.csv` (incl. `boundary_residue_id`) |
| **Orchestrator flags** | `--mpnn_checkpoint`, `--mpnn_chain`, `--redesign_terminus {n,c}`, `--fixed_start_sequence` / `--fixed_end_sequence` |

**What it does.** Stage 1 produced coordinates with no amino acids. ProteinMPNN reads a backbone and emits sequences likely to fold into it. The checkpoint `v_48_020` is the standard ProteinMPNN weight set trained with 0.20 Å backbone noise — the noised variants generalize better to designed (as opposed to crystallographic) backbones, which is what we have here.

**Why *partial* redesign.** Only the latch end is redesigned. The fixed GvpC region must keep its **native sequence**, because it has to remain a functional GvpC that behaves normally in the wetlab assay — let MPNN rewrite it and you get a protein that folds beautifully and is no longer GvpC. The boundary is set by:

- `--redesign_terminus n` — latch at the N-terminus (the Repeat1-2 geometry) → anchor with `--fixed_start_sequence`
- `--redesign_terminus c` — latch at the C-terminus (the Repeat3-4 geometry) → anchor with `--fixed_end_sequence`

The orchestrator validates this pairing and refuses mismatched combinations outright.

### Stage 2.5 — The bridge (inside the orchestrator, no model)

This pipeline deliberately **skips** the earlier `score_latch_monomer.py` (Test 1) stage, which means nothing produces the "Test-1 CSV" that stage 3 expects. The orchestrator synthesizes it from `redesign_summary.csv` instead:

- **Threads** each MPNN sequence onto its backbone, keeping only **N, CA, C, O, CB** atoms. This matters: without rewriting the residue *names*, the RFD3 placeholder residues would propagate into `full_sequence` and every downstream construct would carry the wrong sequence.
- **Derives `binder_start_residue`** — the 1-based *positional* index of the fixed/latch boundary — from `boundary_residue_id`, or from the anchor sequence if that column is missing or stale.
- **Treats the CLI as authoritative**: if `redesign_summary.csv` disagrees with `--redesign_terminus` (an older `batch_partial_mpnn.py`), it warns and uses the CLI value plus the anchor.

Positional (not `seqid.num`) indexing is a scar: an off-by-one bug from residue numbering that is not guaranteed to start at 1 has now been found and fixed **three times** in this codebase, in three different scripts.

### Stage 3 — Fusion construct assembly · `build_gfp_fusion_csv.py` (no model)

| | |
|---|---|
| **Input** | Bridge CSV + a real **target + target-binder complex PDB** |
| **Output** | `gfp_fusion_constructs.csv` — one row per (design × register offset) |
| **Key flags** | `--gfp_binder_pdb`, `--gfp_binder_chain`, `--register_offsets` (default `-2,-1,0,1,2`), `--filler_residue` (default `A`) |

**What it builds.** A single chain, no linkers:

```
redesign_terminus "n":   [target binder][latch, N-terminally trimmed/extended][binder]
redesign_terminus "c":   [binder][latch, C-terminally trimmed/extended][target binder]
```

**Why register offsets — the central trick.** Whether the target-binding module's binding face points *toward* the construct (so the bound target can hit the latch) or *away* from it (so it cannot) depends on the **helical register** at the attachment point. An α-helix rotates roughly **100° per residue**, so trimming or adding 1–2 residues at the latch's free end swings the fused module through most of a full turn. Since stage 4 folds the construct from sequence alone, there is nothing to dock and no rotation to optimize — "try different orientations" reduces entirely to "vary the latch length by a few residues and let AF2 decide". Five offsets per sequence is a cheap sweep over that rotation.

Negative offsets drop existing residues. Positive offsets need residues that do not exist in the backbone and are filled with a generic helix-favouring **alanine**, *not* a redesigned patch — deliberately cheap, because stage 4 re-folds everything anyway, so a ProteinMPNN pass before the first screen would buy nothing.

The target-binder segment and the filler get idealized placeholder coordinates. They are never used as an AF2 template, so their input geometry is irrelevant — but `num_filler_residues` is written to the CSV so stage 4 can exclude the filler from its RMSD.

### Stage 4 — Monomer gate · AlphaFold2 (single chain)

| | |
|---|---|
| **Tool** | `score_gfp_fusion_monomer.py`, FlexCraft env, GPU (JAX) |
| **Model type** | AlphaFold2, **monomer pTM** weights — `model_config("model_1_ptm")`, params `model_<N>_ptm` (`--af_model`, default 1). Not AlphaFold-Multimer; see §2 note 1 |
| **Input** | Stage 3 construct CSV |
| **Output** | `score_all.csv` (everything) + `score_success.csv` (passes) + predicted PDBs |
| **Key flags** | `--af2_params` (default `params/af/`), `--num_recycle` (default 4), `--hotspots`, `--filter_profile` / `--filter_json` |

**The question it asks.** Does this three-segment chain actually fold into the intended *closed* state? AF2 predicts the whole construct from sequence, then the script measures the result segment-wise. Which physical segment counts as which depends on `redesign_terminus`, read per row — the two boundary fields keep their names but **swap meaning** between `n` and `c`.

**The filters, and the reasoning behind each** (`default` profile):

| Threshold | Rejects a design when… | Why it matters |
|---|---|---|
| `min_pLDDT: 0.70` | per-residue confidence is low | Low-confidence regions are usually disordered in reality; a floppy latch cannot hold a defined closed state |
| `min_pTM: 0.45` | the global fold is not confidently placed | pLDDT can look fine while the *relative arrangement* of segments is guesswork — and the arrangement is the entire mechanism |
| `max_Blocking_RMSD: 3.0` Å | the latch moved from its designed position | If AF2 does not reproduce the designed latch pose, the design is not what was designed |
| `max_Binder_RMSD: 1.5` Å | the fixed motif was distorted | The GvpC region must stay native-like; a tighter bound than the latch's, because this part is not supposed to change at all |
| `min_InterfaceOverlap…: 0.25` | the predicted latch/motif interface does not match the designed one | Catches designs that fold fine but contact the *wrong* surface |
| `min_Predicted_HotspotContactCount: 1` | the latch does not touch the intended hotspots | The latch must cover the specific patch it was designed to block |
| `min_Input_HotspotContactCount: 1` | the *input* design already missed its hotspots | A sanity check on construct assembly, not on AF2 |
| `max_GFPSegment_Binder_CAClashes: 0` | the target-binder segment collides with the fixed motif | The module must be *free* to bind target; a collision here is a build error, and is the opposite of the clash we want in stage 5 |

**Filler-aware RMSD.** `Blocking_RMSD` is computed only over *real* latch residues (`is_blocking_real`), excluding positive-offset alanine filler. Comparing predicted coordinates against meaningless placeholder coordinates would corrupt the metric outright.

**What the real data did to these numbers.** On 175 Repeat3-4 constructs, **0 passed**. The sole blocker was `pTM`: observed range 0.31–0.44 (median 0.36), best 0.443 — never reaching 0.45, while pLDDT was close and several rows cleared 0.70. Meanwhile `Predicted_HotspotContactCount` was ≥1 in *all 175 rows*, so the fusion geometry itself was working. Offline re-filtering showed 14/175 would pass at `pTM ≈ 0.34`, from only **2 backbones**. This is what `filters/monomer_loose.json` is for — and it is also the honest reason to expect your own first run to return zero: **the threshold, the architecture, or both.** Look at `score_all.csv`'s distribution before concluding which.

### Stage 5 — Rigid target docking · `score_gfp_rigid_docking.py` (no model)

| | |
|---|---|
| **Input** | Stage 4's AF2-folded structures + the real target+binder complex PDB |
| **Output** | `score_all.csv` / `score_success.csv` + docked PDBs under `docked/` |
| **Key flags** | `--gfp_complex_pdb`, `--gfp_binder_chain`, `--gfp_chain`, `--ca_clash_distance` (default 3.5 Å), `--filter_json` |

**What it does.** It never folds anything. It takes a **real, known** target+binder complex structure as ground truth for how the target sits when genuinely bound, **Kabsch-superimposes** that complex's binder chain onto the construct's target-binder segment (CA atoms only), and carries the target along with the same rigid transform. Then it asks a purely geometric question: *in that "target correctly bound" pose, does the target's real bulk overlap the latch?*

**The inverted success criterion.** Unlike every other filter here, **high clash counts are the goal**. The switch only works if the bound target occupies space the latch currently fills — that overlap *is* the coupling. Zero clashes means the target can bind without disturbing the latch: no coupling, no signal, dead sensor. Hence the filters use `min_` thresholds on clash columns, not `max_`.

| Column | Meaning |
|---|---|
| `GFP_Blocking_CAClashes` | **The headline number** — docked target vs. latch. More is better. |
| `GFP_Binder_CAClashes` | Target vs. the *fixed* motif. A **control**: clashing here is a warning sign, not a win |
| `GFP_BlockingOrBinder_CAClashes` | Sum of both |
| `gfp_binder_alignment_rmsd` | Residual Kabsch fit error. **High value ⇒ do not trust this row's clash counts** |

**Four assumptions the script documents about itself**, all worth respecting:
1. The binder chain in the complex PDB must have an **identical sequence and length** to the construct's target-binder segment — automatic if you pass the same PDB to stages 3 and 5. A mismatch fails loudly rather than silently misaligning.
2. Segment boundaries are applied **positionally**, never by `seqid.num` (the recurring off-by-one).
3. The **rigid-body assumption** — that the module's local backbone is the same free, in the reference complex, and fused into the construct — is a simplification, not a fact. `gfp_binder_alignment_rmsd` is written out precisely so a bad fit is visible. In real Repeat1-2 runs it sat at **3.2–4.3 Å**, which the script's own docs call untrustworthy; a visual check suggested the cause was geometric (the module pointing the wrong way) rather than misfolding, but this remains **unresolved — bitte gegenchecken**.
4. Clash counting is **CA–CA distance only** (3.5 Å) — coarse by design, not an atomic steric detector.

**Calibration status: none.** `min_GFP_Blocking_CAClashes: 1` means "any clash at all" and is a placeholder. The orchestrator prints a warning if you run without `--rigid_filter_json`.

---

### Stage 6 — Partner co-folding · AlphaFold2, multi-chain · **opt-in**

| | |
|---|---|
| **Tool** | `score_gfp_fusion_partners.py`, FlexCraft env, GPU (JAX) |
| **Model type** | Same AF2 **monomer pTM** weights as stage 4, predicting construct + partner chain(s) together. Not AlphaFold-Multimer |
| **Input** | The constructs that passed stages 4+5, as a gate CSV the orchestrator writes |
| **Output** | `score_all.csv` / `score_success.csv`; metrics merged into the final CSV with a `partners_` prefix |
| **Enable with** | `--run_partners` (report-only) · `--partners_required` (make it a gate) |

**What it does.** This is the model-based version of the question stage 5 answers geometrically. Each partner is appended as its own chain, built from a **sequence-only idealized placeholder backbone** (never used as a template or initial guess), and AF2 predicts the whole assembly. Partners are supplied as **amino-acid sequences, not PDBs** — that caught me out, so: `--partner1_sequence`, not a structure file.

One script covers three tests, distinguished only by which partners you pass:

| Test | Partner setup | Question | Wanted for a working switch |
|---|---|---|---|
| **2** | `--partner1_target binder` = the blocked target | Does it still bind the closed construct? | **No** — latch blocks it. `filters/partners_test2_blocked.json` |
| **3** | `--partner1_target gfp` = your detection target | Does the target bind and push the latch off? | **Yes**, with high `Blocking_RMSD` = latch displaced. `filters/partners_test3_target_bound.json` |
| **4** | both partners at once | Can both bind simultaneously? | Both bound — the real switch test |

**Reference geometry** is stage 4's own AF2-folded output (`--gate_pdb_source predicted`), so unlike stage 4 there is no placeholder-filler distortion to correct for: every residue has a real fold behind it. Additionally, once per run per partner, an isolated reference complex (target segment alone + that partner) is predicted to give a "normal binding" baseline for `PartnerN_PositionalDisplacement` and `PartnerN_InterfaceOverlap`.

**Why the filter direction cannot have a sensible default.** Test 2 wants *low* contact, Test 3/4 want *high* contact — the same column, opposite directions. So the script's own profiles (`no_filter`, `permissive_generic`) set **no partner-metric thresholds at all**, only pLDDT/pTM. Gating on them would pass everything, which is why the orchestrator **refuses** `--partners_required` without `--partners_filter_json`. The two shipped JSONs are the note's worked examples, not calibrated thresholds.

**Three things to know before you trust it:**

1. **Hotspots are not derivable.** `--partnerN_hotspots` must be supplied in **target-segment-local** numbering (position 1 = first residue of that segment) — different numbering from both the YAML's `select_hotspots` and stage 4's `--hotspots`. The script cannot infer them.
2. **Test 4 strains AF2.** Multiple simultaneous competing binding events are a known weak spot. The fix is several runs with different `--partners_seed` compared against each other — *not* more recycles. No multi-seed averaging is built in; you orchestrate that yourself.
3. **The historical results are inconclusive, not negative.** Test 2 and Test 3 each rejected 25/25 candidates (15.9.), and a later Test 3 run showed zero hotspot contacts across all 46 rows. **But** an off-by-one bug in the segment masks was fixed on 2026-09-17, and the script's own note says every prior run with the default `--gate_pdb_source predicted` — i.e. all of those — must be repeated before the rejections count as evidence. As far as I can tell that repeat never happened. Treat the earlier "AF2 co-folding doesn't work here" conclusion as **unproven in both directions**.

---

## 4. The orchestrator: batching, stalling, and state

`run_cage_pipeline.py` is stdlib-only (plus `gemmi`) and shells out to both environments by absolute path, so it needs no activated conda env.

- **Success** = passes the monomer gate **and** the rigid-docking filter. With `--dock_all_binders`, every folded construct is docked and success depends on the rigid filter alone — the monomer verdict is still carried into the output as `monomer_gate_passed`. Docking is cheap (no AF2), so this buys information at low cost.
- **With stage 6 enabled**, `--run_partners` alone leaves the success criterion **unchanged** (metrics are recorded as `partners_*` columns plus `partners_gate_passed`, so constructs that fail the partner filter still appear in the final CSV). Adding `--partners_required` makes success = monomer **and** rigid **and** partners, which narrows what counts toward `--target_count` and therefore also affects stalling. Stage 6 runs only on the constructs that already passed stages 4+5 — the orchestrator writes a restricted gate CSV for it rather than handing it the whole batch, because it is the most expensive stage in the pipeline.
- **Stopping.** `--target_count` reached (a batch always finishes completely, so it can overshoot), `--max_batches`, or `--stall_batches` (default **3**) consecutive batches with zero successes. That last one is the useful guard: three empty batches in a row is a filter or architecture problem, not bad luck, and it stops you from burning GPU hours on it.
- **State.** `pipeline_state.json` in the run directory. Re-running with the same `--run_name` resumes after the last *completed* batch; a half-finished batch directory is deleted and redone. `--restart` ignores the state.
- **Results.** `final_passing_constructs.csv`, joining monomer metrics, `rigid_`-prefixed docking metrics and (with stage 6) `partners_`-prefixed co-folding metrics per `task_id`.
- **Logs.** `<run_dir>/<batch>/logs/01_rfd3.log` … `05_score_gfp_rigid_docking.log`, plus `06_score_gfp_fusion_partners.log` with stage 6. The step-6 gate subset is written to `<run_dir>/<batch>/partners_gate.csv`, so you can see exactly which constructs were co-folded.

**Cost.** One batch at defaults = 8 backbones × 8 sequences × 5 offsets = **320 AF2 folds** in stage 4. That is the dominant cost. Stage 6 adds AF2 predictions only for the handful of constructs that passed stages 4+5, plus one reference complex per partner per run — but each of those predictions is larger (construct + partner chains), and Test 4 wants several seeds, so a high pass rate makes stage 6 expensive too.

---

## 5. Prerequisites — what you must install before anything runs

Nothing in this folder runs on its own. The scripts here are **orchestration and scoring logic**; every model lives elsewhere. Budget real time for this section — on a fresh machine the environment setup, not the design work, is the bulk of the effort.

### 5.1 Two separate Python environments

The pipeline deliberately straddles **two** environments, because their dependencies conflict (PyTorch-based diffusion on one side, JAX-based AlphaFold2 on the other). The orchestrator never activates either one — it calls into both by **absolute path** to their binaries.

| Environment | Provides | Used by | Source |
|---|---|---|---|
| **foundry** | `rfd3` (RFdiffusion3) and `batch_partial_mpnn.py` | Stages 1–2 | `RosettaCommons/foundry`, pip package **`rc-foundry`**, with official install instructions. RFdiffusion3 is distributed through it. |
| **FlexCraft** | `flexcraft.structure.af` (AlphaFold2 via JAX), `flexcraft.data`, `flexcraft.structure.metrics`, `flexcraft.utils` | Stages 3–6 | FlexCraft, by **Michael Jendrusch**. As far as I know it is not a public pip package — ask the author for access. *Bitte gegenchecken.* |

Also required:

- **`gemmi`** in the **FlexCraft** env — the orchestrator itself imports it for the bridge (sequence threading), on top of stage 3's use of it. Everything else in the orchestrator is stdlib.
- **`numpy`**, **`jax`** in the FlexCraft env (pulled in by FlexCraft's AF2 wrapper).
- **A GPU** for stages 1, 2, 4 and 6. Stages 3 and 5 are CPU-only — which is exactly why stage 5 exists in its rigid-docking form.
- **Both environments on the same filesystem.** The orchestrator hands file paths from one to the other; it cannot bridge two machines.

### 5.2 The patched `batch_partial_mpnn.py` — easy to miss, breaks silently

Stage 2's driver script needs a **team patch** that the stock version does not have. Required for C-terminal latches (`--redesign_terminus c`):

- the CLI options `--redesign-terminus {n,c}` and `--fixed-end-sequence` (stock is hardwired N-terminal);
- writing the **`boundary_residue_id`** and **`redesign_terminus`** columns into `redesign_summary.csv`, which the bridge reads to locate the latch boundary.

Run an unpatched copy and the MPNN calculation itself is still *correct* — but the summary CSV reports `redesign_terminus=n` and `boundary_residue_id=A1` for genuinely C-terminal designs. That happened to us: 640 rows had to be corrected by hand before scoring was possible. The orchestrator now detects the mismatch, prints a warning, and falls back to the CLI value plus the anchor sequence — so a stale script degrades into a *log line you must notice*, not a crash. Check your copy accepts `--redesign-terminus` before the first `c` run.

### 5.3 Model weights — three separate downloads

None of these ship with the code:

| Weights | Passed as | Note |
|---|---|---|
| **RFdiffusion3** checkpoints | (config/env, per foundry's install docs) | Installed via `rc-foundry` |
| **ProteinMPNN** `proteinmpnn_v_48_020.pt` | `--mpnn_checkpoint` | Ours: `~/.foundry/checkpoints/proteinmpnn_v_48_020.pt`. `v_48_020` = trained with 0.20 Å backbone noise; the noised weights generalize better to designed backbones |
| **AlphaFold2** parameters | `--af2_params` (default `params/af/`) | Standard AF2 params directory, resolved **relative to the FlexCraft repo root** |

### 5.4 Expected directory layout

The layout the defaults assume, with `/path/to/foundry` standing for your own Foundry workspace:

| What | Path |
|---|---|
| Foundry workspace (`--foundry_dir`) | `/path/to/foundry` |
| Motif input PDBs | `<foundry>/input_pdbs/` — e.g. `Ser_GvpC_full.pdb`, `Repeat3.pdb` |
| RFD3 input YAMLs (`--rfd3_yaml`) | `<foundry>/input_jsons/<project>/` — resolved relative to `--foundry_dir` |
| RFD3 / MPNN outputs | `<foundry>/output/<project>/` |
| MPNN driver (`--mpnn_script`) | `<foundry>/scripts/batch_partial_mpnn.py` (default, relative to `--foundry_dir`) |
| ProteinMPNN checkpoint | `~/.foundry/checkpoints/proteinmpnn_v_48_020.pt` |
| The five pipeline scripts + the self-check | `<flexcraft-repo>/flexcraft/pipelines/dual_binder_pipeline/scripts/` |
| Target+binder complex PDBs | `<flexcraft-repo>/flexcraft/pipelines/dual_binder_pipeline/inputs/` |
| Pipeline results (`--results_root`) | `<flexcraft-repo>/flexcraft/pipelines/dual_binder_pipeline/results` (default, relative to `--repo_root`) |

### 5.5 Environment pitfalls on HPC systems

From the team's RFdiffusion **1** setup (RFD3 via `rc-foundry` is reportedly less painful, but the class of problem recurs on HPC):

- A conda-forge PyTorch can silently be a **CPU build** → `NO GPU DETECTED`. Install the CUDA build with `python -m pip` instead.
- pip-installed Torch can drag in an old `libstdc++` → `GLIBCXX_3.4.26 not found` on the scipy import. Fix with `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` and `export PYTHONNOUSERSITE=1` — **also inside Slurm scripts, after `conda activate`.**

---

## 6. Running it on your own target protein

### 6.1 Placement and working directory — the orchestrator is invoked as a module

This trips people up, so to be explicit:

Every path in the command line resolves against one of **two** roots, so the bundle's files do not all go in one place:

```
<flexcraft-repo>/                     ← run the command FROM HERE  (= --repo_root, default ".")
├── filters/                          ← copy this bundle's filters/*.json here
│                                       (--monomer_filter_json / --rigid_filter_json /
│                                        --partners_filter_json resolve against --repo_root)
├── params/af/                        ← AF2 weights (--af2_params)
└── flexcraft/pipelines/dual_binder_pipeline/
    ├── scripts/                      ← all five .py files from this bundle,
    │                                   plus test_pipeline_wiring.py
    ├── inputs/                       ← your target+binder complex PDB
    └── results/                      ← created for you (--results_root)

<foundry-workspace>/                  ← --foundry_dir
├── input_jsons/<project>/            ← your RFD3 YAML  (--rfd3_yaml, relative to --foundry_dir)
├── input_pdbs/                       ← your motif PDB  (referenced by ABSOLUTE path from the YAML)
└── scripts/batch_partial_mpnn.py     ← the PATCHED copy (--mpnn_script, §5.2)
```

Putting `filters/` inside `dual_binder_pipeline/` instead of at the repo root is the easy mistake: the flags resolve against `--repo_root`, not against the script's own directory. Absolute paths avoid the question entirely.

The invocation is `python -m flexcraft.pipelines.dual_binder_pipeline.scripts.run_cage_pipeline`, so:

- **cwd must be the FlexCraft repo root** (that is what puts the `flexcraft` package on `sys.path`), and `--repo_root` defaults to `.` — i.e. to that same directory. `--af2_params` and `--results_root` are resolved relative to it.
- If you put the scripts somewhere else in the package tree, pass the matching `--module_prefix`. All four other scripts are invoked through that same prefix, so a wrong value fails at stage 3, not at startup.

### 6.2 What you actually have to supply

1. **A target + binder complex structure.** This is the one hard input requirement: a PDB containing **your target protein** and **a known binder against it** (nanobody, minibinder, designed binder — experimental or predicted), in their bound pose. Stage 3 reads the binder's sequence verbatim from it; stage 5 uses the complex as ground truth for the bound geometry. Pass it as `--gfp_binder_pdb` (and `--gfp_complex_pdb`, which defaults to the same file), with `--gfp_binder_chain` = the binder's chain and `--gfp_chain` = your target's chain. *The pipeline does not design this binder for you.*
2. **A motif PDB** — the scaffold the latch is built onto, held rigid throughout. Referenced by absolute path from inside the YAML (next item), *not* by a CLI flag.
3. **An RFD3 input YAML** for the latch: contig, fixed atoms, hotspots. This is the real design work, and where your scaffold choice lives. **Two worked configs ship in [`inputs/`](inputs/)** — `example_single_repeat_latch.yaml` (33 aa motif, 16–24 aa latch; the cheaper one to start from) and `example_full_latch.yaml` (164 aa motif, 48–128 aa latch). Both are real configs from actual runs, commented field by field. Note that the **contig order decides the terminus**: generated segment first = N-terminal latch.
4. **The terminus and its anchor**, matching the contig. `--redesign_terminus n` with `--fixed_start_sequence "<start of the fixed sequence>"`, or `c` with `--fixed_end_sequence "<end of the fixed region, directly before the latch>"`. The orchestrator rejects a mismatched pair outright, but it cannot tell whether your *anchor string* matches your *contig* — get that wrong and the boundary lands in the wrong place. The anchor is a literal subsequence of the motif, e.g. `"VTDMSKKTLALLDG"`.
5. **`--hotspots`** — binder-local residue numbers for the patch the latch must cover, for the monomer gate. These are *binder-local* indices, not the motif numbering used in the YAML's `select_hotspots`.

### 6.3 Invocation

Run from the FlexCraft repo root (§6.1). C-terminal latch; `\` are line continuations:

```bash
python -u -m flexcraft.pipelines.dual_binder_pipeline.scripts.run_cage_pipeline \
    --foundry_dir /path/to/workspace/foundry \
    --rfd3_executable /path/to/foundry_env/bin/rfd3 \
    --foundry_python /path/to/foundry_env/bin/python \
    --rfd3_yaml input_jsons/my_target/latch.yaml \
    --mpnn_checkpoint ~/.foundry/checkpoints/proteinmpnn_v_48_020.pt \
    --redesign_terminus c \
    --fixed_end_sequence "<end of the fixed region, directly before the latch>" \
    --gfp_binder_pdb flexcraft/pipelines/dual_binder_pipeline/inputs/my_target_plus_binder_complex.pdb \
    --gfp_binder_chain B \
    --gfp_chain A \
    --hotspots <binder-local hotspot residues, comma-separated> \
    --monomer_filter_json filters/monomer_loose.json \
    --rigid_filter_json filters/rigid_docking_min1.json \
    --target_count 5 \
    --run_name my_target/run01 \
    --af2_params params/af/
```

**Adding stage 6** (Test 3 — does your target bind and displace the latch?) — append:

```bash
    --run_partners \
    --partner1_sequence <your target protein's amino-acid sequence> \
    --partner1_target gfp \
    --partner1_label MyTarget \
    --partner1_hotspots <target-segment-local residues> \
    --partners_filter_json filters/partners_test3_target_bound.json
```

Add `--partners_required` to make it a gate rather than a report. For Test 2, use `--partner1_target binder` with the blocked target's sequence and `filters/partners_test2_blocked.json`; for Test 4, supply `--partner2_*` as well and vary `--partners_seed` across repeated runs.

**Check the wiring without a GPU** — stdlib only, needs no FlexCraft, no weights, no cwd in particular (it imports the orchestrator from its own directory) and executes nothing:

```bash
python3 <path-to>/scripts/test_pipeline_wiring.py
```

### 6.4 Four CLI details that will bite you

1. **Use absolute paths for `--rfd3_executable` and `--foundry_python`.** `--rfd3_executable` defaults to the bare name `rfd3`, which only resolves if the foundry env is already on your `PATH` — and the whole point of the two-env design is that it is not. When you pass an absolute path, the orchestrator prepends that binary's `bin/` directory to the child process's `PATH`, so helper tools the foundry scripts call (`mpnn`, …) are found too. Pass the bare name and stage 1 fails with a confusing "not found".
2. **`--register_offsets` needs the `=` form:** `--register_offsets=-2,-1,0,1,2`. The value starts with `-`, so the space-separated form is parsed as a flag and errors out. (The orchestrator already uses the `=` form for its internal call to stage 3 — the same care is needed on its own CLI. The default is this exact string, so you only hit this if you override it.)
3. **`--rfd3_yaml` resolves against `--foundry_dir`**, not against your cwd — while `--gfp_binder_pdb`, `--af2_params`, all three filter JSONs (`--monomer_filter_json`, `--rigid_filter_json`, `--partners_filter_json`) and `--results_root` resolve against `--repo_root`. Two different bases in one command line. Absolute paths sidestep the whole question.
4. **Paths *inside* the YAML are absolute** and independent of both. The motif PDB is referenced there (`input:`), so a YAML copied from this folder still points at the placeholder `/path/to/foundry` until you edit it.

### 6.5 Bring it up in three steps — do not skip these

```bash
# 1. Print every command without executing anything.
#    Add --dry_run to the call above. Checks path wiring and flag validation;
#    the bridge is skipped (it needs real MPNN output), so stages 3-6 are
#    printed with a bridge CSV path that does not exist yet. Stops after one batch.

# 2. One deliberately tiny real batch: 2 backbones instead of 8.
#    --rfd3_diffusion_batch_size 2 --max_batches 1
#    At defaults a batch is 8 x 8 x 5 = 320 AF2 folds; this keeps the first
#    real run cheap while still exercising every stage end to end.

# 3. Confirm two batches produce DIFFERENT backbones before any long run.
#    Unresolved whether rfd3 accepts seed= (stage 1). If it does not,
#    --rfd3_seed_base is silently ignored and every batch rediscovers the
#    same designs -- the loop would burn GPU hours going nowhere.
```

Then resume the same `--run_name` to continue; the state file handles it.

---

## 7. Calibrating the filters — the step most likely to be skipped

**Do not accept the shipped thresholds.** Both profiles are uncalibrated, and stage 4's `default` rejected 175/175 real constructs. The reliable order of operations:

1. First real run: `--monomer_filter_json filters/monomer_loose.json` and `--rigid_filter_json filters/rigid_docking_min1.json` — loose on purpose, to produce a **distribution** rather than an empty result set.
2. Read `score_all.csv` from both stages. For the monomer gate, check `failure_reason`: if one metric blocks nearly every row, that threshold is the binding constraint and you can re-filter **offline without re-running AF2**.
3. For stage 5, look at the `GFP_Blocking_CAClashes` histogram and set `min_` where the distribution actually separates. Discard rows with high `gfp_binder_alignment_rmsd` first — their clash counts are not meaningful.
4. Then tighten and re-run.

Shipped profiles:

| File | Contents |
|---|---|
| `filters/monomer_loose.json` | pLDDT 0.65 · pTM 0.35 · Blocking_RMSD 4.5 Å · Binder_RMSD 2.5 Å · InterfaceOverlap 0.15 · hotspot contacts ≥1 · ≤2 segment clashes |
| `filters/rigid_docking_min1.json` | `min_GFP_Blocking_CAClashes: 1` (placeholder) |
| `default` (in-code, stage 4) | pLDDT 0.70 · pTM 0.45 · Blocking_RMSD 3.0 Å · Binder_RMSD 1.5 Å · InterfaceOverlap 0.25 · hotspot contacts ≥1 · 0 segment clashes |
| `permissive` (in-code, stage 4) | pLDDT 0.55 · pTM 0.30 · Blocking_RMSD 6.0 Å · Binder_RMSD 3.0 Å · ≤5 segment clashes — for *ranking*, not selection |
| `filters/partners_test3_target_bound.json` | Stage 6, Test 3: pLDDT 0.70 · Partner1 hotspot contacts **≥1** · Partner1 interface overlap **≥0.30** · Blocking_RMSD **≥3.0 Å** (latch displaced) |
| `filters/partners_test2_blocked.json` | Stage 6, Test 2: pLDDT 0.70 · Partner1 hotspot contacts **=0** · interface overlap **≤0.20** · positional displacement **≥3.0 Å** (target stays out) |
| `no_filter` / `permissive_generic` (in-code, stage 6) | pLDDT/pTM only — **no partner-metric thresholds**, so useless as a gate. The orchestrator refuses `--partners_required` without a JSON |

---

## 8. What this pipeline does not do

- **No prediction of the open state unless you enable stage 6.** By default stage 5 is rigid geometry, not a conformational model: no induced fit. `--run_partners` adds the co-folded prediction (§3, stage 6) — but see the caveats there before treating its verdict as truth.
- **No binder design against your target.** You must supply a known binder in complex (§6.2).
- **No experimental validation.** Passing constructs are **hypotheses for wetlab screening**, nothing more. The team's own experience with an earlier binder round is instructive: the designs that made it into the oligo pool were those that passed the generator's filters, and screening them is a separate, long workstream.
- **No Rosetta, no Test 1.** Skipped on purpose; the bridge (§3) exists because Test 1 is skipped. Rosetta ddG for interface energies is part of the broader design concept, but no stage here runs it.
- **Stage 6 is off unless you ask for it**, and even then report-only unless you pass `--partners_required`. A default run therefore still tells you nothing about the bound state. Full description, the three tests and their caveats: §3, stage 6.

  *Correction (2026-10-20):* an earlier version of this document said the partners script does not read `redesign_terminus` and therefore mis-segments C-terminal constructs. **That was wrong** — it was updated on 2026-09-17 to read the column per row, for both the live segment masks and the per-partner reference complex. The stale claim survives in the *other* scripts' docstrings, which were written the same day and never updated; do not trust those cross-references.

---

## 9. Provenance

All scripts in this folder are the team's own work. `scripts/test_pipeline_wiring.py` was written for this bundle (2026-10-20). `inputs/example_full_latch.yaml` and `inputs/example_single_repeat_latch.yaml` are the configs of the 2026-09-15 runs (`gvpc_full_latch.yaml`, `gvpc_repeat3_latch.yaml`) with field-by-field comments added. Their `input:` fields use the placeholder `/path/to/foundry`, which is the first thing to edit (`install.sh` rewrites it for its own installation). The `rc-foundry` source, the MPNN patch history and the environment pitfalls in §5 come from the team's own HPC work.

**Attribution.** RFdiffusion3, ProteinMPNN, and AlphaFold2 are third-party tools by their respective authors — cite them properly in any publication or Wiki page. FlexCraft is by Michael Jendrusch. The pipeline scripts in this folder are the team's own.
