# ATLAS: Drylab pipelines of iGEM Heidelberg 2026

This repository holds the code of the Drylab part of the Heidelberg iGEM 2026 project (team VOYAGE) and an installer that sets up everything the pipelines need. Longer walkthroughs of each pipeline are in the **ATLAS Guide** and on the project wiki: <https://2026.igem.wiki/heidelberg/>.

## The project in short

VOYAGE builds an acoustic platform around **gas vesicles (GVs)**: hollow, gas-filled protein nanostructures (about 250 nm) that scatter ultrasound strongly and can be encoded genetically. The shell is built mainly from **GvpA**; the outer scaffold protein **GvpC** binds to it and sets how much pressure the vesicle survives before the shell buckles and before it collapses (its *collapse pressure*). GvpC is therefore the main engineering handle of the whole project:

- **Observation:** GvpC variants with different collapse pressures make several GV populations distinguishable in one ultrasound image (multiplexing). Analyte-sensitive GvpC constructs turn GVs into biosensors.
- **Organization:** GVs attached to cells (SpyTag/SpyCatcher, DARPin) allow acoustic sorting and patterning.
- **Wet-lab screen:** collapse pressure is measured by hydrostatic pressurized absorbance spectroscopy (OD500 against pressure). The Drylab pipelines below produce the designs that go into this screen and analyse what comes out.

## The five parts

| Part | Folder | What it does | Main tools |
|---|---|---|---|
| **evolution** | [`gvpc-evolution`](gvpc-evolution) | **GvpC evolution**, the in silico evolution loop. Redesigns one GvpC repeat (interface in four tiers, solvent-facing side from a helix palette), scores every sequence three times, runs controls through the same loop and applies a calibrated gate. 11 stages. | LigandMPNN, AlphaFold2 (initial guess), Boltz-2, PyRosetta |
| **gvpa** | [`gvpa-binder-design`](gvpa-binder-design) | De novo binders against GvpA: backbone generation, geometry triage, sequence design, AF2 gate and validation, Rosetta metrics, Boltz-2, ranking, null control. 9 stages. | RFdiffusion3, ProteinMPNN, AlphaFold2 via BindCraft, PyRosetta, Boltz-2 |
| **latch** | [`biosensor-latch`](biosensor-latch) | Cage-latch biosensor design: latch backbone, sequence, fusion with a target binder, monomer gate, rigid docking, partner co-folding. Own [README](biosensor-latch/README.md). | RFdiffusion3, ProteinMPNN, FlexCraft, AlphaFold2 |
| **gvsam** | [`gvsam`](gvsam) | Fine-tunes CellSAM to detect gas vesicle clusters in phase-contrast images. | CellSAM, PyTorch |
| **gas** | [`gas`](gas) | Fits a physics-informed model to ultrasound signal-versus-pressure curves to get a collapse-pressure shift. Notebook only. | NumPy, pandas, matplotlib |

`utils/fix_svg_upload.py` makes SVGs acceptable to the iGEM Registry media uploader.

## Quick start

```bash
git clone https://github.com/Heidelberg-iGEM/ATLAS.git
cd ATLAS
./install.sh --dry-run all      # print what would happen
./install.sh evolution              # or: gvpa | latch | gvsam | gas | all
./atlas.sh doctor evolution        # check that everything is in place
./atlas.sh help
```

The GVSAM weights (about 1.6 GB) are stored with [Git LFS](https://git-lfs.com): run `git lfs install` once before cloning, or `git lfs pull` afterwards.

You need `git`, `wget`, `conda` (or `mamba`: `PKG=mamba ./install.sh ...`) and, for the structure-prediction stages, a CUDA-capable NVIDIA GPU. Plan for several hours and about 20 GB per GPU part. PyRosetta needs a licence (free for academic use, see <https://www.pyrosetta.org/downloads>).

### What `install.sh` does

- fetches the third-party code at the versions pinned in [`third_party.lock`](third_party.lock) into `third_party/`
- builds one environment per toolset (they need conflicting JAX/Torch/NumPy versions), so nothing is activated by hand: every command calls its interpreter by absolute path
- downloads model weights (RFdiffusion3 checkpoint, LigandMPNN weights, AlphaFold2 parameters, CellSAM checkpoint)
- creates a project folder per part under `work/` with the repository's scripts linked in and a ready-to-edit config
- writes `atlas.env` with all paths, which `atlas.sh` reads

| Part | Environments | Third-party code (pinned) |
|---|---|---|
| evolution | `third_party/foundry/.venv` (uv), conda `evolution_boltz`, `BindCraft`, `evolution_rosetta` | rc-foundry 0.2.0 (PyPI), BindCraft, Boltz-2, PyRosetta |
| gvpa | `foundry`, `evolution_boltz`, `BindCraft` (shared with evolution) | as above |
| latch | `foundry`, conda `flexcraft` | rc-foundry, [FlexCraft](https://github.com/mjendrusch/flexcraft) (cloned to `third_party/flexcraft`) |
| gvsam | conda `gvsam` | [CellSAM](https://github.com/vanvalenlab/cellSAM) |
| gas | conda `gas` | none |

**Why third-party code is fetched and not copied into the repository:** foundry and BindCraft are large and have their own licences, and FlexCraft has no licence file that allows redistribution. The pinned versions make the installation reproducible. The pins were taken on 2026-11-04.

All commands in the tutorials are run from the repository root.

---

## Tutorial: evolution (GvpC in silico evolution)

The evolution pipeline needs a structure of a complex, one chain to redesign and a patch on the other chains that the redesigned chain has to keep hitting. It never builds a backbone. A full round is about one GPU-day for 400 sequences.

The stages (as in the guide and `run_pipeline.sh`): **1** LigandMPNN, **2** helix palette, **3** controls, **4** composition filter, **5** Boltz input, **6** Boltz, **7** Boltz ranking, **8** Boltz register, **9** AlphaFold2 initial guess, **10** gate and selection, **11** PyRosetta.

1. **Install:** `./install.sh evolution`
2. **Template:** the Serratia GvpC repeat 3-4 on four GvpA subunits ships with the repository (`gvpc-evolution/inputs/SerGvpCR3_4GvpA_trimmed_repacked.cif` and `.pdb`, plus the precomputed GvpA target MSA `gvpa_target.csv`); `install.sh` links them into `work/evolution/02_input/`. For another target, put your complex as a full mmCIF there and set `template:` in the config (default layout: target chains A–D numbered 1–69, binder chain F numbered 92–124). Check a template without a GPU:
   ```bash
   python3 gvpc-evolution/lib_checks.py gvpc-evolution/inputs/SerGvpCR3_4GvpA_trimmed_repacked.cif
   ```
   A template whose `label_*` and `auth_*` columns disagree would silently renumber the chain, and a missing `_entity_poly_seq` crashes Boltz with a misleading error; the check catches both. `python3 gvpc-evolution/05_template_fix.py raw.cif -o fixed.cif` repairs a repacked template.
3. **Config:** `work/evolution/00_pipeline/config.yaml` is a copy of [`config.example.yaml`](gvpc-evolution/config.example.yaml), the team's L1 configuration for Serratia GvpC (native binder, zones and tiers, epitope, helix palette, gate and selection settings; the comments explain the reasoning). For another target change `template`, `zones`, `epitope` and `native_binder`.
4. **Calibration run:** `./atlas.sh evolution calibrate`. It runs all stages but selects nothing. Read the printed distributions and check the controls: a threshold is only useful if the wild type passes it and the controls fail it.
5. **Set the gate:** enter the thresholds under `gate:` in `config.yaml`, then `./atlas.sh evolution sharp`.
6. **Results** are in `work/evolution/04_loops/L1/`: `master.csv` (everything), `gate.csv` (plus `gate_pass`/`gate_fail`), `selected.csv`, and `order.csv` (what goes to the lab: designs, controls and deliberate failures).
7. **After the wet lab:** add the measurements to `work/evolution/04_loops/experimental.csv` with a `seq_sha8` column, set `project.loop: L2` and run the next round. The first round stays untouched.

Resume at a stage with `./atlas.sh evolution resume 8`, run one stage script with `./atlas.sh evolution py 40_select_gate.py --config config.yaml --open`, and test the stdlib helpers with `./atlas.sh evolution selftest`. To run on a SLURM cluster, submit `run_pipeline.sh` in a job script (partition and time limit are site-specific):

```bash
#!/bin/bash
#SBATCH --job-name=gvpc-evolution
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
cd /path/to/ATLAS && ./atlas.sh evolution sharp
```

Useful helpers: `./atlas.sh evolution py cells_after_filter.py --config config.yaml --loop L1` shows after stage 4 whether every cell of the factorial design is still occupied, and `./atlas.sh evolution py check_controls.py --table ../04_loops/L1/master.csv` tests, before the gate thresholds are frozen, whether the wild type passes and the controls fail.

## Tutorial: gvpa (GvpA binder design)

Nine stages, driven stage by stage with `./atlas.sh gvpa <command>`. Stages 01–03 and 08 run in the `foundry` environment, 04–06 in `BindCraft`, 07 in `boltz`; the commands pick the right interpreter. GPU needed for 01, 04, 05, 07 and the AF2 part of 09; stage 06 is CPU only and the slowest, so shard it over many cores.

**Campaign files** are in `gvpa-binder-design/`: `configs/gvpa_serratia.yaml`, `configs/filters_strict.json`, `configs/filters_relaxed.json` and `inputs/gvpa_ser_5mer.cif` (five GvpA subunits, residues 13–63). Before the first ranking run `cp configs/filters_strict.json configs/filters_bindcraft.json` (the ranker reads that name by default).

```bash
./install.sh gvpa
./atlas.sh doctor gvpa
```

Stage commands (`$RUN` = `<run_root>/<campaign>`, by default `gvpa-binder-design/runs/gvpa_ser`; file names follow the directory layout of the guide, and every command accepts `--help`):

| Stage | Command |
|---|---|
| 01 backbones | `./atlas.sh gvpa make-inputs` then `./atlas.sh gvpa rfd3 --gpus 4` |
| 02 triage | `./atlas.sh gvpa collect` |
| 03 sequences | `./atlas.sh gvpa mpnn` |
| 04 AF2 gate | `./atlas.sh gvpa manifest --out $RUN/af2_gate/manifest.csv` then `./atlas.sh gvpa af2 --manifest $RUN/af2_gate/manifest.csv --out $RUN/af2_gate/scores.csv --mode gate` |
| 05 AF2 full | `./atlas.sh gvpa manifest --out $RUN/af2_full/manifest.csv --gate-scores $RUN/af2_gate/scores.csv` then `./atlas.sh gvpa af2 --manifest $RUN/af2_full/manifest.csv --out $RUN/af2_full/scores.csv --mode full --save-pdb-dir $RUN/af2_full/pdb` |
| 06 Rosetta | `./atlas.sh gvpa rosetta --scores $RUN/af2_full/scores.csv --out $RUN/rosetta.m0.csv --model 0 --n-shards N --shard i` (and `--model 1`) |
| 08 rank, select, report | `./atlas.sh gvpa rank`, `diversity`, `order-list`, `report` |
| 07 Boltz | `./atlas.sh gvpa boltz --accepted $RUN/accepted.csv --out $RUN/boltz/scores.csv`, then rank again with `rank --boltz ...` |
| 09 null control | `./atlas.sh gvpa null make`, score the scrambles with `af2 --mode full`, then `./atlas.sh gvpa null report --null $RUN/null/scores.csv` |

Stage numbers and the gate (tier 1 ≥ 55 residues, tier 2 for shorter binders) are explained in the guide chapters *Pipeline Stages* and *Acceptance and Results*. Without a passed null control the campaign is not reportable.

## Tutorial: latch (biosensor latch design)

Generates a helical latch with RFdiffusion3, designs its sequence, fuses it with a target binder and filters it with AlphaFold2 (monomer gate, rigid docking, optional partner co-folding for "does the target displace the latch").

```bash
./install.sh latch
./atlas.sh latch check          # GPU-free wiring check, runs anywhere
```

1. **Inputs** (layout of the guide): the motif PDB into `third_party/foundry/input_pdbs/` (`Ser_GvpC_full.pdb` and `Repeat3.pdb` are installed there), your RFdiffusion3 YAML into `third_party/foundry/input_jsons/<project>/` (two worked examples are installed under `input_jsons/example/`; paths inside the YAML are absolute and have to be edited), and the target-plus-binder complex PDB into `third_party/flexcraft/flexcraft/pipelines/dual_binder_pipeline/inputs/`.
2. **Dry run** (prints every command, executes nothing):
   ```bash
   ./atlas.sh latch run --dry_run \
       --rfd3_yaml input_jsons/<project>/latch.yaml \
       --redesign_terminus c --fixed_end_sequence "<end of the fixed region>" \
       --gfp_binder_pdb flexcraft/pipelines/dual_binder_pipeline/inputs/<complex>.pdb \
       --gfp_binder_chain B --gfp_chain A --hotspots <binder-local residues> \
       --monomer_filter_json filters/monomer_loose.json \
       --rigid_filter_json filters/rigid_docking_min1.json \
       --target_count 5 --run_name <project>/run01 --af2_params params/af/
   ```
3. **First test batch:** the same call with `--max_batches 1 --rfd3_diffusion_batch_size 2` and without `--dry_run`.
4. **Full run:** the same call with the real `--target_count`. Add `--run_partners --partner1_sequence ... --partner1_target gfp --partners_filter_json filters/partners_test3_target_bound.json` for the displacement test.

`atlas.sh latch run` fills in `--foundry_dir`, `--rfd3_executable`, `--foundry_python` and `--mpnn_checkpoint` for you. The full flag list, filter profiles and pitfalls are in [`biosensor-latch/README.md`](biosensor-latch/README.md). The patched ProteinMPNN driver (`biosensor-latch/foundry/batch_partial_mpnn.py`, recovered from the cluster) and the two motif PDBs of the examples are linked into the foundry workspace by `install.sh`.

## Tutorial: gvsam (gas vesicle cluster detection)

**Segment new images with the trained model.** The fine-tuned weights ship with the repository in [`gvsam/weights/`](gvsam/weights): `decoder.pt` is the complete fine-tuned CellSAM (detector plus mask decoder), `cellfinder.pt` is its detector head alone, the file stage 3 writes. Only the base CellSAM checkpoint is downloaded by `install.sh`.

```bash
./install.sh gvsam
./atlas.sh gvsam stage predict.py /abs/path/to/images --outdir /abs/path/to/preds
```

This writes `<image>_mask.tif` per image (background 0, one integer id per cluster). `--bbox_threshold` (default 0.4) sets the detector confidence; lower finds more clusters. Use absolute paths, because `atlas.sh` runs the script from `work/gvsam/00_pipeline`. The released mask decoder is weak (pixel F1 0.04 on our 10 test images); the CellFinder detector is the useful part (F1 0.19, base CellSAM 0.0007), so check the output before relying on the masks.

**Train your own.**

```bash
./install.sh gvsam
# images:  work/gvsam/01_raw/            (.tif; change data.image_ext for other formats)
# masks:   work/gvsam/02_masks_gv/       (<image>_mask.tif; background 0, one integer id per cluster)
# negatives (optional): work/gvsam/02_masks_gv/negatives.txt  (stems of images checked to hold no GVs)
$EDITOR work/gvsam/00_pipeline/config.yaml
./atlas.sh gvsam run            # split + augment, train CellFinder, evaluate, gallery
./atlas.sh gvsam run 3          # resume at stage 3
```

Stages: **2** split and augment (rotated copies stay out of validation), **3** train CellFinder, **4** evaluate against base CellSAM, **5/6** mask decoder (only if `decoder.enabled: true`), **7** gallery from `visualize.ipynb`. Results are in `work/gvsam/04_runs/<run>/`. A single script: `./atlas.sh gvsam stage 02_eval_cellfinder.py --help`.

## Tutorial: gas (collapse-pressure shift)

```bash
./install.sh gas
./atlas.sh gas notebook
```

The notebook walks from beamformed ultrasound frames over regions of interest and summary statistics to the physics-informed fit, a recovery test on synthetic curves and a checklist for your own run. It expects the `gvv` package, `labbook/` and `data/` next to it (see *Known gaps*).

---

## Command reference

| Command | Effect |
|---|---|
| `./install.sh [--dry-run] [--cuda V] [all\|evolution\|gvpa\|latch\|gvsam\|gas]` | install environments, third-party code, weights, project folders |
| `./atlas.sh doctor [part]` | check the installation; entries marked *repository gap* are files this repository does not contain yet |
| `./atlas.sh evolution calibrate \| sharp [FROM] \| resume N \| selftest \| py SCRIPT` | GvpC evolution rounds |
| `./atlas.sh gvpa make-inputs \| rfd3 \| collect \| mpnn \| manifest \| af2 \| rosetta \| boltz \| rank \| diversity \| order-list \| report \| null \| boltz-check` | GvpA binder design stages |
| `./atlas.sh latch run \| stage \| check` | biosensor latch pipeline |
| `./atlas.sh gvsam run [FROM] \| stage SCRIPT` | GVSAM |
| `./atlas.sh gas notebook` | GAS notebook |

## Repository layout

```
install.sh  atlas.sh  third_party.lock  LICENSE  NOTICE  README.md
gvpc-evolution/      stage scripts 05-60 (incl. 33_boltz_register.py), helpers (cells_after_filter.py,
                     check_controls.py), lib_*.py, run_pipeline.sh, setup_envs.sh, config.example.yaml,
                     inputs/ (template, GvpA target MSA)
gvpa-binder-design/  scripts/ (14 stage scripts), env.sh, configs/ (campaign, filters), inputs/ (target)
biosensor-latch/     scripts/, filters/, inputs/ (+ pdbs/ motif PDBs), foundry/batch_partial_mpnn.py, README.md
gvsam/               01-04 scripts, predict.py, augment.py, lib_*.py, run_pipeline.sh, setup_env.sh, config.yaml,
                     visualize.ipynb, weights/ (fine-tuned cellfinder.pt and decoder.pt, Git LFS)
gas/                 gas-walkthrough.ipynb
utils/               fix_svg_upload.py
third_party/  work/  atlas.env      (created by install.sh, not tracked)
```

## Known gaps

The GvpC evolution scripts `33_boltz_register.py`, `05_template_fix.py`, `cells_after_filter.py` and `check_controls.py`, the real config, the template and the latch MPNN driver were recovered from the team's working copies; the following could not be found:

| Part | Missing file | Consequence / workaround |
|---|---|---|
| gas | `gvv` package, `labbook/`, `data/`, `scripts/40_parallel_claims.py` | the notebook cannot run without them |

## Licence and attribution

The code of this repository is under the [MIT License](LICENSE) (draft: the copyright holder line is "iGEM Heidelberg team (VOYAGE) and contributors"; the team has to confirm it). [`NOTICE`](NOTICE) lists the third-party tools and the one file derived from BindCraft. Third-party tools are fetched by `install.sh` and keep their own licences:

- **FlexCraft has no licence file.** Ask its author (Michael Jendrusch) for permission before this repository goes public, because the latch pipeline needs it.
- **PyRosetta** needs its own licence (free for academic use).
- The licences of the AlphaFold2 parameters, Boltz-2 and the MPNN weights were not checked.

RFdiffusion3, ProteinMPNN/LigandMPNN, AlphaFold2, Boltz-2, PyRosetta, BindCraft, FlexCraft and CellSAM are tools by their respective authors; cite them in any publication.
