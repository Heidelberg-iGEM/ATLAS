# ATLAS: Drylab pipelines of iGEM Heidelberg 2026

Code of the Drylab part of the Heidelberg iGEM 2026 project (team VOYAGE) plus an installer for everything the pipelines need. Longer walkthroughs: **ATLAS Guide** and the wiki, <https://2026.igem.wiki/heidelberg/>.

VOYAGE builds an acoustic platform around **gas vesicles (GVs)**, gas-filled protein nanostructures that scatter ultrasound. Their outer scaffold protein **GvpC** binds the **GvpA** shell and sets the pressure at which the vesicle collapses, so GvpC is the main engineering handle. The pipelines below produce the designs for the wet-lab pressure screen and analyse what comes out.

| Part | Folder | What it does | Main tools |
|---|---|---|---|
| **evolution** | [`gvpc-evolution`](gvpc-evolution) | In silico evolution of one GvpC repeat: redesign, three scorings, controls, calibrated gate (11 stages) | LigandMPNN, AlphaFold2, Boltz-2, PyRosetta |
| **gvpa** | [`gvpa-binder-design`](gvpa-binder-design) | De novo binders against GvpA: backbones, triage, sequences, AF2 gate, Rosetta, Boltz-2, ranking, null control (9 stages) | RFdiffusion3, ProteinMPNN, BindCraft, PyRosetta, Boltz-2 |
| **latch** | [`biosensor-latch`](biosensor-latch) | Cage-latch biosensor: latch backbone, sequence, fusion with a binder, monomer gate, docking, partner co-folding. Own [README](biosensor-latch/README.md) | RFdiffusion3, ProteinMPNN, FlexCraft, AlphaFold2 |
| **gvsam** | [`gvsam`](gvsam) | Fine-tuned CellSAM that finds GV clusters in phase-contrast images | CellSAM, PyTorch |
| **gas** | [`gas`](gas) | Physics-informed fit of ultrasound signal vs. pressure → collapse-pressure shift (notebook) | NumPy, pandas |
| **pas** | [`pas-gui`](pas-gui) | GUI for pressurized absorbance spectroscopy (PAS): pressure ramp, photometer logging, live collapse-pressure (P50) fit. Own [README](pas-gui/README.md) | Tk, NumPy, SciPy, Arduino |

`utils/fix_svg_upload.py` makes SVGs acceptable to the iGEM Registry uploader.

## Install

```bash
git lfs install                 # once; the GVSAM weights (1.6 GB) are in Git LFS
git clone https://github.com/Heidelberg-iGEM/ATLAS.git && cd ATLAS
./install.sh --dry-run all      # print what would happen
./install.sh gvpa               # or: evolution | latch | gvsam | gas | pas | all
atlas doctor gvpa               # check the installation
```

Needs `git`, `wget`, `conda` (or `PKG=mamba`), a CUDA GPU for the structure-prediction stages and about 20 GB per GPU part. PyRosetta needs a licence (free for academic use). `install.sh` fetches third-party code at the versions in [`third_party.lock`](third_party.lock) (pinned 2026-11-04; foundry, BindCraft and FlexCraft are not copied here because of size and licences), builds one environment per toolset, downloads the weights, creates `work/<part>/` and writes `atlas.env`. It also links `atlas.sh` to `~/.local/bin/atlas` (put that folder on your `PATH`, or use `./atlas.sh`). Every `atlas` command prints the line it runs, so any step can be repeated by hand.

## Use

```bash
atlas gvpa --config my_campaign.yaml --gpus 4      # whole GvpA binder campaign
atlas evolution --config my.yaml --sharp           # one GvpC evolution round
atlas latch --rfd3_yaml latch.yaml --gfp_binder_pdb ...   # latch pipeline
atlas gvsam images/ --outdir preds/                # segment images
atlas gas                                          # open the notebook
atlas pas                                          # run a PAS measurement (GUI)
```

Each part has the same shape: `atlas <part> <inputs>` runs it end to end, and `atlas <part> help` lists the single stages.

### gvpa: GvpA binder design

`atlas gvpa` runs all nine stages in order (`1 2 3 4 5 6 8 7 9`: backbones, triage, sequences, AF2 gate, AF2 full, Rosetta, rank/shortlist/report, Boltz plus re-rank, null control).

| Option | Meaning |
|---|---|
| `--config F` | campaign file with target, hotspots, bins (default `gvpa-binder-design/configs/gvpa_serratia.yaml`, target `inputs/gvpa_ser_5mer.cif`) |
| `--filters F` | acceptance filters (default `configs/filters_strict.json`; `filters_relaxed.json` is shipped too) |
| `--gpus N` / `--shards N` | GPUs for RFdiffusion3 / parallel CPU shards for Rosetta, the slowest stage |
| `--from K` | resume at stage K |

Single stages: `atlas gvpa make-inputs\|rfd3\|collect\|mpnn\|manifest\|af2\|rosetta\|boltz\|rank\|diversity\|order-list\|report\|null` (each accepts `--help`; `atlas gvpa help` lists them). Without a passed null control the campaign is not reportable.

### evolution: GvpC in silico evolution

Needs a complex structure, one chain to redesign and a patch on the other chains it must keep hitting; a round is about one GPU-day for 400 sequences. The Serratia GvpC repeat 3-4 template and GvpA target MSA ship in `gvpc-evolution/inputs/` and are linked into `work/evolution/`.

`atlas evolution [--config F] [--sharp] [--from N]` runs all stages (without `--sharp` nothing is selected: the calibration run).

1. `atlas evolution check template.cif` tests a template without a GPU (catches `label_*`/`auth_*` mismatches and a missing `_entity_poly_seq`).
2. Edit `work/evolution/00_pipeline/config.yaml` (copy of [`config.example.yaml`](gvpc-evolution/config.example.yaml); for another target change `template`, `zones`, `epitope`, `native_binder`).
3. `atlas evolution` is the calibration run. Check that the wild type passes your thresholds and the controls fail.
4. Enter the thresholds under `gate:`, then `atlas evolution --sharp`.
5. Results are in `work/evolution/04_loops/L1/`: `master.csv`, `gate.csv`, `selected.csv`, `order.csv` (goes to the lab).
6. After the wet lab, add measurements to `04_loops/experimental.csv` (column `seq_sha8`), set `project.loop: L2`, run again.

Also: `selftest`, `py SCRIPT` (one stage script, e.g. `py 40_select_gate.py --config config.yaml --open`; `py cells_after_filter.py` and `py check_controls.py` help before freezing the gate). On SLURM, put `atlas evolution --sharp` in a GPU job script.

### latch: biosensor latch

```bash
atlas latch check      # GPU-free wiring check
atlas latch --dry_run --rfd3_yaml input_jsons/<project>/latch.yaml --redesign_terminus c \
    --fixed_end_sequence "<end of fixed region>" --gfp_binder_pdb <complex>.pdb \
    --gfp_binder_chain B --gfp_chain A --hotspots <residues> \
    --monomer_filter_json filters/monomer_loose.json --rigid_filter_json filters/rigid_docking_min1.json \
    --target_count 5 --run_name <project>/run01 --af2_params params/af/
```

Drop `--dry_run` for the real run (first with `--max_batches 1`); add `--run_partners ...` for the displacement test. Paths to foundry, rfd3 and the MPNN checkpoint are filled in. Inputs, all flags, filter profiles and pitfalls: [`biosensor-latch/README.md`](biosensor-latch/README.md).

### gvsam: GV cluster detection

`atlas gvsam images/ --outdir preds/` writes `<image>_mask.tif` per image (0 = background, one id per cluster) with the weights in [`gvsam/weights/`](gvsam/weights); `--bbox_threshold` (default 0.4) sets detector confidence. The released mask decoder is weak (pixel F1 0.04 on 10 test images); the CellFinder detector is the useful part (F1 0.19 vs. 0.0007 for base CellSAM), so check the output.

To train your own: put images in `work/gvsam/01_raw/` and masks in `work/gvsam/02_masks_gv/`, edit `work/gvsam/00_pipeline/config.yaml`, run `atlas gvsam run` (`atlas gvsam run 3` resumes at stage 3). Results are in `work/gvsam/04_runs/`.

### gas: collapse-pressure shift

`atlas gas` opens the walkthrough from beamformed frames to the physics-informed fit. It needs the `gvv` package, `labbook/` and `data/`, which are not in this repository yet.

### pas: pressurized absorbance spectroscopy

`atlas pas` opens the GUI for the wet-lab collapse-pressure measurement: it ramps the pressure through an Arduino Nano, logs the photometer and fits the Boltzmann curve (P50) per run. `./install.sh pas` builds its small conda env (Python with Tk, NumPy, SciPy, Matplotlib; no GPU needed), and `atlas doctor pas` checks it. The Arduino sketch is flashed by hand. Method, hardware and a step-by-step run: [`pas-gui/README.md`](pas-gui/README.md).

## Licence

Code: [MIT](LICENSE) (draft; the team has to confirm the copyright holder line). [`NOTICE`](NOTICE) lists third-party tools and the one file derived from BindCraft. Third-party tools keep their licences: **FlexCraft has no licence file** (ask its author, Michael Jendrusch, before this repository goes public), **PyRosetta** needs its own licence, and the licences of the AlphaFold2 parameters, Boltz-2 and the MPNN weights were not checked. Cite RFdiffusion3, ProteinMPNN/LigandMPNN, AlphaFold2, Boltz-2, PyRosetta, BindCraft, FlexCraft and CellSAM in any publication.
