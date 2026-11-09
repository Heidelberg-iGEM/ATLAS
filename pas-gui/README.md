# PAS-GUI: pressurized absorbance spectroscopy

One window to run, log and evaluate a **pressurized absorbance spectroscopy (PAS)** measurement of gas vesicles (GVs): it ramps the pressure, records the photometer, and shows the collapse pressure **P50** a few seconds after the run ends. This is the measurement behind the collapse pressures in the VOYAGE wiki, [Experiments](https://2026.igem.wiki/heidelberg/experiments) and [Results](https://2026.igem.wiki/heidelberg/results) (sections on pressurized absorbance spectroscopy).

## The method in short

Gas-filled GVs scatter light. When enough hydrostatic pressure is applied they collapse, lose their gas and scatter less, so the absorbance at 500 nm (OD500) of a GV sample drops. Not all GVs collapse at the same pressure, so OD500 against pressure is a falling sigmoid. Hydrostatic pressure stands in for acoustic pressure: it is a proxy for acoustic stability, not a direct measure of it.

The sample sits in a flow-through cuvette in a photometer while a regulator raises the air pressure step by step from 0 to 9 bar. Each OD500 reading is assigned the mean pressure of its time interval and the points are fitted with a four-parameter Boltzmann (logistic) function:

```
A(P) = A_low + (A_high − A_low) / (1 + exp((P − P50) / dP))
```

**P50**, the inflection point, is the collapse (midpoint) pressure of the sample; `dP` is the width of the transition. Only the rising-pressure branch is fitted, up to the OD500 minimum. A run whose OD500 drops by less than 5 % gets no P50 (a control that did not collapse). For the sample requirements, the full evaluation and the results, see the wiki pages above.

PAS needs a few mL of sample and takes minutes per run, so it suits characterising variants, not screening large libraries (the wiki describes a separate flotation-based assay for that).

## Hardware

| Part | Role |
|---|---|
| Compressor + air filter | pressure supply |
| SMC electro-pneumatic regulator | sets the pressure; setpoint 0–5 V = 0–9 bar, monitor output 1–5 V = 0–9 bar |
| Arduino Nano + MCP4725 (GY-4725) DAC | drives the setpoint and reads back the monitor voltage over USB serial (115200 baud) |
| Three-way valve, flow-through cuvette | loads the sample and connects it to the pressure line |
| WPA Biowave II+ photometer | reads OD500 in kinetics mode; its USB serial port streams the data (**Auto-Print: On**) |

Wiring, parts and the rig itself are described on the wiki (Experiments, and Engineering for the iterations of the setup). The GUI assumes the calibration in the table; it was not checked against a reference manometer, so absolute bar values carry the regulator's tolerance; comparisons between samples are unaffected.

## Files

| File | Purpose |
|---|---|
| [`pas_gui.py`](pas_gui.py) | the GUI (Tk) |
| [`pas_core.py`](pas_core.py) | serial setup, line formats, Boltzmann fit and `process_run` (used by the GUI; no other dependency in the repository) |
| [`sweep_up_nano/sweep_up_nano.ino`](sweep_up_nano/sweep_up_nano.ino) | Arduino sketch: asks for step size and hold time, then ramps 0 → 5 V once and prints `t_s,dir,setpoint_v,loopback_v,monitor_v` every 100 ms |

## Setup

```bash
./install.sh pas        # conda env "pas": Python with Tk, NumPy, SciPy, Matplotlib
atlas doctor pas        # checks the env
```

Flash `sweep_up_nano/sweep_up_nano.ino` to the Nano (Arduino IDE or `arduino-cli`; pick the right board, processor and port). Connect the Nano and the photometer by USB and switch Auto-Print on at the photometer. The GUI was developed on macOS (`/dev/cu.usbserial-*`, `/dev/cu.usbmodem*` are listed automatically, other ports can be typed in); it uses `termios` and so does not run on Windows.

```bash
atlas pas
```

Without the installer, `python3 pas_gui.py` in this folder works in any environment with NumPy, SciPy, Matplotlib and Tk.

## Doing a run

1. **Setup window**: set the voltage step (default 0.1 V ≙ 0.18 bar), hold time per step (default 3 s), the two serial ports, replicates per sample, the photometer start delay (seconds between the start of the ramp and the photometer's first point, default 8) and the photometer point interval (default 8.16 s, the real sampling interval of the "5 s" kinetics mode). Optionally list sample names. **Continue** saves these to `~/.pas_gui.json`.
2. **Run window**: type the sample name for the next run and press **Setup**. Both ports open and the GUI listens to the photometer (green button = ready).
3. Set the photometer to a kinetics run at 500 nm, then press **Start** in the GUI. The step size and hold time go to the Nano and the ramp begins; the timer and the live pressure (from the monitor output) run.
4. **Press Start on the photometer by hand** right after (the delay in step 1 is the gap you leave). A sound plays 8 min after the ramp start, when the run is nearly over.
5. When the ramp is done the pressure returns to 0. Let the photometer kinetics run to its end: it sends the whole series only then. The GUI collects it, fits the run and plots OD500 (or the normalised, "portion of intact GVs" curve, with the **OD / normalized** button) against pressure, with P50 ± fit error. For several replicates of one sample it shows all curves and the mean P50 ± SD.
6. **Discard run** drops a bad run from the list (its files stay on disk and are marked `discarded`). **Past samples** on the right switches between runs.

Between two runs the photometer cuvette has to be refilled; nothing in the GUI does that.

## Output

Everything is written live to `pas-gui/data/` (created on the first run, ignored by git):

| File | Content |
|---|---|
| `pressure-<stamp>_<sample>_r<rep>.csv` | pressure log from the Nano: time, setpoint, loopback and monitor voltage |
| `photometer/photometer-<stamp>_<sample>_r<rep>.txt` | raw photometer dump, lossless |
| `photometer/photometer-<stamp>_<sample>_r<rep>.csv` | parsed kinetics series (`time,label,value`) |
| `gui_runs.csv` | one row per run: sample, replicate, both file names, P50, fit error, status (`ok`, `no_fit`, `discarded`) |

## Caveats

- The offset between pressure log and photometer is a typed-in number, not measured: a wrong start delay shifts every pressure assignment, and P50 with it. Keep the delay the same within a series.
- Air bubbles in the cuvette distort the first OD500 points; check the curve before trusting a P50.
