#!/usr/bin/env python3
"""PAS run GUI: pressure ramp + photometer logging + live evaluation in one window.

Per run:  Setup (open both serial ports, listen to the photometer, choose sample/replicate)
->  Start (send step size + hold time to the Arduino Nano, the regulator runs the ramp)
->  press Start on the photometer by hand  ->  after the sweep and the photometer dump the run
is evaluated automatically (Boltzmann fit, P50) and plotted.

The sample name is typed for every run in the run window (the sample list in Setup is only an
optional suggestion). Written to data/ next to this script:
    data/pressure-<stamp>_<sample>_r<rep>.csv                       pressure log (t_s,dir,setpoint_v,loopback_v,monitor_v)
    data/photometer/photometer-<stamp>_<sample>_r<rep>.txt/.csv     raw photometer dump / parsed (time,label,value)
    data/gui_runs.csv                                               sample/replicate <-> files, P50 per run

Arduino sketch: sweep_up_nano/ (asks for step size + hold time, ramps 0 -> 5 V once).
Settings are remembered in ~/.pas_gui.json.  Needs numpy, scipy, matplotlib, Tk.
Usage: python3 pas_gui.py
"""
import csv
import glob
import json
import os
import re
import select
import subprocess
import termios
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pas_core import BAUD, DATA_RE, HEADER_RE, INTERVAL_S, XTIME_RE, XVALUE_RE, boltzmann, ist_bar, process_run, set_raw

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CONFIG = Path.home() / ".pas_gui.json"
V_SUPPLY = 5.0                    # as in sweep_up_nano.ino
BEEP_AFTER_S = 480.0              # sound 8 min after the sweep started: run is almost done
PHOT_SILENCE_S = 3.0              # photometer dump counts as finished after this much silence if $PED is missing

DEFAULTS = dict(step_v=0.1, hold_s=3.0, phot_port="", nano_port="", replicates=3,
                samples="Sample A\nSample B", offset_s=8.0, interval_s=INTERVAL_S)

BG, PANEL, INK, MUTED = "#14505c", "#fff1cc", "#1c2b2f", "#b5a678"
READY, ACCENT = "#d3e8d0", "#e8743b"   # READY = green
OLD_COLORS = ["#2f8aa0", "#6aa86f", "#8e6fc2", "#c9a227"]
FONT = "Helvetica Neue"


class Btn(tk.Label):
    """Label as a button: tk.Button ignores the background colour on macOS."""

    def __init__(self, master, text, cmd, size=16, bold=True, **kw):
        super().__init__(master, text=text, bg=PANEL, fg=INK, cursor="hand2",
                         font=(FONT, size, "bold" if bold else "normal"), **kw)
        self.cmd, self.on = cmd, True
        self.bind("<Button-1>", lambda e: self.on and self.cmd())

    def set(self, on=True, bg=PANEL):
        self.on = on
        self.config(fg=INK if on else MUTED, bg=bg, cursor="hand2" if on else "arrow")


def hms(s):
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def mscs(s):
    c = int(s * 100)
    return f"{c // 6000:02d}:{c // 100 % 60:02d}:{c % 100:02d}"


def safe(name):
    """Sample name -> safe for file names."""
    return re.sub(r"[^\w+-]+", "_", name).strip("_") or "sample"


def sidecar(row):
    path = DATA / "gui_runs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "sample", "replicate", "pressure_csv", "photometer_csv",
                        "p50_bar", "p50_err_bar", "status"])
        w.writerow(row)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PAS runs")
        self.geometry("1280x780")
        self.configure(bg=BG)
        try:
            self.cfg = {**DEFAULTS, **json.loads(CONFIG.read_text())}
        except Exception:
            self.cfg = dict(DEFAULTS)
        self.pres_fd = self.phot_fd = None
        self.runs, self.view, self.raw_od = [], None, True
        self.state = "idle"                   # idle | ready | running | waiting
        self.t0 = None                        # monotonic start of the sweep; the timer runs until the photometer data are in
        self.build_setup()
        self.build_run()
        self.show("setup")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self.poll)
        self.tick()

    # ------------------------------------------------------------------ UI --
    def card(self, master, **kw):
        return tk.Frame(master, bg=PANEL, **kw)

    def show(self, name):
        (self.setup_f if name == "setup" else self.run_f).tkraise()

    def build_setup(self):
        self.setup_f = f = tk.Frame(self, bg=BG)
        f.place(relx=0, rely=0, relwidth=1, relheight=1)
        c = self.card(f, padx=36, pady=28)
        c.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(c, text="Setup", bg=PANEL, fg=INK, font=(FONT, 26, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        ports = sorted(glob.glob("/dev/cu.usbserial-*") + glob.glob("/dev/cu.usbmodem*"))
        self.vars = {}
        rows = [("Voltage step [V]", "step_v", None), ("Hold time per step [s]", "hold_s", None),
                ("Microcontroller port", "nano_port", ports), ("Photometer port", "phot_port", ports),
                ("Replicates per sample", "replicates", None),
                ("Photometer start delay [s]", "offset_s", None),
                ("Photometer point interval [s]", "interval_s", None)]
        for i, (lab, key, opts) in enumerate(rows, 1):
            tk.Label(c, text=lab, bg=PANEL, fg=INK, font=(FONT, 13)).grid(
                row=i, column=0, sticky="w", pady=4, padx=(0, 18))
            v = self.vars[key] = tk.StringVar(value=str(self.cfg[key]))
            w = (ttk.Combobox(c, textvariable=v, values=opts, width=30) if opts is not None
                 else ttk.Entry(c, textvariable=v, width=32))
            w.grid(row=i, column=1, sticky="w", pady=4)
        tk.Label(c, text="Samples (optional, one per line)\nor just type each name in the run window",
                 bg=PANEL, fg=INK, font=(FONT, 13), justify="left").grid(row=9, column=0, sticky="nw", pady=(10, 4))
        self.samples_txt = tk.Text(c, width=32, height=7, font=(FONT, 13), relief="solid", bd=1)
        self.samples_txt.insert("1.0", self.cfg["samples"])
        self.samples_txt.grid(row=9, column=1, sticky="w", pady=(10, 4))
        self.setup_err = tk.Label(c, text="", bg=PANEL, fg="#b3261e", font=(FONT, 12))
        self.setup_err.grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))
        Btn(c, "Continue", self.apply_setup, padx=26, pady=8, relief="solid", bd=1).grid(
            row=11, column=1, sticky="e", pady=(10, 0))

    def apply_setup(self):
        try:
            cfg = dict(step_v=float(self.vars["step_v"].get()), hold_s=float(self.vars["hold_s"].get()),
                       nano_port=self.vars["nano_port"].get().strip(),
                       phot_port=self.vars["phot_port"].get().strip(),
                       replicates=int(self.vars["replicates"].get()),
                       offset_s=float(self.vars["offset_s"].get()),
                       interval_s=float(self.vars["interval_s"].get()))
        except ValueError:
            return self.setup_err.config(text="Please enter valid numbers.")
        cfg["samples"] = self.samples_txt.get("1.0", "end").strip()
        names = list(dict.fromkeys(s.strip() for s in cfg["samples"].splitlines() if s.strip()))
        if cfg["step_v"] <= 0 or cfg["hold_s"] <= 0 or cfg["replicates"] < 1:
            return self.setup_err.config(text="Step, hold time and replicates must be set.")
        if not cfg["nano_port"] or not cfg["phot_port"] or cfg["nano_port"] == cfg["phot_port"]:
            return self.setup_err.config(text="Pick two different ports.")
        self.setup_err.config(text="")
        if (cfg["nano_port"], cfg["phot_port"]) != (self.cfg["nano_port"], self.cfg["phot_port"]):
            self.close_ports()
        self.cfg, self.names = cfg, names
        try:
            CONFIG.write_text(json.dumps(cfg, indent=1))
        except OSError:
            pass
        self.state = "idle"
        self.suggest()
        self.refresh()
        self.show("run")

    def build_run(self):
        self.run_f = f = tk.Frame(self, bg=BG)
        f.place(relx=0, rely=0, relwidth=1, relheight=1)
        f.columnconfigure(0, weight=3, uniform="c")
        f.columnconfigure(1, weight=1, uniform="c")
        f.rowconfigure(1, weight=1)
        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=(16, 8), pady=(16, 8))
        self.b_setup = Btn(top, "Setup", self.do_setup, 26, padx=26, pady=18)
        self.b_setup.pack(side="left")
        self.b_start = Btn(top, "Start", self.do_start, 26, padx=26, pady=18)
        self.b_start.pack(side="left", padx=(2, 18))
        tc = self.card(top)
        tc.pack(side="left", padx=(0, 12))
        self.l_timer = tk.Label(tc, text="00:00:00", bg=PANEL, fg=INK, font=("Menlo", 34, "bold"), padx=18, pady=6)
        self.l_timer.pack()
        self.l_total = tk.Label(tc, text="", bg=PANEL, fg=MUTED, font=(FONT, 11))
        self.l_total.pack(pady=(0, 6))
        self.press_card = pc = self.card(top)
        pc.pack(side="left")
        self.l_press = tk.Label(pc, text="0.00 bar", bg=PANEL, fg=INK, font=("Menlo", 34, "bold"), padx=18, pady=6)
        self.l_press.pack()
        tk.Label(pc, text="pressure (monitor)", bg=PANEL, fg=MUTED, font=(FONT, 11)).pack(pady=(0, 6))
        Btn(top, "⚙", self.open_setup, 22, bold=False, padx=12, pady=4).pack(side="right", anchor="n")

        left = tk.Frame(f, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ic = self.card(left, padx=16, pady=10)
        ic.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ic.columnconfigure(0, weight=1)
        self.l_info = tk.Label(ic, text="", bg=PANEL, fg=INK, font=(FONT, 16, "bold"), justify="center")
        self.l_info.grid(row=0, column=0)
        side = tk.Frame(ic, bg=PANEL)
        side.grid(row=0, column=1)
        self.b_view = Btn(side, "OD", self.toggle_view, 11, padx=8, pady=2, relief="solid", bd=1)
        self.b_view.pack(pady=(0, 4))
        self.b_discard = Btn(side, "Discard run", self.discard, 11, padx=8, pady=2, relief="solid", bd=1)
        self.b_discard.pack()
        self.l_status = tk.Label(ic, text="", bg=PANEL, fg=MUTED, font=(FONT, 11))
        self.l_status.grid(row=1, column=0, columnspan=2)
        sr = tk.Frame(ic, bg=PANEL)
        sr.grid(row=2, column=0, columnspan=2, pady=(6, 0))
        tk.Label(sr, text="Sample name for next run:", bg=PANEL, fg=INK, font=(FONT, 13)).pack(side="left")
        self.sample_var = tk.StringVar()
        self.e_sample = ttk.Entry(sr, textvariable=self.sample_var, width=28, font=(FONT, 13))
        self.e_sample.pack(side="left", padx=(8, 0))
        pl = self.card(left)
        pl.grid(row=1, column=0, sticky="nsew")
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pl)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        right = self.card(f)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 16), pady=16)
        tk.Label(right, text="Past samples", bg=PANEL, fg=INK, font=(FONT, 18, "bold")).pack(pady=(14, 8))
        wrap = tk.Frame(right, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.list_cv = tk.Canvas(wrap, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(wrap, command=self.list_cv.yview)
        self.list_cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.list_cv.pack(side="left", fill="both", expand=True)
        self.list_f = tk.Frame(self.list_cv, bg=PANEL)
        win = self.list_cv.create_window(0, 0, window=self.list_f, anchor="nw")
        self.list_cv.bind("<Configure>", lambda e: self.list_cv.itemconfig(win, width=e.width))
        self.list_f.bind("<Configure>", lambda e: self.list_cv.configure(scrollregion=self.list_cv.bbox("all")))
        self.list_cv.bind("<MouseWheel>", lambda e: self.list_cv.yview_scroll(-e.delta, "units"))

    def open_setup(self):
        if self.state in ("idle", "ready"):
            self.show("setup")

    def toggle_view(self):
        self.raw_od = not self.raw_od
        self.refresh()

    # ----------------------------------------------------- run bookkeeping --
    def slots(self):
        return [(s, r) for s in self.names for r in range(1, self.cfg["replicates"] + 1)]

    def next_slot(self):
        done = {(r["sample"], r["rep"]) for r in self.runs}
        return next((s for s in self.slots() if s not in done), None)

    def suggest(self):
        """Prefill the entry with the next planned sample (empty if nothing is planned)."""
        slot = self.next_slot()
        self.sample_var.set(slot[0] if slot else "")

    def siblings(self, run):
        return [r for r in self.runs if r["sample"] == run["sample"]]

    def p50_stats(self, run):
        """-> (mean, error, n) over all replicates of the sample with a fit; n=1: fit error, else SD."""
        fits = [r["res"]["fit"] for r in self.siblings(run) if r["res"] and r["res"]["fit"]]
        p = np.array([f["p50"] for f in fits])
        if not len(p):
            return None, None, 0
        return p.mean(), (p.std(ddof=1) if len(p) > 1 else fits[0]["p50_err"]), len(p)

    def refresh(self):
        run = self.view
        slot = self.next_slot()
        if self.state in ("ready", "running", "waiting") and self.pending:
            s, r = self.pending["sample"], self.pending["rep"]
            head = f"{s} — replicate {r}/{self.cfg['replicates']}"
            line2 = {"ready": "ready: press Start, then Start on the photometer",
                     "running": "running — press Start on the photometer now",
                     "waiting": "sweep done — waiting for photometer data"}[self.state]
        elif run:
            f = run["res"] and run["res"]["fit"]
            p50 = f"P50 = {f['p50']:.2f} ± {f['p50_err']:.2f} bar" if f else "no P50 (no usable fit)"
            head = f"{run['sample']}, {p50}"
            m, e, n = self.p50_stats(run)
            line2 = f"Replicate {run['rep']}/{self.cfg['replicates']}." + (
                f" Mean P50: {m:.2f} ± {e:.2f} bar" + (f" (SD, n={n})" if n > 1 else " (fit error)") if n else "")
        else:
            head, line2 = "No run yet", ""
        if self.state == "idle":
            line2 += ("   |   type a sample name, then Setup" if slot is None
                      else f"   |   next planned: {slot[0]} #{slot[1]}")
        self.l_info.config(text=head + ("\n" + line2 if line2 else ""))
        self.b_setup.set(self.state in ("idle", "ready"), READY if self.state == "ready" else PANEL)
        self.e_sample.config(state="normal" if self.state in ("idle", "ready") else "disabled")
        self.b_start.set(self.state == "ready")
        self.b_view.config(text="OD" if self.raw_od else "normalized")
        self.b_discard.set(run is not None and self.state == "idle")
        self.draw_list()
        self.draw_plot()

    def draw_list(self):
        for w in self.list_f.winfo_children():
            w.destroy()
        for r in reversed(self.runs):
            f = r["res"] and r["res"]["fit"]
            sel = r is self.view
            b = Btn(self.list_f, f"{r['sample']} · #{r['rep']}    {f['p50']:.2f} bar" if f
                    else f"{r['sample']} · #{r['rep']}    –", lambda r=r: self.select(r),
                    13, bold=sel, padx=8, pady=7, anchor="w", relief="solid", bd=1)
            b.config(bg=READY if sel else PANEL)
            b.pack(fill="x", pady=2)
            b.bind("<MouseWheel>", lambda e: self.list_cv.yview_scroll(-e.delta, "units"))

    def select(self, run):
        self.view = run
        self.refresh()

    def draw_plot(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)
        run = self.view
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Pressure [bar]")
        ax.set_ylabel("OD500" if self.raw_od else "Portion of intact GVs [a.u.]")
        if run is None:
            ax.text(0.5, 0.5, "pressure vs. OD of the latest run appears here", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
        else:
            sib = self.siblings(run)
            for i, r in enumerate(sib):
                res = r["res"]
                if not res:
                    continue
                cur = r is run
                col = ACCENT if cur else OLD_COLORS[i % len(OLD_COLORS)]
                f = res["fit"]
                norm = (lambda y, f=f: y) if (self.raw_od or not f) else \
                    (lambda y, f=f: (y - f["a_low"]) / (f["a_high"] - f["a_low"]))
                ax.plot(res["press"], norm(res["a"]), "o", ms=4.5 if cur else 3, color=col,
                        alpha=1 if cur else 0.55, zorder=3 if cur else 2)
                lab = f"#{r['rep']}" + (" (selected)" if cur else "")
                if f:
                    g = np.linspace(0, max(res["press"].max() * 1.05, 0.1), 300)
                    ax.plot(g, norm(boltzmann(g, f["a_high"], f["a_low"], f["p50"], f["dp"])), "-",
                            color=col, lw=2.8 if cur else 1.3, alpha=1 if cur else 0.7, label=lab,
                            zorder=3 if cur else 2)
                    if cur:
                        ax.axvline(f["p50"], ls=":", color=col, lw=1.2)
                else:
                    ax.plot([], [], "o", color=col, label=lab + " (no fit)")
            ax.legend(fontsize=9, frameon=False)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    # --------------------------------------------------------------- run ----
    def status(self, text):
        self.l_status.config(text=text)

    def open_ports(self):
        for attr, key in (("pres_fd", "nano_port"), ("phot_fd", "phot_port")):
            if getattr(self, attr) is None:
                # macOS: the baud rate has to be set on the open fd
                fd = os.open(self.cfg[key], os.O_RDWR | os.O_NOCTTY)
                set_raw(fd, BAUD)
                setattr(self, attr, fd)

    def close_ports(self):
        for attr in ("pres_fd", "phot_fd"):
            if getattr(self, attr) is not None:
                os.close(getattr(self, attr))
                setattr(self, attr, None)

    def close_files(self):
        for name in ("pres_csv", "phot_raw", "phot_csv"):
            f = getattr(self, name, None)
            if f:
                f.close()
            setattr(self, name, None)

    def do_setup(self):
        name = self.sample_var.get().strip()
        if not name:
            return self.status("Enter a sample name first")
        if name not in self.names:
            self.names.append(name)          # ad-hoc sample: counts like a planned one from now on
        rep = sum(r["sample"] == name for r in self.runs) + 1
        try:
            self.open_ports()
        except OSError as e:
            self.close_ports()
            return self.status(f"Cannot open port: {e}")
        self.close_files()
        for fd in (self.pres_fd, self.phot_fd):
            termios.tcflush(fd, termios.TCIFLUSH)     # drop old garbage before the run
        DATA.joinpath("photometer").mkdir(parents=True, exist_ok=True)
        self.tag = f"{safe(name)}_r{rep}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.phot_path = DATA / "photometer" / f"photometer-{stamp}_{self.tag}.csv"
        self.phot_raw = open(self.phot_path.with_suffix(".txt"), "w", buffering=1)
        self.phot_csv = open(self.phot_path, "w", buffering=1)
        self.phot_csv.write("time,label,value\n")
        self.pres_buf = self.phot_buf = b""
        self.pres_path, self.pres_rows, self.pres_done = None, 0, False
        self.kin_time, self.kin_n, self.phot_done, self.phot_last = None, 0, False, 0.0
        self.pending = dict(sample=name, rep=rep)
        self.state = "ready"
        self.t0 = None
        self.l_timer.config(text="00:00:00")
        self.l_press.config(text="0.00 bar")
        self.press_green(False)
        n = round(V_SUPPLY / self.cfg["step_v"]) + 1
        self.l_total.config(text=f"of {hms(n * self.cfg['hold_s'])}")
        self.status("listening to photometer…")
        self.refresh()

    def do_start(self):
        if self.state != "ready":
            return
        self.state, self.start_deadline = "running", time.time() + 8
        os.write(self.pres_fd, f"{self.cfg['step_v']}\n".encode())
        self.after(300, lambda: os.write(self.pres_fd, f"{self.cfg['hold_s']}\n".encode()))
        self.status("waiting for the Nano…")
        self.refresh()

    def press_green(self, on):
        for w in (self.press_card, *self.press_card.winfo_children()):
            w.config(bg=READY if on else PANEL)

    def tick(self):
        if self.t0 is not None:
            t = time.monotonic() - self.t0
            self.l_timer.config(text=mscs(t))
            if t >= BEEP_AFTER_S and not self.beeped:
                self.beeped = True
                try:
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
                except OSError:
                    self.bell()
        self.after(30, self.tick)

    def poll(self):
        try:
            self.pump()
        except Exception as e:
            traceback.print_exc()
            self.status(f"error: {e}")
        self.after(50, self.poll)

    def pump(self):
        fds = [fd for fd in (self.pres_fd, self.phot_fd) if fd is not None]
        if self.state == "idle" or not fds:
            return
        ready, _, _ = select.select(fds, [], [], 0)
        if self.pres_fd in ready:
            self.read_pres()
        if self.phot_fd in ready:
            self.read_phot()
        if self.state == "running" and self.pres_path is None and time.time() > self.start_deadline:
            self.state = "ready"
            self.status("Nano did not answer — check port / sketch, then Start again")
            self.refresh()
        if (self.state in ("running", "waiting") and not self.phot_done and self.kin_n >= 3
                and time.time() - self.phot_last > PHOT_SILENCE_S):
            self.phot_done = True
        if self.state == "waiting" and self.phot_done:
            self.evaluate()

    def lines(self, fd, attr):
        buf = getattr(self, attr) + os.read(fd, 4096)
        *done, rest = buf.split(b"\n")
        setattr(self, attr, rest)
        return [x.decode("utf-8", "replace").strip() for x in done if x.strip()]

    def read_pres(self):
        for line in self.lines(self.pres_fd, "pres_buf"):
            if self.state != "running":
                continue
            if HEADER_RE.match(line):
                self.close_pres()
                self.pres_path = DATA / f"pressure-{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.tag}.csv"
                self.pres_csv = open(self.pres_path, "w", buffering=1)
                self.pres_csv.write(line + "\n")
                self.t0, self.beeped = time.monotonic(), False
                self.status(f"logging → {self.pres_path.name}")
            elif DATA_RE.match(line) and self.pres_csv:
                self.pres_csv.write(line + "\n")
                cols = line.split(",")
                self.l_press.config(text=f"{ist_bar(float(cols[-1])):.2f} bar")
            elif "done" in line and self.pres_csv:      # "Sweep up done, 0V set."
                self.close_pres()
                self.pres_done, self.state = True, "waiting"
                self.l_press.config(text="0.00 bar")
                self.press_green(True)
                self.status("sweep done — waiting for photometer data")
                self.refresh()

    def close_pres(self):
        if self.pres_csv:
            self.pres_csv.close()
            self.pres_csv = None

    def read_phot(self):
        for line in self.lines(self.phot_fd, "phot_buf"):
            if self.phot_raw is None:       # run already evaluated -> ignore the rest
                continue
            self.phot_raw.write(line + "\n")
            if line == "$PED":
                self.phot_done = self.kin_n >= 3
                continue
            m = XTIME_RE.match(line)
            if m:
                self.kin_time = m.group(1)
                continue
            m = XVALUE_RE.match(line)
            if m and self.kin_time is not None:
                self.kin_n += 1
                self.phot_csv.write(f"{self.kin_time},{self.kin_n},{m.group(1)}\n")
                self.kin_time, self.phot_last = None, time.time()
                self.status(f"photometer: {self.kin_n} points received")

    def evaluate(self):
        pres_path, phot_path = self.pres_path, self.phot_path
        self.close_files()
        if self.t0 is not None:
            self.l_timer.config(text=mscs(time.monotonic() - self.t0))
            self.t0 = None
        self.press_green(False)
        self.status("evaluating…")
        try:
            res = process_run(phot_path, pres_path, self.cfg["offset_s"], self.cfg["interval_s"])
        except Exception as e:
            traceback.print_exc()
            res = None
        run = dict(self.pending, res=res, pres=pres_path, phot=phot_path)
        self.runs.append(run)
        f = res and res["fit"]
        sidecar([datetime.now().isoformat(timespec="seconds"), run["sample"], run["rep"], pres_path.name,
                 phot_path.name, f"{f['p50']:.3f}" if f else "", f"{f['p50_err']:.3f}" if f else "",
                 "ok" if f else "no_fit"])
        self.view, self.state, self.pending = run, "idle", None
        self.status("done")
        self.suggest()
        self.refresh()

    def discard(self):
        run = self.view
        if run is None or self.state != "idle":
            return
        self.runs.remove(run)
        sidecar([datetime.now().isoformat(timespec="seconds"), run["sample"], run["rep"],
                 run["pres"].name, run["phot"].name, "", "", "discarded"])
        self.view = self.runs[-1] if self.runs else None
        self.suggest()
        self.refresh()

    def close(self):
        self.close_files()
        self.close_ports()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
