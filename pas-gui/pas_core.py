"""Helpers for pas_gui.py: serial setup, line formats, and the PAS evaluation (Boltzmann fit -> P50).

Kept in one small module so the GUI has no dependency outside this folder.
Needs numpy and scipy; the serial part uses only the standard library (macOS/Linux termios).
"""
import csv
import re
import termios
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

BAUD = "115200"                   # microcontroller and photometer both run 115200 8N1

# SMC regulator calibration: setpoint 0-5 V = 0-9 bar, monitor echo 1-5 V = 0-9 bar.
P_FULL_BAR = 9.0
MONITOR_V_ZERO = 1.0
MONITOR_V_MAX = 5.0

# The Biowave II "5 s" kinetics mode really samples every ~8.16 s (measured by live sync).
INTERVAL_S = 8.16

HEADER_RE = re.compile(r"^t_s,[a-z_,]+$")     # sketch prints this line when a sweep starts
DATA_RE = re.compile(r"^[0-9]")
XTIME_RE = re.compile(r'^\$XC "(\d{2}:\d{2}:\d{2})" 1024$')              # photometer kinetics: time stamp
XVALUE_RE = re.compile(r'^\$XC "(-?\d+(?:\.\d+)?)" 1024$')               # photometer kinetics: A500 value

_K_5_95 = float(np.log(0.95 / 0.05))          # distance from P50 to the 5 % / 95 % point, in units of dP
_K_10_90 = float(2.0 * np.log(9.0))           # 10-90 % width, in units of dP


def set_raw(fd, baud):
    """8N1 raw mode at `baud` on an already-open fd (macOS drops the rate if it is set before open/close)."""
    speed = getattr(termios, f"B{baud}")
    iflag, oflag, cflag, lflag, _, _, cc = termios.tcgetattr(fd)
    iflag &= ~(termios.IXON | termios.IXOFF | termios.IXANY | termios.ICRNL
               | termios.INLCR | termios.IGNCR | termios.ISTRIP | termios.BRKINT)
    oflag &= ~termios.OPOST
    lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
    cflag &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | termios.CRTSCTS)
    cflag |= termios.CS8 | termios.CLOCAL | termios.CREAD
    cc[termios.VMIN], cc[termios.VTIME] = 1, 0
    termios.tcsetattr(fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])
    termios.tcflush(fd, termios.TCIFLUSH)


def ist_bar(v):
    """Monitor voltage -> actual pressure in bar."""
    return (v - MONITOR_V_ZERO) / (MONITOR_V_MAX - MONITOR_V_ZERO) * P_FULL_BAR


def boltzmann(P, a_high, a_low, p50, dp):
    """Falling 4-parameter Boltzmann sigmoid; p50 is the inflection point (collapse pressure)."""
    return a_low + (a_high - a_low) / (1.0 + np.exp((P - p50) / dp))


def _parse_hms(s):
    try:
        h, m, sec = (int(p) for p in s.strip().split(":"))
    except ValueError:
        return None
    return h * 3600 + m * 60 + sec


def load_photometer(path, interval):
    """-> (t, A500). Keeps only the first kinetics series (strictly rising time stamps)."""
    rows = list(csv.DictReader(open(path)))
    out, prev = [], -1
    for r in rows:
        sec = _parse_hms(r["time"])
        if sec is None or sec <= prev:
            break
        out.append(r)
        prev = sec
    rows = out if len(out) >= 3 else rows
    if not rows:
        raise ValueError(f"no data in {path}")
    return np.arange(len(rows)) * interval, np.array([float(r["value"]) for r in rows])


def load_pres(path, offset):
    """-> (t, bar) of the pressure log; `offset` shifts it by the photometer start delay."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"no data in {path}")
    t = np.array([float(r["t_s"]) - offset for r in rows])
    return t, np.array([ist_bar(float(r["monitor_v"])) for r in rows])


def mean_pressure_per_interval(t_phot, interval, t_pres, bar_pres):
    """Mean pressure in each photometer window [t0, t0+interval) -> (press, covered).

    A window behind the end of the pressure log (sweep over, vented) gets press=0, covered=False;
    a window with no coverage at all gets nan."""
    press, covered = [], []
    for t0 in t_phot:
        m = (t_pres >= t0) & (t_pres < t0 + interval)
        if m.any():
            press.append(float(bar_pres[m].mean()))
            covered.append(True)
        else:
            press.append(0.0 if t0 >= t_pres.max() else np.nan)
            covered.append(False)
    return np.array(press), np.array(covered, dtype=bool)


def fit_collapse(press, a):
    """Boltzmann fit of the rising-pressure branch, up to the A500 minimum (A rises again after it in
    some runs: residual kinetics, not part of the sigmoid). -> dict with parameters, errors, R^2."""
    order = np.argsort(press)
    P_all, A_all = np.asarray(press, float)[order], np.asarray(a, float)[order]
    i_min = int(np.argmin(A_all))
    P, A = P_all[:i_min + 1], A_all[:i_min + 1]
    if len(P) < 4:                                   # too few points before the minimum: use everything
        P, A, i_min = P_all, A_all, len(P_all) - 1

    a_high0 = float(np.mean(A[:max(3, len(A) // 5)]))
    a_low0 = float(np.mean(A[-max(3, len(A) // 5):]))
    p50_0 = (float(np.interp(0.5 * (a_high0 + a_low0), A[::-1], P[::-1]))
             if a_high0 != a_low0 else float(np.median(P)))
    span = max(A.max() - A.min(), 1e-3)
    bounds = ([A.min() - span, A.min() - span, P.min() - 1.0, 0.05],
              [A.max() + span, A.max() + span, P.max() + 1.0, max(P.max() - P.min(), 2.0)])
    popt, pcov = curve_fit(boltzmann, P, A, p0=[a_high0, a_low0, p50_0, 1.0], bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    a_high, a_low, p50, dp = popt
    ss_res = float(np.sum((A - boltzmann(P, *popt)) ** 2))
    ss_tot = float(np.sum((A - A.mean()) ** 2))
    return dict(a_high=a_high, a_low=a_low, p50=p50, dp=dp, p50_err=perr[2], dp_err=perr[3],
                p_onset=p50 - _K_5_95 * dp, p_95=p50 + _K_5_95 * dp, width_10_90=_K_10_90 * dp,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                n=len(P), p_min=float(P_all[i_min]), a_min=float(A_all[i_min]))


def process_run(phot, pres, offset, interval):
    """Load and fit one run. -> dict(press, a, fit, drop_pct), or None if there are too few pressure points.
    fit is None for runs without a usable collapse (OD drop under 5 % or failed fit): control runs."""
    t_phot, a = load_photometer(phot, interval)
    t_pres, bar_pres = load_pres(pres, offset)
    press, covered = mean_pressure_per_interval(t_phot, interval, t_pres, bar_pres)
    ramp = covered & np.isfinite(press) & (press > 0.05)
    if ramp.sum() < 5:
        return None
    order = np.argsort(press[ramp])
    P, A = press[ramp][order], a[ramp][order]
    i_min = int(np.argmin(A))
    base = float(np.mean(A[:5]))
    drop_pct = (float(A[i_min]) - base) / base * 100.0
    if drop_pct > -5.0:
        return dict(press=P, a=A, fit=None, drop_pct=drop_pct)
    try:
        fit = fit_collapse(P, A)
    except Exception:
        return dict(press=P, a=A, fit=None, drop_pct=drop_pct)
    return dict(press=P[:i_min + 1], a=A[:i_min + 1], fit=fit, drop_pct=drop_pct)
