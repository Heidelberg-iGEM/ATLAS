"""Shared I/O and geometry for the GvpA binder pipeline.

Deliberately depends on nothing beyond numpy/scipy/pyyaml so that the driver
scripts run in the `foundry` env (which has no Biopython) and the heavyweight
envs -- BindCraft (colabdesign + pyrosetta) and boltz -- are entered only for
the steps that actually need them.

Two structure-format facts this pipeline is built around, both verified against
real files rather than assumed:

1. The mmCIF `atom_site` COLUMN ORDER DIFFERS between the cleaned target
   (`Cartn_x` at field 13) and RFD3's own output (`Cartn_x` at field 22).
   Never index atom_site by position -- always read the loop_ header.

2. RFD3 RELABELS CHAINS on output. With contig
   `<len>,/0,A12-66,/0,B12-66,/0,E12-66` the emitted model is

       out A = the diffused binder
       out B = input chain A   (flanking GvpA subunit)
       out C = input chain B   (the HOTSPOT subunit)
       out D = input chain E   (flanking GvpA subunit)

   so hotspot "B55" in the RFD3 *input* JSON is residue 55 of chain **C** in
   every downstream file. Getting this backwards silently scores the wrong
   epitope, so the mapping lives in the config and is threaded through, never
   hardcoded at a call site.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

#: van der Waals radii used for the solvent-accessible-surface calculation.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
PROBE = 1.4


# --------------------------------------------------------------------------
# structure I/O
# --------------------------------------------------------------------------

@dataclass
class Structure:
    """A flat, column-oriented view of one model. Fast to slice, no object tree."""

    chain: np.ndarray      # (n,) str   auth_asym_id
    resseq: np.ndarray     # (n,) int   auth_seq_id
    resname: np.ndarray    # (n,) str   three-letter
    atom: np.ndarray       # (n,) str   atom name
    element: np.ndarray    # (n,) str
    xyz: np.ndarray        # (n,3) float

    def __len__(self):
        return len(self.chain)

    def sel(self, mask) -> "Structure":
        return Structure(self.chain[mask], self.resseq[mask], self.resname[mask],
                         self.atom[mask], self.element[mask], self.xyz[mask])

    def chains(self, ids) -> "Structure":
        return self.sel(np.isin(self.chain, list(ids)))

    def residues(self, chain_id):
        """Yield (resseq, resname, atom-index-array) in file order for one chain."""
        m = np.where(self.chain == chain_id)[0]
        if len(m) == 0:
            return
        keys = self.resseq[m]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for grp in np.split(m, bounds):
            yield int(self.resseq[grp[0]]), str(self.resname[grp[0]]), grp

    def sequence(self, chain_id) -> str:
        return "".join(THREE_TO_ONE.get(rn, "X") for _, rn, _ in self.residues(chain_id))

    def ca(self, chain_id) -> np.ndarray:
        m = (self.chain == chain_id) & (self.atom == "CA")
        return self.xyz[m]


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_structure(path) -> Structure:
    """Read a .pdb / .cif / .cif.gz into a Structure. mmCIF is header-driven."""
    path = str(path)
    base = path[:-3] if path.endswith(".gz") else path
    if base.endswith(".pdb") or base.endswith(".ent"):
        return _read_pdb(path)
    return _read_cif(path)


def _read_cif(path) -> Structure:
    # `cols` is the header currently being accumulated; `atom_cols` is the one
    # that actually described the ATOM rows we kept. They must be separate:
    # mmCIF blocks end with a bare "#", and folding that terminator back into
    # `cols` would discard the header before it is ever used.
    cols, atom_cols, rows, in_loop = {}, None, [], False
    with _open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("loop_"):
                cols, in_loop = {}, True
                continue
            if in_loop and s.startswith("_atom_site."):
                cols[s.split(".", 1)[1].strip()] = len(cols)
                continue
            if s.startswith("ATOM") or s.startswith("HETATM"):
                if cols:
                    atom_cols = cols
                if atom_cols is not None:
                    rows.append(s.split())
                continue
            if not s or s.startswith("#"):
                in_loop = False
    if not rows or atom_cols is None:
        raise ValueError(f"no atom_site rows in {path}")
    cols = atom_cols

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"{names} not in atom_site header of {path}")

    ci, ri = col("auth_asym_id", "label_asym_id"), col("auth_seq_id", "label_seq_id")
    ni, ai = col("auth_comp_id", "label_comp_id"), col("auth_atom_id", "label_atom_id")
    ei = col("type_symbol")
    xi, yi, zi = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    keep = [r for r in rows if r[0] == "ATOM"]
    return Structure(
        np.array([r[ci] for r in keep]),
        np.array([int(r[ri]) for r in keep]),
        np.array([r[ni] for r in keep]),
        np.array([r[ai].strip('"') for r in keep]),
        np.array([r[ei] for r in keep]),
        np.array([[float(r[xi]), float(r[yi]), float(r[zi])] for r in keep]),
    )


def _read_pdb(path) -> Structure:
    ch, rs, rn, at, el, xyz = [], [], [], [], [], []
    with _open(path) as fh:
        for l in fh:
            if not l.startswith("ATOM"):
                continue
            ch.append(l[21]); rs.append(int(l[22:26]))
            rn.append(l[17:20].strip()); at.append(l[12:16].strip())
            e = l[76:78].strip()
            el.append(e or l[12:16].strip()[0])
            xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return Structure(np.array(ch), np.array(rs), np.array(rn),
                     np.array(at), np.array(el), np.array(xyz, dtype=float))


def write_pdb(st: Structure, path, chain_order=None):
    """Write a Structure as PDB. AF2/colabdesign and PyRosetta both want PDB."""
    order = list(chain_order) if chain_order else list(dict.fromkeys(st.chain))
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for c in order:
            idx = np.where(st.chain == c)[0]
            for i in idx:
                n += 1
                name = st.atom[i]
                nm = f" {name:<3s}" if len(name) < 4 else name
                fh.write(
                    f"ATOM  {n:5d} {nm:4s}{'':1s}{st.resname[i]:>3s} {c:1s}"
                    f"{st.resseq[i]:4d}{'':4s}"
                    f"{st.xyz[i,0]:8.3f}{st.xyz[i,1]:8.3f}{st.xyz[i,2]:8.3f}"
                    f"{1.0:6.2f}{0.0:6.2f}{'':10s}{st.element[i]:>2s}\n"
                )
            fh.write("TER\n")
        fh.write("END\n")
    return str(path)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def min_dist_to_residue(st: Structure, probe_xyz: np.ndarray, chain_id, resseq) -> float:
    """Closest heavy-atom approach from `probe_xyz` to one residue."""
    m = (st.chain == chain_id) & (st.resseq == int(resseq))
    if not m.any() or len(probe_xyz) == 0:
        return float("inf")
    d = np.linalg.norm(probe_xyz[:, None, :] - st.xyz[m][None, :, :], axis=2)
    return float(d.min())


def contact_residues(st: Structure, binder_chain, target_chains, cutoff=4.0):
    """Target residues within `cutoff` A (heavy atom) of the binder.

    Returns {(chain, resseq): resname}. 4.0 A matches BindCraft's
    `hotspot_residues` so interface residue counts are directly comparable.
    """
    from scipy.spatial import cKDTree

    b = st.chain == binder_chain
    t = np.isin(st.chain, list(target_chains))
    if not b.any() or not t.any():
        return {}
    tree = cKDTree(st.xyz[t])
    hits = tree.query_ball_point(st.xyz[b], cutoff)
    ti = np.where(t)[0]
    out = {}
    for close in hits:
        for j in close:
            k = ti[j]
            out[(str(st.chain[k]), int(st.resseq[k]))] = THREE_TO_ONE.get(str(st.resname[k]), "X")
    return out


def sasa(st: Structure, n_points=192) -> dict:
    """Per-residue SASA (Shrake-Rupley, golden-spiral sphere).

    Local rather than freesasa/Biopython: neither is installed in the driver
    env, and the burial numbers are needed early (epitope triage) before any
    heavyweight env is entered.
    """
    from scipy.spatial import cKDTree

    r = np.array([VDW.get(e, 1.7) for e in st.element]) + PROBE
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    th = np.pi * (1 + 5 ** 0.5) * i
    sph = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]

    tree = cKDTree(st.xyz)
    nbrs = tree.query_ball_point(st.xyz, r=r + r.max())
    per_atom = np.zeros(len(st))
    for k in range(len(st)):
        nb = [j for j in nbrs[k] if j != k
              and np.linalg.norm(st.xyz[j] - st.xyz[k]) < r[k] + r[j]]
        pts = st.xyz[k] + sph * r[k]
        if nb:
            d = np.linalg.norm(pts[:, None, :] - st.xyz[nb][None, :, :], axis=2)
            acc = (d >= r[nb][None, :]).all(axis=1)
        else:
            acc = np.ones(n_points, bool)
        per_atom[k] = 4 * np.pi * r[k] ** 2 * acc.mean()
    out = {}
    for c in dict.fromkeys(st.chain):
        for rs, rn, idx in st.residues(c):
            out[(str(c), rs)] = float(per_atom[idx].sum())
    return out


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal superposition. P, Q are (n,3) and already paired."""
    if len(P) != len(Q) or len(P) == 0:
        return float("nan")
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    return float(np.sqrt((((P @ U) - Q) ** 2).sum() / len(P)))


def secondary_structure(ca: np.ndarray) -> dict:
    """Crude helix/sheet/loop fractions from CA geometry alone.

    A stand-in for DSSP in the driver env: helix is detected from the i->i+3
    CA distance (~5.0-5.5 A) and sheet from an extended i->i+2 span (~6.4-7.0 A).
    Used only as a DIVERSITY DESCRIPTOR and a coarse sanity flag. The numbers
    that gate acceptance come from RFD3's own reported ss fractions and from
    DSSP inside the PyRosetta step, never from here.
    """
    n = len(ca)
    if n < 5:
        return {"helix": 0.0, "sheet": 0.0, "loop": 1.0}
    lab = np.array(["L"] * n, dtype="<U1")
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=1)
    d2 = np.linalg.norm(ca[2:] - ca[:-2], axis=1)
    for i, v in enumerate(d3):
        if 4.6 <= v <= 6.3:
            lab[i:i + 4] = "H"
    for i, v in enumerate(d2):
        if v >= 6.4 and lab[i] != "H":
            lab[i:i + 3] = np.where(lab[i:i + 3] == "H", "H", "E")
    tot = float(n)
    return {"helix": float((lab == "H").sum()) / tot,
            "sheet": float((lab == "E").sum()) / tot,
            "loop": float((lab == "L").sum()) / tot}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _expand_env(x):
    """Expand ${VAR} and ~ in every string of the config."""
    if isinstance(x, str):
        return os.path.expanduser(os.path.expandvars(x))
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand_env(v) for v in x]
    return x


def load_config(path=None) -> dict:
    """Load the campaign config.

    Resolution order: explicit `path` > $CAMPAIGN_CONFIG > configs/gvpa_serratia.yaml.
    The env var exists so one interactive run can drive a different campaign
    without every script needing its own --config wiring (run_root() and the
    eleven script invocations in run_interactive.sh all go through here).
    """
    import yaml
    p = Path(path) if path else Path(
        os.environ.get("CAMPAIGN_CONFIG") or repo_root() / "configs" / "gvpa_serratia.yaml")
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg = _expand_env(cfg)
    cfg["_path"] = str(p)
    cfg["_root"] = str(repo_root())
    return cfg


def resolve(cfg: dict, p: str) -> str:
    """Resolve a config path relative to the repo root unless absolute."""
    return p if os.path.isabs(p) else str(Path(cfg["_root"]) / p)


_OFFSET_CACHE: dict = {}


def chain_offsets(cfg: dict) -> dict:
    """First residue number of each design chain in the input target file.

    RFD3 does not only rename chains, it also RENUMBERS every target chain to
    start at 1 (verified from its own diffused_index_map: A12->B1, B52->C41,
    E66->D55). So the full input->output mapping for a hotspot is

        out_chain  = chain_map[in_chain]
        out_resseq = in_resseq - first_resseq[in_chain] + 1

    Dropping the second half silently measures distances to the wrong residue
    -- for this target, "B56" would resolve to a residue that does not exist
    and every hotspot check would quietly read as a miss.
    """
    key = cfg["_path"]
    if key not in _OFFSET_CACHE:
        st = read_structure(resolve(cfg, cfg["target"]["cif"]))
        _OFFSET_CACHE[key] = {
            c: int(st.resseq[st.chain == c].min())
            for c in cfg["target"]["design_chains"]
        }
    return _OFFSET_CACHE[key]


def hotspot_out_chain(cfg: dict, spec: str):
    """Map an RFD3-INPUT hotspot label (e.g. 'B55') to (out_chain, out_resseq)."""
    in_chain, resseq = spec[0], int(spec[1:])
    mapping = cfg["target"]["chain_map"]
    if in_chain not in mapping:
        raise KeyError(f"hotspot {spec}: chain {in_chain} is not a design chain")
    return mapping[in_chain], resseq - chain_offsets(cfg)[in_chain] + 1


def write_two_chain_pdb(st: Structure, path, binder_chain, target_chains):
    """Rewrite a complex as exactly two chains: target = A, binder = B.

    Rosetta's BuriedUnsatHbonds filter with use_ddG_style="true" -- which is
    what BindCraft's n_InterfaceUnsatHbonds comes from -- hard-fails on any
    pose with more than three chains ("use_ddG_style not compatible with
    symmetry or poses with > 3 chains"). A GvpA rib plus a binder is four.

    So the three target subunits are concatenated into one chain and renumbered
    contiguously, giving a two-chain pose of exactly the shape BindCraft scores,
    where set_interface("A_B") is literally BindCraft's own call.

    This is safe for every metric that is gated on, because all of them are
    DIFFERENCES across the binder/target interface: interface_dG is
    (bound - separated) and any strain introduced at the artificial junctions
    between subunits is present identically in both terms and cancels. dSASA,
    shape complementarity, packstat, H-bond and unsat counts are all interface
    quantities and are likewise untouched. binder_score is a chain-B-only
    TotalEnergyMetric and never sees the target at all.

    What it is NOT safe for is any absolute energy of the target, so nothing
    downstream reports one.
    """
    order = [c for c in target_chains if (st.chain == c).any()]
    keep = np.concatenate([np.where(st.chain == c)[0] for c in order]
                          + [np.where(st.chain == binder_chain)[0]])
    n = 0
    with open(path, "w") as fh:
        for label, chains in (("A", order), ("B", [binder_chain])):
            rnum = 0
            for c in chains:
                for _, resname, idx in st.residues(c):
                    rnum += 1
                    for i in idx:
                        n += 1
                        nm = st.atom[i]
                        nm = f" {nm:<3s}" if len(nm) < 4 else nm
                        fh.write(
                            f"ATOM  {n:5d} {nm:4s} {resname:>3s} {label:1s}"
                            f"{rnum:4d}    "
                            f"{st.xyz[i,0]:8.3f}{st.xyz[i,1]:8.3f}{st.xyz[i,2]:8.3f}"
                            f"{1.0:6.2f}{0.0:6.2f}{'':10s}{st.element[i]:>2s}\n")
            fh.write("TER\n")
        fh.write("END\n")
    del keep
    return str(path)


#: Ideal consecutive CA-CA distance, and RFD3's own tolerance on it.
CACA_IDEAL = 3.8
CACA_TOL = 0.75


def ca_bond_defects(st: Structure, chain_id, tol=CACA_TOL):
    """Indices of consecutive CA-CA bonds deviating from ideal by more than tol."""
    ca = st.ca(chain_id)
    if len(ca) < 2:
        return np.array([], dtype=int), np.array([])
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    dev = np.abs(d - CACA_IDEAL)
    return np.where(dev > tol)[0], d


def trim_terminal_defects(st: Structure, chain_id, tol=CACA_TOL, max_trim=2):
    """Drop malformed terminal residues from one chain.

    RFD3 systematically misplaces the FIRST residue of the diffused chain: in
    this campaign every design had an anomalous first CA-CA bond, COMPRESSED to
    2.3-3.1 A against an ideal of 3.8. Designs differed only in whether that
    landed inside RFD3's 0.75 A tolerance (0.66-0.74, reported "clean") or just
    outside it (0.75-1.53, reported as a chainbreak). It is one bad residue at
    the terminus, not a broken chain, and gating on it discarded 62% of
    otherwise good backbones.

    Trimming is preferred over ignoring: the malformed residue would otherwise
    be handed to MPNN and AF2 as if it were real geometry. Only TERMINAL
    defects are trimmed -- an interior break is a genuinely broken backbone and
    must still fail triage.

    Returns (structure, n_trimmed_start, n_trimmed_end).
    """
    resids = [rs for rs, _, _ in st.residues(chain_id)]
    start, end = 0, 0
    for _ in range(max_trim):
        bad, _ = ca_bond_defects(st, chain_id, tol)
        if len(bad) == 0:
            break
        n = len(st.ca(chain_id))
        if 0 in bad:                       # first bond bad -> drop first residue
            drop = resids[start]
            st = st.sel(~((st.chain == chain_id) & (st.resseq == drop)))
            start += 1
        elif (n - 2) in bad:               # last bond bad -> drop last residue
            drop = resids[len(resids) - 1 - end]
            st = st.sel(~((st.chain == chain_id) & (st.resseq == drop)))
            end += 1
        else:
            break                          # interior defect: not ours to fix
    return st, start, end
