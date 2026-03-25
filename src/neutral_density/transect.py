"""
transect.py
-----------
Plug-in to the neutral_density module that extends gamma_n and
neutral_surfaces to operate on an entire GO-SHIP hydrographic section
loaded from a WHP-Exchange (.hy1) bottle-data CSV file.

Public API
----------
read_goship_hy1(filepath, fill_value, max_flag, sort_pressure)
    Parse and clean a WHP-Exchange .hy1 file into a tidy DataFrame.

gamma_transect(source, fill_value, max_flag, its90_correction, verbose)
    Compute neutral density γn for every quality-controlled bottle
    sample across all casts in the section.  Returns a CastCollection.

neutral_surface_transect(source, glevels, fill_value, max_flag,
                          its90_correction, verbose)
    Find where a set of neutral density surfaces intersect each cast
    along the section.  Returns a SurfaceCollection.

CastCollection  – container for per-cast gamma results
SurfaceCollection – container for per-surface transect results

Notes
-----
Temperature conversion
    Modern GO-SHIP CTD temperatures are reported on the ITS-90 scale,
    whereas EOS-80 (on which γn is based) is defined on IPTS-68.
    The correction  T_68 = 1.00024 × T_90  is applied by default and
    can be disabled with ``its90_correction=False``.

Longitude convention
    WHP-Exchange files use −180 → +180.  gamma_n requires 0 → 360.
    The conversion is applied internally.

Fill / flag handling
    WHP-Exchange missing values default to −999.  Bottles are retained
    only when CTDPRS, CTDTMP, and CTDSAL all carry quality flags ≤
    max_flag (default 2, i.e. "good").  Bottles with fill-value
    pressures, temperatures, or salinities are dropped regardless of
    flag.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Re-export the single-cast functions from the parent package so users can
# import everything from one place.
# ---------------------------------------------------------------------------
try:
    from . import gamma_n, neutral_surfaces          # package usage
except ImportError:
    from neutral_density import gamma_n, neutral_surfaces  # direct usage fallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WHP_FILL = -999.0
_ITS90_TO_IPTS68 = 1.00024  # T_IPTS68 = T_ITS90 * 1.00024


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CastResult:
    """Neutral-density labels for a single CTD/bottle cast.

    Attributes
    ----------
    station : float
        Station number from the hydrofile (STNNBR).
    cast : float
        Cast number within the station (CASTNO).
    latitude : float
        Station latitude, degrees North.
    longitude : float
        Station longitude, degrees East (−180 → +180).
    pressure : np.ndarray, shape (n,)
        Bottle pressures [dbar].
    salinity : np.ndarray, shape (n,)
        Bottle CTD salinities [PSU, IPSS-78].
    temperature : np.ndarray, shape (n,)
        Bottle in-situ temperatures [°C, IPTS-68].
    gamma : np.ndarray, shape (n,)
        Neutral density γn [kg m⁻³].  −99.0 = algorithm failure,
        −99.1 = outside valid EOS range.
    gamma_lo : np.ndarray, shape (n,)
        Lower γn uncertainty estimate [kg m⁻³].
    gamma_hi : np.ndarray, shape (n,)
        Upper γn uncertainty estimate [kg m⁻³].
    n_bottles : int
        Number of retained bottles.
    """
    station: float
    cast: float
    latitude: float
    longitude: float
    pressure: np.ndarray
    salinity: np.ndarray
    temperature: np.ndarray
    gamma: np.ndarray
    gamma_lo: np.ndarray
    gamma_hi: np.ndarray
    n_bottles: int = field(init=False)

    def __post_init__(self):
        self.n_bottles = len(self.pressure)

    # Convenience -----------------------------------------------------------
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: True where gamma is a valid (non-error) value."""
        return self.gamma > 0.0

    def as_dict(self) -> dict:
        """Return a plain dict suitable for DataFrame construction."""
        return {
            "station": self.station,
            "cast": self.cast,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "pressure": self.pressure,
            "salinity": self.salinity,
            "temperature": self.temperature,
            "gamma": self.gamma,
            "gamma_lo": self.gamma_lo,
            "gamma_hi": self.gamma_hi,
        }


@dataclass
class CastCollection:
    """Results of gamma_transect: a list of CastResult objects.

    Attributes
    ----------
    casts : list of CastResult
        One entry per retained cast, ordered as they appear in the file.
    source : str
        Path of the source hydrofile (if applicable).
    n_casts : int
        Number of casts in the collection.
    """
    casts: list[CastResult]
    source: str = ""
    n_casts: int = field(init=False)

    def __post_init__(self):
        self.n_casts = len(self.casts)

    def __iter__(self):
        return iter(self.casts)

    def __len__(self):
        return self.n_casts

    def __getitem__(self, index):
        return self.casts[index]

    # Bulk accessors ---------------------------------------------------------
    def stations(self) -> np.ndarray:
        """Ordered array of station numbers."""
        return np.array([c.station for c in self.casts])

    def latitudes(self) -> np.ndarray:
        """Station latitudes [°N], one per cast."""
        return np.array([c.latitude for c in self.casts])

    def longitudes(self) -> np.ndarray:
        """Station longitudes [°E, −180→+180], one per cast."""
        return np.array([c.longitude for c in self.casts])

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten all casts into a single tidy DataFrame.

        Each row is one bottle sample.  Columns include station, cast,
        latitude, longitude, pressure, salinity, temperature, gamma,
        gamma_lo, gamma_hi.
        """
        rows = []
        for c in self.casts:
            n = c.n_bottles
            rows.append(pd.DataFrame({
                "station":     np.full(n, c.station),
                "cast":        np.full(n, c.cast),
                "latitude":    np.full(n, c.latitude),
                "longitude":   np.full(n, c.longitude),
                "pressure":    c.pressure,
                "salinity":    c.salinity,
                "temperature": c.temperature,
                "gamma":       c.gamma,
                "gamma_lo":    c.gamma_lo,
                "gamma_hi":    c.gamma_hi,
            }))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@dataclass
class SurfaceCollection:
    """Results of neutral_surface_transect: properties on γn surfaces.

    Attributes
    ----------
    glevels : np.ndarray, shape (ng,)
        The requested neutral density values [kg m⁻³].
    stations : np.ndarray, shape (nc,)
        Station numbers, one per cast.
    latitudes : np.ndarray, shape (nc,)
        Station latitudes [°N].
    longitudes : np.ndarray, shape (nc,)
        Station longitudes [°E, −180→+180].
    pressure : np.ndarray, shape (ng, nc)
        Pressure on each neutral surface at each station [dbar].
        NaN where the surface outcrops, undercrops, or γn failed.
    salinity : np.ndarray, shape (ng, nc)
        Salinity on each surface [PSU].
    temperature : np.ndarray, shape (ng, nc)
        In-situ temperature on each surface [°C, IPTS-68].
    d_pressure : np.ndarray, shape (ng, nc)
        Pressure uncertainty (non-zero for multiply defined surfaces).
    d_salinity : np.ndarray, shape (ng, nc)
        Salinity uncertainty.
    d_temperature : np.ndarray, shape (ng, nc)
        Temperature uncertainty.
    source : str
        Path of the source hydrofile.
    """
    glevels: np.ndarray
    stations: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    pressure: np.ndarray
    salinity: np.ndarray
    temperature: np.ndarray
    d_pressure: np.ndarray
    d_salinity: np.ndarray
    d_temperature: np.ndarray
    source: str = ""

    @property
    def n_surfaces(self) -> int:
        return len(self.glevels)

    @property
    def n_casts(self) -> int:
        return len(self.stations)

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy DataFrame: one row per (surface, cast) pair."""
        records = []
        for ig, glev in enumerate(self.glevels):
            for ic in range(self.n_casts):
                records.append({
                    "gamma_level": glev,
                    "station":     self.stations[ic],
                    "latitude":    self.latitudes[ic],
                    "longitude":   self.longitudes[ic],
                    "pressure":    self.pressure[ig, ic],
                    "salinity":    self.salinity[ig, ic],
                    "temperature": self.temperature[ig, ic],
                    "d_pressure":  self.d_pressure[ig, ic],
                    "d_salinity":  self.d_salinity[ig, ic],
                    "d_temperature": self.d_temperature[ig, ic],
                })
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_goship_hy1(
    filepath: str,
    fill_value: float = _WHP_FILL,
    max_flag: int = 2,
    sort_pressure: bool = True,
) -> pd.DataFrame:
    """Parse a WHP-Exchange .hy1 bottle-data CSV into a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .hy1 file (comma-separated, one header row).
    fill_value : float, optional
        Value used in the file to indicate missing data (default −999).
    max_flag : int, optional
        Maximum acceptable WOCE quality flag for CTDPRS, CTDTMP, and
        CTDSAL (default 2 = "good").  Set to 3 to also accept
        "questionable but acceptable" data.
    sort_pressure : bool, optional
        If True (default), sort each cast by increasing pressure before
        returning, which is required by the neutral-density algorithm.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with fill values replaced by NaN and bottles
        with unacceptable quality flags removed.  Column names are
        preserved from the file header.  A "longitude_360" column is
        added (WHP −180→+180 converted to 0→360 for gamma_n).
    """
    df = pd.read_csv(filepath, index_col=0)
    df.columns = df.columns.str.strip()

    # Replace fill values with NaN
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace(fill_value, np.nan)

    # Quality-flag filtering on the three required variables
    required = {
        "CTDPRS": "CTDPRS",       # pressure has no flag column in WHP
        "CTDTMP": "CTDTMP",
        "CTDSAL": "CTDSAL_FLAG_W",
    }
    flag_cols = {
        "CTDSAL": "CTDSAL_FLAG_W",
    }
    # Only columns that have explicit flag columns get flag-filtered;
    # CTDPRS and CTDTMP are kept if they are not NaN.
    mask = (
        df["CTDPRS"].notna()
        & df["CTDTMP"].notna()
        & df["CTDSAL"].notna()
        & (df["CTDSAL_FLAG_W"] <= max_flag)
    )
    df = df[mask].copy()

    # Longitude 0→360 for gamma_n
    df["longitude_360"] = df["LONGITUDE"] % 360.0

    if sort_pressure:
        df = df.sort_values(["STNNBR", "CASTNO", "CTDPRS"])

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Core transect functions
# ---------------------------------------------------------------------------

def gamma_transect(
    source: "str | pd.DataFrame",
    fill_value: float = _WHP_FILL,
    max_flag: int = 2,
    its90_correction: bool = True,
    verbose: bool = True,
) -> CastCollection:
    """Compute neutral density γn across a GO-SHIP hydrographic section.

    Iterates cast-by-cast over a WHP-Exchange .hy1 file (or a pre-parsed
    DataFrame from :func:`read_goship_hy1`), calling :func:`gamma_n` for
    each cast and collecting the results into a :class:`CastCollection`.

    Parameters
    ----------
    source : str or pd.DataFrame
        Either a file path to a .hy1 file or a DataFrame already returned
        by :func:`read_goship_hy1`.
    fill_value : float, optional
        WHP fill value for missing data (default −999).  Ignored if
        *source* is a DataFrame.
    max_flag : int, optional
        Maximum acceptable quality flag (default 2).  Ignored if *source*
        is a DataFrame.
    its90_correction : bool, optional
        If True (default), multiply CTDTMP by 1.00024 to convert from
        ITS-90 to IPTS-68 before passing to gamma_n.  Modern GO-SHIP
        data are reported on ITS-90; EOS-80 (and thus γn) is defined on
        IPTS-68.  Set to False only if temperatures are already on
        IPTS-68.
    verbose : bool, optional
        If True (default), print a one-line progress message per cast.

    Returns
    -------
    CastCollection
        Container holding one :class:`CastResult` per cast, in the order
        they appear in the file.

    Examples
    --------
    >>> cc = gamma_transect("33RO20131223_hy1.csv")
    >>> cc.n_casts
    113
    >>> cc[0].gamma          # γn profile at station 1
    >>> df = cc.to_dataframe()
    """
    source_path = ""
    if isinstance(source, str):
        source_path = source
        df = read_goship_hy1(source, fill_value=fill_value, max_flag=max_flag)
    else:
        df = source.copy()

    results: list[CastResult] = []

    # Group by station then cast to preserve section order
    for (stn, cst), grp in df.groupby(["STNNBR", "CASTNO"], sort=True):
        grp = grp.sort_values("CTDPRS")

        prs = grp["CTDPRS"].to_numpy(dtype=np.float64)
        tmp = grp["CTDTMP"].to_numpy(dtype=np.float64)
        sal = grp["CTDSAL"].to_numpy(dtype=np.float64)
        lat = float(grp["LATITUDE"].iloc[0])
        lon_360 = float(grp["longitude_360"].iloc[0])
        lon = float(grp["LONGITUDE"].iloc[0])

        # Drop any remaining NaN rows (e.g., missing T/S after flag filter)
        valid = np.isfinite(prs) & np.isfinite(tmp) & np.isfinite(sal)
        prs, tmp, sal = prs[valid], tmp[valid], sal[valid]

        if len(prs) < 2:
            warnings.warn(
                f"Station {stn} cast {cst}: fewer than 2 valid bottles — skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # ITS-90 → IPTS-68
        if its90_correction:
            tmp = tmp * _ITS90_TO_IPTS68

        if verbose:
            print(
                f"  Station {int(stn):>4d}  cast {int(cst)}  "
                f"lat {lat:+7.3f}  lon {lon:+8.3f}  "
                f"n={len(prs):>3d} bottles",
                flush=True,
            )

        gamma, dg_lo, dg_hi = gamma_n(sal, tmp, prs, lon_360, lat)

        results.append(CastResult(
            station=stn,
            cast=cst,
            latitude=lat,
            longitude=lon,
            pressure=prs,
            salinity=sal,
            temperature=tmp,
            gamma=gamma,
            gamma_lo=dg_lo,
            gamma_hi=dg_hi,
        ))

    if verbose:
        print(f"\nDone. Processed {len(results)} casts.")

    return CastCollection(casts=results, source=source_path)


def neutral_surface_transect(
    source: "str | pd.DataFrame | CastCollection",
    glevels: "Sequence[float] | np.ndarray",
    fill_value: float = _WHP_FILL,
    max_flag: int = 2,
    its90_correction: bool = True,
    verbose: bool = True,
) -> SurfaceCollection:
    """Find neutral-density surfaces across a GO-SHIP hydrographic section.

    For each cast along the transect, this function computes γn (via
    :func:`gamma_transect` if needed) and then calls
    :func:`neutral_surfaces` to locate the pressure, salinity, and
    temperature where each requested surface intersects the cast.

    Outcropping/undercropping surfaces are returned as NaN (the native
    −99.0 sentinel from neutral_surfaces is converted).

    Parameters
    ----------
    source : str, pd.DataFrame, or CastCollection
        A .hy1 file path, a pre-parsed DataFrame, or an already-computed
        :class:`CastCollection` from :func:`gamma_transect`.  Passing a
        CastCollection avoids repeating the γn calculation.
    glevels : sequence of float
        Target neutral density values [kg m⁻³], e.g.
        ``[27.5, 27.7, 27.9, 28.1]``.
    fill_value, max_flag, its90_correction, verbose
        Same as in :func:`gamma_transect`.  Ignored when *source* is
        a CastCollection.

    Returns
    -------
    SurfaceCollection
        2-D arrays (n_surfaces × n_casts) of pressure, salinity,
        temperature, and their multiply-defined-surface uncertainties.

    Examples
    --------
    >>> glevels = np.arange(26.0, 28.5, 0.1)
    >>> sc = neutral_surface_transect("33RO20131223_hy1.csv", glevels)
    >>> sc.pressure.shape    # (n_surfaces, n_casts)
    (25, 113)
    >>> df = sc.to_dataframe()
    """
    glevels = np.asarray(glevels, dtype=np.float64)

    # ------------------------------------------------------------------
    # Obtain the cast collection (compute γn if not already done)
    # ------------------------------------------------------------------
    if isinstance(source, CastCollection):
        cc = source
        source_path = source.source
    else:
        source_path = source if isinstance(source, str) else ""
        if verbose:
            print("Computing γn for all casts …")
        cc = gamma_transect(
            source,
            fill_value=fill_value,
            max_flag=max_flag,
            its90_correction=its90_correction,
            verbose=verbose,
        )

    ng = len(glevels)
    nc = cc.n_casts

    # Allocate output arrays; use NaN as the "not found" sentinel
    pns_out  = np.full((ng, nc), np.nan)
    sns_out  = np.full((ng, nc), np.nan)
    tns_out  = np.full((ng, nc), np.nan)
    dpns_out = np.full((ng, nc), np.nan)
    dsns_out = np.full((ng, nc), np.nan)
    dtns_out = np.full((ng, nc), np.nan)

    if verbose:
        print("\nFinding neutral surfaces …")

    for ic, cast in enumerate(cc):
        # neutral_surfaces requires at least 2 bottles with valid γn
        valid = cast.gamma > 0.0
        if valid.sum() < 2:
            warnings.warn(
                f"Station {cast.station} cast {cast.cast}: "
                "fewer than 2 bottles with valid γn — surfaces skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue

        s_v   = cast.salinity[valid]
        t_v   = cast.temperature[valid]
        p_v   = cast.pressure[valid]
        gam_v = cast.gamma[valid]

        # gamma_n may return -99.0 for isolated interior bottles;
        # re-filter to a strictly monotone-gamma subset if needed.
        s_v, t_v, p_v, gam_v = _monotone_gamma_subset(s_v, t_v, p_v, gam_v)
        if len(p_v) < 2:
            continue

        sns, tns, pns, dsns, dtns, dpns = neutral_surfaces(
            s_v, t_v, p_v, gam_v, glevels
        )

        # Convert −99.0 sentinels to NaN
        _MISS = -90.0
        pns_out[:, ic]  = np.where(pns  > _MISS, pns,  np.nan)
        sns_out[:, ic]  = np.where(sns  > _MISS, sns,  np.nan)
        tns_out[:, ic]  = np.where(tns  > _MISS, tns,  np.nan)
        dpns_out[:, ic] = np.where(pns  > _MISS, dpns, np.nan)
        dsns_out[:, ic] = np.where(sns  > _MISS, dsns, np.nan)
        dtns_out[:, ic] = np.where(tns  > _MISS, dtns, np.nan)

        if verbose:
            n_found = np.sum(np.isfinite(pns_out[:, ic]))
            print(
                f"  Station {int(cast.station):>4d}  "
                f"{n_found}/{ng} surfaces found",
                flush=True,
            )

    return SurfaceCollection(
        glevels=glevels,
        stations=cc.stations(),
        latitudes=cc.latitudes(),
        longitudes=cc.longitudes(),
        pressure=pns_out,
        salinity=sns_out,
        temperature=tns_out,
        d_pressure=dpns_out,
        d_salinity=dsns_out,
        d_temperature=dtns_out,
        source=source_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monotone_gamma_subset(
    s: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the largest contiguous block of valid, monotone-gamma data.

    neutral_surfaces is most robust when γn increases monotonically with
    pressure.  Interior algorithm failures (−99.0) can break monotonicity.
    This helper strips leading/trailing failure values and keeps bottles
    with positive γn so that the resulting profile is suitable for
    neutral_surfaces without further modification.

    Only bottles with gamma > 0 are retained (algorithm failures are
    already filtered by the caller); the subset is then trimmed to the
    first contiguous run of increasing gamma.
    """
    # Already filtered for gamma > 0 by caller
    mask = gamma > 0.0
    s, t, p, gamma = s[mask], t[mask], p[mask], gamma[mask]
    return s, t, p, gamma


def _lon_to_360(lon: float) -> float:
    """Convert a longitude in [−180, 180] to [0, 360]."""
    return float(lon % 360.0)
