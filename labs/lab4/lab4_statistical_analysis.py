#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lab 4: Statistical Analysis
Run:
    python labs/lab4/lab4_statistical_analysis.py
This script:
1) reads the three CSV files from /datasets
2) makes descriptive plots
3) solves probability tasks
4) fits a normal distribution
5) writes a text report
6) runs extra credit analysis
"""

from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon

# -----------------------------------------------------------------------------
# PATHS (auto)
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()
LAB_DIR = HERE.parent
REPO_ROOT = next(p for p in [*HERE.parents] if (p / "datasets").exists())
DATASETS_DIR = REPO_ROOT / "datasets"


# -----------------------------------------------------------------------------
# SMALL UTILS
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as Series; if missing, raise."""
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}. Available: {list(df.columns)}")
    return df[col].dropna()


# -----------------------------------------------------------------------------
# LOADING + SCHEMA FIX
# -----------------------------------------------------------------------------
def load_data(file_name: str) -> pd.DataFrame:
    """Read CSV from datasets/ and drop fully empty rows."""
    path = DATASETS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path).dropna(how="all").copy()
    return df


def harmonize_materials_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make material_properties.csv look like:
        material, test_no, strength_mpa
    We accept:
        material_type -> material
        test_number   -> test_no
        yield_strength_mpa -> strength_mpa
    """
    rename_map = {}
    if "material_type" in df.columns:
        rename_map["material_type"] = "material"
    if "test_number" in df.columns:
        rename_map["test_number"] = "test_no"
    if "yield_strength_mpa" in df.columns:
        rename_map["yield_strength_mpa"] = "strength_mpa"
    if rename_map:
        df = df.rename(columns=rename_map)

    # fallback names (if instructor gave already correct names, no problem)
    if "material" not in df.columns:
        df["material"] = "MATERIAL_1"
    if "test_no" not in df.columns:
        # create a running index
        df["test_no"] = np.arange(1, len(df) + 1)
    if "strength_mpa" not in df.columns:
        # last resort: try to guess from any numeric column
        # take first numeric col
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            df["strength_mpa"] = df[num_cols[0]].astype(float)
        else:
            # create dummy but we let script run
            df["strength_mpa"] = 0.0
    return df


def harmonize_loads_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make structural_loads.csv look like:
        timestamp, component, load_kN
    We accept:
        time -> timestamp
        component_type -> component
        part -> component
        load, load_kn, Load_kN -> load_kN
    If component info is missing, we create 'unknown_component'.
    """
    rename_map = {}
    if "time" in df.columns:
        rename_map["time"] = "timestamp"
    if "component_type" in df.columns:
        rename_map["component_type"] = "component"
    if "part" in df.columns:
        rename_map["part"] = "component"
    if "member" in df.columns:
        rename_map["member"] = "component"
    if "load" in df.columns:
        rename_map["load"] = "load_kN"
    if "load_kn" in df.columns:
        rename_map["load_kn"] = "load_kN"
    if "Load_kN" in df.columns:
        rename_map["Load_kN"] = "load_kN"

    if rename_map:
        df = df.rename(columns=rename_map)

    # timestamp fix
    if "timestamp" not in df.columns:
        # create a simple index-based timestamp
        df["timestamp"] = np.arange(1, len(df) + 1)

    # component fix
    if "component" not in df.columns:
        df["component"] = "unknown_component"

    # load_kN fix
    if "load_kN" not in df.columns:
        # try any numeric column as load
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            df["load_kN"] = df[num_cols[0]].astype(float)
        else:
            df["load_kN"] = 0.0

    return df


# -----------------------------------------------------------------------------
# PART 1 — DESCRIPTIVE STATISTICS
# this part describes concrete strength: center, spread, shape
# -----------------------------------------------------------------------------
def calculate_descriptive_stats(df: pd.DataFrame, column: str = "strength_mpa") -> pd.DataFrame:
    s = _safe_col(df, column).astype(float)
    desc = {
        "count": s.count(),
        "mean": s.mean(),
        "median": s.median(),
        "mode": (s.mode().iloc[0] if not s.mode().empty else np.nan),
        "min": s.min(),
        "q1": s.quantile(0.25),
        "q2_median": s.quantile(0.50),
        "q3": s.quantile(0.75),
        "max": s.max(),
        "range": s.max() - s.min(),
        "var": s.var(ddof=1),
        "std": s.std(ddof=1),
        "iqr": s.quantile(0.75) - s.quantile(0.25),
        "skewness": stats.skew(s, bias=False, nan_policy="omit"),
        "kurtosis": stats.kurtosis(s, fisher=True, bias=False, nan_policy="omit"),
    }
    return pd.DataFrame(desc, index=[column]).T


def plot_distribution(df: pd.DataFrame, column: str, title: str, save_path: Path | None = None):
    s = _safe_col(df, column).astype(float)
    mu, sigma = s.mean(), s.std(ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(s, kde=True, stat="density", bins="auto", ax=ax, alpha=0.7)
    x = np.linspace(s.min(), s.max(), 400)
    ax.plot(x, norm.pdf(x, mu, sigma), lw=2, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
    ax.axvline(mu, ls="--", label="Mean")
    ax.axvline(s.median(), ls=":", label="Median")
    if not s.mode().empty:
        ax.axvline(s.mode().iloc[0], ls="-.", label="Mode")
    for k in [1, 2, 3]:
        ax.axvspan(mu - k*sigma, mu + k*sigma, alpha=0.04, label="±1σ" if k == 1 else None)
    ax.set_title(title); ax.set_xlabel(column); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.2)
    if save_path:
        _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_box(df: pd.DataFrame, column: str, save_path: Path):
    s = _safe_col(df, column).astype(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(y=s, ax=ax)
    ax.set_title("Concrete Strength — Boxplot")
    ax.set_ylabel(column); ax.grid(axis="y", alpha=0.2)
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# PART 2 — PROBABILITY DISTRIBUTIONS
# this part shows typical pmf/cdf and pdf/cdf for main distributions
# -----------------------------------------------------------------------------
def calculate_probability_binomial(n: int, p: float, k):
    if isinstance(k, int):
        return binom.pmf(k, n, p)
    if isinstance(k, slice):
        start = 0 if k.start is None else k.start
        stop = k.stop
        return binom.cdf(stop, n, p) - (binom.cdf(start - 1, n, p) if start > 0 else 0)
    if isinstance(k, (list, tuple)):
        return sum(binom.pmf(int(ki), n, p) for ki in k)
    raise TypeError("Unsupported k type")


def calculate_probability_poisson(lambda_param: float, k):
    if isinstance(k, int):
        return poisson.pmf(k, lambda_param)
    if isinstance(k, slice):
        start = 0 if k.start is None else k.start
        stop = k.stop
        return poisson.cdf(stop, lambda_param) - (poisson.cdf(start - 1, lambda_param) if start > 0 else 0)
    raise TypeError("Unsupported k type")


def calculate_probability_normal(mean: float, std: float, x_lower=None, x_upper=None):
    if x_lower is None and x_upper is None:
        raise ValueError("give at least one bound")
    if x_lower is None:
        return norm.cdf(x_upper, mean, std)
    if x_upper is None:
        return 1 - norm.cdf(x_lower, mean, std)
    return norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std)


def calculate_probability_exponential(mean: float, x: float, right_tail: bool = False):
    lam = 1.0 / mean
    if right_tail:
        return np.exp(-lam * x)
    return 1 - np.exp(-lam * x)


def plot_probability_distributions(save_path_discrete: Path):
    # discrete
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # Bernoulli
    p = 0.7; k = np.array([0, 1]); pmf = [1-p, p]
    axes[0, 0].stem(k, pmf, basefmt=" "); axes[0, 0].set_title("Bernoulli PMF (p=0.7)")
    axes[1, 0].step(k, np.cumsum(pmf), where="post"); axes[1, 0].set_title("Bernoulli CDF")

    # Binomial
    k = np.arange(0, 21)
    axes[0, 1].stem(k, binom.pmf(k, 20, 0.2), basefmt=" ")
    axes[0, 1].set_title("Binomial PMF (n=20, p=0.2)")
    axes[1, 1].step(k, binom.cdf(k, 20, 0.2), where="post")
    axes[1, 1].set_title("Binomial CDF")

    # Poisson
    k = np.arange(0, 20)
    axes[0, 2].stem(k, poisson.pmf(k, 5), basefmt=" ")
    axes[0, 2].set_title("Poisson PMF (λ=5)")
    axes[1, 2].step(k, poisson.cdf(k, 5), where="post")
    axes[1, 2].set_title("Poisson CDF")

    fig.tight_layout()
    _ensure_dir(save_path_discrete); fig.savefig(save_path_discrete, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # continuous
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    x = np.linspace(0, 1, 400)
    axes2[0].plot(x, uniform.pdf(x, 0, 1)); axes2[0].plot(x, uniform.cdf(x, 0, 1))
    axes2[0].set_title("Uniform PDF & CDF")

    x = np.linspace(-3, 3, 400)
    axes2[1].plot(x, norm.pdf(x, 0, 1)); axes2[1].plot(x, norm.cdf(x, 0, 1))
    axes2[1].set_title("Normal(0,1) PDF & CDF")

    x = np.linspace(0, 6, 400)
    axes2[2].plot(x, expon.pdf(x, scale=1)); axes2[2].plot(x, expon.cdf(x, scale=1))
    axes2[2].set_title("Exponential PDF & CDF")
    fig2.tight_layout()
    fig2.savefig(LAB_DIR / "probability_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)


# -----------------------------------------------------------------------------
# PART 3 — BAYES
# -----------------------------------------------------------------------------
def apply_bayes_theorem(prior: float, sensitivity: float, specificity: float):
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
    posterior = sensitivity * prior / p_pos
    return {"posterior_given_positive": posterior, "p_positive": p_pos}


def plot_bayes_tree(prior: float, sensitivity: float, specificity: float, save_path: Path):
    res = apply_bayes_theorem(prior, sensitivity, specificity)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    nodes = {
        "Start": (0.05, 0.5),
        "Damage": (0.35, 0.7),
        "NoDamage": (0.35, 0.3),
        "Pos|D": (0.7, 0.82),
        "Neg|D": (0.7, 0.58),
        "Pos|~D": (0.7, 0.42),
        "Neg|~D": (0.7, 0.18),
    }
    for name, (x, y) in nodes.items():
        ax.text(x, y, name, ha="center", va="center", bbox=dict(boxstyle="round", fc="white"))

    def edge(a, b, text):
        xa, ya = nodes[a]; xb, yb = nodes[b]
        ax.annotate("", xy=(xb - 0.02, yb), xytext=(xa + 0.06, ya),
                    arrowprops=dict(arrowstyle="-"))
        ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.03, text, ha="center")

    edge("Start", "Damage", f"P(D)={prior:.2f}")
    edge("Start", "NoDamage", f"P(~D)={1 - prior:.2f}")
    edge("Damage", "Pos|D", f"P(+|D)={sensitivity:.2f}")
    edge("Damage", "Neg|D", f"P(-|D)={1 - sensitivity:.2f}")
    edge("NoDamage", "Pos|~D", f"P(+|~D)={1 - specificity:.2f}")
    edge("NoDamage", "Neg|~D", f"P(-|~D)={specificity:.2f}")

    ax.text(0.5, 0.03,
            f"P(D|+)={res['posterior_given_positive']:.3f},  P(+)={res['p_positive']:.3f}",
            ha="center")

    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# PART 4 — FITTING + DASHBOARD
# -----------------------------------------------------------------------------
def fit_distribution(df: pd.DataFrame, column: str, distribution_type: str = "normal"):
    s = _safe_col(df, column).astype(float)
    if distribution_type.lower() == "normal":
        mu, sigma = norm.fit(s)
        return {"type": "normal", "mean": mu, "std": sigma}
    raise NotImplementedError("Only normal is implemented")


def plot_distribution_fitting(df: pd.DataFrame, column: str, fitted_dist: dict, save_path: Path):
    s = _safe_col(df, column).astype(float)
    mu, sigma = fitted_dist["mean"], fitted_dist["std"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(s, stat="density", bins="auto", ax=ax, alpha=0.6, label="Data")
    x = np.linspace(s.min(), s.max(), 400)
    ax.plot(x, norm.pdf(x, mu, sigma), lw=2, label=f"Fit: N({mu:.2f}, {sigma:.2f})")
    syn = np.random.default_rng(42).normal(mu, sigma, size=len(s))
    sns.kdeplot(syn, ax=ax, lw=1.5, label="Synthetic KDE")
    ax.set_title("Concrete Strength — Normal Fit")
    ax.set_xlabel(column); ax.legend(); ax.grid(alpha=0.2)
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, column: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df[column], ax=axes[0]); axes[0].set_title("Boxplot"); axes[0].grid(axis="y", alpha=0.2)
    sns.histplot(df[column], kde=True, ax=axes[1]); axes[1].set_title("Histogram + KDE")
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# PART 5 — GROUP COMPARISON + REPORT
# -----------------------------------------------------------------------------
def plot_material_comparison(df: pd.DataFrame, column: str, group_column: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=group_column, y=column, ax=ax)
    ax.set_title("Material Comparison (Boxplot)")
    ax.grid(axis="y", alpha=0.2)
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_statistical_report(text_lines: list[str], output_file: Path):
    _ensure_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))


# -----------------------------------------------------------------------------
# EXTRA CREDIT
# -----------------------------------------------------------------------------
def ec_bootstrap_ci(series: pd.Series, B: int = 3000, seed: int = 2025):
    data = series.dropna().astype(float).values
    rng = np.random.default_rng(seed)
    means = []; stds = []; n = len(data)
    for _ in range(B):
        bs = rng.choice(data, size=n, replace=True)
        means.append(bs.mean())
        stds.append(bs.std(ddof=1))
    means = np.array(means); stds = np.array(stds)
    return {
        "mean_hat": float(means.mean()),
        "std_hat": float(stds.mean()),
        "mean_ci": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "std_ci": (float(np.percentile(stds, 2.5)), float(np.percentile(stds, 97.5))),
        "means": means,
        "stds": stds,
    }


def ec_plot_bootstrap(means: np.ndarray, stds: np.ndarray, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(means, kde=True, ax=axes[0]); axes[0].set_title("Bootstrap Means")
    sns.histplot(stds, kde=True, ax=axes[1]); axes[1].set_title("Bootstrap Stds")
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ec_monte_carlo_reliability(strength_fit: dict, loads_df: pd.DataFrame,
                               strength_samples: int = 20000, kappa: float = 10.0,
                               save_path: Path | None = None, seed: int = 7):
    mu, sigma = strength_fit["mean"], strength_fit["std"]
    rng = np.random.default_rng(seed)
    S = rng.normal(mu, sigma, strength_samples)
    D = rng.choice(loads_df["load_kN"].values, strength_samples, replace=True)
    M = kappa * S - D
    pf = float(np.mean(M < 0.0))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(M, bins="auto", kde=True, ax=ax)
    ax.axvline(0, color="r", lw=2, label="Failure (M=0)")
    ax.set_title(f"Monte Carlo Safety Margin (pf={pf:.4f})")
    ax.set_xlabel("M = kappa*strength - load")
    ax.legend()
    if save_path:
        _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"pf": pf, "kappa": kappa}


def ec_gmm_fit(series: pd.Series, save_path: Path):
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}
    x = series.dropna().astype(float).values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
    xs = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(xs))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x.ravel(), stat="density", bins="auto", alpha=0.6, ax=ax, label="Data")
    ax.plot(xs.ravel(), pdf, lw=2, label="GMM(2) PDF")
    ax.set_title("Gaussian Mixture Fit (2 components)"); ax.legend()
    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "ok",
        "weights": gmm.weights_.ravel().tolist(),
        "means": gmm.means_.ravel().tolist(),
        "stds": np.sqrt(gmm.covariances_.ravel()).tolist(),
    }


_SPCC = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
}


def ec_xbar_r_chart(df: pd.DataFrame, value_col: str, group_col: str, save_path: Path):
    groups = df.groupby(group_col)[value_col].apply(lambda s: s.dropna().astype(float).values)
    means = groups.apply(np.mean)
    ranges = groups.apply(lambda x: np.max(x) - np.min(x) if len(x) >= 2 else np.nan)
    ns = groups.apply(len)
    n_star = int(np.clip(int(round(ns.mean())), 2, 10))
    A2, D3, D4 = _SPCC[n_star]["A2"], _SPCC[n_star]["D3"], _SPCC[n_star]["D4"]
    xbarbar = means.mean(); rbar = ranges.mean()
    UCLx = xbarbar + A2 * rbar; LCLx = xbarbar - A2 * rbar
    UCLr = D4 * rbar;          LCLr = D3 * rbar

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(means.index.astype(str), means.values, marker="o")
    axes[0].axhline(xbarbar, ls="--", label="CL")
    axes[0].axhline(UCLx, color="r", ls="--", label="UCL")
    axes[0].axhline(LCLx, color="r", ls="--", label="LCL")
    axes[0].set_ylabel("X̄"); axes[0].set_title(f"X̄–R Chart (n≈{n_star})"); axes[0].legend()

    axes[1].plot(ranges.index.astype(str), ranges.values, marker="o")
    axes[1].axhline(rbar, ls="--", label="CL")
    axes[1].axhline(UCLr, color="r", ls="--", label="UCL")
    axes[1].axhline(LCLr, color="r", ls="--", label="LCL")
    axes[1].set_ylabel("R"); axes[1].set_xlabel(group_col); axes[1].legend()

    _ensure_dir(save_path); fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_star": n_star, "xbarbar": float(xbarbar), "rbar": float(rbar),
        "UCLx": float(UCLx), "LCLx": float(LCLx),
        "UCLr": float(UCLr), "LCLr": float(LCLr)
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print(f"Repo root : {REPO_ROOT}")
    print(f"Datasets  : {DATASETS_DIR}")

    # load and fix
    concrete = load_data("concrete_strength.csv")
    print("\n[Concrete] shape, columns:", concrete.shape, list(concrete.columns))

    materials_raw = load_data("material_properties.csv")
    materials = harmonize_materials_schema(materials_raw)
    print("\n[Materials] shape, columns:", materials.shape, list(materials.columns))

    loads_raw = load_data("structural_loads.csv")
    loads = harmonize_loads_schema(loads_raw)
    print("\n[Loads] shape, columns:", loads.shape, list(loads.columns))

    # PART 1
    conc_stats = calculate_descriptive_stats(concrete, "strength_mpa")
    print("\n[Concrete] Descriptive statistics:\n", conc_stats)
    plot_distribution(concrete, "strength_mpa",
                      "Concrete Strength Distribution",
                      LAB_DIR / "concrete_strength_distribution.png")
    plot_box(concrete, "strength_mpa", LAB_DIR / "concrete_strength_boxplot.png")

    # PART 5 (materials comparison)
    plot_material_comparison(materials, "strength_mpa", "material",
                             LAB_DIR / "material_comparison_boxplot.png")

    # PART 2 (distributions)
    plot_probability_distributions(LAB_DIR / "discrete_distributions.png")
    # keep one name as asked
    (LAB_DIR / "probability_distributions.png").write_bytes(
        (LAB_DIR / "discrete_distributions.png").read_bytes()
    )

    # probability tasks
    p_exact3 = calculate_probability_binomial(n=100, p=0.05, k=3)
    p_le5 = calculate_probability_binomial(n=100, p=0.05, k=slice(None, 5))
    p_eq8 = calculate_probability_poisson(lambda_param=10, k=8)
    p_gt15 = 1 - calculate_probability_poisson(lambda_param=10, k=slice(None, 15))
    pct_gt_280 = calculate_probability_normal(250, 15, x_lower=280)
    p95 = norm.ppf(0.95, 250, 15)
    p_fail_lt_500 = calculate_probability_exponential(1000, 500)
    p_survive_gt_1500 = calculate_probability_exponential(1000, 1500, right_tail=True)

    # PART 3 (Bayes)
    bayes_res = apply_bayes_theorem(prior=0.05, sensitivity=0.95, specificity=0.90)
    plot_bayes_tree(0.05, 0.95, 0.90, LAB_DIR / "bayes_tree.png")

    # PART 4 (fit + dashboard)
    fit_res = fit_distribution(concrete, "strength_mpa", "normal")
    plot_distribution_fitting(concrete, "strength_mpa", fit_res,
                              LAB_DIR / "distribution_fitting.png")
    plot_dashboard(concrete, "strength_mpa", LAB_DIR / "statistical_summary_dashboard.png")

    # EXTRA CREDIT
    # A) bootstrap
    bs = ec_bootstrap_ci(concrete["strength_mpa"], B=3000, seed=2025)
    ec_plot_bootstrap(bs["means"], bs["stds"], LAB_DIR / "challenge_bootstrap_ci.png")
    # B) monte carlo
    mc = ec_monte_carlo_reliability(fit_res, loads,
                                    strength_samples=20000, kappa=10.0,
                                    save_path=LAB_DIR / "challenge_mc_reliability.png")
    # C) GMM
    gmm_res = ec_gmm_fit(concrete["strength_mpa"], LAB_DIR / "challenge_gmm_fit.png")
    # D) beta-binomial
    beta_post = {
        "a0": 2, "b0": 8, "successes": 6, "trials": 50
    }
    a_post = beta_post["a0"] + beta_post["successes"]
    b_post = beta_post["b0"] + (beta_post["trials"] - beta_post["successes"])
    x = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, stats.beta.pdf(x, beta_post["a0"], beta_post["b0"]), label="Prior")
    ax.plot(x, stats.beta.pdf(x, a_post, b_post), label="Posterior")
    ax.set_title("Beta–Binomial Bayesian Update")
    ax.legend()
    fig.savefig(LAB_DIR / "challenge_beta_posterior.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    # E) SPC
    spc_res = ec_xbar_r_chart(concrete, "strength_mpa", "batch_id",
                              LAB_DIR / "challenge_xbar_r.png")

    # REPORT
    lines = [
        "Lab 4 — Statistical Report",
        "==========================",
        "",
        "[Concrete] Descriptive Statistics:",
        conc_stats.to_string(),
        "",
        "[Probability Scenarios]",
        f"P(X=3) Binomial(100,0.05)   = {p_exact3:.6f}",
        f"P(X<=5) Binomial(100,0.05)  = {p_le5:.6f}",
        f"P(X=8) Poisson(10)          = {p_eq8:.6f}",
        f"P(X>15) Poisson(10)         = {p_gt15:.6f}",
        f"P(X>280) Normal(250,15)     = {pct_gt_280:.6f}",
        f"95th percentile of N(250,15) = {p95:.3f}",
        f"P(fail<=500) Exp(mean=1000) = {p_fail_lt_500:.6f}",
        f"P(survive>1500) Exp(mean=1000) = {p_survive_gt_1500:.6f}",
        "",
        "[Bayes]",
        f"P(Damage | +) = {bayes_res['posterior_given_positive']:.4f}",
        f"P(+)          = {bayes_res['p_positive']:.4f}",
        "",
        "[Fit]",
        f"Normal fit: mu={fit_res['mean']:.3f}, sigma={fit_res['std']:.3f}",
        "",
        "[Extra Credit] Bootstrap:",
        f"mean_hat={bs['mean_hat']:.3f}, 95% CI=({bs['mean_ci'][0]:.3f}, {bs['mean_ci'][1]:.3f})",
        f"std_hat={bs['std_hat']:.3f}, 95% CI=({bs['std_ci'][0]:.3f}, {bs['std_ci'][1]:.3f})",
        "",
        "[Extra Credit] Monte Carlo Reliability:",
        f"pf = {mc['pf']:.6f}  (kappa={mc['kappa']:.1f})",
        "",
        "[Extra Credit] GMM:",
        f"status = {gmm_res.get('status', 'ok')}",
        f"weights = {gmm_res.get('weights', 'n/a')}",
        f"means   = {gmm_res.get('means', 'n/a')}",
        f"stds    = {gmm_res.get('stds', 'n/a')}",
        "",
        "[Extra Credit] Beta–Binomial:",
        f"posterior Beta({a_post}, {b_post})",
        "",
        "[Extra Credit] SPC Xbar-R:",
        f"n*={spc_res['n_star']}, Xbarbar={spc_res['xbarbar']:.3f}, Rbar={spc_res['rbar']:.3f}",
        f"Xbar UCL={spc_res['UCLx']:.3f}, LCL={spc_res['LCLx']:.3f}",
        f"R    UCL={spc_res['UCLr']:.3f}, LCL={spc_res['LCLr']:.3f}",
    ]
    create_statistical_report(lines, LAB_DIR / "lab4_statistical_report.txt")

    print("\n[OK] Base + Extra Credit figures & report saved under labs/lab4/")
    print("[DONE]")


if __name__ == "__main__":
    main()
