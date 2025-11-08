# =============================================
# Lab 4 — Statistical Analysis Toolkit (Patched)
# =============================================

from __future__ import annotations

# ---------- Imports (single block) ----------
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    bernoulli, binom, poisson, uniform, norm, expon,
    skew as _skew_func, kurtosis as _kurtosis_func
)

# ---------- Paths (single source of truth) ----------
try:
    LAB_DIR = Path(__file__).resolve().parent
except Exception:
    LAB_DIR = Path.cwd()

DATASETS_DIR = (LAB_DIR.parent.parent / "datasets").resolve()

# ---------- File helpers ----------
def _resolve_dataset_path(file_name_or_path: str) -> Path:
    """
    Resolve the absolute path to a dataset file.

    Strategy:
      1) If the given path exists, return it.
      2) Else try conventional locations under datasets/.
      3) Else raise FileNotFoundError.

    Error if file is not found.
    """
    p = Path(file_name_or_path)
    if p.exists():
        return p.resolve()

    candidates = [
        DATASETS_DIR / file_name_or_path,
        Path("datasets") / file_name_or_path,
        Path("../datasets") / file_name_or_path,
        Path("../../datasets") / file_name_or_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        f"{file_name_or_path} not found. Please place it under {DATASETS_DIR} or pass a valid path."
    )

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Accepts either a full path or just a file name (looked up under datasets/).
    """
    csv_path = _resolve_dataset_path(file_path)
    return pd.read_csv(csv_path)

# ---------- Common plotting helper ----------
def _save_or_show(save_path: Optional[str] = None) -> None:
    """
    Save figure with tight bounding box (if path given) else show.
    Close figure after save to prevent many windows from piling up.
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# =========================================================
# Part 1A: Data Loading and Exploration
# =========================================================
def explore_datasets() -> None:
    """
    Load and summarize all datasets used in Lab 4.
    """
    datasets = ["concrete_strength.csv", "material_properties.csv", "structural_loads.csv"]

    for file_name in datasets:
        print(f"\n================= {file_name} =================")
        try:
            df = load_data(file_name)
            print(f"✅ Successfully loaded {file_name}")
            print(f"Shape (rows, columns): {df.shape}")
            print("Columns:", list(df.columns))
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nSummary statistics:")
            print(df.describe().round(2))
        except FileNotFoundError:
            print(f"⚠️ {file_name} not found. This dataset may be optional.")
        except Exception as e:
            print(f"❌ Error while reading {file_name}: {e}")

# =========================================================
# Part 1B: Measures of Central Tendency
# =========================================================
def central_tendency_analysis(save_fig: bool = False) -> None:
    """
    Calculate and visualize mean, median, and mode for the concrete strength dataset.
    """
    try:
        df = load_data("concrete_strength.csv")
        if "strength_mpa" not in df.columns:
            raise KeyError("Column 'strength_mpa' not found in concrete_strength.csv")

        df["strength_mpa"] = pd.to_numeric(df["strength_mpa"], errors="coerce")
        df = df.dropna(subset=["strength_mpa"]).copy()
        if df.empty:
            raise ValueError("No valid numeric rows in 'strength_mpa' after cleaning.")

        mean_val = df["strength_mpa"].mean()
        median_val = df["strength_mpa"].median()
        modes = df["strength_mpa"].mode()
        mode_val = modes.iloc[0] if not modes.empty else float("nan")

        print("\n===== Measures of Central Tendency =====")
        print(f"Mean:   {mean_val:.2f}")
        print(f"Median: {median_val:.2f}")
        print(f"Mode:   {mode_val:.2f}")

        print("\nInterpretation:")
        print("- Mean is sensitive to outliers.")
        print("- Median is robust under skew/outliers.")
        print("- Mode is the most frequent value; useful with discrete or multi-modal data.")

        plt.figure(figsize=(8, 5))
        sns.histplot(df["strength_mpa"], bins=20, kde=True, edgecolor="black")
        plt.axvline(mean_val,   linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
        plt.axvline(median_val, linestyle="-.", linewidth=2, label=f"Median = {median_val:.2f}")
        plt.axvline(mode_val,   linestyle=":",  linewidth=2, label=f"Mode = {mode_val:.2f}")
        plt.title("Concrete Strength Distribution with Mean, Median, and Mode")
        plt.xlabel("Compressive Strength (MPa)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        _save_or_show("figs/concrete_strength_central_tendency.png" if save_fig else None)

    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found. Please place it under the datasets folder.")
    except KeyError as e:
        print(f"⚠️ {e}")
    except Exception as e:
        print("❌ Error during analysis:", e)

# =========================================================
# Part 1C: Measures of Spread
# =========================================================
def measures_of_spread_analysis(save_fig: bool = False) -> Dict[str, float] | None:
    """
    Compute variance, standard deviation, range, and IQR for concrete strength.
    Visualize the distribution with ±1σ, ±2σ, ±3σ bands.
    """
    try:
        df = load_data("concrete_strength.csv")
        if "strength_mpa" not in df.columns:
            raise KeyError("Column 'strength_mpa' not found in concrete_strength.csv")

        df["strength_mpa"] = pd.to_numeric(df["strength_mpa"], errors="coerce")
        x = df["strength_mpa"].dropna()
        if x.empty:
            raise ValueError("No valid numeric rows in 'strength_mpa' after cleaning.")

        var_val   = x.var(ddof=1)
        std_val   = x.std(ddof=1)
        data_min  = x.min()
        data_max  = x.max()
        data_rng  = data_max - data_min
        q1        = x.quantile(0.25)
        q3        = x.quantile(0.75)
        iqr_val   = q3 - q1
        mean_val  = x.mean()

        print("\n===== Measures of Spread =====")
        print(f"Variance (sample):         {var_val:.3f}")
        print(f"Std. Deviation (sample):   {std_val:.3f}")
        print(f"Range (max - min):         {data_rng:.3f}  (min={data_min:.3f}, max={data_max:.3f})")
        print(f"IQR (Q3 - Q1):             {iqr_val:.3f}  (Q1={q1:.3f}, Q3={q3:.3f})")

        print("\nEngineering Interpretation:")
        print("- Higher variance/std ⇒ larger variability; potential QC issues.")
        print("- Range is sensitive to extremes; IQR is robust.")
        print("- High variability may require conservative design/QC improvements.")

        plt.figure(figsize=(9, 5.2))
        sns.histplot(x, bins=20, kde=True, edgecolor="black")
        plt.axvline(mean_val, linestyle="--", linewidth=1.8, label=f"Mean = {mean_val:.2f}")
        for k, alpha in zip((1, 2, 3), (0.15, 0.10, 0.06)):
            left, right = mean_val - k*std_val, mean_val + k*std_val
            plt.axvspan(left, right, alpha=alpha, label=f"±{k}σ")
        plt.title("Concrete Strength – Spread with ±1σ, ±2σ, ±3σ")
        plt.xlabel("Compressive Strength (MPa)")
        plt.ylabel("Frequency / Density")
        plt.legend()
        plt.tight_layout()

        _save_or_show("figs/concrete_strength_spread_sigma_bands.png" if save_fig else None)

        return {
            "variance": float(var_val),
            "std": float(std_val),
            "range": float(data_rng),
            "min": float(data_min),
            "max": float(data_max),
            "Q1": float(q1),
            "Q3": float(q3),
            "IQR": float(iqr_val),
            "mean": float(mean_val),
        }

    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found. Please place it under the datasets folder.")
    except KeyError as e:
        print(f"⚠️ {e}")
    except Exception as e:
        print("❌ Error in measures_of_spread_analysis:", e)

    return None

# =========================================================
# Part 1D: Shape Measures (Skewness & Kurtosis)
# =========================================================
def shape_measures_analysis(save_fig: bool = False) -> Dict[str, float] | None:
    """
    Calculate skewness and kurtosis of the concrete strength distribution,
    print an interpretation, and visualize with histogram + density (KDE).
    """
    try:
        df = load_data("concrete_strength.csv")
        if "strength_mpa" not in df.columns:
            raise KeyError("Column 'strength_mpa' not found in concrete_strength.csv")

        df["strength_mpa"] = pd.to_numeric(df["strength_mpa"], errors="coerce")
        x = df["strength_mpa"].dropna()
        if x.empty:
            raise ValueError("No valid numeric rows in 'strength_mpa' after cleaning.")

        skew_val = stats.skew(x, bias=False)
        kurt_excess = stats.kurtosis(x, fisher=True, bias=False)
        mean_val   = x.mean()
        median_val = x.median()

        print("\n===== Shape Measures =====")
        print(f"Skewness: {skew_val:.3f}")
        print(f"Kurtosis (excess): {kurt_excess:.3f}")

        def interpret_skew(s: float) -> str:
            if abs(s) < 0.1:
                return "approximately symmetric"
            return "right-skewed (tail to the right)" if s > 0 else "left-skewed (tail to the left)"

        def interpret_kurt(k: float) -> str:
            if abs(k) < 0.1:
                return "mesokurtic (similar to normal tails)"
            return "leptokurtic (heavier tails / more peak)" if k > 0 else "platykurtic (lighter tails / flatter peak)"

        print("\nInterpretation:")
        print(f"- Skewness suggests the distribution is {interpret_skew(skew_val)}.")
        print(f"- Excess kurtosis suggests it is {interpret_kurt(kurt_excess)}.")
        print("- In engineering: strong skewness/heavy tails may indicate process drift or curing variability.")

        plt.figure(figsize=(9, 5.2))
        sns.histplot(x, bins=20, kde=True, edgecolor="black")
        plt.axvline(mean_val,   linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
        plt.axvline(median_val, linestyle="-.", linewidth=2, label=f"Median = {median_val:.2f}")
        txt = f"Skewness = {skew_val:.2f}\nExcess Kurtosis = {kurt_excess:.2f}"
        plt.gca().text(0.98, 0.95, txt, transform=plt.gca().transAxes,
                       ha="right", va="top",
                       bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
        plt.title("Concrete Strength – Shape (Skewness & Kurtosis)")
        plt.xlabel("Compressive Strength (MPa)")
        plt.ylabel("Frequency / Density")
        plt.legend()
        plt.tight_layout()

        _save_or_show("figs/concrete_strength_shape_skew_kurt.png" if save_fig else None)

        return {
            "skewness": float(skew_val),
            "excess_kurtosis": float(kurt_excess),
            "mean": float(mean_val),
            "median": float(median_val),
        }

    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found. Please place it under the datasets folder.")
    except KeyError as e:
        print(f"⚠️ {e}")
    except Exception as e:
        print("❌ Error in shape_measures_analysis:", e)

    return None

# =========================================================
# Part 1E: Quantiles and Percentiles (Five-Number Summary)
# =========================================================
def quantiles_and_percentiles_analysis(save_fig: bool = False) -> Dict[str, float] | None:
    """
    Compute quartiles (Q1, median, Q3) and five-number summary for concrete strength.
    Create a boxplot to visualize quartiles and potential outliers.
    """
    try:
        df = load_data("concrete_strength.csv")
        if "strength_mpa" not in df.columns:
            raise KeyError("Column 'strength_mpa' not found in concrete_strength.csv")

        df["strength_mpa"] = pd.to_numeric(df["strength_mpa"], errors="coerce")
        x = df["strength_mpa"].dropna()
        if x.empty:
            raise ValueError("No valid numeric rows in 'strength_mpa' after cleaning.")

        q1  = x.quantile(0.25)
        q2  = x.quantile(0.50)
        q3  = x.quantile(0.75)
        data_min = x.min()
        data_max = x.max()
        iqr = q3 - q1

        print("\n===== Quantiles and Percentiles =====")
        print(f"Q1 (25th): {q1:.3f}")
        print(f"Q2 (Median): {q2:.3f}")
        print(f"Q3 (75th): {q3:.3f}")
        print(f"IQR: {iqr:.3f}")

        print("\nFive-Number Summary:")
        print(f"Min: {data_min:.3f}")
        print(f"Q1 : {q1:.3f}")
        print(f"Q2 : {q2:.3f}")
        print(f"Q3 : {q3:.3f}")
        print(f"Max: {data_max:.3f}")

        plt.figure(figsize=(6, 5))
        sns.boxplot(y=x, width=0.3)
        plt.text(0.05, q1, f"Q1 = {q1:.2f}", va="center", fontsize=9)
        plt.text(0.05, q2, f"Median = {q2:.2f}", va="center", fontsize=9)
        plt.text(0.05, q3, f"Q3 = {q3:.2f}", va="center", fontsize=9)
        plt.title("Concrete Strength – Boxplot (Quartiles & Outliers)")
        plt.ylabel("Compressive Strength (MPa)")
        plt.tight_layout()

        _save_or_show("figs/concrete_strength_boxplot_quartiles.png" if save_fig else None)

        return {
            "min": float(data_min),
            "Q1": float(q1),
            "median": float(q2),
            "Q3": float(q3),
            "max": float(data_max),
            "IQR": float(iqr)
        }

    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found. Please place it under the datasets folder.")
    except KeyError as e:
        print(f"⚠️ {e}")
    except Exception as e:
        print("❌ Error in quantiles_and_percentiles_analysis:", e)

    return None

# =========================================================
# Part 2A: Discrete Distributions (Bernoulli, Binomial, Poisson)
# =========================================================
def _plot_pmf_cdf_discrete(x_vals, pmf_vals, cdf_vals, title_prefix: str, xlabel: str, save_path: Optional[str]) -> None:
    """
    Helper: Plot PMF and CDF side-by-side for a discrete distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x_vals, pmf_vals, edgecolor="black")
    axes[0].set_title(f"{title_prefix} – PMF")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Probability")

    axes[1].step(x_vals, cdf_vals, where="post")
    axes[1].set_title(f"{title_prefix} – CDF")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_ylim(-0.02, 1.02)

    plt.tight_layout()
    _save_or_show(save_path)

def bernoulli_analysis(p: float = 0.95, size: int = 2000, save_fig: bool = False) -> None:
    samples = bernoulli.rvs(p, size=size, random_state=42)
    theo_mean = p
    theo_var  = p * (1 - p)
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Bernoulli(p) =====")
    print(f"Parameter p: {p:.3f}")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    x = np.array([0, 1])
    pmf = bernoulli.pmf(x, p)
    cdf = bernoulli.cdf(x, p)

    _plot_pmf_cdf_discrete(
        x, pmf, cdf,
        title_prefix="Bernoulli(p)",
        xlabel="Outcome (0=Fail, 1=Pass)",
        save_path="figs/discrete_bernoulli_pmf_cdf.png" if save_fig else None
    )

    print("\nScenario (QC – single component):")
    print(f"- Each component passes with probability p={p:.2f}.")
    print("- Use Bernoulli to model a single inspection outcome (pass=1 / fail=0).")

def binomial_analysis(n: int = 100, p: float = 0.05, size: int = 20000, save_fig: bool = False) -> None:
    samples = binom.rvs(n, p, size=size, random_state=42)
    theo_mean = n * p
    theo_var  = n * p * (1 - p)
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Binomial(n, p) =====")
    print(f"Parameters n={n}, p={p:.3f}")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    std = np.sqrt(max(theo_var, 0.0))
    if std == 0:
        k_min = k_max = int(round(theo_mean))
    else:
        k_min = max(0, int(np.floor(theo_mean - 4 * std)))
        k_max = min(n, int(np.ceil(theo_mean + 4 * std)))

    x = np.arange(k_min, k_max + 1)
    pmf = binom.pmf(x, n, p)
    cdf = binom.cdf(x, n, p)

    _plot_pmf_cdf_discrete(
        x, pmf, cdf,
        title_prefix=f"Binomial(n={n}, p={p:.2f})",
        xlabel="k (defect count in batch)",
        save_path="figs/discrete_binomial_pmf_cdf.png" if save_fig else None
    )

    print("\nScenario (QC – batch defectives):")
    print(f"- In a batch of n={n} items with defect probability p={p:.2f},")
    print("  K ~ Binomial(n, p) gives the count of defective items (useful for acceptance sampling).")

def poisson_analysis(lam: float = 10.0, horizon: float = 1.0, size: int = 20000, save_fig: bool = False) -> None:
    lam_eff = lam * horizon
    samples = poisson.rvs(mu=lam_eff, size=size, random_state=42)
    theo_mean = lam_eff
    theo_var  = lam_eff
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Poisson(λ) =====")
    print(f"Rate λ={lam:.2f} per unit time, horizon={horizon:.2f} ⇒ λ_eff={lam_eff:.2f}")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    upper = int(np.ceil(lam_eff + 6 * np.sqrt(lam_eff)))
    x = np.arange(0, max(1, upper) + 1)
    pmf = poisson.pmf(x, mu=lam_eff)
    cdf = poisson.cdf(x, mu=lam_eff)

    _plot_pmf_cdf_discrete(
        x, pmf, cdf,
        title_prefix=f"Poisson(λ_eff={lam_eff:.1f})",
        xlabel="k (event count in interval)",
        save_path="figs/discrete_poisson_pmf_cdf.png" if save_fig else None
    )

    print("\nScenario (arrivals/defects per interval):")
    print(f"- If heavy trucks arrive at rate λ={lam:.1f} per hour,")
    print("  then the number of trucks in horizon T follows Poisson(λ·T).")

def discrete_distributions_demo(save_fig: bool = False) -> None:
    bernoulli_analysis(p=0.95, size=2000, save_fig=save_fig)
    binomial_analysis(n=100, p=0.05, size=20000, save_fig=save_fig)
    poisson_analysis(lam=10.0, horizon=1.0, size=20000, save_fig=save_fig)

# =========================================================
# Part 2B: Continuous Distributions (Uniform, Normal, Exponential)
# =========================================================
def _plot_pdf_cdf_continuous(x_vals, pdf_vals, cdf_vals, title_prefix: str, xlabel: str, save_path: Optional[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x_vals, pdf_vals)
    axes[0].set_title(f"{title_prefix} – PDF")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Density")

    axes[1].plot(x_vals, cdf_vals)
    axes[1].set_title(f"{title_prefix} – CDF")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_ylim(-0.02, 1.02)

    plt.tight_layout()
    _save_or_show(save_path)

def uniform_analysis(a: float = 0.0, b: float = 1.0, size: int = 10000, save_fig: bool = False) -> None:
    if b <= a:
        raise ValueError("Uniform requires b > a")

    samples = uniform.rvs(loc=a, scale=(b - a), size=size, random_state=42)
    theo_mean = 0.5 * (a + b)
    theo_var  = (b - a) ** 2 / 12.0
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Uniform(a, b) =====")
    print(f"Parameters a={a:.3f}, b={b:.3f}")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    x = np.linspace(a - 0.05*(b-a), b + 0.05*(b-a), 600)
    pdf = uniform.pdf(x, loc=a, scale=(b - a))
    cdf = uniform.cdf(x, loc=a, scale=(b - a))

    _plot_pdf_cdf_continuous(
        x, pdf, cdf,
        title_prefix=f"Uniform({a:.2f}, {b:.2f})",
        xlabel="x",
        save_path="figs/continuous_uniform_pdf_cdf.png" if save_fig else None
    )

def normal_analysis(mu: float = 40.0, sigma: float = 5.0, size: int = 10000, save_fig: bool = False) -> None:
    if sigma <= 0:
        raise ValueError("Normal requires sigma > 0")
    samples = norm.rvs(loc=mu, scale=sigma, size=size, random_state=42)
    theo_mean = mu
    theo_var  = sigma ** 2
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Normal(μ, σ) =====")
    print(f"Parameters μ={mu:.3f}, σ={sigma:.3f}")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 800)
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    cdf = norm.cdf(x, loc=mu, scale=sigma)

    _plot_pdf_cdf_continuous(
        x, pdf, cdf,
        title_prefix=f"Normal(μ={mu:.1f}, σ={sigma:.1f})",
        xlabel="x",
        save_path="figs/continuous_normal_pdf_cdf.png" if save_fig else None
    )

def exponential_analysis(lmbda: float = 0.2, size: int = 10000, save_fig: bool = False) -> None:
    if lmbda <= 0:
        raise ValueError("Exponential requires λ > 0")
    scale = 1.0 / lmbda
    samples = expon.rvs(scale=scale, size=size, random_state=42)
    theo_mean = 1.0 / lmbda
    theo_var  = 1.0 / (lmbda ** 2)
    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    print("\n===== Exponential(λ) =====")
    print(f"Parameter λ={lmbda:.3f} (scale={scale:.3f})")
    print(f"Theoretical mean: {theo_mean:.4f}, var: {theo_var:.4f}")
    print(f"Empirical   mean: {emp_mean:.4f}, var: {emp_var:.4f}")

    x = np.linspace(0, expon.ppf(0.999, scale=scale), 800)
    pdf = expon.pdf(x, scale=scale)
    cdf = expon.cdf(x, scale=scale)

    _plot_pdf_cdf_continuous(
        x, pdf, cdf,
        title_prefix=f"Exponential(λ={lmbda:.2f})",
        xlabel="x (time)",
        save_path="figs/continuous_exponential_pdf_cdf.png" if save_fig else None
    )

def continuous_distributions_demo(save_fig: bool = False) -> None:
    uniform_analysis(a=0.0, b=1.0, size=10000, save_fig=save_fig)
    normal_analysis(mu=40.0, sigma=5.0, size=10000, save_fig=save_fig)
    exponential_analysis(lmbda=0.2, size=10000, save_fig=save_fig)

# =========================================================
# Part 2C: Distribution Fitting (Normal on concrete strength)
# =========================================================
def normal_fit_analysis(bins: int = 25, save_fig: bool = False) -> Dict[str, float] | None:
    """
    Fit a Normal distribution to 'strength_mpa' and overlay the fitted PDF on the histogram.
    """
    try:
        df = load_data("concrete_strength.csv")
        if "strength_mpa" not in df.columns:
            raise KeyError("Column 'strength_mpa' not found in concrete_strength.csv")

        x = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna()
        if x.empty:
            raise ValueError("No valid numeric rows in 'strength_mpa' after cleaning.")

        sample_mean = x.mean()
        sample_std  = x.std(ddof=1)
        mu_hat, sigma_hat = norm.fit(x)

        print("\n===== Normal Distribution Fitting (Part 2C) =====")
        print(f"Sample mean (ddof=1): {sample_mean:.4f}")
        print(f"Sample std  (ddof=1): {sample_std:.4f}")
        print(f"Fitted mu_hat (MLE):  {mu_hat:.4f}")
        print(f"Fitted sigma_hat:     {sigma_hat:.4f}")

        plt.figure(figsize=(9, 5.2))
        sns.histplot(x, bins=bins, stat="density", kde=False, color="lightgray", edgecolor="black", label="Data (hist)")

        x_min, x_max = float(np.min(x)), float(np.max(x))
        span = (x_max - x_min)
        safe_span = span if span > 0 else 1.0
        grid = np.linspace(x_min - 0.05*safe_span, x_max + 0.05*safe_span, 800)

        fitted_pdf = norm.pdf(grid, loc=mu_hat, scale=sigma_hat)
        plt.plot(grid, fitted_pdf, linewidth=2.2, label=f"Fitted Normal PDF (μ̂={mu_hat:.2f}, σ̂={sigma_hat:.2f})")

        sample_pdf = norm.pdf(grid, loc=sample_mean, scale=sample_std)
        plt.plot(grid, sample_pdf, linestyle="--", linewidth=1.8, label=f"Sample Normal PDF (mean={sample_mean:.2f}, sd={sample_std:.2f})")

        plt.axvline(mu_hat,      linestyle=":",  linewidth=1.8, label="μ̂ (fitted)")
        plt.axvline(sample_mean, linestyle="-.", linewidth=1.8, label="mean (sample)")

        plt.title("Concrete Strength – Normal Fit over Histogram (Part 2C)")
        plt.xlabel("Compressive Strength (MPa)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        _save_or_show("figs/concrete_strength_normal_fit.png" if save_fig else None)

        return {
            "sample_mean_ddof1": float(sample_mean),
            "sample_std_ddof1":  float(sample_std),
            "mu_hat_mle":        float(mu_hat),
            "sigma_hat_mle":     float(sigma_hat),
            "n":                 int(x.size),
            "bins":              int(bins),
        }

    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found. Please place it under the datasets folder.")
    except KeyError as e:
        print(f"⚠️ {e}")
    except Exception as e:
        print("❌ Error in normal_fit_analysis:", e)

    return None

# =========================================================
# Part 3A: Conditional Probability (Engineering QC example)
# =========================================================
def _draw_probability_tree(
    p_defect: float,
    p_detect_given_defect: float,
    p_detect_given_no_defect: float,
    p_fail_given_defect: float,
    p_fail_given_no_defect: float,
    highlight=("Detected", "Failure"),
    save_path: Optional[str] = None
) -> None:
    x0, x1, x2, x3 = 0.0, 1.8, 3.6, 5.4
    y_root = 0.0
    y_def, y_nodef = 1.0, -1.0
    y_D, y_nD = 1.6, 0.4
    y_D2, y_nD2 = -0.4, -1.6
    y_F_def, y_nF_def = 2.2, 0.8
    y_F_nodef, y_nF_nodef = -0.8, -2.2

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    def edge(p1, p2, label, lw=1.6, color="black"):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, color=color)
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my, label, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9))

    ax.scatter([x0], [y_root], s=80)
    ax.text(x0 - 0.1, y_root + 0.15, "Start", fontsize=10)

    edge((x0, y_root), (x1, y_def),      f"P(Defect)={p_defect:.2f}")
    edge((x0, y_root), (x1, y_nodef),    f"P(NoDefect)={1-p_defect:.2f}")
    ax.scatter([x1, x1], [y_def, y_nodef], s=80)
    ax.text(x1 + 0.1, y_def + 0.1, "Defect", fontsize=10)
    ax.text(x1 + 0.1, y_nodef - 0.2, "NoDefect", fontsize=10)

    edge((x1, y_def),   (x2, y_D),   f"P(D|Def)={p_detect_given_defect:.2f}",
         color=("tab:blue" if "Detected" in highlight else "black"), lw=(2.2 if "Detected" in highlight else 1.6))
    edge((x1, y_def),   (x2, y_nD),  f"P(~D|Def)={1-p_detect_given_defect:.2f}")
    edge((x1, y_nodef), (x2, y_D2),  f"P(D|~Def)={p_detect_given_no_defect:.2f}",
         color=("tab:blue" if "Detected" in highlight else "black"), lw=(2.2 if "Detected" in highlight else 1.6))
    edge((x1, y_nodef), (x2, y_nD2), f"P(~D|~Def)={1-p_detect_given_no_defect:.2f}")
    ax.scatter([x2]*4, [y_D, y_nD, y_D2, y_nD2], s=60)

    ax.text(x2 + 0.1, y_D + 0.1,  "Detected", fontsize=9)
    ax.text(x2 + 0.1, y_nD - 0.2, "NotDetected", fontsize=9)
    ax.text(x2 + 0.1, y_D2 + 0.1, "Detected", fontsize=9)
    ax.text(x2 + 0.1, y_nD2 - 0.2,"NotDetected", fontsize=9)

    edge((x1, y_def), (x3, y_F_def),   f"P(F|Def)={p_fail_given_defect:.2f}",
         color=("tab:red" if "Failure" in highlight else "black"), lw=(2.2 if "Failure" in highlight else 1.6))
    edge((x1, y_def), (x3, y_nF_def),  f"P(~F|Def)={1-p_fail_given_defect:.2f}")
    edge((x1, y_nodef), (x3, y_F_nodef),   f"P(F|~Def)={p_fail_given_no_defect:.2f}",
         color=("tab:red" if "Failure" in highlight else "black"), lw=(2.2 if "Failure" in highlight else 1.6))
    edge((x1, y_nodef), (x3, y_nF_nodef),  f"P(~F|~Def)={1-p_fail_given_no_defect:.2f}")

    ax.scatter([x3]*4, [y_F_def, y_nF_def, y_F_nodef, y_nF_nodef], s=50)
    ax.text(x3 + 0.1, y_F_def + 0.1,     "Failure", fontsize=9)
    ax.text(x3 + 0.1, y_nF_def - 0.2,    "NoFailure", fontsize=9)
    ax.text(x3 + 0.1, y_F_nodef + 0.1,   "Failure", fontsize=9)
    ax.text(x3 + 0.1, y_nF_nodef - 0.2,  "NoFailure", fontsize=9)

    ax.set_axis_off()
    ax.set_title("Probability Tree – Defect, Detection, and Failure", fontsize=12)
    plt.tight_layout()
    _save_or_show(save_path)

def conditional_failure_given_detected(
    p_defect: float = 0.08,
    p_detect_given_defect: float = 0.92,
    p_detect_given_no_defect: float = 0.03,
    p_fail_given_defect: float = 0.30,
    p_fail_given_no_defect: float = 0.01,
    save_fig: bool = False
) -> Dict[str, float]:
    for name, p in [
        ("p_defect", p_defect),
        ("p_detect_given_defect", p_detect_given_defect),
        ("p_detect_given_no_defect", p_detect_given_no_defect),
        ("p_fail_given_defect", p_fail_given_defect),
        ("p_fail_given_no_defect", p_fail_given_no_defect),
    ]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {p}")

    p_nd = 1 - p_defect
    p_S_D   = p_defect * p_detect_given_defect
    p_Sbar_D  = p_nd    * p_detect_given_no_defect
    p_D = p_S_D + p_Sbar_D

    p_F_and_D = (
        p_defect * p_detect_given_defect     * p_fail_given_defect +
        p_nd     * p_detect_given_no_defect  * p_fail_given_no_defect
    )
    p_F_given_D = p_F_and_D / p_D if p_D > 0 else np.nan
    p_F = p_defect * p_fail_given_defect + p_nd * p_fail_given_no_defect

    print("\n===== Conditional Probability (Part 3A) =====")
    print(f"P(Detected)             = {p_D:.4f}")
    print(f"P(Failure)              = {p_F:.4f}")
    print(f"P(Failure ∧ Detected)   = {p_F_and_D:.4f}")
    print(f"⇒ P(Failure | Detected) = {p_F_given_D:.4f}")

    _draw_probability_tree(
        p_defect, p_detect_given_defect, p_detect_given_no_defect,
        p_fail_given_defect, p_fail_given_no_defect,
        highlight=("Detected", "Failure"),
        save_path="figs/conditional_probability_tree.png" if save_fig else None
    )

    return {
        "P_Detected": float(p_D),
        "P_Failure": float(p_F),
        "P_F_and_D": float(p_F_and_D),
        "P_F_given_D": float(p_F_given_D),
    }

# =========================================================
# Part 3B: Bayes' Theorem (Structural damage diagnostic)
# =========================================================
def bayes_diagnostic_posterior(
    prevalence: float = 0.04,
    sensitivity: float = 0.90,
    specificity: float = 0.96,
    population: int = 10000,
    save_fig: bool = False
) -> Dict[str, Any]:
    for name, p in [("prevalence", prevalence), ("sensitivity", sensitivity), ("specificity", specificity)]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {p}")

    prior = prevalence
    fp_rate = 1.0 - specificity
    numerator   = sensitivity * prior
    denominator = sensitivity * prior + fp_rate * (1 - prior)
    posterior   = numerator / denominator if denominator > 0 else np.nan
    posterior_no_damage_given_pos = 1 - posterior

    denom_neg = specificity * (1 - prior) + (1 - sensitivity) * prior
    npv = (specificity * (1 - prior)) / denom_neg if denom_neg > 0 else np.nan
    ppv = posterior

    n_damage     = population * prior
    n_nodamage   = population - n_damage
    n_true_pos   = n_damage   * sensitivity
    n_false_pos  = n_nodamage * fp_rate
    n_true_neg   = n_nodamage * specificity
    n_false_neg  = n_damage   * (1 - sensitivity)
    n_positive   = n_true_pos + n_false_pos
    n_negative   = n_true_neg + n_false_neg

    print("\n===== Bayes' Theorem (Part 3B) =====")
    print(f"Posterior (PPV)     P(Damage | +)  = {ppv:.4f}")
    print(f"P(NoDamage | +)                     = {posterior_no_damage_given_pos:.4f}")
    print(f"NPV                 P(NoDamage | -) = {npv:.4f}")

    # Bar: prior vs posterior
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar(["Prior P(Damage)", "Posterior P(Damage | +)"], [prior, posterior], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Bayes Update: Prior vs Posterior")
    for i, v in enumerate([prior, posterior]):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center")
    plt.tight_layout()
    _save_or_show("figs/bayes_prior_vs_posterior.png" if save_fig else None)

    # Stacked bar: positives composition
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(["Positives"], [n_true_pos], label="True Positive (Damage & +)", edgecolor="black")
    ax.bar(["Positives"], [n_false_pos], bottom=[n_true_pos], label="False Positive (NoDamage & +)", edgecolor="black")
    ax.set_ylabel("Count in cohort")
    ax.set_title("Positive Tests: True vs False Origin")
    ax.legend()
    ax.text(0, n_positive + population*0.02, f"Total + = {n_positive:.0f}", ha="center")
    ax.text(0, n_true_pos/2, f"TP ≈ {n_true_pos:.0f}", ha="center")
    ax.text(0, n_true_pos + (n_false_pos/2), f"FP ≈ {n_false_pos:.0f}", ha="center")
    plt.tight_layout()
    _save_or_show("figs/bayes_positive_composition.png" if save_fig else None)

    return {
        "prior": float(prior),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "posterior_ppv": float(ppv),
        "npv": float(npv),
        "counts": {
            "true_positive": float(n_true_pos),
            "false_positive": float(n_false_pos),
            "true_negative": float(n_true_neg),
            "false_negative": float(n_false_neg),
            "total_positive": float(n_positive),
            "total_negative": float(n_negative),
            "population": int(population),
        }
    }

# =========================================================
# Part 3C: Basic Comparison (group-wise)
# =========================================================
def _pick_categorical_column(
    df: pd.DataFrame,
    preferred: Optional[List[str]] = None,
    max_unique: int = 12
) -> Optional[str]:
    if preferred:
        for c in preferred:
            if c in df.columns and 2 <= df[c].nunique(dropna=True) <= max_unique:
                return c

    # non-numeric candidates
    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in non_num:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= max_unique:
            return c

    # low-cardinality numeric-as-category (e.g., batch_id)
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[c]) and 2 <= nun <= max_unique:
            return c

    return None

def basic_group_comparison(
    value_col: str = "strength_mpa",
    group_col: Optional[str] = None,
    preferred_groups: Optional[List[str]] = None,
    save_fig: bool = False
) -> pd.DataFrame:
    """
    Compare distribution of 'value_col' across groups in 'group_col'.
    - Prints descriptive stats per group
    - Draws boxplot, violin, KDE and ECDF overlays.
    """
    df = load_data("concrete_strength.csv")

    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in concrete_strength.csv")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).copy()
    if df.empty:
        raise ValueError(f"No valid numeric rows in '{value_col}' after cleaning.")

    if group_col is None:
        group_col = _pick_categorical_column(
            df, preferred=preferred_groups or ["material_type", "mix_type", "mix_id", "batch", "supplier"]
        )
    if group_col is None or group_col not in df.columns:
        raise KeyError(
            "No suitable grouping column found automatically. "
            "Pass a valid 'group_col' (e.g., 'material_type', 'mix_type', 'batch')."
        )

    grp_df = df.dropna(subset=[group_col]).copy()
    nun = grp_df[group_col].nunique()
    if nun < 2:
        raise ValueError(f"Grouping column '{group_col}' has fewer than 2 distinct values after cleaning.")
    if nun > 12:
        raise ValueError(f"Too many groups ({nun}). Please reduce to ≤12 for readability.")

    # cast numeric low-cardinality group labels to string for categorical plotting
    if pd.api.types.is_numeric_dtype(grp_df[group_col]):
        grp_df[group_col] = grp_df[group_col].astype(str)

    def iqr(s: pd.Series) -> float:
        return float(s.quantile(0.75) - s.quantile(0.25))

    summary = (
        grp_df.groupby(group_col)[value_col]
        .agg(count="count", mean="mean", std="std", median="median",
             Q1=lambda s: s.quantile(0.25), Q3=lambda s: s.quantile(0.75), IQR=iqr, min="min", max="max")
        .sort_values("mean")
        .round(3)
    )

    print("\n===== Basic Comparison (Part 3C) =====")
    print(f"Value column     : {value_col}")
    print(f"Grouping column  : {group_col}")
    print("\nDescriptive statistics by group:")
    print(summary)

    # Boxplot
    plt.figure(figsize=(9.5, 5.2))
    sns.boxplot(data=grp_df, x=group_col, y=value_col)
    plt.title(f"{value_col} by {group_col} – Boxplot")
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save_or_show(f"figs/basic_cmp_boxplot_{value_col}_by_{group_col}.png" if save_fig else None)

    # Violin
    plt.figure(figsize=(9.5, 5.2))
    sns.violinplot(data=grp_df, x=group_col, y=value_col, inner="quartile", cut=0)
    plt.title(f"{value_col} by {group_col} – Violin (with quartiles)")
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save_or_show(f"figs/basic_cmp_violin_{value_col}_by_{group_col}.png" if save_fig else None)

    # KDE overlays (common_norm=False as requested)
    plt.figure(figsize=(9.5, 5.2))
    for g, sub in grp_df.groupby(group_col):
        sns.kdeplot(sub[value_col], label=str(g), linewidth=1.8, common_norm=False)
    plt.title(f"{value_col} – KDE overlays by {group_col}")
    plt.xlabel(value_col)
    plt.ylabel("Density")
    plt.legend(title=group_col)
    plt.tight_layout()
    _save_or_show(f"figs/basic_cmp_kde_{value_col}_by_{group_col}.png" if save_fig else None)

    # ECDF overlays (skip if statsmodels missing)
    try:
        from statsmodels.distributions.empirical_distribution import ECDF
        plt.figure(figsize=(9.5, 5.2))
        for g, sub in grp_df.groupby(group_col):
            ecdf = ECDF(sub[value_col].to_numpy())
            xs = np.linspace(sub[value_col].min(), sub[value_col].max(), 500)
            ys = ecdf(xs)
            plt.step(xs, ys, where="post", label=str(g))
        plt.title(f"{value_col} – ECDF overlays by {group_col}")
        plt.xlabel(value_col)
        plt.ylabel("Cumulative probability")
        plt.legend(title=group_col)
        plt.tight_layout()
        _save_or_show(f"figs/basic_cmp_ecdf_{value_col}_by_{group_col}.png" if save_fig else None)
    except Exception:
        pass

    return summary

# =========================================================
# Part 4 & 5: Utilities, Fits, Probabilities, Visuals, Report
# =========================================================
def calculate_descriptive_stats(data: pd.DataFrame, column: str = "strength_mpa") -> pd.Series:
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not in DataFrame.")
    x = pd.to_numeric(data[column], errors="coerce").dropna()
    if x.empty:
        raise ValueError(f"No valid numeric values in '{column}'.")
    desc = x.describe(percentiles=[0.25, 0.5, 0.75])
    q1, q2, q3 = desc["25%"], desc["50%"], desc["75%"]
    iqr = q3 - q1
    skew_val = float(_skew_func(x, bias=False))
    kurt_excess = float(_kurtosis_func(x, fisher=True, bias=False))
    return pd.Series({
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]),
        "min": float(desc["min"]),
        "Q1": float(q1),
        "median": float(q2),
        "Q3": float(q3),
        "max": float(desc["max"]),
        "IQR": float(iqr),
        "skew": skew_val,
        "excess_kurtosis": kurt_excess
    })

def plot_distribution(
    data: pd.DataFrame, column: str, title: str, save_path: Optional[str] = None
) -> None:
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not found.")
    x = pd.to_numeric(data[column], errors="coerce").dropna()
    if x.empty:
        raise ValueError(f"No valid numeric values in '{column}'.")
    mu_hat, sigma_hat = norm.fit(x)
    # Güvenli grid aralığı (Series.ptp yerine)
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    safe_span = float(span if span > 0 else 1.0)

    plt.figure(figsize=(8.6, 5.0))
    sns.histplot(x, bins=25, stat="density", color="lightgray", edgecolor="black", label="Data (hist)")
    grid = np.linspace(xmin - 0.05*safe_span, xmax + 0.05*safe_span, 800)
    plt.plot(grid, norm.pdf(grid, mu_hat, sigma_hat), linewidth=2.0,
             label=f"Normal fit (μ̂={mu_hat:.2f}, σ̂={sigma_hat:.2f})")
    plt.axvline(x.mean(), linestyle="--", linewidth=1.6, label=f"Mean={x.mean():.2f}")
    plt.axvline(x.median(), linestyle="-.", linewidth=1.6, label=f"Median={x.median():.2f}")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)

def fit_distribution(
    data: pd.DataFrame, column: str, distribution_type: str = "normal"
) -> Dict[str, float]:
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not in DataFrame.")
    x = pd.to_numeric(data[column], errors="coerce").dropna().to_numpy()
    if x.size == 0:
        raise ValueError(f"No valid numeric values in '{column}'.")
    d = distribution_type.lower().strip()
    if d == "normal":
        mu, sigma = norm.fit(x)
        return {"type": "normal", "mu": float(mu), "sigma": float(sigma)}
    elif d == "exponential":
        loc, scale = expon.fit(x, floc=0)
        return {"type": "exponential", "loc": float(loc), "scale": float(scale), "lambda": float(1.0/scale if scale > 0 else np.nan)}
    elif d == "uniform":
        loc, scale = uniform.fit(x)
        a, b = loc, loc + scale
        return {"type": "uniform", "a": float(a), "b": float(b)}
    else:
        raise ValueError("distribution_type must be 'normal', 'exponential', or 'uniform'.")

def calculate_probability_binomial(n: int, p: float, k: int) -> float:
    if n < 0 or not (0 <= p <= 1) or k < 0:
        raise ValueError("Invalid n, p, or k.")
    return float(binom.pmf(k, n, p))

def calculate_probability_normal(
    mean: float, std: float, x_lower: Optional[float] = None, x_upper: Optional[float] = None
) -> float:
    if std <= 0:
        raise ValueError("std must be > 0.")
    if x_lower is None and x_upper is None:
        raise ValueError("Provide at least x_lower or x_upper.")
    dist = norm(loc=mean, scale=std)
    if x_lower is None:
        return float(dist.cdf(x_upper))
    if x_upper is None:
        return float(1.0 - dist.cdf(x_lower))
    if x_upper < x_lower:
        raise ValueError("x_upper must be >= x_lower.")
    return float(dist.cdf(x_upper) - dist.cdf(x_lower))

def calculate_probability_poisson(lambda_param: float, k: int) -> float:
    if lambda_param < 0 or k < 0:
        raise ValueError("lambda_param and k must be non-negative.")
    return float(poisson.pmf(k, mu=lambda_param))

def calculate_probability_exponential(mean: float, x: float) -> Dict[str, float]:
    if mean <= 0:
        raise ValueError("mean must be > 0.")
    if x < 0:
        raise ValueError("x must be >= 0.")
    scale = mean
    return {
        "cdf": float(expon.cdf(x, scale=scale)),
        "sf":  float(expon.sf(x, scale=scale)),
        "pdf": float(expon.pdf(x, scale=scale))
    }

def apply_bayes_theorem(prior: float, sensitivity: float, specificity: float) -> Dict[str, float]:
    for name, p in [("prior", prior), ("sensitivity", sensitivity), ("specificity", specificity)]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be in [0,1].")
    fp_rate = 1 - specificity
    num = sensitivity * prior
    den = sensitivity * prior + fp_rate * (1 - prior)
    ppv = float(num / den) if den > 0 else np.nan
    den_neg = specificity * (1 - prior) + (1 - sensitivity) * prior
    npv = float(specificity * (1 - prior) / den_neg) if den_neg > 0 else np.nan
    return {"PPV": ppv, "NPV": npv, "false_positive_rate": float(fp_rate)}

def plot_material_comparison(
    data: pd.DataFrame, column: str, group_column: str, save_path: Optional[str] = None
) -> None:
    if column not in data.columns or group_column not in data.columns:
        raise KeyError("column or group_column not in DataFrame.")
    df = data.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column, group_column])
    nun = df[group_column].nunique(dropna=True)
    if nun < 2:
        raise ValueError(f"Grouping column '{group_column}' has fewer than 2 distinct values.")
    if nun > 12:
        raise ValueError(f"Too many groups ({nun}). Please reduce to ≤12 for readability.")
    if pd.api.types.is_numeric_dtype(df[group_column]):
        df[group_column] = df[group_column].astype(str)
    plt.figure(figsize=(10, 5.2))
    sns.boxplot(data=df, x=group_column, y=column)
    plt.title(f"{column} by {group_column}")
    plt.xlabel(group_column)
    plt.ylabel(column)
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save_or_show(save_path)

def plot_distribution_fitting(
    data: pd.DataFrame, column: str, fitted_dist: Optional[Dict[str, float]] = None, save_path: Optional[str] = None
) -> Dict[str, float]:
    if fitted_dist is None:
        fitted_dist = fit_distribution(data, column, distribution_type="normal")
    x = pd.to_numeric(data[column], errors="coerce").dropna()
    if x.empty:
        raise ValueError(f"No valid numeric values in '{column}'.")
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    safe_span = float(span if span > 0 else 1.0)

    plt.figure(figsize=(8.8, 5.0))
    sns.histplot(x, bins=25, stat="density", color="lightgray", edgecolor="black", label="Data (hist)")
    grid = np.linspace(xmin - 0.05*safe_span, xmax + 0.05*safe_span, 800)
    d_type = fitted_dist.get("type", "normal").lower()
    if d_type == "normal":
        mu, sigma = fitted_dist["mu"], fitted_dist["sigma"]
        plt.plot(grid, norm.pdf(grid, mu, sigma), linewidth=2.0, label=f"Normal fit (μ̂={mu:.2f}, σ̂={sigma:.2f})")
    elif d_type == "exponential":
        scale = fitted_dist["scale"]
        plt.plot(grid, expon.pdf(grid, scale=scale), linewidth=2.0, label=f"Exponential fit (mean={scale:.2f})")
    elif d_type == "uniform":
        a, b = fitted_dist["a"], fitted_dist["b"]
        plt.plot(grid, uniform.pdf(grid, loc=a, scale=(b - a)), linewidth=2.0, label=f"Uniform fit (a={a:.2f}, b={b:.2f})")
    else:
        raise ValueError("Unsupported fitted distribution type.")
    plt.title(f"{column} – Distribution & Fitted Curve")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)
    return fitted_dist

def plot_pdf_cdf_comparisons(
    save_path: Optional[str] = None,
    normal_params: Tuple[float, float] = (40.0, 5.0),
    expon_mean: float = 5.0,
    uniform_bounds: Tuple[float, float] = (0.0, 1.0)
) -> None:
    mu, sigma = normal_params
    a, b = uniform_bounds
    if sigma <= 0 or not (b > a) or expon_mean <= 0:
        raise ValueError("Invalid parameters for comparisons.")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6))
    x_n = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
    axes[0, 0].plot(x_n, norm.pdf(x_n, mu, sigma)); axes[0, 0].set_title("Normal PDF")
    axes[1, 0].plot(x_n, norm.cdf(x_n, mu, sigma)); axes[1, 0].set_title("Normal CDF")

    scale = expon_mean
    x_e = np.linspace(0, expon.ppf(0.999, scale=scale), 600)
    axes[0, 1].plot(x_e, expon.pdf(x_e, scale=scale)); axes[0, 1].set_title("Exponential PDF")
    axes[1, 1].plot(x_e, expon.cdf(x_e, scale=scale)); axes[1, 1].set_title("Exponential CDF")

    x_u = np.linspace(a - 0.05*(b - a), b + 0.05*(b - a), 600)
    axes[0, 2].plot(x_u, uniform.pdf(x_u, loc=a, scale=(b - a))); axes[0, 2].set_title("Uniform PDF")
    axes[1, 2].plot(x_u, uniform.cdf(x_u, loc=a, scale=(b - a))); axes[1, 2].set_title("Uniform CDF")

    for ax in axes.flat:
        ax.set_xlabel("x"); ax.set_ylabel("density / probability")
    plt.tight_layout()
    _save_or_show(save_path)

def build_stat_dashboard(
    data: pd.DataFrame, column: str = "strength_mpa", save_path: Optional[str] = None
) -> pd.Series:
    stats_s = calculate_descriptive_stats(data, column)
    x = pd.to_numeric(data[column], errors="coerce").dropna()
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    safe_span = float(span if span > 0 else 1.0)

    fig = plt.figure(figsize=(12, 6.5))

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    mu_hat, sigma_hat = norm.fit(x)
    sns.histplot(x, bins=25, stat="density", color="lightgray", edgecolor="black", ax=ax1)
    grid = np.linspace(xmin - 0.05*safe_span, xmax + 0.05*safe_span, 800)
    ax1.plot(grid, norm.pdf(grid, mu_hat, sigma_hat), linewidth=2.0, label="Normal fit")
    ax1.set_title("Histogram + Normal fit"); ax1.legend()

    ax2 = plt.subplot2grid((2, 3), (1, 0))
    sns.boxplot(y=x, ax=ax2, color="lightblue", width=0.35)
    ax2.set_title("Boxplot")

    ax3 = plt.subplot2grid((2, 3), (1, 1))
    sns.kdeplot(x, ax=ax3, linewidth=2.0)
    ax3.set_title("KDE")

    ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax4.axis("off")
    txt = (
        f"count = {stats_s['count']}\n"
        f"mean = {stats_s['mean']:.2f}\n"
        f"std = {stats_s['std']:.2f}\n"
        f"min/Q1/median/Q3/max\n"
        f"{stats_s['min']:.2f} / {stats_s['Q1']:.2f} / {stats_s['median']:.2f} / "
        f"{stats_s['Q3']:.2f} / {stats_s['max']:.2f}\n"
        f"IQR = {stats_s['IQR']:.2f}\n"
        f"skew = {stats_s['skew']:.2f}\n"
        f"excess kurtosis = {stats_s['excess_kurtosis']:.2f}"
    )
    ax4.text(0.0, 0.9, "Statistical Summary", fontsize=12, weight="bold")
    ax4.text(0.0, 0.8, txt, fontsize=10, family="monospace")

    plt.tight_layout()
    _save_or_show(save_path)
    return stats_s

def plot_probability_tree(
    prior: float, sensitivity: float, specificity: float, save_path: Optional[str] = None
) -> None:
    fp = 1 - specificity
    x0, x1, x2 = 0.0, 2.0, 4.0
    y_root = 0.0
    y_D, y_nD = 1.0, -1.0
    y_pos_D, y_neg_D = 1.7, 0.3
    y_pos_nD, y_neg_nD = -0.3, -1.7

    fig, ax = plt.subplots(figsize=(10, 5.5))

    def edge(p1, p2, label, lw=1.7, color="black"):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, color=color)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9), fontsize=9)

    ax.scatter([x0], [y_root], s=80)
    ax.text(x0 - 0.1, y_root + 0.15, "Start", fontsize=10)

    edge((x0, y_root), (x1, y_D),   f"P(Damage)={prior:.2f}")
    edge((x0, y_root), (x1, y_nD), f"P(NoDamage)={1-prior:.2f}")
    ax.scatter([x1, x1], [y_D, y_nD], s=70)
    ax.text(x1 + 0.1, y_D + 0.05, "Damage", fontsize=10)
    ax.text(x1 + 0.1, y_nD - 0.2, "NoDamage", fontsize=10)

    edge((x1, y_D),   (x2, y_pos_D), f"P(+|D)={sensitivity:.2f}", color="tab:blue", lw=2.2)
    edge((x1, y_D),   (x2, y_neg_D), f"P(-|D)={1-sensitivity:.2f}")
    edge((x1, y_nD),  (x2, y_pos_nD), f"P(+|~D)={fp:.2f}", color="tab:blue", lw=2.2)
    edge((x1, y_nD),  (x2, y_neg_nD), f"P(-|~D)={specificity:.2f}")

    ax.scatter([x2]*4, [y_pos_D, y_neg_D, y_pos_nD, y_neg_nD], s=60)
    ax.text(x2 + 0.1, y_pos_D + 0.05,  "Positive", fontsize=9)
    ax.text(x2 + 0.1, y_neg_D - 0.2,   "Negative", fontsize=9)
    ax.text(x2 + 0.1, y_pos_nD + 0.05, "Positive", fontsize=9)
    ax.text(x2 + 0.1, y_neg_nD - 0.2,  "Negative", fontsize=9)

    ax.set_axis_off()
    ax.set_title("Probability Tree – Bayes Test Outcome", fontsize=12)
    plt.tight_layout()
    _save_or_show(save_path)

def create_statistical_report(
    data: pd.DataFrame,
    output_file: str = "lab4_statistical_report.txt",
    value_column: str = "strength_mpa",
    group_column: Optional[str] = None
) -> str:
    stats_s = calculate_descriptive_stats(data, value_column)

    fit_norm = fit_distribution(data, value_column, "normal")
    try:
        fit_exp = fit_distribution(data, value_column, "exponential")
    except Exception:
        fit_exp = {"type": "exponential", "error": "fit failed"}
    try:
        fit_uni = fit_distribution(data, value_column, "uniform")
    except Exception:
        fit_uni = {"type": "uniform", "error": "fit failed"}

    group_summary = None
    if group_column and group_column in data.columns:
        df = data.copy()
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna(subset=[value_column, group_column])
        if df[group_column].nunique() >= 2 and df[value_column].notna().sum() > 0:
            group_summary = (
                df.groupby(group_column)[value_column]
                .agg(count="count", mean="mean", std="std", median="median")
                .round(3)
            )

    lines = []
    lines.append("==== Lab 4 – Statistical Report ====\n")
    lines.append(f"Target column: {value_column}\n")
    lines.append("Descriptive statistics:\n")
    for k, v in stats_s.items():
        lines.append(f"  {k:>16}: {v}\n")

    lines.append("\nFitted distributions:\n")
    lines.append(f"  Normal:      mu={fit_norm.get('mu'):.4f}, sigma={fit_norm.get('sigma'):.4f}\n")
    if "error" in fit_exp:
        lines.append("  Exponential: fit failed\n")
    else:
        lines.append(f"  Exponential: mean={fit_exp.get('scale'):.4f} (lambda={fit_exp.get('lambda'):.4f})\n")
    if "error" in fit_uni:
        lines.append("  Uniform:     fit failed\n")
    else:
        lines.append(f"  Uniform:     a={fit_uni.get('a'):.4f}, b={fit_uni.get('b'):.4f}\n")

    if group_summary is not None:
        lines.append("\nGroup comparison (basic):\n")
        lines.append(str(group_summary))
        lines.append("\n")

    lines.append("Key findings & interpretations:\n")
    lines.append("- Central tendency & spread indicate typical strength and variability.\n")
    lines.append("- Skew/kurtosis indicate tail behavior; heavy tails → check QC.\n")
    if group_summary is not None:
        lines.append("- Group mean/median gaps may indicate supplier/mix shifts.\n")

    lines.append("\nEngineering implications:\n")
    lines.append("- High variability → tighter QC, revise mix design, conservative φ.\n")
    lines.append("- Heavy tails → inspect spec compliance and reliability margins.\n")
    lines.append("- Persistent group differences → root cause (curing, w/c ratio, aggregates).\n")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    return str(out_path.resolve())

def generate_all_required_visuals(
    df: pd.DataFrame,
    value_column: str = "strength_mpa",
    group_column: Optional[str] = None,
    outdir: str = "figs"
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    # 1) Histogram + Normal
    p1 = str(Path(outdir) / f"{value_column}_hist_normal.png")
    plot_distribution(df, value_column, f"{value_column} – Histogram + Normal", save_path=p1)
    out["hist_normal"] = p1

    # 2) Grup kutu grafiği (eğer uygunsa)
    if group_column and group_column in df.columns and df[group_column].nunique() >= 2:
        p2 = str(Path(outdir) / f"{value_column}_by_{group_column}_box.png")
        plot_material_comparison(df, value_column, group_column, save_path=p2)
        out["boxplot_groups"] = p2

    # 3) PDF/CDF karşılaştırmaları
    p3 = str(Path(outdir) / "pdf_cdf_comparisons.png")
    plot_pdf_cdf_comparisons(save_path=p3)
    out["pdf_cdf_comparisons"] = p3

    # 4) Dashboard  (DÜZELTME: column=value_column)
    p4 = str(Path(outdir) / f"{value_column}_dashboard.png")
    build_stat_dashboard(df, column=value_column, save_path=p4)
    out["dashboard"] = p4

    # 5) Bayes olasılık ağacı
    p5 = str(Path(outdir) / "bayes_tree.png")
    plot_probability_tree(prior=0.06, sensitivity=0.90, specificity=0.96, save_path=p5)
    out["bayes_tree"] = p5

    return out

# =========================================================
# Single entry point (only one __main__)
# =========================================================
def main() -> None:
    # Part 1A quick exploration
    explore_datasets()

    # Run Part 1 analyses (no file I/O if save_fig=False)
    central_tendency_analysis(save_fig=True)
    measures = measures_of_spread_analysis(save_fig=True)
    shape_measures_analysis(save_fig=True)
    quantiles_and_percentiles_analysis(save_fig=True)

    # Discrete / Continuous demos
    discrete_distributions_demo(save_fig=True)
    continuous_distributions_demo(save_fig=True)
    normal_fit_analysis(bins=25, save_fig=True)

    # Conditional & Bayes
    conditional_failure_given_detected(save_fig=True)
    bayes_diagnostic_posterior(save_fig=True)

    # Group comparison (auto-pick)
    try:
        basic_group_comparison(save_fig=True)
    except Exception as e:
        print("Group comparison skipped:", e)

    # Full visuals + report
    try:
        df = load_data("concrete_strength.csv")
    except FileNotFoundError:
        print("⚠️ concrete_strength.csv not found; skipping visuals bundle & report.")
        return

    figs = generate_all_required_visuals(
        df,
        value_column="strength_mpa",
        group_column=("material_type" if "material_type" in df.columns else None),
        outdir="figs"
    )
    print("Generated figures:", figs)

    report_path = create_statistical_report(
        df, output_file="lab4_statistical_report.txt",
        value_column="strength_mpa",
        group_column=("material_type" if "material_type" in df.columns else None)
    )
    print("Report written to:", report_path)

if __name__ == "__main__":
    main()