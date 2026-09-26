"""Episode 02: When does LeBron first reach his season-high score?

Offline analysis of verified NBA caches. Run fetch_data.py first if absent.
The primary target is the team's actual regular-season game number.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = ROOT / "images"
FEATURES = ["age_at_debut", "debut_points"]
TARGET = "first_high_team_game_number"


def load_data():
    path = DATA / "season_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Run python/fetch_data.py to download the NBA data first.")
    table = pd.read_csv(path).sort_values("season").reset_index(drop=True)
    validate(table)
    return table


def validate(table):
    required = FEATURES + [TARGET, "season", "team_games", "player_games", "debut_fgm",
                           "season_high_points", "first_high_player_game_number", "first_high_82_equivalent"]
    if table[required].isna().any().any():
        raise ValueError("Missing required observations")
    expected = [f"{y}-{(y + 1) % 100:02d}" for y in range(2003, 2026)]
    if table["season"].tolist() != expected:
        raise ValueError("Expected exactly the 23 seasons 2003-04 through 2025-26")
    if not (table[TARGET].ge(1) & table[TARGET].le(table["team_games"])).all():
        raise ValueError("Peak outside team schedule")
    if not table["first_high_player_game_number"].le(table[TARGET]).all():
        raise ValueError("A player cannot appear in more games than his team has played")
    if not table["season_high_points"].ge(table["debut_points"]).all():
        raise ValueError("Season high below debut score")
    if not table["first_high_82_equivalent"].between(1, 82).all():
        raise ValueError("Invalid normalized position")


def empirical_distribution(values, upper=82):
    """Integer support includes zero-frequency outcomes; CDF is right-continuous."""
    support = pd.Index(range(1, upper + 1), name="game_number")
    if values.empty or not values.between(1, upper).all() or not values.eq(values.round()).all():
        raise ValueError("Expected nonempty integer game positions in [1, upper]")
    pmf = values.value_counts(normalize=True).reindex(support, fill_value=0.0)
    return pd.DataFrame({"pmf": pmf, "cdf": pmf.cumsum()})


def correlation_table(table):
    """Pearson: linear association. Spearman: monotonic rank association."""
    rows = []
    for outcome in ["debut_fgm", "season_high_points"]:
        x, y = table["age_at_debut"], table[outcome]
        rows.append({"x": "age_at_debut", "y": outcome, "n": len(table),
                     "pearson_r": float(x.corr(y)),
                     "spearman_rho": float(x.rank().corr(y.rank()))})
    return pd.DataFrame(rows)


def fit_ols(table, target=TARGET):
    """Fit intercept + age + debut PTS by ordinary least squares."""
    design = np.column_stack([np.ones(len(table)), table[FEATURES].to_numpy(float)])
    y = table[target].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Age and debut points do not identify all coefficients")
    fitted = design @ beta
    ss_total = np.sum((y - y.mean()) ** 2)
    return {"target": target, "n": len(table), "features": FEATURES,
            "intercept": float(beta[0]), "age_coefficient": float(beta[1]),
            "debut_points_coefficient": float(beta[2]),
            "training_r2": float(1 - np.sum((y - fitted) ** 2) / ss_total),
            "training_mae": float(np.mean(np.abs(y - fitted)))}


def predict(model, age, debut_points):
    return (model["intercept"] + model["age_coefficient"] * age
            + model["debut_points_coefficient"] * debut_points)


def walk_forward(table, target=TARGET, min_train=10):
    """Predict each held-out season from earlier seasons only, after its debut."""
    rows = []
    for i in range(min_train, len(table)):
        train, test = table.iloc[:i], table.iloc[i]
        model = fit_ols(train, target)
        rows.append({"test_season": test["season"], "train_through": train.iloc[-1]["season"],
                     "train_n": len(train), "actual": float(test[target]),
                     "prediction": float(predict(model, test["age_at_debut"], test["debut_points"])),
                     "past_mean": float(train[target].mean()),
                     "past_median": float(train[target].median())})
    return pd.DataFrame(rows)


def evaluate(predictions):
    metrics = {"n_test_seasons": len(predictions)}
    actual = predictions["actual"].to_numpy()
    for column in ["prediction", "past_mean", "past_median"]:
        errors = actual - predictions[column].to_numpy()
        metrics[column] = {"mae_games": float(np.abs(errors).mean()),
                           "rmse_games": float(np.sqrt(np.mean(errors ** 2)))}
    return metrics


def setup_style():
    plt.rcParams.update({"figure.facecolor": "#10191e", "axes.facecolor": "#15232b",
                        "savefig.facecolor": "#10191e", "text.color": "#eaf0f2",
                        "axes.labelcolor": "#eaf0f2", "xtick.color": "#b9c7cd", "ytick.color": "#b9c7cd",
                        "axes.edgecolor": "#5f737d", "grid.color": "#34464f", "font.size": 11,
                        "axes.titlesize": 14, "legend.facecolor": "#15232b", "legend.edgecolor": "#5f737d"})


def save_plot(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / f"{name}.png", dpi=165, bbox_inches="tight")
    fig.savefig(IMAGES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def create_plots(table, distribution, correlations, predictions):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    axes[0].bar(distribution.index, distribution.pmf, color="#6baee8", width=0.8)
    axes[0].set(title="Empirical PMF: exactly game k", xlabel="First season-high team game #",
                ylabel="Relative frequency", xlim=(0, 83))
    for k in table[TARGET].mode():
        axes[0].annotate(f"{k}", (k, distribution.loc[k, "pmf"]), xytext=(0, 5),
                         textcoords="offset points", ha="center", color="#ffbf69")
    axes[1].step([0, *distribution.index], [0, *distribution.cdf], where="post", color="#ffbf69", linewidth=2)
    for k in [20, 41, 60]:
        value = float(distribution.loc[k, "cdf"])
        axes[1].plot(k, value, "o", color="#ffbf69")
        axes[1].annotate(f"F({k}) = {value:.1%}", (k, value), xytext=(-25, -23),
                         textcoords="offset points", fontsize=10)
    axes[1].set(title="Empirical CDF: by game k", xlabel="Team game # k", ylabel="Cumulative relative frequency",
                xlim=(0, 83), ylim=(-0.02, 1.08))
    for ax in axes: ax.grid(axis="y", alpha=0.4)
    fig.suptitle("23 seasons, one first-maximum position per season (actual schedule lengths)", fontsize=13)
    save_plot(fig, "01-empirical-pmf-cdf")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    for ax, (_, row) in zip(axes, correlations.iterrows()):
        y = row["y"]
        ax.scatter(table.age_at_debut, table[y], color="#80d2b0", s=45)
        a, b = np.polyfit(table.age_at_debut, table[y], 1)
        xs = np.array([table.age_at_debut.min(), table.age_at_debut.max()])
        ax.plot(xs, a * xs + b, "--", color="#ffbf69", label="Bivariate descriptive line")
        title = "Debut FGM" if y == "debut_fgm" else "Season-high points"
        ax.set(title=f"{title} vs. age: r = {row.pearson_r:+.3f}", xlabel="Age at season debut (years)",
               ylabel="Field goals made" if y == "debut_fgm" else "Points in highest-scoring game")
        ax.grid(alpha=0.3)
    fig.suptitle("Within one player's career: association is not an age effect", fontsize=13)
    save_plot(fig, "02-age-correlations")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    x = np.arange(len(predictions))
    axes[0].plot(x, predictions.actual, "o-", color="#eaf0f2", label="Actual first high")
    axes[0].plot(x, predictions.prediction, "o-", color="#6baee8", label="Age + debut PTS")
    axes[0].plot(x, predictions.past_mean, "--", color="#ffbf69", label="Past-season mean")
    axes[0].set_xticks(x, predictions.test_season, rotation=55, ha="right", fontsize=9)
    axes[0].set(title="Expanding-window predictions", xlabel="Held-out season", ylabel="Team game #")
    axes[0].legend(fontsize=9)
    errors = evaluate(predictions)
    maes = [errors[col]["mae_games"] for col in ["prediction", "past_mean", "past_median"]]
    bars = axes[1].bar(["OLS", "Past mean", "Past median"], maes,
                       color=["#6baee8", "#ffbf69", "#80d2b0"])
    axes[1].bar_label(bars, fmt="%.1f", padding=4)
    axes[1].set(title="Out-of-sample MAE (lower is better)", ylabel="Mean absolute error (games)",
                ylim=(0, max(maes) * 1.2))
    save_plot(fig, "03-regression-backtest")

    full = table.loc[table.team_games.eq(82)]
    full_dist = empirical_distribution(full[TARGET])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    axes[0].step([0, *distribution.index], [0, *distribution.cdf], where="post", label="All 23 seasons", color="#6baee8")
    axes[0].step([0, *full_dist.index], [0, *full_dist.cdf], where="post", label=f"Only 82-game seasons (n={len(full)})", color="#ffbf69")
    axes[0].set(title="Sensitivity to schedule length", xlabel="First season-high team game #", ylabel="Empirical CDF", xlim=(0, 83), ylim=(0, 1.06))
    axes[0].legend(fontsize=9)
    axes[1].hist(table.first_high_season_fraction, bins=np.linspace(0, 1, 7), density=True,
                 color="#80d2b0", edgecolor="#15232b")
    axes[1].set(title="Density histogram: binned approximation", xlabel="Season position (game # - 1) / (N - 1)",
                ylabel="Density (area sums to 1)", xlim=(0, 1))
    fig.suptitle("A density histogram is an approximation; exact game numbers use a PMF", fontsize=13)
    save_plot(fig, "04-schedule-and-density")


def main(age=None, debut_points=None):
    table = load_data()
    distribution = empirical_distribution(table[TARGET])
    correlations = correlation_table(table)
    model = fit_ols(table)
    predictions = walk_forward(table)
    full = table.loc[table.team_games.eq(82)].reset_index(drop=True)
    normalized_target = "first_high_82_equivalent"
    results = {
        "sample_seasons": len(table), "full_82_game_seasons": len(full),
        "target": TARGET, "age_definition": "days since 1984-12-30 at debut / 365.2425",
        "short_seasons": table.loc[table.team_games.ne(82), ["season", "team_games"]].to_dict("records"),
        "distribution": {"median": float(table[TARGET].median()), "mean": float(table[TARGET].mean()),
                         "modes": table[TARGET].mode().tolist(),
                         "cdf_at_20": float(distribution.loc[20, "cdf"]),
                         "cdf_at_41": float(distribution.loc[41, "cdf"]),
                         "cdf_at_60": float(distribution.loc[60, "cdf"])},
        "correlations": correlations.to_dict("records"), "model": model,
        "walk_forward": evaluate(predictions),
        "sensitivity_82_game_seasons": {"model": fit_ols(full), "walk_forward": evaluate(walk_forward(full))},
        "sensitivity_normalized_82": {"model": fit_ols(table, normalized_target),
                                      "walk_forward": evaluate(walk_forward(table, normalized_target))},
    }
    if (age is None) != (debut_points is None):
        raise ValueError("Provide both --age and --debut-points")
    if age is not None:
        if not np.isfinite([age, debut_points]).all() or age <= 0 or debut_points < 0:
            raise ValueError("Age must be positive and debut points nonnegative and finite")
        raw = predict(model, age, debut_points)
        results["user_scenario"] = {"age": age, "debut_points": debut_points,
            "predicted_team_game_raw": raw, "nearest_game": int(np.rint(raw)),
            "outside_82_game_range": not 1 <= raw <= 82,
            "extrapolates_training_features": not (
                table.age_at_debut.min() <= age <= table.age_at_debut.max()
                and table.debut_points.min() <= debut_points <= table.debut_points.max()),
            "note": "A user-supplied scenario, not an observed 2026-27 debut. Raw OLS is not clipped."}
    distribution.to_csv(DATA / "empirical_distribution.csv")
    empirical_distribution(full[TARGET]).to_csv(DATA / "empirical_distribution_82_only.csv")
    correlations.to_csv(DATA / "correlations.csv", index=False)
    predictions.to_csv(DATA / "walk_forward_predictions.csv", index=False)
    (DATA / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    create_plots(table, distribution, correlations, predictions)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    # Only the command line needs a headless backend; importing must leave a notebook's inline plotting alone.
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", type=float)
    parser.add_argument("--debut-points", type=float)
    args = parser.parse_args()
    main(args.age, args.debut_points)
