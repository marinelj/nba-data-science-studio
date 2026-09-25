"""Build the notebook, timed narration, teleprompter parts and course HTML."""

import base64
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBTITLES = ROOT / "subtitles"
HTML = ROOT / "html"
PYTHON = ROOT / "python"
AUDIO = ROOT / "audio"
IMAGES = ROOT / "images"

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def timestamp(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"00:{minutes:02d}:{secs:02d},000"


def audio_name(cue_number):
    return f"episode-02-cue-{cue_number:02d}.mp3"


def data_uri(path, mime):
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def narration_clips(cues):
    manifest = json.loads((AUDIO / "narration-manifest.json").read_text())
    if [clip["text"] for clip in manifest["clips"]] != [cue["text"] for cue in cues]:
        raise ValueError("Narration audio is stale. Run: uv run python/build_narration_audio.py")
    return [data_uri(AUDIO / clip["file"], "audio/mpeg") for clip in manifest["clips"]]


def integer_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return TENS[tens] + (f"-{ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{ONES[hundreds]} hundred" + (f" {integer_words(rest)}" if rest else "")
    raise ValueError(f"No spoken form for {n}")


def spoken_number(value, decimals=3):
    whole, fraction = f"{abs(value):.{decimals}f}".split(".")
    sign = "minus " if value < 0 else ""
    return f"{sign}{integer_words(int(whole))} point " + " ".join(ONES[int(d)] for d in fraction)


def build_cues(results):
    mapping = {
        "cdf41": f"{100 * results['distribution']['cdf_at_41']:.1f}",
        "r_fgm_spoken": spoken_number(results["correlations"][0]["pearson_r"]),
        "r_peak_spoken": spoken_number(results["correlations"][1]["pearson_r"]),
        "intercept_spoken": spoken_number(results["model"]["intercept"], 2),
        "age_coefficient_spoken": spoken_number(abs(results["model"]["age_coefficient"]), 3),
        "points_coefficient_spoken": spoken_number(abs(results["model"]["debut_points_coefficient"]), 3),
        "mae_spoken": spoken_number(results["walk_forward"]["prediction"]["mae_games"], 2),
        "baseline_mae_spoken": spoken_number(results["walk_forward"]["past_mean"]["mae_games"], 2),
    }
    template = json.loads((SUBTITLES / "narration-template.json").read_text())
    cues = [{**cue, "text": cue["text"].format(**mapping)} for cue in template]
    if cues[0]["start"] != 0 or cues[-1]["end"] != 420:
        raise ValueError("Narration must cover exactly seven minutes")
    for current, following in zip(cues, cues[1:]):
        if current["end"] != following["start"]:
            raise ValueError("Narration has a gap or overlap")
    return cues


def build_subtitles(cues):
    SUBTITLES.mkdir(parents=True, exist_ok=True)
    srt = "\n\n".join(
        f"{i}\n{timestamp(cue['start'])} --> {timestamp(cue['end'])}\n{cue['text']}"
        for i, cue in enumerate(cues, 1)
    ) + "\n"
    (SUBTITLES / "episode-02-season-high-en.srt").write_text(srt)
    section_labels = {
        "Opening": "00:00–00:30 · Opening",
        "Hand-drawn intuition": "00:30–02:00 · Hand-drawn intuition",
        "Python walkthrough": "02:00–06:30 · Python walkthrough",
        "Closing": "06:30–07:00 · Closing",
    }
    lines = ["# Episode 02 Recording Script", ""]
    previous = None
    for cue in cues:
        if cue["section"] != previous:
            lines += [f"## {section_labels[cue['section']]}", ""]
            previous = cue["section"]
        lines += [cue["text"], ""]
    (SUBTITLES / "episode-02-script.md").write_text("\n".join(lines))
    full = " ".join(cue["text"] for cue in cues)
    (SUBTITLES / "episode-02-teleprompter-full.txt").write_text(full + "\n")
    for old in SUBTITLES.glob("episode-02-teleprompter-[0-9][0-9].txt"):
        old.unlink()
    chunks, chunk = [], ""
    for cue in cues:
        candidate = f"{chunk} {cue['text']}".strip()
        if len(candidate) > 980 and chunk:
            chunks.append(chunk)
            chunk = cue["text"]
        else:
            chunk = candidate
    if chunk:
        chunks.append(chunk)
    for i, text in enumerate(chunks, 1):
        if len(text) > 980:
            raise ValueError(f"Teleprompter part {i} is over the 980-character limit")
        (SUBTITLES / f"episode-02-teleprompter-{i:02d}.txt").write_text(text + "\n")
    return chunks


def code_steps(results):
    cdf = results["distribution"]["cdf_at_41"]
    return [
        {"title": "1. Load verified seasons", "start": 120, "end": 145,
         "description": "Read the prepared table. The separate download script preserves official responses and reconciliation controls.",
         "code": "from episode_02_season_high import load_data\n\nseasons = load_data()\nseasons.shape\nseasons.head(3)",
         "expected": "Expected: 23 rows, one completed season per row."},
        {"title": "2. Define the target", "start": 145, "end": 175,
         "description": "Sort every team schedule first. Join player appearances by game ID. Select the earliest tied maximum.",
         "code": "joined = player_games.merge(\n    team_games[[\"game_id\", \"team_game_number\"]],\n    on=\"game_id\", validate=\"one_to_one\"\n)\npeak_points = joined[\"PTS\"].max()\nfirst_peak = joined.loc[joined[\"PTS\"].eq(peak_points)].iloc[0]",
         "expected": "Target: first_high_team_game_number. Player appearance number stays separate."},
        {"title": "3. Audit the source", "start": 175, "end": 210,
         "description": "Check that detailed player logs reproduce NBA season totals and that every player game matches a team game.",
         "code": "assert len(player_games) == career_total[\"GP\"]\nassert player_games[\"PTS\"].sum() == career_total[\"PTS\"]\nassert player_games[\"FGM\"].sum() == career_total[\"FGM\"]\nassert joined[\"team_game_number\"].notna().all()",
         "expected": "All 23 seasons pass game-count, points, FGM and game-ID checks."},
        {"title": "4. Estimate PMF and CDF", "start": 210, "end": 255,
         "description": "Game number is discrete. Reindexing exposes unobserved game numbers as zero mass before accumulation.",
         "code": "support = pd.Index(range(1, 83), name=\"game_number\")\npmf = (seasons[\"first_high_team_game_number\"]\n       .value_counts(normalize=True)\n       .reindex(support, fill_value=0.0))\ncdf = pmf.cumsum()",
         "expected": f"At team game 41, the empirical CDF is {cdf:.1%}."},
        {"title": "5. Correlate age", "start": 255, "end": 300,
         "description": "Pearson measures linear association. Spearman checks rank association without claiming causality.",
         "code": "r_age_debut_fgm = seasons[\"age_at_debut\"].corr(seasons[\"debut_fgm\"])\nr_age_high_points = seasons[\"age_at_debut\"].corr(seasons[\"season_high_points\"])\n\nprint(r_age_debut_fgm, r_age_high_points)",
         "expected": f"Pearson r: age/debut FGM {results['correlations'][0]['pearson_r']:+.3f}; age/season-high PTS {results['correlations'][1]['pearson_r']:+.3f}."},
        {"title": "6. Fit OLS", "start": 300, "end": 335,
         "description": "Age and debut total points are available after the first appearance. Add an intercept and solve least squares.",
         "code": "X = np.column_stack([\n    np.ones(len(seasons)),\n    seasons[[\"age_at_debut\", \"debut_points\"]]\n])\ny = seasons[\"first_high_team_game_number\"].to_numpy()\nbeta, *_ = np.linalg.lstsq(X, y, rcond=None)",
         "expected": f"Game # = {results['model']['intercept']:.3f} {results['model']['age_coefficient']:+.3f} × age {results['model']['debut_points_coefficient']:+.3f} × debut PTS."},
        {"title": "7. Test forward in time", "start": 335, "end": 370,
         "description": "Begin with ten seasons, predict the next, expand the training window, and compare with a past-mean baseline.",
         "code": "for i in range(10, len(seasons)):\n    train = seasons.iloc[:i]\n    test = seasons.iloc[i]\n    model = fit_ols(train)\n    prediction = predict(model, test.age_at_debut, test.debut_points)",
         "expected": f"Held-out MAE: OLS {results['walk_forward']['prediction']['mae_games']:.2f}; past mean {results['walk_forward']['past_mean']['mae_games']:.2f} games."},
        {"title": "8. Try a scenario", "start": 370, "end": 380,
         "description": "Enter hypothetical inputs. Keep the raw continuous output visible, then round only for communication.",
         "scenario": True,
         "expected": "The app flags values outside the historical feature range or the 1–82 target range."},
        {"title": "9. Read the figures", "start": 380, "end": 390,
         "description": "Use about ten seconds of playback. The charts show description, association and out-of-sample error.",
         "gallery": True,
         "expected": "The regression only slightly improves MAE and worsens RMSE versus the simple mean baseline."},
    ]


def build_notebook(results):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python (NBA Data Science Studio)",
                                               "language": "python", "name": "nba-data-science-studio"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    cells = [
        nbf.v4.new_markdown_cell("# Episode 02: When does LeBron first reach his season high?\n\nWe estimate an empirical PMF and CDF for the **team game number of the first occurrence of each season's highest-scoring game**. We then measure age relationships and test an OLS prediction using age plus debut total points."),
        nbf.v4.new_markdown_cell("## 1. Set up the independent episode folder\n\nThe notebook works when Jupyter starts from the repository root, this episode folder, or its `python` folder. `fetch_data.py` is intentionally separate: it downloads official NBA responses, preserves source URLs, and verifies detailed logs against season totals."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport sys\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display\n\ncwd = Path.cwd().resolve()\nif cwd.name == 'python':\n    EPISODE_ROOT = cwd.parent\nelif (cwd / 'data' / 'season_summary.csv').exists():\n    EPISODE_ROOT = cwd\nelse:\n    EPISODE_ROOT = cwd / 'episodes' / 'episode-02-season-high'\n\nsys.path.insert(0, str(EPISODE_ROOT / 'python'))\nfrom episode_02_season_high import (\n    correlation_table, empirical_distribution, evaluate, fit_ols,\n    load_data, predict, walk_forward\n)"),
        nbf.v4.new_markdown_cell("## 2. Load and verify 23 season-level observations\n\nOne row represents one completed regular season, from 2003–04 through 2025–26. `team_game_number` counts team games, even when LeBron did not appear. If the season high is tied, the earliest occurrence wins."),
        nbf.v4.new_code_cell("seasons = load_data()\ncolumns = [\n    'season', 'team_games', 'player_games', 'age_at_debut',\n    'debut_fgm', 'debut_points', 'season_high_points',\n    'first_high_team_game_number', 'first_high_player_game_number'\n]\ndisplay(seasons[columns])\nprint('Rows:', len(seasons))\nprint('82-game seasons:', seasons['is_82_game_season'].sum())"),
        nbf.v4.new_markdown_cell("## 3. Empirical PMF, CDF and density\n\nThe target is an integer game number, so exact probabilities use a **PMF**. The CDF answers: “By team game $k$, what fraction of historical first season highs had occurred?” A histogram of normalized season position can approximate a density, but its bar height is not $P(G=k)$."),
        nbf.v4.new_code_cell("distribution = empirical_distribution(seasons['first_high_team_game_number'])\ndisplay(distribution.loc[[10, 20, 30, 41, 49, 60, 70, 82]])\nprint('PMF sums to:', distribution['pmf'].sum())\nprint('CDF at game 41:', f\"{distribution.loc[41, 'cdf']:.1%}\")"),
        nbf.v4.new_code_cell("fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\naxes[0].bar(distribution.index, distribution.pmf, color='steelblue')\naxes[0].set(title='Empirical PMF', xlabel='First season-high team game #', ylabel='Relative frequency')\naxes[1].step([0, *distribution.index], [0, *distribution.cdf], where='post', color='darkorange')\naxes[1].set(title='Empirical CDF', xlabel='Team game # k', ylabel='P(G ≤ k)', ylim=(0, 1.05))\nplt.tight_layout(); plt.show()"),
        nbf.v4.new_markdown_cell("## 4. Age correlations\n\nAge is measured at LeBron's first appearance that season. `debut_fgm` is field goals made. `season_high_points` is the maximum single-game total points in that completed regular season. Pearson describes linear association; Spearman describes rank association. Neither establishes a causal age effect."),
        nbf.v4.new_code_cell("correlations = correlation_table(seasons)\ndisplay(correlations)"),
        nbf.v4.new_markdown_cell("## 5. Linear regression\n\nFit $\\hat G = b_0 + b_1 \\times age + b_2 \\times debut\\ points$. These predictors are available only after the season debut, so this is an after-debut forecast."),
        nbf.v4.new_code_cell("model = fit_ols(seasons)\nmodel"),
        nbf.v4.new_markdown_cell("## 6. Chronological evaluation\n\nTraining fit is optimistic. For an honest historical check, train on the first ten seasons, predict the next season, expand the window, and repeat. Compare the model against the mean and median of earlier target values."),
        nbf.v4.new_code_cell("predictions = walk_forward(seasons)\ndisplay(predictions)\nevaluate(predictions)"),
        nbf.v4.new_markdown_cell("## 7. Try a hypothetical input\n\nThe inputs below are a scenario, not observed 2026–27 data. Keep the raw OLS output before rounding; linear regression does not enforce the valid range of games 1 through 82."),
        nbf.v4.new_code_cell("age = 41.8\ndebut_points = 25\nraw_prediction = predict(model, age, debut_points)\nprint(f'Raw prediction: {raw_prediction:.2f}')\nprint(f'Nearest team game: {round(raw_prediction)}')"),
        nbf.v4.new_markdown_cell("## 8. Interpret the result\n\nThe fitted equation exists, but predictive usefulness is weak: the expanding-window MAE is about 18 games and only slightly better than the historical-mean baseline. Age and debut points leave most of the timing variation unexplained. See `data/results.json` and the four generated figures for the exact results and sensitivity analyses."),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, PYTHON / "episode_02_season_high.ipynb")


def build_html(results, seasons, cues, steps):
    template = (PYTHON / "course_template.html").read_text()
    replacements = {
        "__RESULTS_JSON__": json.dumps(results, separators=(",", ":")),
        "__SEASONS_JSON__": json.dumps(seasons, separators=(",", ":")),
        "__CUES_JSON__": json.dumps(cues, separators=(",", ":")),
        "__STEPS_JSON__": json.dumps(steps, separators=(",", ":")),
        # Embedded, because Jupyter's /files/ sandboxes HTML pages and their own requests arrive logged out.
        "__AUDIO_JSON__": json.dumps(narration_clips(cues)),
        "__FIGURES_JSON__": json.dumps([data_uri(IMAGES / name, "image/png") for name in [
            "01-empirical-pmf-cdf.png", "02-age-correlations.png",
            "03-regression-backtest.png", "04-schedule-and-density.png"]]),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Expected one marker: {marker}")
        template = template.replace(marker, value)
    if re.search(r"__[A-Z_]+__", template):
        raise ValueError("Unresolved HTML marker")
    HTML.mkdir(parents=True, exist_ok=True)
    (HTML / "episode-02-recording-lab.html").write_text(template)


def main():
    results = json.loads((DATA / "results.json").read_text())
    # Preserve numeric types for the browser instead of serializing CSV strings.
    import pandas as pd
    seasons = pd.read_csv(DATA / "season_summary.csv").to_dict("records")
    cues = build_cues(results)
    chunks = build_subtitles(cues)
    steps = code_steps(results)
    build_notebook(results)
    build_html(results, seasons, cues, steps)
    print(f"Built 42 timed cues, {len(chunks)} Prompt+ parts, notebook and recording lab.")
    print("Prompt+ character counts:", [len(chunk) for chunk in chunks])


if __name__ == "__main__":
    main()
