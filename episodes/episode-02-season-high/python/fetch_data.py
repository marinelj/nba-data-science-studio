"""Download auditable NBA responses, then build one observation per season.

Run from the repository root. Cached responses make subsequent runs offline.
Only --refresh performs new requests when a valid cache exists.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonplayerinfo, playercareerstats, playergamelog, teamgamelog,
)
from nba_api.stats.static import teams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PLAYER_ID = 2544
SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2003, 2026)]


def cached_response(filename, factory, refresh=False):
    path = RAW / filename
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    error = None
    for attempt in range(3):
        try:
            response = factory().nba_response
            body = response.get_dict()
            if not body.get("resultSets"):
                raise ValueError(f"Missing resultSets: {filename}")
            envelope = {
                "source_url": response.get_url(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "nba_api_version": version("nba_api"),
                "response": body,
            }
            RAW.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(envelope, indent=2) + "\n")
            temporary.replace(path)
            time.sleep(0.7)
            return envelope
        except Exception as exc:
            error = exc
            print(f"Retry {attempt + 1}/3: {filename}: {exc}", flush=True)
            time.sleep(1 + attempt)
    raise RuntimeError(f"Cannot fetch {filename}; no substitute data used") from error


def frame(envelope, name):
    result_sets = envelope["response"]["resultSets"]
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    result = next(item for item in result_sets if item["name"] == name)
    return pd.DataFrame(result["rowSet"], columns=result["headers"])


def prepare_log(log):
    log = log.rename(columns={"Game_ID": "game_id", "GAME_ID": "game_id"}).copy()
    log["game_id"] = log["game_id"].astype(str).str.zfill(10)
    if log.empty or log["game_id"].duplicated().any():
        raise ValueError("Empty log or duplicate game IDs")
    if not log["game_id"].str.startswith("002").all():
        raise ValueError("Unexpected non-regular-season game in log")
    log["game_date"] = pd.to_datetime(log["GAME_DATE"], format="mixed")
    return log.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def season_summary(player, team, season, birthdate):
    """Order first; use the earliest tied maximum and the TEAM game number."""
    player, team = prepare_log(player), prepare_log(team)
    team["team_game_number"] = range(1, len(team) + 1)
    player["player_game_number"] = range(1, len(player) + 1)
    joined = player.merge(
        team[["game_id", "team_game_number", "game_date"]],
        on="game_id", how="left", validate="one_to_one", suffixes=("", "_team"),
    )
    if joined["team_game_number"].isna().any():
        raise ValueError(f"{season}: unmatched player game")
    if not joined["game_date"].eq(joined["game_date_team"]).all():
        raise ValueError(f"{season}: player/team game dates disagree")
    debut = joined.iloc[0]
    peak_points = int(joined["PTS"].max())
    peaks = joined.loc[joined["PTS"].eq(peak_points)]
    first_peak = peaks.iloc[0]
    g = int(first_peak["team_game_number"])
    n = len(team)
    age = (debut["game_date"] - birthdate).days / 365.2425
    row = {
        "season": season,
        "team": debut["MATCHUP"].split()[0],
        "team_games": n,
        "player_games": len(player),
        "debut_date": debut["game_date"].date().isoformat(),
        "age_at_debut": age,
        "debut_team_game_number": int(debut["team_game_number"]),
        "debut_fgm": int(debut["FGM"]),
        "debut_points": int(debut["PTS"]),
        "season_high_points": peak_points,
        "first_high_date": first_peak["game_date"].date().isoformat(),
        "first_high_game_id": first_peak["game_id"],
        "first_high_team_game_number": g,
        "first_high_player_game_number": int(first_peak["player_game_number"]),
        "tied_high_games": len(peaks),
        "first_high_season_fraction": (g - 1) / (n - 1),
        "first_high_82_equivalent": 1 + 81 * (g - 1) / (n - 1),
        "is_82_game_season": n == 82,
        "player_log_url": f"https://www.nba.com/stats/player/2544/boxscores-traditional?Season={season}",
        "first_high_boxscore_url": f"https://www.nba.com/game/{first_peak['game_id']}/box-score",
    }
    joined.insert(0, "season", season)
    return row, joined


def main(refresh=False):
    info = cached_response("player_info.json", lambda: commonplayerinfo.CommonPlayerInfo(
        player_id=PLAYER_ID, timeout=25), refresh)
    birthdate = pd.Timestamp(frame(info, "CommonPlayerInfo").iloc[0]["BIRTHDATE"])
    career = cached_response("career_totals.json", lambda: playercareerstats.PlayerCareerStats(
        player_id=PLAYER_ID, timeout=25), refresh)
    totals = frame(career, "SeasonTotalsRegularSeason")
    team_ids = {item["abbreviation"]: item["id"] for item in teams.get_teams()}
    rows, logs, checks = [], [], []
    for season in SEASONS:
        response = cached_response(f"player_{season}.json", lambda: playergamelog.PlayerGameLog(
            player_id=PLAYER_ID, season=season, season_type_all_star="Regular Season", timeout=25), refresh)
        player = frame(response, "PlayerGameLog")
        abbreviations = player["MATCHUP"].str.split().str[0].unique()
        if len(abbreviations) != 1:
            raise ValueError(f"{season}: multi-team season requires explicit schedule stitching")
        team_id = team_ids[abbreviations[0]]
        response = cached_response(f"team_{season}.json", lambda: teamgamelog.TeamGameLog(
            team_id=team_id, season=season, season_type_all_star="Regular Season", timeout=25), refresh)
        team = frame(response, "TeamGameLog")
        row, log = season_summary(player, team, season, birthdate)
        total = totals.loc[totals["SEASON_ID"].eq(season) & totals["TEAM_ID"].eq(team_id)]
        if len(total) != 1:
            raise ValueError(f"{season}: no unique career-total control row")
        total = total.iloc[0]
        checks.append({"season": season, "games": len(player), "points": int(player["PTS"].sum()),
                       "fgm": int(player["FGM"].sum()), "career_games": int(total["GP"]),
                       "career_points": int(total["PTS"]), "career_fgm": int(total["FGM"])})
        for measured, expected in [(len(player), total["GP"]), (player["PTS"].sum(), total["PTS"]),
                                   (player["FGM"].sum(), total["FGM"])]:
            if int(measured) != int(expected):
                raise ValueError(f"{season}: incomplete log; totals disagree: {measured} != {expected}")
        if len(team) != int((team["W"] + team["L"]).max()):
            raise ValueError(f"{season}: incomplete team log")
        rows.append(row)
        logs.append(log)
        print(f"{season}: {len(player)} appearances / {len(team)} team games; "
              f"high {row['season_high_points']} PTS first at team game {row['first_high_team_game_number']}", flush=True)
    summary = pd.DataFrame(rows)
    if len(summary) != 23 or summary["season"].duplicated().any():
        raise ValueError("Expected 23 unique completed seasons")
    summary.to_csv(DATA / "season_summary.csv", index=False, float_format="%.8f")
    pd.concat(logs, ignore_index=True).to_csv(DATA / "lebron_regular_season_games.csv", index=False)
    pd.DataFrame(checks).to_csv(DATA / "source_reconciliation.csv", index=False)
    manifest = {"player_id": PLAYER_ID, "birthdate": str(birthdate.date()), "seasons": SEASONS,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "raw_files": []}
    for path in sorted(RAW.glob("*.json")):
        envelope = json.loads(path.read_text())
        manifest["raw_files"].append({"file": str(path.relative_to(DATA)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_url": envelope["source_url"], "retrieved_at_utc": envelope["retrieved_at_utc"]})
    (DATA / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved 23 verified seasons, joined game logs, reconciliation and source manifest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    main(parser.parse_args().refresh)
