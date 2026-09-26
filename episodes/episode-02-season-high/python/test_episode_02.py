"""Focused checks for the statistical definition and leakage boundaries."""

import base64
import json
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_assets import AUDIO, DATA, audio_name, build_cues, data_uri, integer_words, narration_clips, spoken_number
from fetch_data import prepare_log, season_summary
from episode_02_season_high import empirical_distribution, fit_ols, load_data, walk_forward


class EpisodeTests(unittest.TestCase):
    def test_tied_maximum_and_missed_team_game(self):
        # Appearance two is TEAM game three; the later tied maximum must not win.
        team = pd.DataFrame({"Game_ID": ["0020000004", "0020000002", "0020000001", "0020000003"],
                             "GAME_DATE": ["2020-01-04", "2020-01-02", "2020-01-01", "2020-01-03"]})
        player = pd.DataFrame({"Game_ID": ["0020000004", "0020000001", "0020000003"],
                               "GAME_DATE": ["2020-01-04", "2020-01-01", "2020-01-03"],
                               "MATCHUP": ["LAL vs. BOS"] * 3, "PTS": [40, 20, 40], "FGM": [15, 8, 14]})
        row, _ = season_summary(player, team, "fixture", pd.Timestamp("1984-12-30"))
        self.assertEqual(row["first_high_team_game_number"], 3)
        self.assertEqual(row["first_high_player_game_number"], 2)
        self.assertEqual(row["tied_high_games"], 2)
        self.assertEqual(row["debut_points"], 20)
        self.assertEqual(row["first_high_82_equivalent"], 55)

    def test_duplicate_ids_rejected(self):
        duplicate = pd.DataFrame({"Game_ID": ["0020000001"] * 2, "GAME_DATE": ["2020-01-01"] * 2})
        with self.assertRaises(ValueError): prepare_log(duplicate)

    def test_pmf_cdf_exact_values_and_empty_support(self):
        result = empirical_distribution(pd.Series([10, 30, 30, 70]))
        self.assertEqual(result.loc[30, "pmf"], 0.5)
        self.assertEqual(result.loc[30, "cdf"], 0.75)
        self.assertEqual(result.loc[29, "cdf"], 0.25)
        self.assertEqual(result.loc[31, "pmf"], 0)
        self.assertEqual(result.loc[82, "cdf"], 1)
        self.assertAlmostEqual(result.pmf.sum(), 1)

    def test_chronological_evaluation_ignores_future_targets(self):
        table = load_data()
        original = walk_forward(table)
        changed = table.copy()
        changed.loc[10:, "first_high_team_game_number"] = 82
        modified = walk_forward(changed)
        self.assertAlmostEqual(original.iloc[0].prediction, modified.iloc[0].prediction)
        self.assertTrue((original.train_through < original.test_season).all())
        self.assertEqual(len(original), 13)

    def test_ols_recovers_known_coefficients(self):
        ages = np.arange(18, 38)
        points = np.array([12, 20, 8, 40, 16, 22, 31, 27, 17, 35] * 2)
        table = pd.DataFrame({"age_at_debut": ages, "debut_points": points,
                              "first_high_team_game_number": 80 - ages + 0.25 * points})
        model = fit_ols(table)
        np.testing.assert_allclose([model["intercept"], model["age_coefficient"],
                                    model["debut_points_coefficient"]], [80, -1, 0.25], atol=1e-10)

    def test_23_season_dataset_and_debut_reconciliation(self):
        table = load_data()
        self.assertEqual(len(table.loc[table.team_games.eq(82)]), 20)
        from pathlib import Path
        episode_one = Path(__file__).resolve().parents[2] / "episode-01-cdf-pmf" / "data"
        earlier = pd.read_csv(episode_one / "lebron_season_debut_fgm_2003_2025.csv")
        joined = table.merge(earlier, on="season", validate="one_to_one")
        self.assertTrue(joined.debut_fgm.eq(joined.fgm).all())

    def test_spoken_numbers_use_natural_english(self):
        self.assertEqual(spoken_number(75.728, 2), "seventy-five point seven three")
        self.assertEqual(spoken_number(-0.0854), "minus zero point zero eight five")
        self.assertEqual(spoken_number(18.43, 2), "eighteen point four three")
        self.assertEqual(integer_words(40), "forty")
        self.assertEqual(integer_words(115), "one hundred fifteen")
        with self.assertRaises(ValueError): integer_words(1000)

    def test_narration_clips_match_subtitles_and_embed_as_mp3(self):
        cues = build_cues(json.loads((DATA / "results.json").read_text()))
        clips = narration_clips(cues)
        self.assertEqual(len(clips), len(cues))
        for number, clip in enumerate(clips, 1):
            header, payload = data_uri(AUDIO / clip["file"], "audio/mpeg").split(",", 1)
            self.assertEqual(header, "data:audio/mpeg;base64", number)
            self.assertEqual(base64.b64decode(payload), (AUDIO / audio_name(number)).read_bytes(), number)

    def test_word_timings_cover_each_cue_in_order(self):
        for number, clip in enumerate(narration_clips(build_cues(json.loads((DATA / "results.json").read_text()))), 1):
            starts = [start for start, _ in clip["words"]]
            self.assertEqual("".join(word for _, word in clip["words"]).strip(), clip["text"], number)
            self.assertEqual(starts, sorted(starts), f"cue {number} word timings move backwards")
            self.assertGreaterEqual(starts[0], 0, number)
            self.assertLess(starts[-1], clip["seconds"], f"cue {number} starts a word after the audio ends")

    def test_stale_narration_is_rejected(self):
        cues = build_cues(json.loads((DATA / "results.json").read_text()))
        cues[0] = {**cues[0], "text": cues[0]["text"] + " Edited."}
        with self.assertRaisesRegex(ValueError, "stale"):
            narration_clips(cues)


if __name__ == "__main__":
    unittest.main()
