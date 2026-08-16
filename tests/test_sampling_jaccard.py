import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from sampling_jaccard import (
    create_length_conditioned_jaccard_figure,
    group_tokens_by_length,
    jaccard_score,
    pairwise_jaccard_by_length,
    plot_length_conditioned_jaccard,
)


class SamplingJaccardTest(unittest.TestCase):
    def test_grouping_uses_stored_length(self):
        grouped = group_tokens_by_length(["a", "a", "Ġx", "Ã©"])

        self.assertEqual(grouped, {1: {"a"}, 2: {"Ġx", "Ã©"}})

    def test_pairwise_results_keep_each_pair_and_only_defined_lengths(self):
        vocabularies = [
            {"a", "b", "cc"},
            {"a", "dd", "eee"},
            {"z", "cc"},
        ]
        results = pairwise_jaccard_by_length(vocabularies)

        self.assertEqual(list(results), ["1", "2", "3"])
        self.assertEqual(
            results["1"],
            [
                {"sample_i": 0, "sample_j": 1, "jaccard": 0.5},
                {"sample_i": 0, "sample_j": 2, "jaccard": 0.0},
                {"sample_i": 1, "sample_j": 2, "jaccard": 0.0},
            ],
        )
        self.assertEqual(
            results["2"],
            [
                {"sample_i": 0, "sample_j": 1, "jaccard": 0.0},
                {"sample_i": 0, "sample_j": 2, "jaccard": 1.0},
                {"sample_i": 1, "sample_j": 2, "jaccard": 0.0},
            ],
        )
        # Only sample 1 has a length-three token, so pairs not involving it are
        # omitted instead of receiving an artificial empty/empty score.
        self.assertEqual(
            results["3"],
            [
                {"sample_i": 0, "sample_j": 1, "jaccard": 0.0},
                {"sample_i": 1, "sample_j": 2, "jaccard": 0.0},
            ],
        )

    def test_jaccard_score_keeps_existing_intersection_over_union_behavior(self):
        self.assertEqual(
            jaccard_score(
                ["a", "bb", "shared"],
                ["a", "cc", "shared"],
            ),
            0.5,
        )

    def test_figure_has_method_panels_pair_traces_and_gaps(self):
        results = {
            "lp_det": {
                "1": [
                    {"sample_i": 0, "sample_j": 1, "jaccard": 0.8},
                    {"sample_i": 0, "sample_j": 2, "jaccard": 0.6},
                ],
                "2": [
                    {"sample_i": 0, "sample_j": 1, "jaccard": 0.4},
                ],
            },
            "bpe": {
                "1": [
                    {"sample_i": 0, "sample_j": 1, "jaccard": 0.7},
                ],
                "3": [
                    {"sample_i": 0, "sample_j": 1, "jaccard": 0.2},
                ],
            },
        }
        figure = create_length_conditioned_jaccard_figure(results, title="test")
        try:
            visible_axes = [axis for axis in figure.axes if axis.get_visible()]
            self.assertEqual(
                [axis.get_title() for axis in visible_axes],
                ["lp det", "bpe"],
            )
            self.assertEqual(len(visible_axes[0].lines), 2)
            self.assertEqual(len(visible_axes[1].lines), 1)
            second_pair_values = list(visible_axes[0].lines[1].get_ydata())
            self.assertEqual(second_pair_values[0], 0.6)
            self.assertTrue(math.isnan(second_pair_values[1]))
            bpe_values = list(visible_axes[1].lines[0].get_ydata())
            self.assertEqual(bpe_values[0], 0.7)
            self.assertTrue(math.isnan(bpe_values[1]))
            self.assertEqual(bpe_values[2], 0.2)
        finally:
            plt.close(figure)

    def test_plot_is_written_as_a_nonempty_png(self):
        results = {
            "lp_det": {
                "1": [
                    {"sample_i": 0, "sample_j": 1, "jaccard": 0.5},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "plot.png"
            returned_path = plot_length_conditioned_jaccard(
                results,
                str(output_path),
                title="test",
            )

            self.assertEqual(returned_path, str(output_path))
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
