import csv
import tempfile
import unittest
from pathlib import Path

from lp_tokenizer.celex import (
    EnglishCelex,
    is_morphology_violation,
    validate_morphology_rho,
    write_unmatched_report,
)


def _lemma_row(lemma_id, head, structure="", alternative_structure=""):
    fields = [""] * (44 if alternative_structure else 25)
    fields[0] = str(lemma_id)
    fields[1] = head
    fields[21] = structure
    if alternative_structure:
        fields[40] = alternative_structure
    return fields


def _wordform_row(wordform_id, surface, lemma_id, transformation):
    return [str(wordform_id), surface, "", str(lemma_id), "", transformation]


class EnglishCelexTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        lemma_dir = root / "english" / "eml"
        wordform_dir = root / "english" / "emw"
        lemma_dir.mkdir(parents=True)
        wordform_dir.mkdir(parents=True)

        lemma_rows = [
            _lemma_row(
                1,
                "happiness",
                "((happy)[A],(ness)[N|A.])[N]",
                "(happiness)[N]",
            ),
            _lemma_row(2, "make", "(make)[V]"),
            _lemma_row(3, "making", "(making)[N]"),
            _lemma_row(4, "study", "(study)[V]"),
            _lemma_row(5, "child", "(child)[N]"),
        ]
        with (lemma_dir / "eml.cd").open("w", encoding="ascii", newline="") as handle:
            csv.writer(handle, delimiter="\\", lineterminator="\n").writerows(
                lemma_rows
            )

        wordform_rows = [
            _wordform_row(1, "happiness", 1, "@"),
            _wordform_row(2, "making", 2, "@-e+ing"),
            _wordform_row(3, "making", 3, "@"),
            _wordform_row(4, "studies", 4, "@-y+ies"),
            _wordform_row(5, "children", 5, "IRR"),
        ]
        with (wordform_dir / "emw.cd").open("w", encoding="ascii", newline="") as handle:
            csv.writer(handle, delimiter="\\", lineterminator="\n").writerows(
                wordform_rows
            )
        self.celex = EnglishCelex.load(str(root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preferred_parse_and_spelling_projection(self):
        match = self.celex.match_pretoken("happiness")
        self.assertFalse(match.unmatched)
        self.assertEqual(match.aligned_spans, frozenset({(0, 5), (5, 9)}))

    def test_inflection_and_homograph_analyses_are_preserved(self):
        studies = self.celex.match_pretoken("studies")
        self.assertEqual(studies.aligned_spans, frozenset({(0, 4), (4, 7)}))

        children = self.celex.match_pretoken("children")
        self.assertEqual(children.aligned_spans, frozenset({(0, 8)}))

        making = self.celex.match_pretoken("making")
        self.assertEqual(
            making.aligned_spans,
            frozenset({(0, 3), (3, 6), (0, 6)}),
        )

    def test_bytelevel_prefix_and_safe_casefold(self):
        match = self.celex.match_pretoken("ĠHappiness")
        self.assertFalse(match.unmatched)
        self.assertEqual(
            match.aligned_spans,
            frozenset({(0, 6), (1, 6), (6, 10)}),
        )

    def test_unmatched_and_edge_classification(self):
        unmatched = self.celex.match_pretoken("xyzzy")
        self.assertTrue(unmatched.unmatched)
        self.assertEqual(unmatched.reason, "no_celex_entry")

        spans = {(0, 5), (5, 9)}
        self.assertFalse(is_morphology_violation(0, 5, False, spans))
        self.assertTrue(is_morphology_violation(0, 9, False, spans))
        self.assertTrue(is_morphology_violation(1, 5, False, spans))
        self.assertFalse(is_morphology_violation(0, 9, True, spans))

    def test_unmatched_report_is_complete_and_sorted(self):
        rows = [
            {
                "pretoken": "zeta",
                "celex_decoded_form": "zeta",
                "frequency": 3,
                "celex_unmatched_reason": "no_celex_entry",
                "celex_unmatched": True,
            },
            {
                "pretoken": "matched",
                "celex_decoded_form": "matched",
                "frequency": 100,
                "celex_unmatched_reason": "matched",
                "celex_unmatched": False,
            },
            {
                "pretoken": "alpha",
                "celex_decoded_form": "alpha",
                "frequency": 8,
                "celex_unmatched_reason": "no_celex_entry",
                "celex_unmatched": True,
            },
        ]
        report_path = Path(self.temp_dir.name) / "report.tsv"
        self.assertEqual(write_unmatched_report(rows, str(report_path)), 2)
        self.assertEqual(
            report_path.read_text(encoding="utf-8").splitlines(),
            [
                "pretoken\tdecoded_form\tfrequency\treason",
                "alpha\talpha\t8\tno_celex_entry",
                "zeta\tzeta\t3\tno_celex_entry",
            ],
        )

    def test_rho_validation(self):
        self.assertEqual(validate_morphology_rho(0), 0.0)
        self.assertEqual(validate_morphology_rho(0.5), 0.5)
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_morphology_rho(invalid)


if __name__ == "__main__":
    unittest.main()
