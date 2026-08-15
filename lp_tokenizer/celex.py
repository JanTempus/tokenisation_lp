"""English CELEX morphology loading and ByteLevel pre-token matching."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from tokenizers.decoders import ByteLevel


BoundaryAnalysis = Tuple[int, ...]
EncodedSpan = Tuple[int, int]


@dataclass(frozen=True)
class MorphologicalAnalysis:
    morpheme_spans: Tuple[EncodedSpan, ...]


@dataclass(frozen=True)
class PretokenMorphology:
    decoded_form: str
    unmatched: bool
    reason: str
    analyses: Tuple[MorphologicalAnalysis, ...]

    @property
    def morphology_weight(self) -> float:
        return 0.0 if self.unmatched else 1.0


def validate_morphology_rho(value: float) -> float:
    rho = float(value)
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError("morphology_rho must be a finite, non-negative number.")
    return rho


def endpoint_morphology_penalty(
    position: int,
    morpheme_spans: Sequence[EncodedSpan],
) -> float:
    position = int(position)
    for left, right in morpheme_spans:
        if position == left or position == right:
            return 0.0
        if left < position < right:
            distance = min(position - left, right - position)
            return 2.0 * distance / (right - left)
    return 0.0


def edge_morphology_penalty(
    start: int,
    end: int,
    analyses: Sequence[MorphologicalAnalysis],
) -> float:
    if not analyses:
        return 0.0
    return min(
        endpoint_morphology_penalty(start, analysis.morpheme_spans)
        + endpoint_morphology_penalty(end, analysis.morpheme_spans)
        for analysis in analyses
    )


def default_celex_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "CELEX_V2"


def _celex_files(celex_dir: Optional[str]) -> Tuple[Path, Path]:
    root = Path(celex_dir).expanduser().resolve() if celex_dir else default_celex_dir()
    lemma_file = root / "english" / "eml" / "eml.cd"
    wordform_file = root / "english" / "emw" / "emw.cd"
    missing = [str(path) for path in (lemma_file, wordform_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "English CELEX morphology is enabled, but required files are missing: "
            + ", ".join(missing)
        )
    return lemma_file, wordform_file


def _edit_prefix_costs(source: str, target: str) -> List[int]:
    previous = list(range(len(target) + 1))
    for source_char in source:
        current = [previous[0] + 1]
        for target_index, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous


def _project_boundaries(
    source: str,
    source_boundaries: Sequence[int],
    target: str,
) -> Optional[BoundaryAnalysis]:
    """Project source split points to target with deterministic edit alignment."""
    if not source_boundaries:
        return ()
    if source == target:
        return tuple(int(boundary) for boundary in source_boundaries)

    projected = []
    for boundary in source_boundaries:
        prefix_costs = _edit_prefix_costs(source[:boundary], target)
        reversed_suffix_costs = _edit_prefix_costs(
            source[boundary:][::-1], target[::-1]
        )
        candidates = []
        for target_boundary in range(1, len(target)):
            suffix_cost = reversed_suffix_costs[len(target) - target_boundary]
            candidates.append(
                (
                    prefix_costs[target_boundary] + suffix_cost,
                    suffix_cost,
                    -target_boundary,
                    target_boundary,
                )
            )
        if not candidates:
            return None
        projected.append(min(candidates)[-1])

    if any(left >= right for left, right in zip(projected, projected[1:])):
        return None
    return tuple(projected)


_CLASS_LABEL = re.compile(r"\[[^]]*\]")


def _deep_morphemes(structure: str) -> List[str]:
    without_classes = _CLASS_LABEL.sub("", structure)
    flat = without_classes.replace("(", "").replace(")", "")
    return [part for part in flat.split(",") if part]


def _lemma_analysis(head: str, structure: str) -> Optional[BoundaryAnalysis]:
    if not structure:
        return ()
    morphemes = _deep_morphemes(structure)
    if not morphemes:
        return None
    source = "".join(morphemes)
    source_boundaries = []
    cursor = 0
    for morpheme in morphemes[:-1]:
        cursor += len(morpheme)
        source_boundaries.append(cursor)
    return _project_boundaries(source, source_boundaries, head)


def _apply_inflection(lemma: str, transformation: str) -> Optional[Tuple[str, Optional[int]]]:
    if not transformation.startswith("@"):
        return None
    stem = lemma
    addition = ""
    operations = transformation[1:]
    cursor = 0
    while cursor < len(operations):
        operator = operations[cursor]
        if operator not in "+-":
            return None
        cursor += 1
        end = cursor
        while end < len(operations) and operations[end] not in "+-":
            end += 1
        operand = operations[cursor:end].replace("@", "'")
        if operator == "-":
            if not operand or not stem.endswith(operand):
                return None
            stem = stem[:-len(operand)]
        else:
            addition += operand
        cursor = end
    boundary = len(stem) if addition else None
    return stem + addition, boundary


class EnglishCelex:
    def __init__(
        self,
        analyses: Mapping[str, Iterable[BoundaryAnalysis]],
        known_surfaces: Iterable[str],
        projection_failures: Iterable[str],
    ):
        self.analyses: Dict[str, FrozenSet[BoundaryAnalysis]] = {
            surface: frozenset(tuple(boundaries) for boundaries in boundary_sets)
            for surface, boundary_sets in analyses.items()
        }
        self.known_surfaces = frozenset(known_surfaces)
        self.projection_failures = frozenset(projection_failures)
        folded = defaultdict(set)
        for surface, boundary_sets in self.analyses.items():
            key = surface.casefold()
            folded[key].update(
                (surface, boundaries) for boundaries in boundary_sets
            )
        self._folded_analyses = {
            key: frozenset(entries) for key, entries in folded.items()
        }
        self._folded_known_surfaces = frozenset(
            surface.casefold() for surface in self.known_surfaces
        )
        self._decoder = ByteLevel()

    @classmethod
    def load(cls, celex_dir: Optional[str] = None) -> "EnglishCelex":
        lemma_file, wordform_file = _celex_files(celex_dir)
        analyses = defaultdict(set)
        lemma_entries = defaultdict(list)
        known_surfaces: Set[str] = set()
        projection_failures: Set[str] = set()

        with lemma_file.open("r", encoding="ascii", newline="") as handle:
            reader = csv.reader(handle, delimiter="\\")
            for fields in reader:
                if len(fields) < 2:
                    continue
                lemma_id, head = fields[0], fields[1]
                structure = fields[21] if len(fields) > 21 else ""
                known_surfaces.add(head)
                boundaries = _lemma_analysis(head, structure)
                if boundaries is None:
                    projection_failures.add(head)
                    continue
                analyses[head].add(boundaries)
                lemma_entries[lemma_id].append((head, boundaries))

        with wordform_file.open("r", encoding="ascii", newline="") as handle:
            reader = csv.reader(handle, delimiter="\\")
            for fields in reader:
                if len(fields) < 4:
                    continue
                surface, lemma_id = fields[1], fields[3]
                transformation = fields[5] if len(fields) > 5 else ""
                known_surfaces.add(surface)
                entries = lemma_entries.get(lemma_id, ())
                added = False
                for lemma, lemma_boundaries in entries:
                    transformations = transformation.split()
                    if not transformations or transformation == "IRR":
                        projected = _project_boundaries(lemma, lemma_boundaries, surface)
                        if projected is not None:
                            analyses[surface].add(projected)
                            added = True
                        continue
                    for variant in transformations:
                        inflected = _apply_inflection(lemma, variant)
                        if inflected is None:
                            continue
                        generated, inflection_boundary = inflected
                        if generated != surface:
                            continue
                        stem_end = inflection_boundary if inflection_boundary is not None else len(surface)
                        projected = _project_boundaries(
                            lemma, lemma_boundaries, surface[:stem_end]
                        )
                        if projected is None:
                            continue
                        boundaries = list(projected)
                        if inflection_boundary not in (None, 0, len(surface)):
                            boundaries.append(inflection_boundary)
                        boundaries = tuple(sorted(set(boundaries)))
                        if all(0 < boundary < len(surface) for boundary in boundaries):
                            analyses[surface].add(boundaries)
                            added = True
                if entries and not added and surface not in analyses:
                    projection_failures.add(surface)

        return cls(analyses, known_surfaces, projection_failures)

    def _surface_candidate(self, decoded: str) -> Tuple[Optional[str], int]:
        candidates = [(decoded, 0)]
        if decoded and not decoded[0].isalnum():
            candidates.append((decoded[1:], len(decoded[0].encode("utf-8"))))
        for surface, byte_offset in candidates:
            if surface in self.analyses or surface in self.known_surfaces:
                return surface, byte_offset
        return candidates[-1]

    @staticmethod
    def _encoded_analysis(
        surface: str,
        boundaries: BoundaryAnalysis,
        byte_offset: int,
    ) -> Optional[MorphologicalAnalysis]:
        non_increasing = any(
            left >= right for left, right in zip(boundaries, boundaries[1:])
        )
        out_of_range = any(
            boundary <= 0 or boundary >= len(surface)
            for boundary in boundaries
        )
        if non_increasing or out_of_range:
            return None
        byte_positions = [
            byte_offset + len(surface[:boundary].encode("utf-8"))
            for boundary in boundaries
        ]
        positions = [
            byte_offset,
            *byte_positions,
            byte_offset + len(surface.encode("utf-8")),
        ]
        spans = tuple(zip(positions, positions[1:]))
        return MorphologicalAnalysis(spans)

    def match_pretoken(self, pretoken: str) -> PretokenMorphology:
        try:
            decoded = self._decoder.decode([pretoken])
        except Exception:
            return PretokenMorphology(pretoken, True, "decode_failure", ())

        surface, byte_offset = self._surface_candidate(decoded)
        if not surface or any(character.isspace() for character in surface):
            return PretokenMorphology(
                decoded, True, "unsupported_nonlexical_form", ()
            )

        exact_analyses = {
            analysis
            for boundaries in self.analyses.get(surface, ())
            for analysis in (self._encoded_analysis(surface, boundaries, byte_offset),)
            if analysis is not None
        }
        if exact_analyses:
            return PretokenMorphology(
                decoded,
                False,
                "exact_celex_match",
                tuple(sorted(exact_analyses, key=lambda item: item.morpheme_spans)),
            )

        folded_entries = self._folded_analyses.get(surface.casefold(), ())
        folded_analyses = set()
        for source_surface, source_boundaries in folded_entries:
            projected = _project_boundaries(
                source_surface, source_boundaries, surface
            )
            if projected is None:
                continue
            analysis = self._encoded_analysis(surface, projected, byte_offset)
            if analysis is not None:
                folded_analyses.add(analysis)
        if folded_analyses:
            return PretokenMorphology(
                decoded,
                False,
                "casefold_celex_match",
                tuple(sorted(folded_analyses, key=lambda item: item.morpheme_spans)),
            )

        has_celex_candidate = (
            surface in self.known_surfaces
            or surface.casefold() in self._folded_known_surfaces
        )
        reason = (
            "boundary_projection_failure"
            if has_celex_candidate
            else "no_celex_entry"
        )
        return PretokenMorphology(decoded, True, reason, ())


def write_unmatched_report(rows: Iterable[Mapping[str, object]], output_path: str) -> int:
    unmatched = [
        (
            str(row["pretoken"]),
            str(row["celex_decoded_form"]),
            int(row["frequency"]),
            str(row["celex_unmatched_reason"]),
        )
        for row in rows
        if bool(row["celex_unmatched"])
    ]
    unmatched.sort(key=lambda item: (-item[2], item[0]))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pretoken", "decoded_form", "frequency", "reason"])
        writer.writerows(unmatched)
    temporary.replace(path)
    return len(unmatched)
