import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "hf_baseline_tokenizers"
    / "train_frequent_pretokens.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "train_frequent_pretokens", RUNNER_PATH
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)

NANOCHAT_SPECIAL_TOKENS = RUNNER.NANOCHAT_SPECIAL_TOKENS
build_equal_score_tokenizer = RUNNER.build_equal_score_tokenizer
collect_ranked_pretokens = RUNNER.collect_ranked_pretokens
parse_vocab_sizes = RUNNER.parse_vocab_sizes
rank_pretokens = RUNNER.rank_pretokens
select_vocabulary = RUNNER.select_vocabulary
tokenizer_kwargs_for_mode = RUNNER.tokenizer_kwargs_for_mode
train_frequency_baselines = RUNNER.train_frequency_baselines


BYTE_ALPHABET = sorted(ByteLevel.alphabet())


class FrequentPretokenSelectionTests(unittest.TestCase):
    def test_parse_vocab_sizes_sorts_and_deduplicates(self):
        self.assertEqual(parse_vocab_sizes("32768, 8192,32768"), [8192, 32768])

    def test_ranking_breaks_ties_lexicographically(self):
        ranking = rank_pretokens(
            ["beta", "alpha", "gamma"],
            [3, 3, 1],
        )
        self.assertEqual(ranking, [("alpha", 3), ("beta", 3), ("gamma", 1)])

    def test_selection_is_exact_and_skips_reserved_duplicates(self):
        ranking = [("a", 100), ("alpha", 10), ("beta", 9), ("gamma", 8)]
        vocab = select_vocabulary(
            ranking,
            vocab_size=7,
            special_tokens=["<unk>", "<bos>"],
            byte_alphabet=["a", "b"],
        )
        self.assertEqual(
            vocab,
            ["<unk>", "<bos>", "a", "b", "alpha", "beta", "gamma"],
        )
        self.assertEqual(len(vocab), len(set(vocab)))

    def test_selection_allows_a_required_tokens_only_vocabulary(self):
        vocab = select_vocabulary(
            [("extra", 10)],
            vocab_size=3,
            special_tokens=["<unk>"],
            byte_alphabet=["a", "b"],
        )
        self.assertEqual(vocab, ["<unk>", "a", "b"])

    def test_frequency_collection_calls_lp_pipeline_once(self):
        class FakeLPTokenizer:
            corpus = ["unused"]

            def __init__(self):
                self.calls = 0

            def pretokenize_and_prepare_corpus(self, corpus):
                self.calls += 1
                return {
                    "pretoken": ["beta", "alpha"],
                    "frequency": [3, 3],
                }

        tokenizer = FakeLPTokenizer()
        ranking = collect_ranked_pretokens(tokenizer)
        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(ranking, [("alpha", 3), ("beta", 3)])

    def test_special_token_roles_match_supported_modes(self):
        nanochat = tokenizer_kwargs_for_mode("nanochat")
        self.assertEqual(nanochat["bos_token"], "<|bos|>")
        self.assertEqual(nanochat["unk_token"], "<|unk|>")

        apertus = tokenizer_kwargs_for_mode("apertus")
        self.assertEqual(apertus["unk_token"], "[UNK]")
        self.assertEqual(apertus["eos_token"], "[EOS]")
        self.assertEqual(apertus["pad_token"], "[PAD]")


class EqualScoreUnigramTests(unittest.TestCase):
    @staticmethod
    def build_components():
        return (
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ByteLevelDecoder(),
        )

    def test_tokenizer_scores_roles_roundtrip_and_reload(self):
        pretokenizer, decoder = self.build_components()
        vocab = [*NANOCHAT_SPECIAL_TOKENS, *BYTE_ALPHABET, "ab"]
        tokenizer = build_equal_score_tokenizer(
            vocab,
            pretokenizer,
            decoder,
            tokenizer_kwargs_for_mode("nanochat"),
        )

        self.assertEqual(len(tokenizer), len(vocab))
        self.assertEqual(tokenizer.bos_token, "<|bos|>")
        self.assertEqual(tokenizer.unk_token, "<|unk|>")
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode("ab", add_special_tokens=False)
            ),
            ["ab"],
        )

        samples = [
            " leading space\tand tab\n",
            "naive cafe 中文 Ελληνικα العربية 😀🚀",
            "def f(x):\n    return x**2  # square\n",
            "an unseen pretoken still round-trips through byte tokens",
        ]
        for sample in samples:
            ids = tokenizer.encode(sample, add_special_tokens=False)
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), sample)

        with tempfile.TemporaryDirectory() as temporary_directory:
            save_path = Path(temporary_directory)
            tokenizer.save_pretrained(save_path)
            tokenizer_data = json.loads(
                (save_path / "tokenizer.json").read_text(encoding="utf-8")
            )
            scores = [score for _, score in tokenizer_data["model"]["vocab"]]
            self.assertEqual(scores, [-1.0] * len(vocab))

            reloaded = PreTrainedTokenizerFast.from_pretrained(save_path)
            self.assertEqual(len(reloaded), len(vocab))
            self.assertEqual(reloaded.bos_token, "<|bos|>")
            self.assertEqual(reloaded.unk_token, "<|unk|>")

    def test_sweep_collects_frequencies_once_and_writes_each_size(self):
        pretokenizer, decoder = self.build_components()

        class BackendWrapper:
            pass

        class PipelinePretokenizer:
            pass

        backend = BackendWrapper()
        backend.pre_tokenizer = pretokenizer
        backend.decoder = decoder
        pipeline_pretokenizer = PipelinePretokenizer()
        pipeline_pretokenizer.backend_tokenizer = backend

        class FakeLPTokenizer:
            calls = 0

            def __init__(self, corpus, **kwargs):
                self.corpus = corpus

            def pretokenize_and_prepare_corpus(self, corpus):
                type(self).calls += 1
                return {
                    "pretoken": ["ab", "abc", "a"],
                    "frequency": [10, 9, 100],
                }

        required_count = len(set([*NANOCHAT_SPECIAL_TOKENS, *BYTE_ALPHABET]))
        sizes = [required_count + 1, required_count + 2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_dir = Path(temporary_directory)
            train_frequency_baselines(
                dataset=["unused"],
                vocab_sizes=sizes,
                save_dir=save_dir,
                mode="nanochat",
                pipeline_pretokenizer=pipeline_pretokenizer,
                lp_tokenizer_class=FakeLPTokenizer,
                byte_alphabet=BYTE_ALPHABET,
            )

            self.assertEqual(FakeLPTokenizer.calls, 1)
            for size in sizes:
                output_dir = save_dir / f"frequent_pretoken_{size}"
                tokenizer_data = json.loads(
                    (output_dir / "tokenizer.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(tokenizer_data["model"]["vocab"]), size)


if __name__ == "__main__":
    unittest.main()
