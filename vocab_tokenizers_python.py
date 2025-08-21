from transformers import PreTrainedTokenizer, AutoTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path
import json
import os


class ByteLevelPreTrainedTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, base_model="EleutherAI/pythia-70m-deduped",
                 revision="step3000", cache_dir="./pythia-70m-deduped/step3000",
                 unk_token="[UNK]", pad_token="[PAD]", eos_token="<|endoftext|>"):

        vocab_file = Path(vocab_file)
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found at {vocab_file}")

        # Load vocab
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Build reverse vocab
        self.ids_to_tokens = {id_: tok for tok, id_ in self.vocab.items()}

        # Load Eleuther pretokenizer (byte-level)
        base_tok = AutoTokenizer.from_pretrained(
            base_model, revision=revision, cache_dir=cache_dir
        )
        self.pretokenizer = base_tok.backend_tokenizer.pre_tokenizer
        self.decoder = ByteLevelDecoder(add_prefix_space=True)

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        """Split text into tokens using EleutherAI’s pretokenizer"""
        splits = []
        for token, _ in self.pretokenizer.pre_tokenize_str(text):
            splits.append(token if token in self.vocab else self.unk_token)
        return splits

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return self.decoder.decode([(tok, (0, 0)) for tok in tokens])

    def save_vocabulary(self, save_directory, filename_prefix=None):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        out_path = save_directory / ((filename_prefix or "") + "vocab.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (str(out_path),)


def bytelevel_vocab_to_tokenizer(vocab_path: str, vocab_size: int, raw_dataset_path: str, save_dir: str):
    """Construct and save a slow Python Hugging Face tokenizer."""
    hf_tokenizer = ByteLevelPreTrainedTokenizer(vocab_file=vocab_path)

    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{raw_dataset_path}")
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face slow tokenizer at {save_path}")


if __name__ == '__main__':
    dataset_path = "finewebedu_data"
    save_dir = "tokenizers"

    vocab_4096="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_4096.json"
    vocab_8192="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_8192.json"
    vocab_32768="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_32768.json"

    bytelevel_vocab_to_tokenizer(vocab_4096, 4096, dataset_path, save_dir)
    bytelevel_vocab_to_tokenizer(vocab_8192, 8192, dataset_path, save_dir)
    bytelevel_vocab_to_tokenizer(vocab_32768, 32768, dataset_path, save_dir)
