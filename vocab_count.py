import json
from pathlib import Path

def print_vocab_size(tokenizer_json_path: str | Path) -> None:
    tokenizer_json_path = Path(tokenizer_json_path)

    with tokenizer_json_path.open("r", encoding="utf-8") as f:
        tok = json.load(f)

    model = tok.get("model", {})

    if "vocab" not in model:
        raise ValueError("No 'model.vocab' field found in tokenizer.json")

    vocab = model["vocab"]

    # Unigram: vocab is a list of [token, score]
    if isinstance(vocab, list):
        vocab_size = len(vocab)

    # BPE / WordPiece: vocab is a dict {token: id}
    elif isinstance(vocab, dict):
        vocab_size = len(vocab)

    else:
        raise TypeError(f"Unexpected vocab type: {type(vocab)}")

    print(f"Vocab size: {vocab_size}")

if __name__ == "__main__":
    # example usage
    print_vocab_size("/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/rounded_tokenizers/lp_1024_det/tokenizer.json")
