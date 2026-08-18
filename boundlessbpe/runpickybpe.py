from boundlessbpe import PickyBPE
import time

filepath = "data/minipile.jsonl"

num_lines =  1000000 # 100000   # now stop due to max_bytes of 1GB
vocab_size = 131072 # 1000 # 131072 # 500 # 40960 # 256 + 50, 131072 ~ 128k
tau = 0.9 # deletion threshold 

# note if you set tau > 1, then this code can just be used as a regular BPE
# tau = 1.1 # deletion threshold 

print("tau:", tau)  # TODO: have two values for each
outprefix = f"./models/pickybpe_0.9_{num_lines}_{vocab_size}_{tau}"
print("outprefix:", outprefix)
recalc = 100

tokenizer = PickyBPE()
tokenizer.train(filepath, outprefix, num_lines, vocab_size, recalc)
tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
# print(tokenizer.encode("<|endoftext|>hello world", allowed_special="all"))

print("saving")
tokenizer.save(outprefix)

# make sure it loads
print("loading")
tokenizer2 = PickyBPE()
tokenizer2.load(outprefix + ".model")

print("done")
    
# python -u runpickybpe.py 2>&1 | tee logfile_pickbpe_0.9_1GB.txt
