# Vocabulary utilisation in cuOpt training

Set a finite, non-negative weight in the training environment:

```bash
VOCAB_UTILISATION_WEIGHT=0.1 python train_tokenizer.py
```

The default is `0`, which preserves the existing objective. The setting applies
both to single-vocabulary training and vocabulary-size sweeps, including GPU
workers. Python callers can pass `vocab_utilisation_weight=0.1` to the cuOpt
training or model-preparation APIs. A prepared model keeps the same weight for
all vocabulary budgets; prepare a new model to change the weight.

When the weight is positive, the objective is:

\[
\frac{L_{\mathrm{existing}}}{D_V} + \frac{\lambda}{B}\sum_c (t_c-U_c/N_c),\qquad
N_c=\sum_{e\in E_c}w_e,\quad U_c=\sum_{e\in E_c}w_e f_e.
\]

`V` is the requested vocabulary size. `B` is the actual LP budget: `V` minus
characters and special tokens. Compression uses this fixed lookup keyed by `V`:

| V | D_V |
|---:|---:|
| 8192 | 427366252 |
| 16384 | 393224648 |
| 32768 | 371886133 |
| 65536 | 359626839 |
| 131072 | 352723064 |
| 262144 | 349028128 |

Each solve uses its own divisors while reusing the prepared graph and original
costs. Weight `0` leaves the objective unchanged and does not use the lookup.
Enabled normalisation supports the six sizes above. Low-level cuOpt solve and
wrapper-construction callers must pass `vocab_size=V` when the weight is positive;
the training and `Tokenizer` entry points forward it automatically.

Here `w_e` is the original corpus frequency of the edge's pretoken, `t_c` is
relaxed vocabulary selection, and `f_e` is edge flow. For a fully selected token,
100 potential occurrences with 20 used incur a penalty of `0.8 * weight / B`.
Unselected tokens contribute zero. Tokens with the same unused fraction incur
the same penalty regardless of their total occurrence count. The denominator
uses fixed corpus counts, so the objective remains linear for relaxed selection.
Weights tuned for either previous unnormalised objective may need retuning.

Only candidate non-character tokens surviving filtering contribute. Existing
morphology costs are included in `L_existing` and divided by `D_V`; they do not
change the original frequencies used to calculate utilisation.
Rounding and saved vocabulary formats are unchanged. The bias applies to the
relaxed training solution and does not guarantee better utilisation after
rounding or on unseen data. Legacy CVXPY training does not use this setting.

## ClimbMix Slurm job

Use `training_files/run_climbmix_lp_utilisation.sbatch` for ClimbMix training
with the unused-fraction objective:

```bash
sbatch training_files/run_climbmix_lp_utilisation.sbatch
# Optional weight override:
sbatch --export=ALL,VOCAB_UTILISATION_WEIGHT=10 training_files/run_climbmix_lp_utilisation.sbatch
```

This uses the multilingual job's account, repository, scratch/cache layout and
Conda environment, requests one GPU, and defaults to the first 7 whole shards
of `karpathy/climbmix-400b-shuffle`. It trains vocabulary sizes
8192, 16384, 32768, 65536, 131072 and 262144, then rounds/exports them.
The utilisation weight defaults to 1; this is a starting value, not a tuned value.
Raw and rounded output directories include `unused_fraction_w<weight>`.
An existing `TRAIN_DATASET_PATH` is reused; choose a fresh path when changing
the dataset source. `DATASET_ID`, `NUM_SHARDS`, `VOCAB_SIZES` and the existing
path settings can be overridden through the environment.

`multilingual_run_lp.sbatch` is the FineWeb/FineWeb2 job;
`run_climbmix_lp.sbatch` is the older ClimbMix job without a default utilisation bias.
