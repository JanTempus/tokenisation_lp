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

The objective is:

\[
L_{\mathrm{existing}} + \lambda\sum_c (t_c-U_c/N_c),\qquad
N_c=\sum_{e\in E_c}w_e,\quad U_c=\sum_{e\in E_c}w_e f_e.
\]

Here `w_e` is the original corpus frequency of the edge's pretoken, `t_c` is
relaxed vocabulary selection, and `f_e` is edge flow. For a fully selected token,
100 potential occurrences with 20 used incur a penalty of `0.8 * weight`.
Unselected tokens contribute zero. Tokens with the same unused fraction incur
the same penalty regardless of their total occurrence count. The denominator
uses fixed corpus counts, so the objective remains linear for relaxed selection.
Weights tuned for the previous unused-count objective may need retuning.

Only candidate non-character tokens surviving filtering contribute. Existing
morphology costs remain additive; they do not change these frequency weights.
Rounding and saved vocabulary formats are unchanged. The bias applies to the
relaxed training solution and does not guarantee better utilisation after
rounding or on unseen data. Legacy CVXPY training does not use this setting.
