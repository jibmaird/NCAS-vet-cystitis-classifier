# Synthetic data

The three CSVs in this directory (`dev-utd-predicted.csv`, `dev-all-syndromes.csv`,
`test-all-syndromes.csv`) are **entirely synthetic**. They were produced by
[`tools/generate_synthetic_data.py`](../../tools/generate_synthetic_data.py),
which composes fabricated veterinary-note text from a small set of
author-written template sentences -- no real clinical note, patient record,
or consultation ever appears here or in that script.

They exist so the rest of the pipeline (data preparation, the Stage-1 UTI
gate, Stage-2 cross-validation, the single-stage baselines, and the two-stage
evaluation) can be run and inspected end-to-end without access to the real,
private annotation data the paper's results are based on.

**Numbers produced by running this pipeline on this synthetic data are
meaningless and will not resemble the paper's reported results.** The
synthetic classes are deliberately easy to separate (distinct vocabulary per
class), so you should expect near-perfect scores -- that only demonstrates
that the code runs correctly, not that the method works.

Regenerate with different sizes/seed:

```bash
python tools/generate_synthetic_data.py --seed 1 --n_train 200 --n_dev 100 --n_test 100
```

## Using your own (real) data instead

Replace these three files with your own data sharing the same column schema:

```
consult_table_id, patient_species, patient_sex, text,
utd_spor_cyst, utd_other, utd_journey, urine_collected, cytology,
enteropathy_acute_diar, enteropathy_other, skin_sup_pyo, skin_other, other
```

Only `text`, `utd_spor_cyst`, `utd_other`, `utd_journey`, `consult_table_id`,
and `patient_species` are actually read by the scripts in this repo; the
remaining columns are kept for schema compatibility with the original
annotation format. `utd_spor_cyst == 1` marks a confirmed Sporadic Cystitis
record; `utd_other` (or `utd_journey`) holds the other-UTD sub-type label
(e.g. `FIC`, `FLUTD`, `urolithiasis`, `recurrent cystitis`,
`urethral obstruction`) for is-UTD-but-not-SC records.
