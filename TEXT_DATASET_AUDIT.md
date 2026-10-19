# Supplied text datasets: audit and external diagnostic

Audited locally on 18 October 2026. No inference API was used. Archive contents were treated as data, with no execution of files or embedded instructions.

`News-_dataset.zip` adds **no new news articles**. Its ZIP container has a different fingerprint from `data/isot.zip`, but both contained CSV files are byte-for-byte identical:

| Member | SHA-256 |
| --- | --- |
| Fake.csv | `bebf8bcfe95678bf2c732bf413a2ce5f621af0102c82bf08083b2e5d3c693d0c` |
| True.csv | `ba0844414a65dc6ae7402b8eee5306da24b6b56488d6767135af466c7dcb2775` |

The supplied ZIP SHA-256 is `1846d4ca7abfedb53a4f7c233489d23a0e6c8bb79e7f56fb8345c4faceae5268`; the previous ZIP SHA-256 is `38bf7e046c3a59b2c746c38f2fb74f4603493c6ed9c997f2f2ddb3969705fb42`. Repacking a dataset does not create an independent benchmark. The previous 98.10% ISOT test accuracy cannot be reported as a new improvement from these files.

The LIAR archive contains genuine new English political-claim data. Its SHA-256 is `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`. The [original paper](https://aclanthology.org/P17-2067/) describes the task; the supplied archive README restricts dataset use to research and says original publishers retain copyright. Those dataset terms are separate from the paper's publication license.

| Original label | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| pants-fire | 839 | 116 | 92 |
| false | 1,995 | 263 | 249 |
| barely-true | 1,654 | 237 | 212 |
| half-true | 2,114 | 248 | 265 |
| mostly-true | 1,962 | 251 | 241 |
| true | 1,676 | 169 | 208 |
| Total | 10,240 | 1,284 | 1,267 |

The archive has short statements and metadata, with **no evidence sentences or full verdict reports**. It cannot populate an evidence-verification model merely by relabeling its claims. Speaker credit-history columns include the current statement, according to the supplied README; using those fields would risk leaking the target. This audit feeds only claim text to the model.

Punctuation/case/whitespace-normalized statement auditing found 21 duplicate rows within training and one within validation; six statements overlap training/validation and five overlap training/test. IDs do not overlap. The preparation script retains test statements first, then validation, then training, and writes disjoint six-label CSVs with 10,208 / 1,283 / 1,267 rows. This only removes normalized exact duplicates; it does not establish independent speakers, events or near-duplicate statements.

A separate strict binary preparation maps `true` to real and `false` / `pants-fire` to fake. It excludes **all** partly true categories, including `mostly-true`. Prepared strict train/validation/test counts are 4,488 / 547 / 549. The original six labels are also retained so that their distinctions are not silently discarded.

## Unchanged ISOT BERT on LIAR

The existing local BERT artifact was evaluated without retraining or changing its 0.5 threshold or its ISOT-validation temperature. The complete original LIAR test set was used for per-label diagnostics. The existing ISOT splits had no identical normalized statement overlaps with that LIAR test set.

| Diagnostic mapping | Test count | Accuracy | Balanced accuracy | ROC AUC | Majority baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict: true vs false/pants-fire; partly true omitted | 549 | **56.83%** | 51.19% | 0.5433 | 62.11% |
| Directional sensitivity check: true/mostly-true vs false/pants-fire; half/barely true omitted | 790 | 47.59% | 50.83% | 0.5382 | 56.84% |

The strict diagnostic has a 95% accuracy interval of 52.65–60.91%, a Brier score of 0.3524 and expected calibration error of 0.3256. Its confusion matrix (rows and columns real, fake) is `[[58,150],[87,254]]`. It underperforms the majority baseline and shows weak discrimination on political claims. **The 98.10% historical article score does not transfer to short claims, and neither text diagnostic supports a general 90% accuracy claim.** A binary article model cannot predict LIAR's six truth levels, so six-class accuracy is intentionally not reported.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m ml.audit_text_archives --liar-archive C:\Users\roshi\Downloads\liar_dataset.zip --news-archive C:\Users\roshi\Downloads\News-_dataset.zip --threads 3
```

The script reads only named archive members, uses local model files, preserves the original labels, and writes `data/text-liar/audit.json`, per-statement prediction diagnostics, and prepared CSVs. It does not modify deployed model weights or configuration. Model-card and source-archive fingerprints are recorded for reproducibility.

## Additional evidence data

[FEVER](https://fever.ai/dataset/fever.html) is a relevant potential source for a separate claim-plus-evidence verifier. Its annotations identify evidence sentences in a dated Wikipedia snapshot; the annotations alone do not contain those sentences. Its `SUPPORTS`, `REFUTES` and `NOT ENOUGH INFO` outcomes should remain distinct. A useful experiment would require the matching snapshot, retrieval training/evaluation, an entailment model, and an independent test, rather than treating claim wording as proof. FEVER was inspected but not downloaded or represented as an installed model.

The loose supplied README describes an aggregated multi-million-row CSV that was not supplied. Its claimed strong class imbalance means a high raw accuracy could come from predicting the majority class, and overlap with ISOT could contaminate testing. No such corpus was downloaded or counted as new training data. No new Hindi/Telugu, AI-written text or dated fact-check evidence corpus was present in the supplied text files.
