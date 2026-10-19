Model artifacts are generated locally, never fabricated or committed with credentials.

The selected local visual artifact is `image-v2/model.joblib` plus `image-v2/backbone.pt`. It contains the fine-tuned CNN configuration and calibrated inference metadata. Its measured test accuracy is 73.90%, below the requested 85% target, and the interface marks it experimental. The initial `image/` baseline and pooled `image-all/` experiment are retained for review. See `../EVALUATION.md` for all results.

`image/model.joblib` contains a fitted scikit-learn head, a calibration temperature, and an evaluation card. `image/backbone.pt` contains the CNN weights. `image/model-card.json` records the audited dataset fingerprint, split sizes, measured metrics, calibration error, and limitations. Only load artifacts created by a trusted administrator; joblib is not a safe format for untrusted uploads.

`text/` must contain a local Hugging Face checkpoint, tokenizer, and `model-card.json` declaring `fake_label`. `ml.train_text` produces these after training on labeled text CSVs. No text training data was included in the supplied ZIP.

The visual model does not establish AI generation, news truth, or universal deepfake detection. Video inference reuses its frame predictions; the implemented temporal score variation is descriptive, not a learned temporal detector.
