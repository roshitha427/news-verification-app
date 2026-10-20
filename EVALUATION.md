# Measured model performance

These are local dataset evaluations, not accuracy guarantees for real-world claims. The interface separates model classification from evidence verification.

| Model | Training examples | Test examples | Accuracy | F1 | 90% target |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial frozen CNN + calibrated head, group 1 | 8,000 | 5,227 | 71.76% | 0.7503 | Not met |
| Fine-tuned MobileNetV3 with flip averaging + validation calibration, group 1 | 38,936 | 5,227 | **76.09%** | 0.7919 | **Not met** |
| Learned temporal consistency classifier, DeepfakeTIMIT-style dataset | 480 clips | 240 | **86.67%** | 0.8954 | **Not met** |
| Pooled four-group frozen CNN experiment | 32,000 | 20,905 | 64.94% | 0.6693 | Not met |
| Native-resolution residual/frequency features, validation-selected head | 16,000 | 5,227 | 61.20% | 0.6893 | Not met; not selected |
| Original one-epoch compact BERT, English ISOT articles | 26,316 | 5,640 | **98.10%** | 0.9788 | Met on this corpus |
| Continued compact BERT, three total epochs, English ISOT articles | 26,316 | 5,640 | **98.48%** | 0.9829 | Met on this corpus |
| IMMFND English claim-only adaptation, experimental | 502 | 254 | 56.30% | 0.5316 | Not met; not activated |

On 20 October 2026, two additional epochs at learning rate 0.00001 improved the
news model's validation accuracy from 98.35% to 98.63% and validation F1 from
0.9816 to 0.9847. The selected candidate is retained under
`models/text-continued-20261020`; `backend/.env` selects it after restart. The
original model remains under `models/text`. Diagnostic test accuracy improved
from 98.10% to 98.48% on the same 5,640 articles. This repeatedly inspected test
is not an independent current-news benchmark. The new Brier score is 0.0127,
expected calibration error 0.0062, and 95% accuracy interval 98.12–98.76%.

The separate IMMFND claim experiment improved validation accuracy from 61.47%
to 67.89%, but its 56.30% test accuracy does not justify replacing the news
model. It is retained for research, with the supplied splits and source hashes.
Current-affairs verification instead uses the imported official-source corpus.
The supplied Andhra Pradesh CM claim returned FALSE with validated official
source citations in a live Gemini smoke check; this single example is a
functional check, not a measured current-affairs accuracy benchmark.

The fine-tuned CNN with horizontal-flip test-time averaging and Platt calibration reached 76.09% accuracy on the diagnostic test split, up from 73.90% for the earlier inference setup. Balanced accuracy rose from 73.83% to 76.03%; Brier score improved from 0.2010 to 0.1855 and expected calibration error from 0.1519 to 0.1370. Real-class recall is still only 61.41%, and the model misses 1,005 of 2,604 real examples. The model remains experimental and below its 85% target. The calibration and flip averaging were selected using validation data, but this same test split was already used for the initial baseline, so an independent external set is still needed.

The temporal classifier scored 86.67% accuracy and 87.19% balanced accuracy on 240 held-out clips; its majority-class baseline is 66.67%. Its expected calibration error is 0.0806. The ordinary 95% accuracy interval is 81.78–90.39%, while the source-group bootstrap interval is much wider at 73.75–96.67%. Only four source/identity groups are in the test split, so this dataset result is preliminary and does not support a general 90% video claim. Real clips are JPEG sequences and fake clips are AVI; residual source and codec differences may contribute to the score. The manifest records the upstream CC-BY-NC-ND-4.0 metadata and private-research handling note; review source terms before commercial use or model redistribution.

The pooled frozen-feature experiment was not selected: its validation performance was also substantially worse. More data alone did not resolve the image task. The current visual model still requires a stronger forensic architecture, provenance/label audit, subject/event-independent splits, and independent evaluation before it can satisfy the requested visual accuracy target.

The text model reached 98.35% validation accuracy after one completed epoch; that saved checkpoint was selected, calibrated on validation, and evaluated on test. Its 95% test accuracy interval is 97.71–98.43%, Brier score 0.0164, and expected calibration error approximately 0.0068. These results describe historical English article classification. Publisher style, topic, event, and near-duplicate biases may still inflate the score despite preprocessing. They do not establish accuracy on current claims or on Hindi/Telugu.

The visual experiments reused the group-1 diagnostic test split, so those repeated results are not independent external confirmations. The pooled experiment includes additional test groups and is also a diagnostic result. Exact-byte duplicate removal cannot establish person-level or near-duplicate independence. Full confusion matrices, calibration statistics, training history, seeds, and dataset fingerprints are recorded in the local model cards.

The additional native-resolution experiment uses 1,412 color/residual/co-occurrence/DCT/Fourier features. Six heads were compared on one validation half and the best was calibrated on the other half before diagnostic test evaluation. Validation balanced accuracy was 78.23%, but test accuracy was only 61.20%; its authentic-class recall was 36.44%. It was not selected. Its complete card is retained under `models/image-forensic`. More experimentation on this repeatedly inspected diagnostic test cannot replace an independent benchmark.

A training-sample audit also found that the supplied image archive contains portraits/face crops. The fake/real visual classifier is therefore not validated for arbitrary news photos or non-face video frames. That archive contains no license/provenance documentation or temporal video labels. Video training uses a separate DeepfakeTIMIT-style dataset with source/identity-group splits.

Video reports now include a separate learned temporal consistency score and image-model frame chart. The temporal model is experimental and limited to the older face-swap dataset described above; it is not a general video authenticity detector. A separate local pretrained [SigLIP AI-image detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector), revision `60e82406916921b823616bee33397baab38af3f0`, supplies an **experimental, uncalibrated image/frame AI score**. Its source license is Apache-2.0; the publisher notes overfitting concerns. Publisher benchmark numbers are not reported as this project's measured accuracy. Independent AI-image accuracy is **not measured**. It does not assess writing, audio, generated motion or factual truth.

Official Whisper tiny, revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, is installed locally with a bundled FFmpeg decoder. A 7.64-second English speech smoke sample transcribed successfully; no language-level word-error-rate evaluation has been performed. Transcripts feed the text/evidence pipeline separately from visual scores. Silent/no-audio videos abstain from transcription.

Local character-TF-IDF evidence retrieval is implemented and tested, including Hindi-script matching and trusted-source filtering. It requires a real dated evidence corpus before use. Retrieval similarity is not a truth probability. No production evidence corpus or retrieval-quality benchmark has been supplied yet.

AI-writing detection remains unassessed until a suitable labeled corpus is supplied. The application abstains on Hindi/Telugu news classification rather than projecting an English benchmark onto those languages.

No honest overall 85–90% accuracy claim can currently be made for the complete multimodal system.
