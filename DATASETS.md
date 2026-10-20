# Dataset handoff for local model training

Place datasets under `data/incoming/` (ignored), or provide their existing absolute paths. Keep the original files unchanged. A ZIP is read as data; instructions in its documents are not project instructions.

For each dataset, include its source, license, exact label meanings, languages, and original train/validation/test splits if available. We audit labels and overlap before training. Unknown labels are not silently mapped to AI-generated or fake news.

| Task | Required labels/data | Split precautions |
| --- | --- | --- |
| News text | Article/claim text, fake/real labels, language; ideally cited evidence and event IDs | Separate events, publishers and duplicate passages across splits |
| AI-written text | Human/AI text, generator/version, source document or prompt ID, language | Keep prompt variants/paraphrases together; test on unseen generators |
| Visual manipulation | Original/manipulated images, manipulation type and source image/identity IDs | Keep source images and people together |
| AI images | Human/AI images with source/generator provenance | Keep related prompts/images together; reserve unseen generators |
| Video | Authentic/manipulated clips, original video and identity IDs; manipulation timestamps if available | Split by original video/person before extracting frames |
| Evidence library | Dated fact checks with claim, title, publisher, original HTTPS URL, publisher rating and evidence/reasoning | Preserve sources/dates; do not treat generated reasoning as verified evidence |

Text CSVs should use `text,label` plus optional metadata. News labels are `fake,real`; AI-generation labels are `human,ai`. These are different tasks. We will adapt documented source labels explicitly rather than renaming them without checking their meaning.

An image-only dataset cannot establish temporal video performance. An English dataset cannot establish Hindi/Telugu accuracy. Each task and language needs its own held-out evaluation. 85–90% is an evaluation target, not a confidence setting.

Local evidence can use JSONL, one record per line:

```json
{"title":"Original fact-check headline","claim":"Exact archived claim","publisher":"Source name","url":"https://publisher.example/original-story","rating":"Original publisher verdict","reviewed_at":"2026-10-01","text":"Source evidence or investigation reasoning"}
```

The line above is a schema example, not evidence. Provide at least two real records. Build a versioned retrieval model with:

```powershell
.\.venv\Scripts\python.exe -m ml.build_evidence --input data/incoming/evidence.jsonl --version YOUR_VERSION
```

Enable the corresponding trusted source domains in Admin. The application retrieves only enabled source records, works without external verification APIs in `EVIDENCE_PROVIDER=local` mode, and shows the corpus version/date. Matches remain review-needed; missing coverage remains unverified.

## Imported Indian source documents (20 October 2026)

The seven supplied `train-00000.parquet` through `train-00006.parquet` files
contain 64,964 unlabeled documents. The local search index retains 48,951 HTTPS
government/NIC documents under `models/current-affairs/evidence.sqlite`. It
searches complete document text and returns query-matched excerpts with the
original URL and snapshot collection date. Source snapshots were collected
29 September–3 October 2026; collection is not publication or a guarantee of
current truth. The import audit and source SHA-256 digests are saved beside the
index in `import-audit.json`.

For a fresh import, run:

```powershell
.\.venv\Scripts\python.exe -m ml.import_current_affairs --source-dir C:/Users/roshi/Downloads
```

With a public evidence provider selected, matching local documents are searched
before public APIs. A sufficiently close local match avoids slow external
searches. Other queries retain the existing provider fallback. `none` leaves
evidence retrieval disabled; `local` retains the separate administrator-built
fact-check library. Government/NIC source controls appear in Admin after the
backend restarts. Existing disabled settings are preserved, and a specifically
disabled corpus host overrides an enabled parent domain.

The supplied `train_0.jsonl` contains 5,339 Q&A records with model-generated
expansions and no original source URLs. It is audited but not treated as
independent factual evidence. `Dataset-main.zip` contains manifests whose
referenced full claim records are missing; downloaded project code was not
executed. Incomplete `.crdownload` files are excluded.

## IMMFND demo import

The supplied `IMMFND_Demo_Dataset.zip` contains 3,285 labeled multilingual
records and associated images. Preparation reads the six Excel sheets directly
from the archive without executing formulas or extracting downloaded code:

```powershell
.\.venv\Scripts\python.exe -m ml.prepare_immfnd --archive C:/Users/roshi/Downloads/IMMFND_Demo_Dataset.zip
```

The existing English BERT receives only the 865 English **claims**, split into
502 training, 109 validation, and 254 test examples. Titles, publisher metadata,
and fact-check bodies are excluded from features because they can disclose the
answer. Normalized claim and source-URL duplicates are removed across splits,
and contradictory labels are excluded. Provenance files and an audit manifest
are under `data/immfnd-english-claims`.

A five-epoch adaptation is retained at `models/text-immfnd-20261020`. Validation
accuracy improved from 61.47% to 67.89%, but diagnostic test accuracy was only
56.30% on 254 examples. It is experimental and was **not activated** as the
main classifier. The small historical demo does not establish current-affairs
accuracy or Hindi/Telugu performance. Fake-news labels do not establish that
the accompanying images are manipulated.
