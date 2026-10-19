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
