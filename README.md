# AI-Powered Fake News Detection and Verification System

**AI-Powered Fake News Detection and Verification System**

React frontend, Flask API, MongoDB Atlas persistence, local PyTorch visual inference, BERT text classification, local evidence retrieval, Whisper transcription, and PDF reports. Detection runs on this computer; it does not call paid inference APIs. Deployment is deferred.

## Run on this computer

From `C:\Users\roshi\OneDrive\Desktop\SR`:

```powershell
.\scripts\run.ps1
```

Open **http://127.0.0.1:5000**. The built React app and API share the same origin. Keep the terminal open; press Ctrl+C to stop.

For frontend development, run these in separate terminals:

```powershell
npm run api
npm run dev
```

Vite prints its local URL and proxies `/api` to Flask on port 5000.

On a fresh machine, install Python 3.12+ and Node.js 24, create `.venv`, install `requirements-lock.txt`, and run `npm ci --prefix frontend` followed by `npm run build`. The checked local environment is project-local and does not depend on the machine's old Python 3.7 installation.

## Configuration

The ignored file `backend/.env` holds local settings. Start with `backend/.env.example` on another machine. Real credentials are never included in the source, browser bundle, reports, or model artifacts.

| Setting | Purpose |
| --- | --- |
| `DATABASE_BACKEND=mongo` | Use MongoDB Atlas. Connection failures stop startup; there is no silent fallback. |
| `MONGODB_URI` | Atlas driver connection string; URL-encode special characters in the database password. |
| `MONGODB_DATABASE` | Database name; defaults to `veritas`. |
| `SECRET_KEY` | Long random session signing key. Required explicitly for production; development creates an ignored local key. |
| `EVIDENCE_PROVIDER=public` | Selected free mode: search supported public fact-check archives and publisher feeds. No API key or billing account is required. |
| `EVIDENCE_PROVIDER=local` | Use an administrator-built local evidence index. No external verification calls. |
| `EVIDENCE_MODEL_PATH` | Local evidence index; defaults to `models/evidence/index.joblib`. Build from real dated source records before use. |
| `GOOGLE_FACT_CHECK_API_KEY` | Optional, only used with `EVIDENCE_PROVIDER=google` or `auto`. |
| `GEMINI_API_KEY` | Optional backend-only key for concise report explanations and report Q&A from retrieved passages. It is never sent to the browser. |
| `GEMINI_MODEL` | Gemini model for report explanations and Q&A; defaults to `gemini-3.5-flash-lite`. |
| `IMAGE_MODEL_PATH` | Local trusted visual model artifact. |
| `TEXT_MODEL_PATH` | Local BERT model and tokenizer directory. |
| `AI_IMAGE_MODEL_PATH` | Local pretrained SigLIP image/frame AI scorer. Experimental; independent accuracy is not measured. |
| `WHISPER_MODEL` | Local Whisper checkpoint; installed as `models/whisper-tiny`. FFmpeg comes from `imageio-ffmpeg`. Blank disables speech. |
| `COOKIE_SECURE=true` | Required behind HTTPS in production; leave false for local HTTP. |
| `PUBLIC_BASE_URL` | Public HTTPS app URL used to build password-reset links; set it before production. |
| `MAIL_HOST`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_FROM` | SMTP delivery for password-reset email. Use a mail-provider app password, not an account password. |
| `MAIL_USE_TLS` / `MAIL_USE_SSL` | Select STARTTLS (587) or implicit TLS (465); enable only one. |
| `PASSWORD_RESET_TTL_SECONDS` | Reset-link lifetime; defaults to one hour. Tokens are single-use and stored only as hashes. |
| `DATABASE_BACKEND=sqlite` | Explicit local development/testing mode. This is not Atlas. |

Password recovery uses the configured SMTP server. Set the mail values in `backend/.env`; `/api/health` and Admin show whether delivery is configured. Production startup requires SMTP plus an HTTPS `PUBLIC_BASE_URL`. Unknown email addresses receive the same response as registered accounts.

Test Atlas without displaying the URI:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_atlas
```

This authenticates, pings, and writes/reads/deletes one temporary marker. Atlas must allow your current IP and your database user must have read/write permissions for the chosen database.

Create your own administrator:

```powershell
.\.venv\Scripts\python.exe -m flask --app backend.app:create_app create-admin
```

The command asks for a name, email, and password privately. There are no default administrator credentials in the main application. Sign in normally to see the Admin navigation item. Users cannot assign themselves admin privileges through registration.

## What works

- Registration, login, logout, profile name/email/password updates (current-password checks for email and password changes), hashed passwords, HTTP-only session cookies, CSRF protection, account disabling, and per-process request throttling.
- Private text/image/video submissions, bounded background jobs, automatic report polling, report history/search/type filtering/deletion, and PDF downloads.
- Separate text, image, and video analysis paths. Video uses a learned temporal consistency model plus the image-model frame chart; all three model cards expose measured accuracy and limitations.
- A structured evidence report with an overall verdict, up to five sentence-level claim checks for longer submissions, supporting/contradicting source IDs, retrieval status, technical warnings, and model limitations. The report panel has expandable source excerpts; the PDF contains the full retrieval details.
- Local Hugging Face BERT classification when its trained checkpoint is present. Models abstain on unvalidated scripts.
- Local evidence retrieval from dated source records, filtered to enabled trusted domains. External archive/RSS/Google/GDELT providers remain optional and are not called in local mode.
- A separate experimental image/frame AI-generation score and cached local Whisper video transcription. Spoken text is analyzed separately from frame probabilities.
- Administrator user access management, trusted-source creation/toggling, versioned dataset metadata, report monitoring, and model readiness.
- Responsive English/Hindi/Telugu controls and result explanations, sticky navigation, a signed-in sidebar for verification/reports/profile, icon-guided uploads, browser read-aloud, reduced-motion support, and a fixed-headline illustration slideshow that pauses on hover/focus.

## How to interpret results

**Fake/real percentages** are complementary class probabilities within the installed model's training task. They are not percentages of false sentences or proof of an event. **AI-generated percentage** is independent: images and sampled frames use an experimental, uncalibrated pretrained model score; writing remains unassessed. This project's independent AI-generation accuracy is not measured. **Confidence** is the model's strongest class probability after its saved calibration step; **accuracy** is measured against labeled test examples. They are not interchangeable, and calibration quality differs by model. See [EVALUATION.md](EVALUATION.md) for measured accuracy and calibration error.

**Evidence verdicts** are separate from model classifications. Text submissions are split into at most five candidate claims and each claim is searched separately. GDELT DOC 2.0 searches a rolling three-month news window; fact-check publisher archives and feeds provide additional coverage. The system fetches bounded excerpts from enabled HTTPS sources when available, then ranks them for claim relevance and rejects numeric mismatches before summarization. Gemini receives the claim-specific excerpts, source URLs and metadata and returns a schema-constrained TRUE, FALSE, or UNVERIFIED assessment with source IDs. The backend validates claim coverage, cited IDs, relevance, source type, verdicts and aggregation. Invalid responses fall back to the deterministic assessment. Without Gemini, deterministic TRUE/FALSE still requires a close match to a published fact-check with a clear rating. Conflicting reviews or insufficient evidence remain UNVERIFIED. Search results are not proof: confirm the linked original pages, especially when the report says metadata only or a page could not be retrieved.

For text claims with no retrieved excerpts, the evidence verdict remains UNVERIFIED. If Gemini responds, the report also shows a separate best-effort answer from its pretrained knowledge, clearly labeled as not source verified. That model-only answer does not change the evidence verdict.

Image and video analysis scores describe media manipulation signals only. A caption and any available transcript are checked as text against retrieved sources; this does not establish whether the media depicts the described event. Switching content type clears the previous text/caption and report so results do not carry between modalities.

The text model's English historical news benchmark cannot substantiate an 85–90% claim for current news, unseen publishers/events, Hindi/Telugu, images, videos, or AI-generated writing. Each modality and language needs an independent evaluation. The interface works in three languages; that is distinct from validated multilingual model performance.

## Datasets and reproducible training

The supplied image ZIP is read as data only. It contains 230,335 JPEGs in four fake/real groups and no text or video samples. The selected fine-tuned visual model uses group 1. Video uses a separately downloaded DeepfakeTIMIT-style set of 960 clips from paired authentic/fake sources, split into train, validation, and test identity groups. Zenodo metadata lists the DeepfakeTIMIT/VidTIMIT source archives as CC-BY-NC-ND-4.0; preserve their notices, keep this preprocessing for private research, and review the upstream terms before commercial use or model redistribution. A learned temporal model is installed, but its four-group test set is small and its 86.67% result is not a general video accuracy claim. A separate pooled-image-groups experiment is retained for comparison. The image audit removes exact duplicates and records a SHA-256 fingerprint; it does not prove identity-level, event-level, or near-duplicate independence. Video provenance, splits, license metadata, and limitations are recorded in its model card.

```powershell
.\.venv\Scripts\python.exe -m ml.audit_dataset --archive "C:\Users\roshi\Downloads\archive.zip"
$env:TORCH_HOME = Join-Path $PWD '.tools\torch'
.\.venv\Scripts\python.exe -m ml.train_image --archive "C:\Users\roshi\Downloads\archive.zip" --test 0
.\.venv\Scripts\python.exe -m ml.finetune_image --archive "C:\Users\roshi\Downloads\archive.zip" --epochs 5
.\.venv\Scripts\python.exe -m scripts.register_dataset
```

The initial model uses a frozen MobileNetV3 CNN plus a fitted head and validation temperature scaling. Fine-tuning adapts later CNN layers and the classification head; the selected image artifact averages original and horizontally flipped predictions and applies Platt calibration fitted on a held-out validation half. Model cards record split sizes, seed, confusion matrix, accuracy, F1, ROC AUC, calibration error, and confidence interval. Epochs and calibration are chosen on validation, not test. The image test split was already used to report the earlier baseline, so an independent external test remains necessary before any production accuracy claim.

Text data comes from the [ISOT corpus](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) via the [Kaggle mirror](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). The mirror lists **CC BY-NC-SA 4.0**; retain attribution and review license suitability before commercial deployment. Its real class is largely Reuters and its period/topics are limited. Preprocessing removes direct Reuters markers, URLs, HTML, short articles, conflicting labels, and duplicate leading passages before a stratified article split. Residual publisher/topic/near-duplicate bias is documented.

```powershell
# Download data/isot.zip from the Kaggle dataset endpoint, then:
.\.venv\Scripts\python.exe -m ml.prepare_text
.\.venv\Scripts\python.exe -m ml.train_text --train data/text/train.csv --validation data/text/validation.csv --test data/text/test.csv --base prajjwal1/bert-tiny --batch-size 32 --max-length 192 --epochs 3
.\.venv\Scripts\python.exe -m scripts.register_dataset --manifest data/text/manifest.json
```

The compact [BERT checkpoint](https://huggingface.co/prajjwal1/bert-tiny) makes real transformer fine-tuning practical on this CPU. The generic script also supports larger BERT checkpoints when suitable compute is available. No subject/date label columns are passed as model inputs. Reports use only locally installed checkpoints; no news text is sent to Hugging Face at inference.

The selected local BERT artifact uses the first completed epoch checkpoint (`checkpoint-823`), chosen after reaching 98.35% validation accuracy. The initial three-epoch run was stopped during the second epoch; the saved first-epoch checkpoint was calibrated and tested separately. Its actual training state and metrics are retained. To finalize a completed checkpoint explicitly:

```powershell
.\.venv\Scripts\python.exe -m ml.finalize_text --checkpoint models/text/checkpoint-823
```

## Local evidence model (optional offline mode)

`EVIDENCE_PROVIDER=local` calls no external verification service. It fits character TF-IDF features on a versioned library of real fact-check records and retrieves source matches locally. Source links, publisher ratings, dates, corpus version and build time are preserved. It is a retrieval model, not an automatic current-truth classifier. Matches need review; missing coverage remains unverified.

No production evidence corpus has been supplied yet, so the current library is explicitly **not configured**. See [DATASETS.md](DATASETS.md) for the input schema and training-data handoff. To build it when the source records arrive:

```powershell
.\.venv\Scripts\python.exe -m ml.build_evidence --input data/incoming/evidence.jsonl --version YOUR_VERSION
```

The library supports within-language text matching, including English/Hindi/Telugu scripts, without automatically translating claims. Its search quality needs evaluation on representative claim/evidence pairs. Updating the corpus is required for newer events. MongoDB Atlas still needs internet for account/report persistence; local inference does not make the entire Atlas-backed application offline.

## Optional free verification alternatives

`EVIDENCE_PROVIDER=public` needs **no Google account, API key, credit card, or billing setup**. It searches the public read-only WordPress archives of enabled Alt News and Factly sources. If no related archive result is found, it checks recent public RSS content from supported enabled publishers (BBC, Alt News, Factly). This is optional external retrieval. It returns titles, dates, original links, source availability, and a review-needed/unverified status; it does not fabricate normalized publisher ratings or automatically infer truth from search matches.

Alternative modes:

| Value | Behavior |
| --- | --- |
| `local` | Local versioned evidence retrieval; optional offline mode, requires a corpus. |
| `public` | Optional fact-check publisher archives + RSS fallback. |
| `rss` | Recent publisher feeds only. |
| `gdelt` | Keyless GDELT DOC 2.0 news search (rolling three-month window), with attribution. Related reporting, not fact-check ratings. |
| `google` | Google Fact Check Tools API; requires your key. |
| `auto` | Google if a key is configured, otherwise public archives + RSS. |
| `none` | Explicit offline verification mode; model inference still works. |

Public endpoints may change, rate-limit, or become unavailable. Archive coverage is limited to connected publishers; RSS coverage is limited to recent feed items. Responses are cached for ten minutes. GDELT timed out in the live environment test, so it is an optional provider rather than the default. The public archive and RSS endpoints responded successfully without credentials. WordPress archive search uses its [documented public posts search](https://developer.wordpress.org/rest-api/reference/posts/).

When `EVIDENCE_PROVIDER=gdelt` is selected and GDELT is unreachable, the backend now falls back to the same free fact-check archives and RSS feeds used by `public`. Results identify which providers responded; matching articles still require review and do not automatically verify a claim.

Check a free provider:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_evidence --provider public --query "NASA moon landing fake"
.\.venv\Scripts\python.exe -m scripts.check_evidence --provider rss
```

Admin source toggles apply to the free providers too. Public archive API calls go only to fixed supported publisher endpoints; admin-entered URLs are not fetched by the server. English keywords currently work best. Automatic Hindi/Telugu-to-English translation is not installed.

## Test evidence verdicts

Run the offline assessment tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_report_summary.py -q
```

These tests use clearly labeled mock fact-check records to cover TRUE, FALSE, conflicting ratings, stale records for current claims, insufficient evidence, multi-claim submissions, Gemini failures, and citation validation. They do not represent live source results. In the running app, a source search can return UNVERIFIED even for a well-known fact when its configured providers have no close-matching fact-check record. That result means the configured evidence did not settle the claim; it is not an estimate of truth.

## Optional Google Fact Check setup

Google is not required for the default local verification workflow. To select it, set `EVIDENCE_PROVIDER=google` and:

1. In your Google Cloud project, enable [Fact Check Tools API](https://console.cloud.google.com/apis/library/factchecktools.googleapis.com).
2. Open [Credentials](https://console.cloud.google.com/apis/credentials), create an API key, and restrict it to Fact Check Tools API.
3. Set `GOOGLE_FACT_CHECK_API_KEY` in `backend/.env`, then restart the server.

Run `.\.venv\Scripts\python.exe -m scripts.check_factcheck` to check the key without displaying it.

The backend implements the [official claims search endpoint](https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search). Google Cloud account/project access is required to create a key; it cannot be created by this repository itself.

## Tests and browser QA

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npm run doctor
npm run build
# Separate terminal, isolated SQLite data and QA-only credentials:
.\.venv\Scripts\python.exe -m scripts.qa_server
# Then:
cd frontend
npx playwright test
```

Playwright uses the installed Microsoft Edge executable on this Windows machine. Screenshots and the sample PDF go into ignored `artifacts/`. Browser QA never creates a default admin in the Atlas database. Backend tests use isolated temporary databases and do not contact Atlas or fact-check services. A separate Atlas round-trip check tests the actual database connection.

The live free-provider browser check is opt-in. Start `python -m scripts.qa_server --evidence-provider public`, then run `$env:VERITAS_LIVE_EVIDENCE='1'; npx playwright test live-evidence.spec.js` from `frontend/`. It exercises real public archive search, source links, source availability, and PDF download using isolated local accounts.

See [EVALUATION.md](EVALUATION.md) for the measured accuracy shortfall and what remains unvalidated. The text classifier met the corpus-specific target; the current visual classifier did not.

## Repository layout

```text
frontend/        React interface, translations, responsive CSS, browser tests
backend/         Flask routes, authentication, storage, inference orchestration, evidence, PDF
ml/              ZIP audit, text preparation, CNN/BERT training, shared visual inference
scripts/         Launch, Atlas diagnostics, dataset registration, isolated browser QA
tests/           API, privacy, media, admin, and verification tests
models/          Ignored local model artifacts and evaluation cards
data/            Ignored manifests, training CSVs, local development runtime data
```

## Deployment preparation

The supplied Dockerfile builds React and serves it with Flask through Waitress. Mount trusted model artifacts read-only at `/app/models`; provide secrets through environment variables. Set `APP_ENV=production`, Atlas configuration, a strong `SECRET_KEY`, and `COOKIE_SECURE=true`; terminate HTTPS at your reverse proxy. The application intentionally binds only to loopback in the local launch script.

The current background executor is for **one application process**. On restart, unfinished reports are marked failed; they are not falsely marked complete. Before horizontal scaling, move jobs to a durable queue such as Redis/RQ or Celery, use shared rate limits, add monitoring/backups and retention policies, and perform an external model evaluation. Raw uploads are private temporary files and are deleted after each completed/failed job; reports persist until deleted. Unexpected hard process termination can leave OS temporary files, so production needs scheduled temporary-file cleanup.

PDFs currently provide an English summary; original Unicode text remains available in the web report. Browser read-aloud depends on installed voices. A learned temporal deepfake model, validated AI-writing detector, and validated Hindi/Telugu news classifier are not installed. Local Whisper tiny and the separate experimental SigLIP AI-image scorer are installed on this machine; they do not establish an overall accuracy guarantee.

To recreate the local model downloads on another machine:

```powershell
.\.venv\Scripts\python.exe -m scripts.setup_generation
.\.venv\Scripts\python.exe -m scripts.setup_speech --configure
```

These commands download pinned snapshots of public weights; inference uses local files afterward. The AI-image source is [Ateeqq/ai-vs-human-image-detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (Apache-2.0; publisher notes overfitting concerns). Speech uses the official [OpenAI Whisper tiny](https://huggingface.co/openai/whisper-tiny) checkpoint. No external model inference service is called.
