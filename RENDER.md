# Deploy from GitHub to Render

[Deploy to Render](https://render.com/deploy?repo=https://github.com/roshitha427/news-verification-app)

This setup builds the React frontend and Flask API together using the Dockerfile.
The deploy link opens setup; it is not a live application URL. After a successful
deployment, Render provides the public HTTPS URL in its dashboard.

The Blueprint requests a paid Standard service and a 20 GB model disk. Review
the current price before applying. Keep one instance because analysis jobs and
rate limits currently live in one process. Monitor memory during concurrent
inference and increase compute if required.

1. Connect Render to the GitHub repository and open the deploy link.
2. Provide MongoDB and SMTP settings from the local `backend/.env` through
   Render's secret environment settings. Never commit that file. Match the mail
   port and TLS/SSL settings to the mail provider. Set `PUBLIC_BASE_URL` to the
   service's actual HTTPS URL before starting production. Set `GEMINI_API_KEY`
   separately if report summaries and Q&A should use Gemini.
3. Allow the service's outbound IPs in MongoDB Atlas Network Access.
4. Populate the persistent `/app/models` disk through Render SSH with trusted
   local artifacts: `image-v2`, `text-continued-20261020`, `video-temporal`,
   `ai-image`, `ai-text`, `whisper-tiny`, and `current-affairs`. Preserve relative
   directory names. Transfer only the selected model's final weights, tokenizer,
   configuration and model cards; training checkpoints and raw datasets are not
   needed. The source repository intentionally contains no trained weights.
5. Restart after transferring models. Check `/api/health` for model readiness,
   then verify registration, login, an analysis, PDF download and reset email.

A successful build alone does not install the private models. Missing models
remain unavailable in the app. Do not share the app as fully working until the
model readiness and end-to-end checks pass. This Docker image listens on port
5000 on all interfaces, which Render can route to over HTTPS.

See [Render's Blueprint documentation](https://render.com/docs/blueprint-spec)
for the hosting configuration format.
