# Deploy on Netlify

Netlify hosts the React frontend. The Flask API, background analysis jobs,
MongoDB connection, and local AI models require a separate backend server.
Uploading the source ZIP does not install or run Flask on Netlify.

## Upload the built website

1. Extract `SR-netlify-frontend.zip`.
2. Sign in at https://app.netlify.com/drop.
3. Drag the extracted folder containing `index.html`, `_redirects`, and `assets`
   into the upload area.

This publishes the frontend. Until a backend is connected, login, analysis,
and reports return a clear backend-unavailable error.

## Build from a Git repository

The root `netlify.toml` configures the base directory as `frontend`, build command
as `npm run build:netlify`, and publish directory as `dist` relative to that base.
Connect this repository in Netlify and deploy. Do not upload `backend/.env`.

## Connect the backend

Host the existing Dockerfile on a server that supports a long-running Python
process and mounted model files. Mount the trained models at `/app/models` and
provide backend environment variables through that hosting service. The
Dockerfile serves port 5000; the host must provide public HTTPS routing to it.

Production backend settings include `APP_ENV=production`, `COOKIE_SECURE=true`,
`DATABASE_BACKEND=mongo`, a random `SECRET_KEY`, MongoDB credentials, SMTP settings,
and `PUBLIC_BASE_URL=https://YOUR-SITE.netlify.app`. The backend validates these
settings on startup; see README.md and backend/.env.example for the full list.
Keep database, mail, and API credentials on the backend host.

Set `BACKEND_URL=https://YOUR-BACKEND-HOST` in Netlify's build environment and
redeploy. Use only the HTTPS origin, without an `/api` suffix. The build generates
an `/api/*` proxy before the SPA fallback so login cookies, CSRF tokens, uploads,
and PDF downloads use the same browser origin.

For a manual upload, rebuild from PowerShell after setting the backend URL:

```powershell
$env:BACKEND_URL = 'https://YOUR-BACKEND-HOST'
npm.cmd --prefix frontend run build:netlify
```

Upload `frontend/dist` to Netlify. A previously generated ZIP is not updated by
this command. Without `BACKEND_URL`, the build deliberately returns HTTP 404 for
API requests instead of returning the frontend HTML as an API response.

## Verify after connecting

Open `/api/health` on the Netlify URL and check for a successful database status.
Then register or sign in, submit an analysis, and download its PDF. Check password
reset email delivery as well. Large media uploads and slow backend requests are
subject to Netlify proxy limits; verify those with representative files before
using the deployment for production.
