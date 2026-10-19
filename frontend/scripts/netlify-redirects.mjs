import { writeFile } from 'node:fs/promises';

// Keep requests on the frontend origin so session cookies and CSRF still work.
const backend = process.env.BACKEND_URL?.trim();
let apiRule = '/api/* /api-unavailable.json 404!';

if (backend) {
  const url = new URL(backend);
  if (url.protocol !== 'https:' || url.username || url.password ||
      url.search || url.hash || url.pathname !== '/') {
    throw new Error('BACKEND_URL must be an HTTPS origin, such as https://api.example.com (without /api).');
  }
  apiRule = `/api/* ${url.origin}/api/:splat 200!`;
}

await writeFile(new URL('../dist/_redirects', import.meta.url), `${apiRule}\n/* /index.html 200\n`);
await writeFile(new URL('../dist/api-unavailable.json', import.meta.url), JSON.stringify({
  error: 'The backend is not deployed yet. Login, analysis, and reports will be available after the server is connected.',
}));
console.log(backend ? 'Netlify API proxy configured.' : 'Frontend only: set BACKEND_URL and rebuild after hosting Flask.');
