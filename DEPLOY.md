# OILTRACE — Deployment

OILTRACE deploys as two independent services on separate domains:

```
  https://oiltrace.example.com          (frontend — Next.js)
            │  HTTPS, cross-origin
            ▼
  https://api.oiltrace.example.com      (backend — FastAPI in Docker)
```

The browser loads the frontend, then calls the backend directly at
`NEXT_PUBLIC_API_URL`. The backend allows the frontend's origin via an explicit
CORS list (`ALLOWED_ORIGINS`). No proxy, no wildcard origins.

---

## Environment variables

| Variable | Applies to | Purpose | Default | Example |
| --- | --- | --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | frontend (**build time**) | Backend API base URL the browser calls | `http://localhost:8000` | `https://api.oiltrace.example.com` |
| `ALLOWED_ORIGINS` | backend (runtime) | Comma-separated frontend origins allowed by CORS | `http://localhost:3000` | `https://oiltrace.example.com,https://www.oiltrace.example.com` |
| `PORT` | backend (runtime) | Port the backend listens on (binds `0.0.0.0`) | `8000` | `8080` |

Notes:
- `NEXT_PUBLIC_API_URL` is a `NEXT_PUBLIC_*` variable, so it is **inlined into
  the browser bundle at build time**. Changing it requires rebuilding the
  frontend, not just restarting it.
- `ALLOWED_ORIGINS` is read at backend startup. Whitespace around each
  comma-separated origin is trimmed. It is never `*`.
- With no variables set, both services run in their local-development defaults.

---

## Local development (no environment variables)

Backend:

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -r requirements.txt
python -m app.main            # listens on 0.0.0.0:8000; CORS allows http://localhost:3000
```

Frontend:

```bash
cd frontend
npm install
npm run dev                   # http://localhost:3000; calls http://localhost:8000
```

Nothing else is required: the frontend defaults to `http://localhost:8000` and
the backend defaults to allowing `http://localhost:3000`.

---

## Backend — Docker

The image is self-contained: it bakes in the three scene fixtures
(`scene_normal`, `scene_ambiguous`, `scene_calm`) at build time and needs no
volume mounts and no network access at runtime.

Build:

```bash
cd backend
docker build -t oiltrace-api:latest .
```

Run (default port 8000):

```bash
docker run --rm -p 8000:8000 oiltrace-api:latest
```

Run on a custom port (the platform sets `PORT`):

```bash
docker run --rm -e PORT=8080 -p 8080:8080 oiltrace-api:latest
```

Run with production CORS:

```bash
docker run --rm \
  -e ALLOWED_ORIGINS=https://oiltrace.example.com \
  -e PORT=8000 -p 8000:8000 \
  oiltrace-api:latest
```

Offline / self-contained check (no mounts, no network):

```bash
docker run --rm --network none -p 8000:8000 oiltrace-api:latest
# then, from the host or an exec shell: GET /health and POST /analyze all work
```

---

## Frontend — production build

The backend URL is baked in at build time, so build **after** the backend URL
is known:

```bash
cd frontend
NEXT_PUBLIC_API_URL=https://api.oiltrace.example.com npm run build
npx next start -p 3000        # or deploy the .next output to your Node host
```

Verify the URL was embedded (and no localhost leaked into the served bundle):

```bash
grep -rl "api.oiltrace.example.com" .next/static   # -> found in the page chunk
grep -rE "localhost:8000" .next/static .next/server # -> no matches
```

---

## Health endpoint

`GET /health` is a lightweight probe (no pipeline, no image processing, no
network). Point your host's health check at it.

```json
{ "status": "ok", "scenes": ["normal", "ambiguous", "calm"] }
```

The `scenes` list is derived from the fixture directories actually present in
the image, so it never advertises a scene whose data is missing.

---

## Scene selection

`POST /analyze` selects a scene via the request body; default is `normal`:

```bash
curl -X POST https://api.oiltrace.example.com/analyze \
  -H 'Content-Type: application/json' -d '{"scene":"ambiguous"}'
```

Valid scenes: `normal`, `ambiguous`, `calm`. Unknown values return HTTP 422.

---

## Deployment ordering

Because the frontend bakes in the backend URL and the backend must allow the
frontend origin, deploy in this order:

1. **Deploy the backend** (Docker image) and note its public URL, e.g.
   `https://api.oiltrace.example.com`.
2. **Build + deploy the frontend** with
   `NEXT_PUBLIC_API_URL=https://api.oiltrace.example.com`. Note its URL, e.g.
   `https://oiltrace.example.com`.
3. **Set the backend's `ALLOWED_ORIGINS`** to the frontend URL and restart the
   backend so CORS lets the browser through.

If the frontend URL changes later, update `ALLOWED_ORIGINS` (restart backend).
If the backend URL changes, rebuild the frontend.

---

## Render blueprint (optional)

`render.yaml` at the repo root defines both services. In the Render dashboard:
**New + → Blueprint → connect this repo → Apply**. The two URL variables are
left unset (`sync: false`) because of the ordering above:

1. Apply the blueprint; both services get URLs.
2. Set `oiltrace-web`'s `NEXT_PUBLIC_API_URL` to the `oiltrace-api` URL and
   trigger a redeploy of the web service.
3. Set `oiltrace-api`'s `ALLOWED_ORIGINS` to the `oiltrace-web` URL.

Render's free plan spins services down when idle, so the first request after a
quiet period is slow (cold start). Fine for a demo; use a paid plan or a keep-
warm pinger during judging.

---

## Verifying a deployment

```bash
# backend
curl https://api.oiltrace.example.com/health
curl -X POST https://api.oiltrace.example.com/analyze \
  -H 'Content-Type: application/json' -d '{"scene":"calm"}'

# CORS from the real frontend origin
curl -i -X OPTIONS https://api.oiltrace.example.com/analyze \
  -H 'Origin: https://oiltrace.example.com' \
  -H 'Access-Control-Request-Method: POST'
# -> 200 with Access-Control-Allow-Origin: https://oiltrace.example.com

# frontend: open it, switch Normal / Ambiguous / Calm, confirm no console errors
```
