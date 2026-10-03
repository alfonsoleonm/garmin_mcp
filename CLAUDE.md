# Context
Fork of Taxuspt/garmin_mcp deployed on Google Cloud Run (streamable-http transport).
Auth is a secret path segment handled by src/garmin_mcp/http_auth.py.

# Task
Add GET /api/dashboard returning JSON (sleep, HRV, Body Battery, training readiness,
resting HR, training load) using @fastmcp.custom_route and the existing Garmin client.

# Rules
- Auth: Authorization: Bearer <DASHBOARD_API_KEY> (env var, injected from Secret Manager),
  compared with hmac.compare_digest. Missing/invalid key -> 401.
- CORS restricted to https://alfonsoleonm.github.io (never "*").
- Never print or log secrets, tokens, or health data.
- Do not change stdio mode or existing tests; add new tests using mocks.
- Stay on branch cloud-run-deploy. Do not deploy or run gcloud without asking me.
