# Golden payloads

`payloads.json` is committed. **`expected.json` is deliberately absent.**

Expected outputs must be recorded against the **real promoted artifact**, served by the
real container — never against a fixture-trained stand-in. A fixture-derived
`expected.json` makes the post-deploy job compare the live endpoint to a *different model*
and fail on every run.

Runbook **step 8** records it, against the real artifact served by local uvicorn, before
the image is built. The CI `golden` job skips with a message while the file is absent, and
enforces once it exists.

## Recording it

```bash
python tests/golden/record.py --endpoint http://127.0.0.1:8000
git add tests/golden/expected.json
```

## Three-point parity

The same `payloads.json` is sent to three places and the outputs compared to the decimal:

1. local `uvicorn`   — runbook step 8
2. `docker run`      — runbook step 10
3. through the LB    — runbook step 18, and automatically on every deploy thereafter

A container returning a different number from local is a corrupt artifact. Both would
still return HTTP 200.

`payloads.json` includes a **null-categorical case that must be REJECTED**, not imputed.
`check_live.py` asserts the 4xx as strictly as it asserts the 200s.
