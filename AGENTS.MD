# Agent Guidance

## PowerShell and Encoding Discipline

This repository contains Thai and mixed-language filenames, comments, logs, and documentation. Before reading those files through PowerShell, always set UTF-8 for the session and read explicitly as UTF-8:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
chcp 65001
Get-Content -Encoding UTF8 <path>
```

Use this before inspecting Markdown files, `.env` files, Thai filenames under `app/asset`, test data under `tests/mock_data`, or any mixed-language source file. If text appears as mojibake, stop and fix the read/encoding path before drawing conclusions from that file.

## User Working Style

The user expects implementation-first, evidence-driven work. Do not provide polished guesses or broad summaries when the repo can be checked directly. Inspect the live implementation, Docker/runtime files, environment config, tests, logs, and current documentation before making claims.

If the user asks for understanding first, do a deep code-first read before proposing edits. For dependency maps or call graphs, present function-level parent/child relationships rather than stopping at a module summary.

When the user asks to continue backlog or finish a module, first verify the backlog against live code and the relevant owner files. Do not assume a note is still true. Identify what is already implemented, what is missing, and what needs code, docs, tests, or runtime verification before choosing the work.

When drift is found, keep going and reconcile it unless the user explicitly asks for analysis only. A good pass usually updates the code, tests, configuration examples, Docker/runtime behavior, and documentation together so the repo returns to a consistent state.

For runtime behavior changes, close the loop in one pass when feasible: code, `.env.example`, Docker or compose files, focused unit tests, and direct smoke tests with mock data. Do not stop at code-only changes when configuration or deployment behavior also changed.

Use `config/.env` as the real runtime configuration for local runs, Docker smoke tests, and debugging. Treat `config/.env.example` only as a mockup/template that documents which keys need to be set for another environment. Do not assume `.env.example` values are the active configuration.

This repo now uses a single `requirements.txt` for both local and Docker installs. Do not reintroduce a second requirements file for Docker-only dependencies; if Docker needs a special install path, keep `requirements.txt` as the source of truth and adjust the Dockerfile logic instead.

The user values source-of-truth discipline. Do not scatter the same policy across multiple docs. Keep the main docs explicit, usable, and owned by the right file.

For anomaly retraining, treat `config/retrain_anomaly.json` as the single default source of truth for both `POST /anomaly/retrain/` and `AnomalyDetection/scripts/retrain_from_config.py`. Do not hardcode a second default model list or batch/report/index policy in docs or helper scripts.

When documenting runtime routes, re-check `app/main.py` and keep all three V2 routes straight: `/v2/upload_pdf/`, `/v2/detection_pig`, and `/v2/detection_pig_follicle`. The follicle route is gate-plus-Gemini output, not the same contract as the DB-writing V2 routes.

When describing health behavior, note that `/health` now returns safe runtime config summary alongside app metadata and DB status. Do not document it as version-only or DB-only if the payload has changed.

The user prefers direct execution over instructions. If they ask to run tests, build Docker, reset data, or try a mock file, do it and report the exact command and result. If a command fails, inspect the concrete failure and continue with the next practical verification path.

When the user corrects a misunderstood change, first undo the wrong scoped change before continuing. Then re-read the relevant code/docs and verify the intended target rather than building on the mistaken assumption.

The user is sensitive to operational drift. Do not remove working command examples unless replacing them with a verified equivalent. Preserve useful local and Docker command forms when they are already documented or used during verification.

Encoding and documentation readability matter. If a documentation file becomes mojibake or unreadable, fix the source directly and preserve UTF-8. For large documentation rewrites, create a `.bak` copy first so old wording can be recovered.
