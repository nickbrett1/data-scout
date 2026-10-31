# data-scout

A data-scout project generated with genproj

## Capabilities

This project includes the following capabilities:

- **Doppler Secrets Management**: Integrates Doppler for secure secrets management. Enables the various MCP servers that rely on privileged tokens to access their services (e.g. Buildkite, GitHub, SonarQube).
- **AI Coding Agents**: Sets up the AI coding agents in the devcontainer: goose (config, MCP servers and spec-first recipes) plus the Antigravity CLI.
- **Container Agent**: Every generated devcontainer brings up and registers its own a2a-goose agent (`<repo>-dev`), reached over the tailnet by the LiteLLM proxy; reuses the a2a-goose GitHub release channel, so the container has the same self-update path as a host.
- **Secret Scanning (gitleaks)**: Required on every project: gitleaks scans staged changes in a pre-commit hook and the full history in CI. Free, offline, no account and no quota.
- **Docker**: Adds Docker support for containerised builds and tooling.
- **Python DevContainer**: Sets up a VS Code DevContainer with Python environment.
- **Docker Container**: Containerize the project and publish to the GitHub Container Registry (GHCR) for deployment to a NAS or self-hosted host via Docker Compose. Mutually exclusive with other deployment systems. Set `workload: batch` for a scheduled job image (runs and exits; no port, no healthcheck, no Watchtower) instead of a long-running service.
- **Buildkite Integration**: Runs CI on a self-hosted Buildkite agent (Apple silicon) instead of a metered cloud fleet. The pipeline and its GitHub webhook are created during generation, so there is no manual "set up project" step. It can coexist with an existing CI provider, so a repository can migrate without a flag day.
- **Ruff (Python code quality)**: Adds fast, zero-configuration Python linting with Ruff (rules live in pyproject.toml [tool.ruff]). Lint locally with `ruff check`. Requires a Python devcontainer. The generated CI pipeline also runs `ruff check`.
- **Editor Configuration**: Shared VS Code extensions and workspace settings for consistent tooling across the team.
- **Shell & Terminal**: Zsh shell with the Powerlevel10k prompt and productivity plugins.

## Setup

1. Clone the repository
2. Create a virtualenv and install the package with dev extras:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the checks:

   ```bash
   ruff check src tests
   pytest -v
   ```

## Secret scanning

Every commit is scanned for credentials by `gitleaks`, and CI scans the
**full history**. The local hook is convenience; CI is the gate you cannot skip
with `git commit --no-verify`.

```bash
scripts/setup-git-hooks.sh   # installs gitleaks and sets core.hooksPath
```

If a commit is blocked, **rotate the credential first** — a finding means it may
already be exposed. Then remove it from the staged changes and commit again.

Only for a verified false positive, add its fingerprint to `.gitleaksignore`,
which is a committed and reviewable diff. Never bypass the hook with
`--no-verify`: it leaves no record and the next commit is unscanned too.

This is detection, not storage (Doppler holds secrets) and not rotation (that is
a human step). Allowlist edits in `.gitleaks.toml` are reviewable changes on
purpose.

## Doppler

This project uses Doppler for secrets from the shared `common` project
(config `dev`) — no per-repo Doppler project is created. First use (links
the shared project and `dev` config):

```bash
doppler setup --project common --config dev
```

If your repo needs app-specific secrets that shouldn't live in the shared
`common` project, regenerate it with the doppler capability set to
`projectStrategy: "new"` to get a dedicated project.

The Doppler CLI is installed in the devcontainer — it must be on PATH for the
VS Code extension and `doppler run` to work. Auth is persisted via the host
`~/.doppler` bind-mount.

### Env-var precedence (read this if `doppler run` hits the wrong project)

Doppler resolves its target as **environment variables > `doppler.yaml` >
`~/.doppler` scoped config**. If your shell — or the session that launched
the devcontainer (e.g. an agent runtime) — exports `DOPPLER_PROJECT` /
`DOPPLER_CONFIG` / `DOPPLER_ENVIRONMENT`, those silently override this
repo's `doppler.yaml` and every `doppler` command targets the wrong
project. The devcontainer's post-create setup pins this repo's context
(`common`/`dev`) in `~/.bashrc` and `~/.zshrc` and warns at
setup if resolution still mismatches. To force the correct context manually:

```bash
unset DOPPLER_PROJECT DOPPLER_CONFIG DOPPLER_ENVIRONMENT
doppler setup --no-interactive --project common --config dev
```

## Deployment

See `deploy/README.md` for the deployment runbook (Buildkite -> GHCR ->
Watchtower -> Docker host). Deploy with:

```bash
docker compose up -d
```

## The container's agent

This devcontainer brings up its own `a2a-goose` agent, registered in the hub as
`data-scout-dev` - one agent per repo, so a restart reclaims the same entry
instead of adding a second one. Turns are billed through the LiteLLM proxy
configured in Doppler (`LITELLM_BASE_URL`).

```bash
scripts/agent-dev.sh start    # write secrets + config, fetch the launcher, run it
scripts/agent-dev.sh status   # running or not, the card URL, the log tail
scripts/agent-dev.sh stop     # SIGTERM, wait for a clean deregister, confirm gone
```

`start` runs from the devcontainer's post-start hook, so the agent is normally
already up when you arrive. It fails open: with no network on a first start it
prints why it did not start and leaves the project usable. Secrets come from
Doppler into `~/.config/a2a-goose/env` (mode 0600) and never into the image or
`containerEnv`.

## The toolbelt

`data_scout/tools/` holds fifteen deterministic instruments — forensics, structure,
content, semantic, redundancy, sampling and the ledger. They are **a Python library
first**: the MCP façade is generated from `data_scout/registry.py` afterwards and is
not built yet. The rule that shapes everything else: an MCP result never becomes a
record, so anything that ends up in a report is a library call in a notebook cell.

```python
from data_scout.tools import pdf_forensics, structure_mix, candidate_density

pdf_forensics("/datasets/my-dataset")    # zero fonts => no text layer
text_quality(sample_of_pages)              # is this text, and are the chars in words
```

Tier 2 (`embed_sample`, `cluster_topics`, `image_inventory`) needs the optional extra
and a gateway key, and **fails loudly** without them rather than reporting zero:

```bash
pip install -e ".[semantic]"
```

### Two roots, and one writable one

| env | what | default |
|---|---|---|
| `DATA_SCOUT_DATASETS_ROOT` | the datasets, mounted **read-only** | `/datasets` |
| `DATA_SCOUT_WORKSPACE` | notebooks, `findings.jsonl`, samples, cache, reports | `/workspace` |

A dataset id that does not exist under the root fails loudly **naming the resolved
root**; a dataset with no `frame.md` fails as *unprofiled* rather than being profiled
under an inferred id. Fail loudly, never infer.

### Rendering — `report-render`

Rendering runs in the same container, as a second entrypoint rather than a second
service. It executes the notebook, stamps `metadata.kernelspec`, re-executes it into
a scratch copy of the workspace and diffs `findings.jsonl` modulo `ts`, and **refuses
to render** if the re-execution does not reproduce the lines.

```bash
docker compose exec data-scout report-render <dataset-id> [--session N] [--verify-only]
report-render <dataset-id>          # in the dev environment, after `pip install -e .`
```

The frozen artifacts are served on the single published port: `/report/…` (HTML) and
`/healthz`, alongside the MCP façade at `/mcp`. `/a2a` is a documented seam, not an agent
loop — the tools are the attachable surface, and the intelligence is a recipe driving them.

### Tools over MCP — the belt as one server

The same fifteen instruments are served as an MCP tool server on the same port, so an
agent can call what a notebook calls. There is no second deployment and no second port:
the façade is generated from the registry (`mcp_server.build_server()`), and a test
asserts the tool set *is* `registry.BELT` — a sixteenth tool cannot appear here without
appearing in the registry first.

```
http://data-scout:8710/mcp        # streamable HTTP, dialed by container name
```

Two properties worth knowing before wiring a UI to it:

- **A refusal is an error, not a zero.** A missing extra, a missing binary, an
  unconfigured gateway, or a path outside the two roots returns an `isError` result whose
  text carries the install command or both roots. The SDK's default would reduce that to
  "Error executing tool <name>" and keep the text on the server; `_deliberate` re-raises
  it so the caller can act on it.
- **An MCP result never becomes a record.** `findings_append` is exposed, but nothing here
  appends on your behalf: the ledger still only grows when a cell or a human decides what
  a measurement means.

Paths given to a path-taking tool must resolve under the datasets root or the workspace
root; this server does not read the rest of the filesystem, and the dataset root stays
read-only.

### The devcontainer trap (read before adding a tool)

`devcontainer-python` has no configuration schema, so **the generated devcontainer
cannot be pointed at the project's own image**. `data_scout` imports fine there and
`pdffonts`, `exiftool` and `file` may simply not be installed — the failure is silent
at import and only shows at runtime. `.devcontainer/Dockerfile` carries a `LOCAL EDITS`
block that installs the tier-0 belt for exactly this reason; keep it in step with the
runtime `Dockerfile`.

If you do not want to maintain that block, the better shape is VS Code's *Attach to
Running Container* against the compose service: the session loop runs in the deployed
container anyway, and developing a tool against the real image is closer to production
than a devcontainer is.

## Generated by genproj

This project was generated using the genproj tool.
