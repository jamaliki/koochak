# Config System Overhaul (OmegaConf + Dataclasses)

## Summary
Overhaul the configuration system to be OmegaConf-first, section-based, and predictable with typed dataclasses: a single root config loaded once, clear section ownership of keys, explicit defaults layering, and dataclass-backed validation. Training and other components should receive only the section configs they need (e.g., `train` for the loop), with cross-section reuse handled via OmegaConf interpolation.

## Context
- The current config flow mixes full-root config and section dicts, uses ad hoc access helpers, and makes it unclear where keys should live or be accessed.
- Users and contributors want simplicity and predictability: a single template should show where keys live and where they are read.
- OmegaConf is already in use and should be the base for resolving/interpolation and defaults, with structured dataclasses for clarity.

## Goals
- Simple, predictable key placement: each key has a single canonical location.
- Consistent access pattern for contributors (section-based configs with typed fields).
- Clear defaults layering: explicit user values override defaults; defaults only fill missing keys.
- Preserve config summary and OmegaConf resolution.
- Minimal changes to training logic (focus on config flow and interfaces).

## Non-Goals
- Changing training logic, data pipelines, or runtime behavior beyond config handling.
- Automatic detection of where keys should live.
- Complex schema enforcement or type coercion beyond basic sanity.

## Requirements
- Functional:
  - OmegaConf is the base config object (DictConfig) with structured dataclasses.
  - Keep current top-level section names (train/data/optim/logging/wandb/entry).
  - Provide a single, canonical template config that makes key ownership obvious.
  - Ensure defaults layering is deterministic: `defaults < user YAML < overrides`.
  - Provide a pre-run config summary (sections present, unknown keys, strictness).
  - Pass section configs to components; `training_loop` receives `train` only.
  - If a feature needs extra config beyond `train`, add explicit arguments.
- Non-functional:
  - Simplicity and ease of discovery over maximal compatibility.
  - Performance-neutral config handling.
  - Allow a hard break from previous config behavior if needed.

## Proposed Options
### Option A: Structured Dataclasses per Section (Recommended)
- Description: Define dataclasses for each section (train/data/optim/logging/wandb/entry) and use `OmegaConf.structured` for defaults, validation, and docs. A small `koochak.config` module loads root config via OmegaConf, merges defaults, resolves interpolation, validates unknown keys (via structured configs), and exposes `get_section(...)` helpers. Each component receives only its section config. `training_loop` accepts `train_cfg` only.
- Pros:
  - Clear ownership and discoverability via typed fields.
  - Better validation with minimal extra logic.
  - Matches current section layout and keeps training logic stable.
- Cons:
  - Requires defining/maintaining dataclasses.
  - Slightly more boilerplate for new keys.
- Risks:
  - Overly strict typing could hinder experimentation if not designed for extension.

### Option B: Section-Based DictConfig + Simple Schema
- Description: A small `koochak.config` module loads root config via OmegaConf, merges defaults, resolves interpolation, validates unknown keys against a minimal schema, and exposes `get_section(...)` helpers. Each component receives only its section config. `training_loop` accepts `train_cfg` only.
- Pros:
  - Simple mental model; matches current section layout.
  - Minimal changes to training logic.
  - Easy to document with a single template and config map.
- Cons:
  - Less type safety than structured configs.
  - Validation remains shallow (unknown keys only).
- Risks:
  - Users may still misplace keys without strict enforcement.

### Option C: Single Root Config Passed Everywhere
- Description: Pass the root DictConfig to all components and access dotted keys directly.
- Pros:
  - Less plumbing of section configs.
- Cons:
  - Encourages unclear key ownership and scattered access patterns.
- Risks:
  - Reintroduces the main confusion this overhaul aims to fix.

## Decision
Choose Option A. It keeps the system simple and predictable while improving discoverability and validation via typed fields. It also supports hard-break migration with minimal behavioral risk.

## Detailed Design

### Config Module
Create `koochak/config.py` (or `koochak/config/__init__.py`) as the single entry point. All config handling goes through it.

API (proposed):
- `load_config(path: str, overrides: dict | None = None) -> DictConfig`
  - `OmegaConf.load(path)`
  - Merge defaults (structured): `cfg = OmegaConf.merge(STRUCTURED_DEFAULTS, cfg)`
  - Apply overrides (optional): `cfg = OmegaConf.merge(cfg, overrides)`
  - Resolve: `OmegaConf.resolve(cfg)`
- `get_section(cfg: DictConfig, name: str, *, required: bool = True) -> DictConfig`
  - Returns the section (DictConfig), optionally raising if missing.
- `summarize(cfg: DictConfig, *, schema: dict, strict: bool) -> None`
  - Prints sections present, unknown keys, strict flag.
  - Unknown keys are computed via a simple schema map of allowed keys.
- `validate_unknown_keys(cfg: DictConfig, schema: dict) -> set[str]`
  - Returns unknown dotted paths for logging/strict mode.

Defaults and schema:
- `STRUCTURED_DEFAULTS`: structured dataclasses per section used as defaults and for validation.
- `SCHEMA`: optional extra schema for loose sections or extension points (if needed).
- Keep typing flexible where experimentation is expected (e.g., `Dict[str, Any]` for extension hooks).

### Section Ownership and Access
- Canonical sections: `train`, `data`, `optim`, `logging`, `wandb`, `entry`.
- Code must access keys through its section config only.
- Shared values must live in one section and be referenced elsewhere via OmegaConf interpolation, e.g.:
  - `logging.csv_path: ${train.out_dir}/log.csv`
  - `data.batch_size: ${train.batch_size}` (if desired)

### Training Loop Interface
- `training_loop` signature uses only `train_cfg` as config input.
- If a feature needs other section data, add explicit arguments (e.g., `logging_cfg`, `ddp_cfg`, `data_cfg`).
- This prevents hidden coupling to unrelated config keys.

### CLI and Examples
- CLI loads root config via `load_config`.
- CLI calls `summarize` before training.
- CLI passes the appropriate sections to builders:
  - `build_optimizer(..., optim_cfg)`
  - `make_stdout_hooks(logging_cfg, train_cfg)` if needed
  - `training_loop(..., config=train_cfg)`
- Examples follow the same pattern; no custom config access helpers.

### Template and Documentation
- Provide a single, canonical template config file (e.g., `examples/config_template.yaml`) showing all supported keys.
- Add a “Config Map” section to README:
  - A brief description of each section.
  - Links to where each section is consumed (loop/optim/logging/CLI).

### Error Handling and Strictness
- `train.strict_config` and `train.config_warn_unknown` remain supported.
- Strict mode errors on unknown keys before training starts.
- Warnings are rank-0 only in DDP contexts.

## Rollout / Migration
- Hard break is acceptable.
- Steps:
  1) Add new config module and template file.
  2) Update CLI and examples to use the new loader and section access.
  3) Update `training_loop` call sites to pass `train_cfg` only.
  4) Remove old config helpers or leave as minimal wrappers (documented as deprecated).
  5) Update README to point to the new template and access rules.

## Observability
- Pre-run summary prints:
  - Sections present
  - Unknown keys (if any)
  - Strict mode status
- Success criteria:
  - Template-driven config is self-explanatory.
  - Contributors can locate key usage by section with minimal searching.

## Testing Plan
- Unit tests for:
  - `load_config` layering order (defaults < YAML < overrides).
  - `summarize` output for unknown keys.
  - `get_section` required behavior.
- No broad integration tests required; verify in a real workload after rollout.

## Risks and Mitigations
- Risk: Hard break impacts existing YAMLs.
  - Mitigation: Provide clear template and migration guidance in README.
- Risk: Overly loose validation allows typos.
  - Mitigation: Keep a minimal schema and strict mode default-on.

## Checklist
- [x] All interview questions answered.
- [x] No open questions remain.
- [x] Option decision documented with rationale.
- [x] Testing plan covers critical paths.

## Open Questions
- None.
