# Koochak and Scruffy Training Reliability Plan

## Purpose

This plan makes Koochak campaigns continue safely across Scruffy controller
restarts, allocation replacement, and deliberate operator evacuation. It fixes
the infrastructure once rather than adding a Hierarchical Kaveh-specific
supervisor that reconstructs scheduler state from files.

The implementation is split across three repositories:

- **Scruffy** owns allocation lifecycle, durable workflow state, resource
  placement, evacuation, and bounded scheduler-level retries.
- **Koochak** owns reproducible commands, safe training interruption, validated
  resume, and immutable artifact publication.
- **Hierarchical Kaveh** owns its scientific campaign DAG, resource choices,
  output contracts, and ledger-ready result generation.

The implementation must leave each repository usable and tested after every
phase. Tokyo production campaigns are migrated only after a one-GPU canary has
survived manual evacuation and a real allocation replacement.

## Decisions

1. Do not build a resident `latent_supervisor`, a second retry database, or a
   second mutable campaign registry in Hierarchical Kaveh.
2. Submit complete campaign DAGs atomically through Scruffy. Do not submit a DAG
   by looping over individual `submit_scruffy` calls.
3. Use lifecycle dependencies only when producer process success is the actual
   condition. Use exact artifact conditions for checkpoints, samples, folds,
   and analysis products.
4. Automatically retry only tasks that explicitly declare themselves
   restartable, and initially only for infrastructure reasons.
5. Re-run the same immutable command for every attempt. Koochak and the project
   entrypoint must make that command safe for both a fresh start and a resume.
6. Treat graceful evacuation as the normal allocation-end path. Unexpected
   allocation loss remains a recoverable fallback.
7. Keep ledger publication outside the scheduler control loop. Cluster jobs
   produce immutable ledger bundles; a separate guarded importer updates Git.

## Non-goals

- Automatically changing batch size, optimizer behavior, or scientific
  configuration after an OOM.
- Automatically repairing parser bugs or invalid resource profiles.
- Automatically clearing a GPU quarantine.
- Retrying arbitrary nonzero application exits.
- Polling the filesystem from blocked jobs.
- Committing directly to `main` from a Tokyo controller process.
- Preserving compatibility with Koochak/Scruffy interfaces introduced during
  this implementation. The repositories will advance together to one tested
  protocol.

## Target operator experience

### Deliberate allocation evacuation

The normal explicit command is:

```bash
scruffy --root "$SCRUFFY_ROOT" evacuate --wait
```

It performs the same operation as automatic deadline evacuation:

1. Disable new launches for the allocation.
2. Select running jobs with an evacuation policy.
3. Signal each selected job once.
4. Wait for checkpoint publication and process exit up to each job's grace
   period.
5. Create durable replacement attempts for jobs whose retry policy permits it.
6. Leave the queue drained for allocation handover.

For a controlled test or rolling restart within the current allocation:

```bash
scruffy --root "$SCRUFFY_ROOT" evacuate \
  --project kaveh-ce20-20260806 \
  --workflow WORKFLOW_ID \
  --wait \
  --resume-after
```

`--resume-after` clears the evacuation drain only when every selected job either
completed normally or produced an eligible replacement attempt. A partial or
timed-out evacuation remains drained and exits nonzero.

Narrow scopes are supported:

```bash
scruffy --root "$SCRUFFY_ROOT" evacuate --job JOB_ID --wait --resume-after
scruffy --root "$SCRUFFY_ROOT" evacuate --project PROJECT_ID --wait
scruffy --root "$SCRUFFY_ROOT" evacuate --project PROJECT_ID --workflow WORKFLOW_ID --wait
```

The command is idempotent when the caller supplies the same `--request-id`.
Without one, the CLI generates and prints a request ID before submitting the
command. Repeating an uncertain mutation uses the printed ID.

### Status

Existing views remain the primary interface:

```bash
scruffy --root "$SCRUFFY_ROOT" summary --project PROJECT_ID
scruffy --root "$SCRUFFY_ROOT" running --project PROJECT_ID
scruffy --root "$SCRUFFY_ROOT" blocked --project PROJECT_ID
scruffy --root "$SCRUFFY_ROOT" explain JOB_ID
```

`summary`, `status`, `explain`, the dashboard, and MCP projections gain the
active or most recent evacuation, predecessor/successor attempt links, retry
exhaustion, and checkpoint evidence. There is no separate mutable `latentctl`
view.

## End-to-end state model

### Scruffy task recovery policy

Workflow task specifications gain an optional strict object:

```json
{
  "recovery": {
    "max_attempts": 4,
    "retry_on": [
      "allocation_replaced",
      "allocation_incarnation_changed",
      "evacuated"
    ],
    "evacuation": {
      "signal": "USR1",
      "grace_seconds": 600
    }
  }
}
```

Rules:

- Absence means no automatic retry and no evacuation signal.
- Version 1 supports automatic recovery only for workflow tasks, because
  `(project_id, workflow_id, task_id)` is their stable logical identity.
- `max_attempts` includes the first attempt and is bounded to a small protocol
  maximum, initially 10.
- Version 1 accepts only `allocation_replaced`,
  `allocation_incarnation_changed`, and `evacuated`. Add `node_lost` or
  site-specific preemption only after Scruffy can classify them authoritatively.
- `signal` is initially restricted to `USR1`.
- A successor copies the predecessor's immutable argv, cwd, environment,
  resources, dependencies, artifact conditions, and recovery policy.
- A retry never changes scientific configuration or resources.

Scruffy assigns the successor the next task attempt and a deterministic internal
request ID derived from project, workflow, task, and attempt number. It records
`predecessor_job_id`, `retry_reason`, and `successor_job_id`. Controller replay
must observe an already-admitted successor rather than create a duplicate.

### Evacuation lifecycle

Do not add a new active job state. A selected job remains `running` while it
checkpoints and exits, so it continues to own its resources. Scruffy stores an
allocation-level evacuation record and per-target progress:

```text
requested -> signalling -> waiting -> complete
                                  \-> partial
```

Per-target outcomes are:

- `completed`: the job succeeded before interruption;
- `checkpointed`: a new typed checkpoint was published after the request;
- `retry_queued`: a successor attempt was admitted;
- `not_restartable`: the target had no evacuation policy;
- `timed_out`: the grace period expired;
- `lost`: the allocation disappeared before a clean exit was observed.

After Koochak exits with the reserved evacuation exit code, Scruffy records the
existing terminal state `failed` with `reason=evacuated`. This avoids a breaking
expansion of the public job-state enum. The retry relationship makes the
recoverable outcome explicit. Ordinary nonzero exits remain `failed` with their
existing reason and do not retry.

### Automatic deadline evacuation

Scruffy keeps its existing pre-deadline drain. A new
`--evacuate-before-end-seconds` setting invokes the same internal operation as
`scruffy evacuate`, with allocation-wide scope and `resume_after=false`.

Rollout behavior:

1. Ship the setting with default `0` while local and Tokyo canaries run.
2. Enable it explicitly at 900 seconds on the canary allocation.
3. After a real rollover succeeds, make 900 seconds the Scruffy default.

Deadline evacuation never signals a job without an explicit evacuation policy.
Non-restartable jobs continue running and are reported as such.

### Koochak interruption contract

Koochak installs a minimal `SIGUSR1` handler when evacuation is enabled. The
handler performs no I/O and only sets an in-memory flag. At the end of each
completed optimizer update, the training loop:

1. Reconciles the stop flag across DDP ranks.
2. Finishes pending EMA and scheduler work for that update.
3. Writes a terminal checkpoint whose `next_step` is the first unexecuted step.
4. Atomically writes the checkpoint ready manifest.
5. Publishes the typed Scruffy artifact event.
6. Flushes bounded logs and W&B state.
7. Exits with reserved code 75.

The checkpoint must be published before the exit becomes visible to Scruffy.
If interruption arrives during import, compilation, or before the first safe
boundary, Koochak exits only when it can preserve a valid cursor. If the grace
period or allocation ends first, normal allocation-loss recovery uses the last
previously published checkpoint.

### Safe replay and resume

A restartable Hierarchical Kaveh training task always invokes one immutable
command, including on its first attempt:

```text
python -m hierarchical_kaveh.train --config CONFIG --resume auto
```

`--resume auto` means:

- start from step zero when no published numbered checkpoint exists;
- otherwise select the highest numbered checkpoint with a valid ready manifest;
- validate manifest schema, path, size, digest, and checkpoint resume cursor;
- never select `latest.pt` or a pre-created milestone directory as evidence.

The run directory and W&B run identity are stable across attempts. W&B uses a
resume mode that permits the first creation and requires subsequent attempts to
join the same identity; this behavior needs an explicit first-run/resume test.

## Artifact contracts

### Training checkpoints

Use the current Koochak typed checkpoint publication as the base contract:

- exact numbered checkpoint;
- absolute immutable path;
- byte size and SHA-256;
- atomic `.ready.json` manifest;
- typed `workload.artifact` publication.

Artifact IDs remain scoped by Scruffy producer task identity, so
`checkpoint/step000100000.pt` is unambiguous inside a workflow.

### Sampling, ESMFold, all-atom, and analysis

Koochak gains a small generic artifact module and a declared-output contract for
non-checkpoint stages. `PreparedRun` lists the artifact IDs and ready-manifest
paths the command must produce. A stage uses the generic module to validate its
output and atomically write each manifest. The manifest contains:

- artifact ID and logical stage;
- code commit and workflow/task identity;
- absolute immutable paths and SHA-256 values;
- expected and observed record counts;
- stage-specific dimensions such as arm, checkpoint, length, and variant;
- creation timestamp and schema version.

Directory artifacts use a deterministic manifest of contained files rather
than a directory timestamp. For runs with declared outputs, Koochak's runner
keeps ownership of the worker process, forwards signals to the child process
group, and waits for it. After a zero exit it validates every declared manifest
and publishes the corresponding typed Scruffy events. A missing, invalid, or
unpublishable artifact makes the runner exit nonzero, so Scruffy cannot record a
successful producer whose contract was not met. Scruffy never scans artifact
storage and releases a waiter only from a strict typed event.

Consumers use artifact conditions without also requiring producer success when
the immutable artifact is the complete dependency. This allows a 100k sampler,
for example, to run after the trainer published 100k and was later evacuated.

## Atomic Koochak workflows

Koochak gains these public concepts:

```python
PreparedTask(
    task_id=...,
    run=prepared_run,
    resources=...,
    needs=...,
    wait_for=...,
    recovery=...,
)

PreparedWorkflow(
    request_id=...,
    workflow_id=...,
    project_id=...,
    tasks=(...),
)

submit_scruffy_workflow(workflow, root=...)
```

Submission stages every immutable Koochak run artifact, constructs one Scruffy
workflow document, validates it, and calls `scruffy.submit_workflow`. Staging is
idempotent; a staging failure admits no jobs. Scruffy then admits every task or
none.

Koochak directly targets one pinned Scruffy protocol. Artifact gates and
recovery policies fail closed when unsupported. Do not silently drop keywords
after inspecting a client signature.

## Implementation phases

### Phase 0: Establish clean worktrees and protocol baselines

Repositories:

- Scruffy: `/Users/kiarash.jamali/Documents/ChatGPT/Cluster management`
- Koochak: its canonical repository, not a project submodule worktree
- Hierarchical Kaveh: a new `codex/` branch from current `main`

Actions:

1. Record current Scruffy, Koochak, Pazuzu, and project commits installed on
   Tokyo.
2. Create independent worktrees for Scruffy and Koochak changes.
3. Confirm the current Scruffy test suite and Koochak test suite pass unchanged.
4. Add a cross-repository compatibility fixture or pin so tests never exercise
   an accidental ambient Scruffy installation.

Exit gate: clean baseline tests and an explicit commit matrix.

### Phase 1: Scruffy recovery policy and automatic attempts

Primary files:

- `src/scruffy/submissions.py`: validate and persist `recovery`.
- `src/scruffy/client.py`: accept the policy in job and workflow APIs.
- `src/scruffy/workflows.py`: retain policy across attempts and resolution.
- `src/scruffy/controller.py`: classify eligible terminal outcomes and admit
  exactly one successor.
- `src/scruffy/storage.py`: retain retry linkage in hot and archived records.
- `src/scruffy/summary.py`: expose attempt chains and exhaustion.
- `src/scruffy/protocol.py` and `docs/client-v1.md`: define the wire contract.

Tests:

- omitted policy never retries;
- invalid policy is rejected before admission;
- allocation replacement creates one successor for an eligible running task;
- controller crash between successor journaling and snapshot write does not
  duplicate it;
- max attempts stop the chain;
- non-infrastructure failure does not retry;
- newest task attempt satisfies downstream workflow resolution;
- artifact waiters remain blocked rather than skipped when a producer attempt
  is lost;
- archive compaction retains predecessor/successor identity.

Exit gate: automatic retry works in Scruffy local-mode tests without Koochak.

### Phase 2: Scruffy operator and deadline evacuation

Primary files:

- `src/scruffy/client.py`: `request_evacuation` and bounded waiting.
- `src/scruffy/cli.py`: `evacuate`, scope flags, `--wait`, and
  `--resume-after`.
- `src/scruffy/controller.py`: one evacuation state machine used by manual and
  deadline requests.
- `src/scruffy/slurm.py`: signal an exact owned Slurm step without signalling
  the outer allocation.
- `src/scruffy/runtime.py`: local-mode signal support.
- `src/scruffy/summary.py`, dashboard, and MCP projections: evacuation status.

Safety rules:

- signal only active jobs whose persisted policy permits evacuation;
- prove the job still owns the recorded launch token and allocation
  incarnation before signalling;
- signal once and journal the decision before the side effect;
- never use a bare PID or mutate the parent allocation;
- do not kill a timed-out job automatically in version 1;
- keep the queue drained after any partial evacuation.

Tests:

- manual single-job, project, workflow, and allocation scopes;
- duplicate request ID is idempotent and conflicting reuse fails;
- `--wait` succeeds only after retry admission or normal completion;
- `--resume-after` resumes only a complete evacuation;
- non-restartable jobs are reported and untouched;
- deadline and manual paths produce the same state transitions;
- stale launch tokens and replacement incarnations fail closed;
- Slurm signalling addresses the worker step, not the outer allocation.

Exit gate: a local fake workload catches `SIGUSR1`, exits 75, and is restarted
exactly once by `scruffy evacuate --wait --resume-after`.

### Phase 3: Koochak safe interruption and validated auto-resume

Primary files:

- `koochak/loop.py`: signal flag, DDP reconciliation, safe-boundary stop, and
  terminal checkpoint.
- `koochak/storage/checkpoint.py`: highest valid published checkpoint lookup.
- `koochak/logging/events.py`: evacuation milestone and publication ordering.
- `koochak/jobs/manifest.py`: recovery policy in immutable run metadata.
- `koochak/jobs/backends.py`: pass recovery policy without compatibility
  fallbacks.
- Koochak CLI/config: explicit evacuation enablement and exit code.

Tests:

- signal during single-process training saves the correct `next_step`;
- resume executes the first previously unexecuted update;
- DDP ranks agree to stop and preserve per-rank RNG state;
- signal during a checkpoint cannot leave a ready manifest for partial bytes;
- artifact publication precedes process exit;
- corrupt, missing, copied, or stale checkpoint scaffolding is rejected;
- first attempt with auto-resume starts cleanly;
- later attempt resumes the same W&B identity;
- Scruffy worker variables prevent any mutation of the parent Slurm allocation.

Exit gate: a Koochak integration test under Scruffy local mode survives a
manual evacuation and matches an uninterrupted reference run at the next
checkpoint.

### Phase 4: Atomic workflow and generic output publication

Primary Koochak work:

- add `PreparedTask`, `PreparedWorkflow`, and
  `submit_scruffy_workflow`;
- add generic file/directory artifact manifest construction and publication;
- add declared outputs to `PreparedRun` and its immutable launch manifest;
- add a managed-child runner path that forwards signals, preserves the child
  exit, and validates/publishes declared outputs before reporting success;
- pin the compatible Scruffy commit in Koochak packaging;
- test the real Scruffy API, including artifact gates and recovery policy.

Tests:

- complete workflow admission is all-or-nothing;
- retrying an uncertain submission with the same request ID deduplicates;
- staging failure admits no jobs;
- missing declared artifact fails the producer contract and keeps consumers
  blocked;
- strict publication releases exactly the intended consumer;
- a producer's later evacuation does not revoke published evidence.
- the managed runner forwards `SIGUSR1` to the exact child process group and
  never signals its parent allocation;

Exit gate: one synthetic train -> sample -> fold -> analysis DAG completes with
artifact-only edges and survives evacuation of the trainer.

### Phase 5: Hierarchical Kaveh integration

Primary files:

- update the `external/koochak` gitlink to the tested Koochak commit;
- `hierarchical_kaveh/train.py` and `hierarchical_kaveh/training.py`: implement
  validated `--resume auto`;
- replace one current campaign launcher with an atomic Python workflow builder;
- update sampling, ESMFold, all-atom, and analysis scripts to publish strict
  artifacts;
- remove `wait_for_checkpoint_and_sample.py` from production launch paths;
- replace `condition: succeeded` checkpoint edges with exact artifact waits;
- keep training configuration YAML as scientific configuration, not mutable
  scheduler state.

The first migrated workflow is a small one-arm canary with:

- one restartable trainer;
- two numbered checkpoint milestones;
- one sampler waiting on the first checkpoint;
- one fold stage waiting on samples;
- one analysis stage waiting on folds;
- stable W&B identity and a small wall-time/resource request.

Exit gate: local project tests and dry-run output prove the exact task graph,
artifact IDs, recovery policies, commits, and run paths.

### Phase 6: Tokyo canary and allocation handover

Before each remote mutation, follow the Tokyo health and routing checks. Use a
committed launcher through Koochak; do not submit raw shell-built jobs.

Canary sequence:

1. Submit the one-GPU workflow to a healthy, non-draining allocation.
2. Wait for training progress and at least one durable checkpoint.
3. Run scoped manual evacuation with `--wait --resume-after`.
4. Verify the first attempt exited as `reason=evacuated`, exactly one successor
   exists, W&B resumed, and training passed the interruption step.
5. Verify the already-published checkpoint released its sampler exactly once.
6. Restart the Scruffy controller inside the same allocation and verify live
   step reattachment remains unchanged.
7. Run a real allocation handover using the stable queue root, without
   `--resume-after`.
8. Verify the replacement controller adopts queued retries, no task is
   duplicated, and downstream artifact gates continue resolving.

Stop and diagnose on any mismatch. Do not migrate production campaigns from a
submission response alone.

Exit gate: numbered checkpoints, attempt linkage, samples, fold outputs,
analysis result, and Scruffy terminal states all agree after a real handover.

### Phase 7: Production migration and default-on deployment

1. Migrate the length-256 scratch campaigns first, one arm at a time.
2. Recover existing lost tasks as explicit new attempts under the new workflow
   contract; do not rewrite their historical records.
3. Migrate other active Koochak campaigns after the scratch canary remains
   healthy for one checkpoint interval.
4. Enable automatic deadline evacuation at 900 seconds.
5. Promote a generic Scruffy allocation entrypoint that always starts the
   controller as the outer allocation foreground process with the stable queue
   root.
6. If the site cannot requeue the outer allocation reliably, add one generic
   allocation renewer in Scruffy/Pazuzu infrastructure. Do not put it in a
   scientific project repository.
7. Remove obsolete polling helpers only after no active workflow references
   them.

Exit gate: one complete production allocation rollover requires no agent or
user action and loses at most the work since the most recent safe checkpoint.

### Phase 8: Ledger bundle publication

Analysis produces an immutable ledger bundle containing:

- machine-readable rows and aggregate metrics;
- charts;
- code, Koochak, and Scruffy commits;
- workflow and task attempt identities;
- artifact manifests and hashes;
- explicit limitations and failed/missing panels.

A separate importer runs from a clean current `main`, validates the bundle,
writes `docs/experimental-ledger`, and either creates a results PR or performs a
guarded fast-forward commit under a repository lock. Git failure leaves the
bundle pending but does not alter campaign state.

Exit gate: importing the same bundle twice is a no-op, conflicting provenance
fails closed, and concurrent user changes are never overwritten.

## Validation commands

Use the repositories' own environments. Expected local commands are:

```bash
# Scruffy
PYTHONPATH=src python -m unittest discover -s tests -v
python -m ruff check src tests

# Koochak
pytest -q

# Hierarchical Kaveh
pytest -q
```

Add targeted commands to each implementation commit description. The final
cross-repository validation must pin exact commits rather than importing any
ambient site package.

## Failure policy

| Outcome | Automatic action |
| --- | --- |
| Deliberate evacuation | Checkpoint, publish, exit, retry within limit |
| Allocation replaced/incarnation changed | Retry declared restartable task from latest valid publication |
| Missing or corrupt checkpoint | Fail closed and alert; do not fall back to scaffolding |
| OOM | Record and alert; no automatic scientific/resource change |
| Ordinary nonzero exit | Record and alert; no retry in version 1 |
| Declared artifact missing after child exit | Koochak runner exits nonzero; consumer remains blocked |
| No capacity | Remain queued without consuming resources |
| GPU quarantine | Preserve Scruffy health authority; do not clear automatically |
| Evacuation timeout | Leave queue drained, report partial, require diagnosis |
| Git/ledger conflict | Retain immutable pending bundle; do not mutate `main` |

## Acceptance criteria

The work is complete only when all of the following are true:

1. A complete Koochak campaign is admitted atomically into Scruffy.
2. Every dependency on a checkpoint or result uses a strict typed artifact.
3. `scruffy evacuate --wait` can deliberately checkpoint eligible running jobs
   and queue exactly one successor attempt.
4. `--resume-after` provides a safe same-allocation end-to-end test path.
5. Automatic pre-deadline evacuation uses the same tested state machine.
6. A replacement allocation automatically recovers declared restartable tasks
   from valid numbered checkpoints.
7. Existing artifact consumers are not skipped merely because a producer
   attempt was evacuated or lost.
8. Retry limits, non-restartable failures, and partial evacuations fail closed.
9. The Scruffy dashboard and CLI show the complete attempt and evacuation
   history without a project-specific supervisor.
10. A Tokyo canary and one production rollover complete without agent or user
    intervention.
11. Ledger publication cannot overwrite concurrent repository changes.

## Recommended implementation order

The critical path is:

```text
Scruffy retry policy
  -> Scruffy manual evacuation
  -> Koochak checkpoint-on-signal and auto-resume
  -> Koochak atomic workflows and generic artifacts
  -> Hierarchical Kaveh canary
  -> Tokyo manual evacuation
  -> Tokyo allocation replacement
  -> production migration
  -> ledger importer
```

Do not start with the project campaign migration. Until Scruffy can create a
durable successor and Koochak can safely replay the same command, project-side
reconciliation would only hide the missing infrastructure contract.
