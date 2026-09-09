# Running experiments with Koochak and Scruffy

This note is for agents launching Hierarchical Kaveh experiments on Sandpit
Tokyo. The shortest correct summary is:

> **Koochak prepares the experiment; Scruffy schedules it inside an active
> allocation; Pazuzu provides the connection and standalone-Slurm fallback.**

Do not treat Scruffy as a shell wrapper around `sbatch`, and do not bypass a
healthy Scruffy allocation with ad-hoc Slurm jobs.

## 1. The three layers

| Layer | Owns | Use it for |
| --- | --- | --- |
| **Koochak** | Resolved configs, immutable run manifests, environment profiles, worker preflights, checkpoints, typed artifacts, recovery hooks | Building a reproducible training/evaluation run |
| **Scruffy** | Scheduling, GPU reservations, task dependencies, artifact gates, lifecycle, allocation handover, retry policy | Running a complete DAG inside one active allocation |
| **Pazuzu** | The persistent connection to the Tokyo login host and typed standalone Slurm access | Remote inspection, staging committed launchers, or submitting when no suitable Scruffy allocation exists |

For this repository, GPU experiments and their CPU evaluation stages should be
submitted through `koochak.jobs.submit_scruffy_workflow`. Use the same workflow
for training, sampling, folding, and analysis so that dependencies are
authoritative rather than implemented with polling.

## 2. Mandatory pre-submission sequence

Before every new submission:

1. Call Pazuzu `connection_health(probe=true)`.
2. Call Scruffy `overview`.
3. Require all of the following:
   - the call succeeded and its heartbeat is fresh;
   - the allocation state is `running`;
   - `draining` and `launches_paused` are false;
   - healthy GPU capacity, CPU, memory, and remaining walltime fit the whole
     experiment;
   - the controller release matches the Scruffy client pinned by the
     environment profile.
4. Inspect the target checkout, Koochak commit, environment profile, parent
   configs, and output root.
5. Submit one complete, validated DAG.
6. Inspect Scruffy state after submission. Do not submit one task at a time.

If the Scruffy MCP bridge fails, that does **not** prove that the allocation is
gone. Use Pazuzu's bounded Slurm status to distinguish a dead allocation from
a broken Scruffy bridge. Do not launch duplicate standalone jobs against a
possibly live allocation.

## 3. Prepare a reproducible launch

The launch program must be a committed Python file in the project repository.
It should contain:

- a stable workflow ID and request ID;
- explicit resource requests for every stage;
- immutable parent config paths and SHA-256 hashes;
- a small, explicit set of `ConfigPatch` values;
- a versioned output root that is never silently reused;
- a strict environment profile with absolute Python/compiler paths;
- a preflight task for each training cell;
- numbered checkpoint artifacts and typed downstream dependencies;
- a recovery policy for allocation replacement/evacuation;
- a resolved-config-diff artifact recording what changed.

The normal construction pattern is:

```python
from koochak.jobs import ConfigPatch, PreparedTask, PreparedWorkflow
from koochak.jobs import load_environment_profile, prepare_run
from koochak.jobs import submit_scruffy_workflow

# In the real launcher, build these from the project profile and resource
# helpers; they are abbreviated here for readability.
SCRUFFY_ROOT = "/shared/path/to/scruffy-queue"
train_resources = ...
sample_resources = ...

train = prepare_run(
    name="my-training-cell",
    profile=load_environment_profile("environments/tokyo-gpu.yaml"),
    python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}",
                 "--resume", "auto"],
    cwd="/remote/project/checkout",
    run_dir="/remote/runs/campaign-v1/train/cell-a",
    base_config="/remote/parents/cell-a/config.yaml",
    patches=[
        ConfigPatch("train.max_steps", 500_000),
        ConfigPatch("logging.csv_path", "/remote/runs/.../metrics.csv"),
    ],
)

# Build `sample` as another PreparedRun with the same profile/config contract.
sample = ...

workflow = PreparedWorkflow(
    request_id="project/campaign-v1/attempt-1",
    workflow_id="campaign-v1",
    project_id="project-name",
    tasks=(
        PreparedTask(
            "train-cell-a", train, train_resources,
            recovery={
                "max_attempts": 3,
                "retry_on": [
                    "allocation_replaced",
                    "allocation_incarnation_changed",
                    "evacuated",
                ],
                "evacuation": {"signal": "USR1", "grace_seconds": 600},
            },
        ),
        PreparedTask(
            "sample-cell-a-step100k", sample, sample_resources,
            wait_for=({
                "kind": "artifact",
                "task_id": "train-cell-a",
                "artifact_id": "checkpoint/step000100000.pt",
            },),
        ),
    ),
)

submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
```

The actual project launchers use helper functions for profiles, manifests,
config hashing, resource requests, and evaluation fanout. Follow those local
patterns rather than copying this abbreviated example literally.

### Config discipline

Use structured config patches, not `sed`, string replacement, or shell-built
YAML. For a factorial comparison, the child config should differ from its
parent only in the declared scientific factors plus output identity. Verify
this by flattening and comparing resolved OmegaConf mappings.

For every cell, record:

- parent config path and hash;
- child config hash;
- every changed dotted path and its before/after values;
- code, Koochak, and Scruffy commits;
- seed, horizon, milestones, sample count, and evaluation versions.

## 4. Artifact gates, not sleeps

Submit consumers in the same workflow as producers and gate them on exact,
numbered artifacts:

- `checkpoint/step000100000.pt`, not `latest.pt`;
- a sample directory with a durable ready manifest;
- an ESMFold output directory with its declared record count;
- a Progres analysis JSON with its declared schema.

Koochak publishes an artifact only after the file and ready manifest are
durable. Scruffy then keeps the consumer blocked without reserving resources
until the gate is satisfied. Never put filesystem polling or `sleep` loops in
a blocked task.

The expected shape for a 500k panel is usually:

```text
preflight(cell)
train(cell)
  -> sample(cell, step050k), esmfold(cell, step050k), analysis(cell, step050k)
  -> sample(cell, step100k), esmfold(cell, step100k), analysis(cell, step100k)
  -> ...
```

Evaluation should use the exact training config, checkpoint, sample seed, and
analysis input contract. Loss is not a sufficient scientific endpoint for
this project: compare designability, Progres cluster count/diversity, and
activation/weight diagnostics at matched checkpoints.

## 5. Data locality is part of correctness

The L128 Atom14 mixture is large enough that random Lustre access can dominate
GPU time. Use the project's resident-cache policy:

- assign physical shards disjointly across global `(rank, worker)` owners;
- preload each worker's complete owned shard set when the measured unique
  decoded footprint fits the cgroup;
- preload small conditioning sidecars, including Progres embeddings, under the
  same ownership rule;
- keep `data.shard_cache_size: null` unless a production-shaped gate proves a
  bounded cache is necessary;
- never infer memory from a duplicated worker layout;
- record peak RSS, cache hits/misses, prefetch waits, p50/p90 step time, and
  finite-loss evidence for any changed data policy.

A hit rate near `cache_size / owned_shards`, high prefetch waits, or highly
variable step time is an I/O regression until disproven. Do not “fix” it by
changing model architecture or GPU settings first.

## 6. Monitoring Scruffy

Prefer Scruffy MCP operational views:

- `overview`: allocation health and aggregate counts;
- `running_jobs`: active work;
- `blocked_jobs`: dependency/resource blockers;
- `inspect_job`: authoritative state and reason for one job;
- `tail_job_output`: bounded stdout/stderr diagnosis;
- `wait_job` or `wait_for_updates`: event-driven waiting.

If using the CLI, the equivalent commands are conceptually:

```bash
scruffy --root "$SCRUFFY_ROOT" summary --project "$SCRUFFY_PROJECT"
scruffy --root "$SCRUFFY_ROOT" running --project "$SCRUFFY_PROJECT"
scruffy --root "$SCRUFFY_ROOT" blocked --project "$SCRUFFY_PROJECT"
scruffy --root "$SCRUFFY_ROOT" explain JOB_ID
scruffy --root "$SCRUFFY_ROOT" logs JOB_ID --stream stderr --tail 200
```

For incremental observation, keep the opaque cursor private to the observing
agent and replace it with every returned `next_cursor`. If `reset` is true,
rebuild from a fresh overview. If `more` is true, page immediately. Do not
share cursors between agents or assume an old allocation ID is still valid.

Interpret states this way:

| State | Meaning |
| --- | --- |
| `running` | The task has a live placement |
| `blocked` | A dependency, artifact gate, or resource condition is unsatisfied |
| `skipped` | A required-success dependency ended unsuccessfully |
| `failed` | The task ended non-successfully; inspect its authoritative reason |
| `lost` | Allocation/controller loss has not yet been reconciled |
| `succeeded` | The task completed and its declared outputs passed their contract |

Scheduler lifecycle state is authoritative. Workload stdout/stderr is evidence
for diagnosis, not a substitute for Scruffy state.

## 7. Recovery and uncertain operations

Ordinary inspection can be repeated after a connection repair. Submission,
cancellation, Git writes, and other mutations must not be blindly replayed.

- If a Scruffy submission response is uncertain, reconcile by the same stable
  request ID and identical workflow specification. Scruffy deduplicates an
  identical request; changing resources, argv, dependencies, or profile with
  that ID is rejected.
- If the response is definitely rejected before admission, fix the cause and
  use a new versioned output root/request identity when the specification
  changes.
- If a task is `lost` or the allocation is replaced, inspect the handover and
  let the declared Koochak recovery policy resume from the highest valid
  numbered checkpoint.
- Do not cancel the outer Slurm allocation to stop or evacuate one worker.
  Use Scruffy's scoped evacuation controls.
- Do not manually signal a worker on a compute node or launch a successor
  against the same output directory without reconciling Scruffy state.

Koochak's `--resume auto` is safe only when the checkpoint/manifest contract is
used. It must select a valid numbered checkpoint, never incomplete scaffolding
or an unverified `latest.pt`.

## 8. Common mistakes

1. **Raw `sbatch` for a Koochak experiment.** This bypasses environment,
   artifacts, dependencies, and recovery. Use the committed Koochak launcher.
2. **Training on the login node.** Pazuzu is for bounded inspection and
   submission, not GPU work, compilation, broad scans, or data preparation.
3. **Ambient imports.** Workers use isolated Python; an installed or login-node
   `scruffy` is not a valid handoff. Pin the exact compute-visible Scruffy site
   first in `PYTHONPATH` and declare `scruffy: "*"` in the profile.
4. **One submission per task.** Build one complete DAG and submit it atomically.
5. **Filesystem polling.** Use typed `wait_for` artifact dependencies.
6. **`latest.pt` dependencies.** Use immutable numbered checkpoints.
7. **Arbitrary small caches.** Measure production-shaped memory and I/O before
   changing the resident-cache contract.
8. **Reusing an output root.** Version the campaign identity; never mix a retry,
   changed config, or new code commit into old artifacts.
9. **Reading loss as the experiment result.** For Kaveh, designability,
   diversity, cluster structure, and internal stability are the endpoint.

## 9. Final launch checklist

- [ ] Clean committed checkout and pinned Koochak submodule.
- [ ] Environment profile has absolute Python/PATH/compiler values and the
      compute-visible Scruffy site first in `PYTHONPATH`.
- [ ] Parent configs and hashes are verified.
- [ ] Resolved child diffs contain only declared factors and output identity.
- [ ] Allocation passed fresh Pazuzu and Scruffy health checks.
- [ ] Complete DAG validates before submission.
- [ ] Resources include explicit GPU, CPU, memory, and walltime requests.
- [ ] Resident shard/sidecar ownership follows the data contract.
- [ ] Consumers gate on exact numbered artifacts.
- [ ] Recovery policy and artifact acknowledgement timeout are explicit.
- [ ] Submission uses one stable request ID and one output root.
- [ ] Immediate post-submit Scruffy state has been inspected.

Existing project references: [robust job operations](robust-job-operations.md),
[training infrastructure reliability](training_infrastructure_reliability_plan.md),
and [data loading](data.md).
