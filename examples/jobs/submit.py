"""Site-neutral examples of the two supported Python submission paths."""

from __future__ import annotations

from koochak.jobs import load_environment_profile, prepare_run


def prepared_run():
    """Return one immutable launch shared by either scheduler backend."""

    return prepare_run(
        name="example-training",
        profile=load_environment_profile("examples/jobs/environment.yaml"),
        python_args=["-m", "examples.mnist.main", "--config", "{config}"],
        cwd="/shared/project/repo",
        run_dir="/shared/project/runs/example-training",
        base_config="examples/mnist/config.yaml",
    )


async def submit_standalone() -> object:
    """Submit through a running Pazuzu gateway."""

    from pazuzu import PazuzuClient, SlurmResources

    from koochak.jobs import submit_pazuzu

    prepared = prepared_run()
    return await submit_pazuzu(
        PazuzuClient(),
        prepared,
        resources=SlurmResources(
            nodes=1,
            gpus_per_node=1,
            cpus_per_task=14,
            memory_gb_per_node=128,
            time_limit="02:00:00",
        ),
        log_dir=f"{prepared.run_dir}/logs",
    )


def submit_to_allocation() -> dict[str, object]:
    """Submit where the Scruffy root and run directory are mounted."""

    from scruffy import ResourceRequest

    from koochak.jobs import submit_scruffy

    return submit_scruffy(
        prepared_run(),
        root="/shared/queues/allocation",
        resources=ResourceRequest(
            nodes=1,
            gpus_per_node=1,
            cpus_per_node=14,
            memory_gb_per_node=128,
            time_limit_seconds=7200,
        ),
        request_id="example/training/attempt-1",
        project_id="example",
    )
