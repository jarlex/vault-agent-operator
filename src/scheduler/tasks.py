"""Built-in task definitions for the scheduler.

Provides pre-defined task templates that can be referenced by configuration
or used programmatically.  These are NOT automatically registered — they
serve as documented defaults and convenience constants.

The actual task registration happens in ``SchedulerEngine.start()`` based on
the ``scheduler.tasks`` section of ``config/default.yaml``.

Usage::

    from src.scheduler.tasks import BUILTIN_TASKS, HEALTH_CHECK

    # Access a specific built-in task
    print(HEALTH_CHECK)
    # {'id': 'health_check', 'cron': '*/5 * * * *', 'prompt': '...', 'enabled': True}

    # Iterate all built-in tasks
    for task in BUILTIN_TASKS:
        print(task["id"])
"""

from __future__ import annotations

from typing import TypedDict


class TaskDefinition(TypedDict):
    """Type for built-in task definition dictionaries."""

    id: str
    cron: str
    prompt: str
    enabled: bool


# ---------------------------------------------------------------------------
# Built-in task definitions
# ---------------------------------------------------------------------------

HEALTH_CHECK: TaskDefinition = {
    "id": "health_check",
    "cron": "*/5 * * * *",
    "prompt": "Check Vault server health and report status",
    "enabled": True,
}

ROTATION_REMINDER: TaskDefinition = {
    "id": "rotation_reminder",
    "cron": "0 9 * * 1",
    "prompt": (
        "List all KV secrets and identify any that may need rotation "
        "based on their age"
    ),
    "enabled": False,
}

SECRET_AUDIT: TaskDefinition = {
    "id": "secret_audit",
    "cron": "0 2 * * *",
    "prompt": (
        "Audit all secret engine mounts and report any misconfigurations, "
        "disabled engines, or unusual access patterns"
    ),
    "enabled": False,
}

# All built-in tasks — can be used to populate defaults or documentation
BUILTIN_TASKS: list[TaskDefinition] = [
    HEALTH_CHECK,
    ROTATION_REMINDER,
    SECRET_AUDIT,
]
