"""Scheduler package for vault-operator-agent.

Provides periodic task execution using APScheduler's AsyncIOScheduler,
integrated with FastAPI lifespan events.

Exports:
    - **SchedulerEngine**: Main scheduler class that manages periodic tasks.
    - **BUILTIN_TASKS**: Pre-defined task templates (health check, etc.).
"""

from src.scheduler.engine import SchedulerEngine
from src.scheduler.tasks import BUILTIN_TASKS

__all__ = [
    "SchedulerEngine",
    "BUILTIN_TASKS",
]
