"""Scheduler Engine — APScheduler-based periodic task execution.

Integrates with FastAPI lifespan events to start/stop the scheduler.
Each scheduled task calls ``agent.execute(prompt)`` and logs the result.
Failed tasks are logged at ERROR level but never stop the scheduler.

Usage::

    from src.scheduler.engine import SchedulerEngine
    from src.config.models import SchedulerConfig
    from src.agent.core import AgentCore

    engine = SchedulerEngine(config=scheduler_config, agent=agent)
    engine.start()   # During FastAPI startup
    engine.stop()    # During FastAPI shutdown
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.logging import get_logger

if TYPE_CHECKING:
    from src.agent.core import AgentCore
    from src.config.models import SchedulerConfig

logger = get_logger(__name__)


class SchedulerEngine:
    """APScheduler-based engine for periodic agent tasks.

    The scheduler runs on the same asyncio event loop as FastAPI / Uvicorn,
    so it does **not** block the API server.  Each job is an async coroutine
    that invokes the agent's reasoning loop with a configured prompt.

    Parameters
    ----------
    config:
        Scheduler configuration containing enabled flag and task definitions.
    agent:
        The AgentCore instance used to execute scheduled prompts.
    """

    def __init__(self, config: SchedulerConfig, agent: AgentCore) -> None:
        self._config = config
        self._agent = agent
        self._scheduler: AsyncIOScheduler | None = None

    @property
    def running(self) -> bool:
        """Return ``True`` if the scheduler is currently running."""
        return self._scheduler is not None and self._scheduler.running

    def start(self) -> None:
        """Initialise the AsyncIOScheduler and register jobs from config.

        Only tasks with ``enabled=True`` are registered.  If the scheduler
        itself is disabled in config (``scheduler.enabled=false``), this
        method is a no-op.

        Should be called during FastAPI lifespan startup.
        """
        if not self._config.enabled:
            logger.info("scheduler.disabled", message="Scheduler is disabled in configuration")
            return

        self._scheduler = AsyncIOScheduler()

        registered = 0
        for task in self._config.tasks:
            if not task.enabled:
                logger.info(
                    "scheduler.task.skipped",
                    task_id=task.id,
                    reason="disabled",
                )
                continue

            self.add_task(task_id=task.id, cron=task.cron, prompt=task.prompt)
            registered += 1

        self._scheduler.start()

        logger.info(
            "scheduler.started",
            total_tasks=len(self._config.tasks),
            registered_tasks=registered,
        )

    def stop(self) -> None:
        """Gracefully shut down the scheduler.

        Waits for currently running jobs to finish before stopping.
        Should be called during FastAPI lifespan shutdown.
        """
        if self._scheduler is None or not self._scheduler.running:
            logger.info("scheduler.stop.noop", message="Scheduler is not running")
            return

        self._scheduler.shutdown(wait=True)
        logger.info("scheduler.stopped")

    def add_task(self, task_id: str, cron: str, prompt: str) -> None:
        """Add a periodic task that sends a prompt to the agent.

        Parameters
        ----------
        task_id:
            Unique identifier for the task (used as APScheduler job ID).
        cron:
            Standard 5-field cron expression (minute hour day month weekday).
        prompt:
            Natural-language prompt to execute via ``agent.execute()``.

        Raises
        ------
        RuntimeError
            If the scheduler has not been initialised (``start()`` not called
            or scheduler is disabled).
        ValueError
            If the cron expression is invalid.
        """
        if self._scheduler is None:
            raise RuntimeError(
                "Cannot add task: scheduler has not been initialised. "
                "Call start() first or ensure scheduler is enabled in config."
            )

        try:
            trigger = CronTrigger.from_crontab(cron)
        except ValueError as exc:
            logger.error(
                "scheduler.task.invalid_cron",
                task_id=task_id,
                cron=cron,
                error=str(exc),
            )
            raise

        self._scheduler.add_job(
            func=self._run_task,
            trigger=trigger,
            id=task_id,
            name=f"scheduled-task-{task_id}",
            kwargs={"task_id": task_id, "prompt": prompt},
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs of the same task
        )

        logger.info(
            "scheduler.task.registered",
            task_id=task_id,
            cron=cron,
            prompt_preview=prompt[:80],
        )

    async def _run_task(self, task_id: str, prompt: str) -> None:
        """Execute a single scheduled task.

        Calls ``agent.execute(prompt)`` and logs the result.  Any exception
        is caught and logged at ERROR level — it never propagates to the
        scheduler, so one failing task cannot bring down other tasks or the
        scheduler itself.

        Parameters
        ----------
        task_id:
            The task identifier (for logging).
        prompt:
            The prompt to send to the agent.
        """
        logger.info(
            "scheduler.task.start",
            task_id=task_id,
            prompt_preview=prompt[:80],
        )

        try:
            result = await self._agent.execute(prompt=prompt)

            logger.info(
                "scheduler.task.completed",
                task_id=task_id,
                status=result.status,
                iterations=result.iterations,
                tool_call_count=len(result.tool_calls),
                model_used=result.model_used,
                result_preview=result.result[:200] if result.result else "",
                warning=result.warning,
            )

        except asyncio.CancelledError:
            # Re-raise cancellation — this is expected during shutdown
            logger.info("scheduler.task.cancelled", task_id=task_id)
            raise

        except Exception as exc:
            # Log at ERROR but do NOT re-raise — the scheduler must continue
            logger.error(
                "scheduler.task.failed",
                task_id=task_id,
                error=str(exc),
                exc_type=type(exc).__name__,
            )
