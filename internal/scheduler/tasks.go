package scheduler

import (
	"context"
	"time"

	"github.com/jarlex/vault-agent-operator/internal/agent"
	"github.com/jarlex/vault-agent-operator/internal/config"
)

// defaultTaskTimeout is the maximum duration for a single scheduled task.
const defaultTaskTimeout = 5 * time.Minute

// executeTask runs a single scheduled task with a timeout. Called by the cron
// scheduler in its own goroutine.
func (e *Engine) executeTask(taskDef config.ScheduledTaskDef) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTaskTimeout)
	defer cancel()

	logger := e.logger.With().Str("task_id", taskDef.ID).Logger()
	logger.Info().
		Str("prompt", taskDef.Prompt).
		Msg("scheduler.task.start")

	startTime := time.Now()

	result, err := e.executor.Execute(ctx, agent.ExecuteRequest{
		Prompt: taskDef.Prompt,
	})

	duration := time.Since(startTime)

	if err != nil {
		logger.Error().
			Err(err).
			Dur("duration", duration).
			Msg("scheduler.task.error")
		return
	}

	logger.Info().
		Str("status", result.Status).
		Int("iterations", result.Iterations).
		Int64("duration_ms", result.DurationMS).
		Int("tool_calls", len(result.ToolCalls)).
		Dur("duration", duration).
		Msg("scheduler.task.complete")
}
