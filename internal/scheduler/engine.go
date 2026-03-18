package scheduler

import (
	"context"

	"github.com/robfig/cron/v3"
	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/agent"
	"github.com/jarlex/vault-agent-operator/internal/config"
)

// TaskExecutor runs agent tasks. Satisfied by *agent.AgentCore.
type TaskExecutor interface {
	Execute(ctx context.Context, req agent.ExecuteRequest) (*agent.AgentResult, error)
}

// Engine manages cron-based periodic task execution.
type Engine struct {
	cron     *cron.Cron
	executor TaskExecutor
	config   config.SchedulerConfig
	logger   zerolog.Logger
}

// NewEngine creates a new scheduler engine. Tasks are not started until Start
// is called.
func NewEngine(cfg config.SchedulerConfig, executor TaskExecutor, logger zerolog.Logger) *Engine {
	return &Engine{
		cron:     cron.New(),
		executor: executor,
		config:   cfg,
		logger:   logger.With().Str("component", "scheduler").Logger(),
	}
}

// Start registers all enabled tasks from configuration and starts the cron
// scheduler. It returns immediately — the cron scheduler runs in its own
// goroutine.
func (e *Engine) Start() error {
	if !e.config.Enabled {
		e.logger.Info().Msg("scheduler.disabled")
		return nil
	}

	registered := 0
	for _, taskDef := range e.config.Tasks {
		if !taskDef.Enabled {
			e.logger.Debug().
				Str("task_id", taskDef.ID).
				Msg("scheduler.task.skipped_disabled")
			continue
		}

		// Capture taskDef for the closure.
		td := taskDef
		_, err := e.cron.AddFunc(td.Cron, func() {
			e.executeTask(td)
		})
		if err != nil {
			e.logger.Error().
				Err(err).
				Str("task_id", td.ID).
				Str("cron", td.Cron).
				Msg("scheduler.task.invalid_cron")
			continue
		}

		e.logger.Info().
			Str("task_id", td.ID).
			Str("cron", td.Cron).
			Msg("scheduler.task.registered")
		registered++
	}

	e.logger.Info().
		Int("registered_tasks", registered).
		Msg("scheduler.start")

	e.cron.Start()
	return nil
}

// Stop halts the scheduler and waits for any running tasks to complete.
func (e *Engine) Stop() {
	e.logger.Info().Msg("scheduler.stop")
	ctx := e.cron.Stop()
	<-ctx.Done()
	e.logger.Info().Msg("scheduler.stopped")
}
