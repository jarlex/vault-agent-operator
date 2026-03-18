package scheduler

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"github.com/jarlex/vault-agent-operator/internal/agent"
	"github.com/jarlex/vault-agent-operator/internal/config"
)

// --- Mock TaskExecutor ---

type mockExecutor struct {
	mu          sync.Mutex
	calls       []agent.ExecuteRequest
	result      *agent.AgentResult
	err         error
	callCount   atomic.Int32
	executeChan chan struct{} // signaled on each Execute call
}

func newMockExecutor(result *agent.AgentResult, err error) *mockExecutor {
	return &mockExecutor{
		result:      result,
		err:         err,
		executeChan: make(chan struct{}, 100),
	}
}

func (m *mockExecutor) Execute(_ context.Context, req agent.ExecuteRequest) (*agent.AgentResult, error) {
	m.mu.Lock()
	m.calls = append(m.calls, req)
	m.mu.Unlock()
	m.callCount.Add(1)
	m.executeChan <- struct{}{}
	return m.result, m.err
}

func (m *mockExecutor) getCalls() []agent.ExecuteRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]agent.ExecuteRequest, len(m.calls))
	copy(result, m.calls)
	return result
}

// --- NewEngine ---

func TestNewEngine(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "test", Cron: "*/5 * * * *", Prompt: "test prompt", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	if engine == nil {
		t.Fatal("expected non-nil engine")
	}
	if engine.cron == nil {
		t.Error("expected cron to be initialized")
	}
	if engine.executor == nil {
		t.Error("expected executor to be set")
	}
}

// --- Start with disabled scheduler ---

func TestEngine_Start_Disabled(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: false,
		Tasks: []config.ScheduledTaskDef{
			{ID: "test", Cron: "*/5 * * * *", Prompt: "test prompt", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	// Should not have registered any tasks.
	engine.Stop()
}

// --- Start with no tasks ---

func TestEngine_Start_NoTasks(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks:   []config.ScheduledTaskDef{},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
	engine.Stop()
}

// --- Start with disabled tasks ---

func TestEngine_Start_AllTasksDisabled(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "test1", Cron: "*/5 * * * *", Prompt: "test", Enabled: false},
			{ID: "test2", Cron: "0 * * * *", Prompt: "test2", Enabled: false},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
	engine.Stop()
}

// --- Start with invalid cron expression ---

func TestEngine_Start_InvalidCron(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "bad_cron", Cron: "not a cron expression", Prompt: "test", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	// Start should not return an error — invalid cron is logged and skipped.
	err := engine.Start()
	if err != nil {
		t.Errorf("expected no error even with invalid cron, got: %v", err)
	}
	engine.Stop()
}

// --- Start with mixed valid/invalid crons ---

func TestEngine_Start_MixedCrons(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "valid", Cron: "*/5 * * * *", Prompt: "valid prompt", Enabled: true},
			{ID: "invalid", Cron: "bad cron", Prompt: "invalid prompt", Enabled: true},
			{ID: "disabled", Cron: "*/10 * * * *", Prompt: "disabled prompt", Enabled: false},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
	engine.Stop()
}

// --- Start and Stop lifecycle ---

func TestEngine_StartStop(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "test", Cron: "*/1 * * * *", Prompt: "test", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(&agent.AgentResult{
		Status: "completed", Result: "ok",
	}, nil)

	engine := NewEngine(cfg, executor, logger)

	err := engine.Start()
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Stop should not panic or block indefinitely.
	engine.Stop()
}

// --- executeTask success ---

func TestEngine_ExecuteTask_Success(t *testing.T) {
	result := &agent.AgentResult{
		Status:     "completed",
		Result:     "Vault is healthy",
		ModelUsed:  "gpt-4o",
		DurationMS: 500,
		Iterations: 1,
		ToolCalls: []agent.ToolCallRecord{
			{ToolName: "vault_health", DurationMS: 100},
		},
	}
	executor := newMockExecutor(result, nil)

	cfg := config.SchedulerConfig{Enabled: true}
	logger := zerolog.Nop()
	engine := NewEngine(cfg, executor, logger)

	taskDef := config.ScheduledTaskDef{
		ID:     "health_check",
		Cron:   "*/5 * * * *",
		Prompt: "Check Vault health",
	}

	// Execute the task directly.
	engine.executeTask(taskDef)

	calls := executor.getCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Prompt != "Check Vault health" {
		t.Errorf("expected prompt='Check Vault health', got %q", calls[0].Prompt)
	}
}

// --- executeTask error ---

func TestEngine_ExecuteTask_Error(t *testing.T) {
	executor := newMockExecutor(nil, errors.New("agent crashed"))

	cfg := config.SchedulerConfig{Enabled: true}
	logger := zerolog.Nop()
	engine := NewEngine(cfg, executor, logger)

	taskDef := config.ScheduledTaskDef{
		ID:     "failing_task",
		Cron:   "*/5 * * * *",
		Prompt: "This will fail",
	}

	// Should not panic even when executor returns an error.
	engine.executeTask(taskDef)

	calls := executor.getCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
}

// --- Task with very fast cron (every second via @every) ---

func TestEngine_CronFiresTask(t *testing.T) {
	result := &agent.AgentResult{
		Status:     "completed",
		Result:     "ok",
		DurationMS: 10,
		Iterations: 1,
	}
	executor := newMockExecutor(result, nil)

	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "fast", Cron: "@every 1s", Prompt: "fast task", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	engine := NewEngine(cfg, executor, logger)

	err := engine.Start()
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	// Wait for at least one execution.
	select {
	case <-executor.executeChan:
		// Task was executed.
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for cron task to fire")
	}

	engine.Stop()

	if executor.callCount.Load() < 1 {
		t.Error("expected at least 1 executor call")
	}

	calls := executor.getCalls()
	if calls[0].Prompt != "fast task" {
		t.Errorf("expected prompt='fast task', got %q", calls[0].Prompt)
	}
}

// --- Multiple enabled tasks ---

func TestEngine_MultipleTasks(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "task1", Cron: "*/5 * * * *", Prompt: "prompt1", Enabled: true},
			{ID: "task2", Cron: "0 * * * *", Prompt: "prompt2", Enabled: true},
			{ID: "task3", Cron: "0 9 * * 1", Prompt: "prompt3", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(&agent.AgentResult{Status: "completed"}, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	engine.Stop()
}

// --- defaultTaskTimeout ---

func TestDefaultTaskTimeout(t *testing.T) {
	if defaultTaskTimeout != 5*time.Minute {
		t.Errorf("expected defaultTaskTimeout=5m, got %v", defaultTaskTimeout)
	}
}

// --- Stop without Start ---

func TestEngine_Stop_WithoutStart(t *testing.T) {
	cfg := config.SchedulerConfig{Enabled: true}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	// Stop on a non-started engine should not panic.
	engine.Stop()
}

// --- Start returns nil error ---

func TestEngine_Start_ReturnsNil(t *testing.T) {
	cfg := config.SchedulerConfig{
		Enabled: true,
		Tasks: []config.ScheduledTaskDef{
			{ID: "test", Cron: "*/5 * * * *", Prompt: "test", Enabled: true},
		},
	}
	logger := zerolog.Nop()
	executor := newMockExecutor(nil, nil)

	engine := NewEngine(cfg, executor, logger)
	err := engine.Start()
	if err != nil {
		t.Errorf("expected nil error from Start, got: %v", err)
	}
	engine.Stop()
}
