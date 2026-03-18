"""API route modules for vault-operator-agent.

Sub-modules:
- tasks: POST /api/v1/tasks
- health: GET /api/v1/health
- models: GET /api/v1/models
"""

from src.api.routes import health, models, tasks

__all__ = ["health", "models", "tasks"]
