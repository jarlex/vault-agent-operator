# LLM Multi-Model Support: Research Report

**Project**: vault-operator-agent (PeTe-agent)
**Date**: 2026-03-17
**Context**: The agent needs a single abstraction layer to support multiple LLM models (GPT-4o, Claude, Llama, Gemini, etc.) with **mandatory tool calling / function calling** support for MCP tool invocation against HashiCorp Vault.

---

## Table of Contents

1. [Critical Requirements](#critical-requirements)
2. [Option Analysis](#option-analysis)
   - [1. LiteLLM](#1-litellm)
   - [2. GitHub Models API](#2-github-models-api)
   - [3. OpenRouter](#3-openrouter)
   - [4. Portkey](#4-portkey)
   - [5. Ollama](#5-ollama)
   - [6. LangChain / LangChain Core](#6-langchain--langchain-core)
   - [7. Azure AI Model Catalog (Azure AI Inference)](#7-azure-ai-model-catalog-azure-ai-inference)
   - [8. Amazon Bedrock](#8-amazon-bedrock)
   - [9. Google Vertex AI](#9-google-vertex-ai)
   - [10. Direct Provider SDKs (Custom Abstraction)](#10-direct-provider-sdks-custom-abstraction)
   - [11. Instructor](#11-instructor)
   - [12. Haystack (deepset)](#12-haystack-deepset)
3. [Comparison Table](#comparison-table)
4. [Tiered Recommendations](#tiered-recommendations)
5. [Recommended Architecture](#recommended-architecture)
6. [Final Recommendation](#final-recommendation)

---

## Critical Requirements

| Requirement | Priority | Notes |
|---|---|---|
| Tool calling / function calling | **P0** | Agent invokes 19 MCP tools against Vault. Non-negotiable. |
| Python SDK quality | **P0** | Agent is Python 3.12 + FastAPI |
| Multi-model switching via config | **P0** | GPT-4o, Claude, Llama, Gemini at minimum |
| Docker-friendly | **P1** | Runs in Docker containers |
| Config-driven (YAML/env) | **P1** | No code changes to switch models |
| Fallback / retry support | **P2** | Graceful degradation if primary model is unavailable |
| Cost tracking | **P3** | Visibility into LLM spend |
| Self-hostable | **P3** | Optional; cloud APIs are acceptable |

---

## Option Analysis

### 1. LiteLLM

| Attribute | Details |
|---|---|
| **URL** | https://github.com/BerriAI/litellm |
| **Stars** | 39,400+ |
| **License** | MIT (core SDK); Enterprise features separate |
| **Backed by** | Y Combinator W23 |
| **Used by** | Stripe, Google ADK, Netflix, OpenHands |

**How it works**: Python SDK that provides a unified `litellm.completion()` / `litellm.acompletion()` interface translating to 100+ LLM providers. Can also be deployed as a standalone **proxy server** (AI Gateway) with a REST API compatible with OpenAI's format.

**Models supported**: 100+ providers including OpenAI, Anthropic, Google (Gemini/Vertex), AWS Bedrock, Azure, Ollama, Hugging Face, Cohere, Mistral, Groq, Together AI, Fireworks, Replicate, and critically: **GitHub Models** via `github/` prefix.

**Tool calling support**: **Full support**. Provides `litellm.supports_function_calling(model="...")` helper to check model capability at runtime. Translates OpenAI-format tool definitions to each provider's native format. Also has an **experimental MCP bridge** (`experimental_mcp_client.load_mcp_tools()`) that can load MCP tools directly into the completion call.

**Pricing**: Free (MIT licensed SDK). Proxy server is also free. Enterprise features (SSO, audit logs, advanced analytics) are paid.

**Python SDK quality**: Excellent. Pure Python, async-first (`acompletion`), type-hinted, well-documented. 500+ contributors.

**Docker-friendliness**: Official Docker images available. Proxy server has `docker-compose.yaml` examples. Reports 8ms P95 latency at 1,000 RPS.

**Configuration**: YAML-based `config.yaml` with `model_list` supporting aliases, fallbacks, routing, and load balancing:
```yaml
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "github/gpt-4o"
      api_key: "os.environ/GITHUB_TOKEN"
  - model_name: "claude"
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"
      api_key: "os.environ/ANTHROPIC_API_KEY"
```

**Pros**:
- Largest provider coverage (100+)
- Mature, battle-tested at scale (Stripe, Netflix)
- Native GitHub Models support (`github/model-name`)
- Built-in MCP bridge for tool loading
- Fallback chains, retries, load balancing out of the box
- Cost tracking per request
- Can be used as SDK (embedded) or proxy (separate service)
- Config-driven model switching without code changes

**Cons**:
- Adds a dependency (~30MB installed)
- Proxy mode adds operational complexity (if used)
- Some provider-specific edge cases in tool calling translation
- Enterprise features require paid license

---

### 2. GitHub Models API

| Attribute | Details |
|---|---|
| **URL** | https://github.com/marketplace/models |
| **Type** | Hosted API (Microsoft/Azure-backed) |
| **Auth** | GitHub PAT with `models:read` scope |

**How it works**: Free API endpoint at `https://models.inference.ai.azure.com` providing access to multiple models through an OpenAI-compatible interface. Uses the Azure AI Inference SDK under the hood.

**Models supported**: GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet (preview), Llama 3.x, Phi-4, DeepSeek-R1, Grok-3, Mistral, Cohere, and others. Model availability changes as GitHub adds/removes from marketplace.

**Tool calling support**: Supported for models that natively support it (GPT-4o, Claude, Llama 3.1+). Uses standard OpenAI `tools` parameter.

**Pricing (Free Tier)**:
| Tier | Rate Limit | Daily Limit | Token Limits |
|---|---|---|---|
| Low (GPT-4o, Claude) | 10-15 RPM | 50-150 req/day | 4,000-8,000 in/out |
| High (Phi, Llama small) | 15-30 RPM | 150-300 req/day | 4,000-8,000 in/out |
| Embedding | 15 RPM | 150 req/day | 64,000 in |

Paid tier available through Azure billing (opt-in).

**Python SDK quality**: Uses standard `openai` Python SDK (just change `base_url`). Also has `azure-ai-inference` SDK.

**Docker-friendliness**: No container needed; it's a cloud API. Just set `GITHUB_TOKEN` env var.

**Pros**:
- Free with GitHub Copilot subscription
- OpenAI-compatible — works with existing `openai` SDK code
- Multiple models in one endpoint
- Zero infrastructure to manage

**Cons**:
- **Severely rate-limited** on free tier (50-150 req/day for GPT-4o/Claude)
- Token limits are restrictive (4K-8K in/out)
- Model availability is curated by GitHub (can't access everything)
- Not viable as sole production provider without paid tier
- No fallback/retry/load-balancing built in
- Vendor dependency on GitHub/Microsoft

**Verdict**: Excellent for prototyping and development. Should be one provider behind LiteLLM, not the sole abstraction layer.

---

### 3. OpenRouter

| Attribute | Details |
|---|---|
| **URL** | https://openrouter.ai |
| **Type** | Hosted API gateway |

**How it works**: API gateway that routes requests to multiple LLM providers. Endpoint at `https://openrouter.ai/api/v1/chat/completions` is OpenAI-compatible.

**Models supported**: Hundreds of models from OpenAI, Anthropic, Google, Meta, Mistral, etc. Aggregates across providers.

**Tool calling support**: Supported for models that support it natively. Uses OpenAI-format `tools` parameter.

**Pricing**: Pay-per-token, passed through from underlying providers with a small markup. Some free models available. Credit-based system.

**Python SDK quality**: Uses standard `openai` SDK by changing `base_url` to `https://openrouter.ai/api/v1`. Also has community SDKs.

**Docker-friendliness**: Cloud API; no self-hosting option.

**Pros**:
- Very large model selection
- OpenAI-compatible — trivial integration
- Pay-per-use, no subscriptions
- Model comparison and routing features
- Can try new models without separate API keys

**Cons**:
- No self-hosting — all traffic routes through OpenRouter
- Adds latency (extra hop)
- Pricing markup over direct provider access
- Single point of failure if OpenRouter goes down
- Less control over provider-specific parameters
- Documentation gaps (tool calling docs returned 404 during research)

---

### 4. Portkey

| Attribute | Details |
|---|---|
| **URL** | https://github.com/Portkey-AI/gateway |
| **Stars** | 10,900+ |
| **License** | MIT (gateway) |

**How it works**: Open-source AI Gateway (TypeScript/Node.js) that sits between your app and LLM providers. Routes requests with <1ms latency overhead (122KB footprint). Also available as a managed cloud service.

**Models supported**: 250+ LLMs across all major providers.

**Tool calling support**: Supported, passes through to underlying providers.

**Pricing**: Gateway is free (MIT). Managed platform has free tier (10K req/month), paid tiers above.

**Python SDK quality**: `portkey_ai` package available. Also works with standard `openai` SDK by changing base URL.

**Docker-friendliness**: Can be self-hosted via Docker (but it's a Node.js/TypeScript service, not Python).

**Features**: Retries, fallbacks, load balancing, guardrails, semantic caching, request logging, MCP Gateway support.

**Pros**:
- Open source, self-hostable
- Very low latency overhead
- Rich feature set (guardrails, caching, observability)
- MCP Gateway support (can expose LLM as MCP tools)
- Works with OpenAI SDK

**Cons**:
- Gateway is **Node.js/TypeScript** — adds a non-Python service to the Docker Compose stack
- Operational overhead of running a separate gateway service
- Managed platform has request limits on free tier
- Smaller community than LiteLLM
- Adds architectural complexity for what could be SDK-level abstraction

---

### 5. Ollama

| Attribute | Details |
|---|---|
| **URL** | https://github.com/ollama/ollama |
| **Stars** | 165,000+ |
| **License** | MIT |

**How it works**: Local model runner that downloads and serves open-source LLMs (Llama, Mistral, Phi, Gemma, etc.) via a REST API. Provides an OpenAI-compatible endpoint at `http://localhost:11434/v1/`.

**Models supported**: Hundreds of open-source models. No proprietary models (no GPT-4o, no Claude).

**Tool calling support**: Supported since July 2024. Works with OpenAI-format `tools` parameter. Streaming tool calls supported.

**Pricing**: Completely free. Self-hosted. Costs only compute resources.

**Python SDK quality**: `ollama-python` library available. Also works with `openai` SDK by changing `base_url`.

**Docker-friendliness**: Official Docker image. Easy to run: `docker run ollama/ollama`.

**Pros**:
- Completely free, no API keys needed
- Full data privacy — nothing leaves your network
- Massive community (165K stars)
- OpenAI-compatible API
- Great for development and testing
- Tool calling works with capable models (Llama 3.1+, Mistral, etc.)

**Cons**:
- **Not a multi-provider abstraction** — only serves local models
- Requires GPU for good performance (CPU inference is slow)
- Model quality lower than GPT-4o/Claude for complex reasoning
- Tool calling reliability varies by model
- Only open-source models — no access to proprietary models

**Verdict**: Excellent as one **provider** behind LiteLLM (e.g., for offline/development use), but not a solution by itself for multi-model support.

---

### 6. LangChain / LangChain Core

| Attribute | Details |
|---|---|
| **URL** | https://github.com/langchain-ai/langchain |
| **Stars** | 100,000+ |
| **License** | MIT |

**How it works**: Comprehensive AI application framework with provider abstractions (`ChatOpenAI`, `ChatAnthropic`, `ChatOllama`, etc.), tool/agent interfaces, chain composition, memory, and more.

**Models supported**: All major providers via `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-aws`, `langchain-community`, etc.

**Tool calling support**: Full support via `bind_tools()` and agent executors. Standardized tool interface across providers.

**Pricing**: Free (MIT). LangSmith (observability platform) is paid.

**Python SDK quality**: Large, well-documented. Active development. However, API surface is very large and has had breaking changes between versions.

**Docker-friendliness**: Pure Python; no special Docker considerations.

**Pros**:
- Massive ecosystem and community
- Standardized tool calling across providers
- Rich agent abstractions
- Extensive documentation and examples
- Can handle complex multi-step reasoning out of the box

**Cons**:
- **Heavyweight** — pulls in significant dependencies and framework opinions
- Our proposal already chose a simpler architecture (openai SDK + custom reasoning loop)
- Abstraction layers can obscure provider-specific behavior
- Frequent API changes and version churn
- Overkill for "just" LLM abstraction — we'd use 5% of LangChain's features
- Debugging through multiple abstraction layers is painful
- Learning curve is steep

**Verdict**: Not recommended for this project. The agent's architecture is deliberately simple (direct tool calling loop). LangChain would add significant complexity for minimal benefit. If we only need model abstraction, LiteLLM does it without the framework baggage.

---

### 7. Azure AI Model Catalog (Azure AI Inference)

| Attribute | Details |
|---|---|
| **URL** | https://ai.azure.com/explore/models |
| **Type** | Cloud service (Microsoft Azure) |

**How it works**: Azure's model marketplace offering serverless API endpoints for multiple model families. Access via Azure AI Inference SDK or OpenAI-compatible endpoint.

**Models supported**: GPT-4o, GPT-4o-mini, Llama 3.x, Mistral, Phi-4, Cohere, and others deployed on Azure infrastructure.

**Tool calling support**: Supported for compatible models via standard OpenAI-format or Azure AI Inference SDK.

**Pricing**: Pay-per-token. Pricing varies by model. Some models have free tiers. Requires Azure subscription.

**Python SDK quality**: `azure-ai-inference` SDK. Also works with `openai` SDK via Azure-specific configuration. Well-maintained by Microsoft.

**Docker-friendliness**: Cloud API; no container needed.

**Pros**:
- Enterprise-grade reliability and SLA
- Same models as GitHub Models but with production rate limits
- OpenAI-compatible endpoint
- Integrates with Azure ecosystem (RBAC, networking, monitoring)
- Natural upgrade path from GitHub Models (same underlying infrastructure)

**Cons**:
- **Requires Azure subscription and billing**
- Vendor lock-in to Microsoft/Azure ecosystem
- More complex authentication (Azure AD, managed identities) vs. simple API keys
- Model availability depends on Azure region
- Not a multi-provider abstraction by itself — only Azure-hosted models
- Overkill if you don't need Azure's enterprise features

**Verdict**: Good option for production if the organization is already on Azure. Best accessed as a provider through LiteLLM rather than used directly as the abstraction layer.

---

### 8. Amazon Bedrock

| Attribute | Details |
|---|---|
| **URL** | https://aws.amazon.com/bedrock |
| **Type** | Cloud service (AWS) |

**How it works**: AWS's managed service for foundation models. Access multiple models through a unified AWS API using the Bedrock Runtime SDK (`boto3`).

**Models supported**: Claude (Anthropic), Llama (Meta), Mistral, Cohere Command, Amazon Titan, Stability AI, and others. Model access must be explicitly enabled per-region.

**Tool calling support**: Supported via Bedrock's Converse API (`converse()` / `converse_stream()`). Uses Bedrock-specific tool format (not OpenAI-compatible).

**Pricing**: Pay-per-token. Pricing varies by model. On-demand and provisioned throughput options. Requires AWS account.

**Python SDK quality**: `boto3` with Bedrock Runtime client. Well-maintained but verbose. Not OpenAI-compatible format.

**Docker-friendliness**: Cloud API. Needs AWS credentials (IAM roles, access keys).

**Pros**:
- Enterprise-grade with AWS SLA
- Access to Claude models with higher rate limits than direct Anthropic API (in some cases)
- AWS ecosystem integration (IAM, VPC, CloudWatch)
- Guardrails feature for content filtering
- Knowledge Bases and Agents features (if needed later)

**Cons**:
- **Requires AWS account and billing**
- **Non-OpenAI-compatible API** — Bedrock has its own tool calling format
- Vendor lock-in to AWS
- Model availability varies by region
- More complex auth (IAM roles, STS)
- Not a multi-provider abstraction — only AWS-hosted models
- Verbose SDK (`boto3` is not the most ergonomic)

**Verdict**: Good if the organization is on AWS. Best accessed through LiteLLM, which handles the format translation automatically.

---

### 9. Google Vertex AI

| Attribute | Details |
|---|---|
| **URL** | https://cloud.google.com/vertex-ai |
| **Type** | Cloud service (Google Cloud) |

**How it works**: Google Cloud's ML platform offering access to Gemini models, PaLM, and third-party models (Claude, Llama) through Model Garden. Unified API via `google-cloud-aiplatform` SDK.

**Models supported**: Gemini 1.5/2.0 (Pro, Flash, Ultra), Claude (via Model Garden), Llama (via Model Garden), Mistral, and others.

**Tool calling support**: Full support in Gemini models via `function_calling` parameter. Uses Google's own format (not OpenAI-compatible).

**Pricing**: Pay-per-token. Gemini has a generous free tier via Google AI Studio. Vertex AI requires GCP billing.

**Python SDK quality**: `google-cloud-aiplatform` and `google-generativeai` SDKs. Well-maintained but Google-specific format. Also `vertexai` SDK.

**Docker-friendliness**: Cloud API. Needs GCP credentials (service accounts, application default credentials).

**Pros**:
- Access to Gemini models (strong tool calling, large context windows)
- Third-party models via Model Garden
- GCP ecosystem integration
- Generous free tier on Google AI Studio (Gemini API)
- Strong multimodal capabilities

**Cons**:
- **Non-OpenAI-compatible API** (Google's own format for tool calling)
- Requires GCP account for Vertex AI
- Vendor lock-in to Google Cloud
- Multiple SDK options are confusing (`generativeai` vs `aiplatform` vs `vertexai`)
- Not a multi-provider abstraction

**Verdict**: Worth having as a provider option (especially for Gemini models). Best accessed through LiteLLM, which handles format translation.

---

### 10. Direct Provider SDKs (Custom Abstraction)

| Attribute | Details |
|---|---|
| **Type** | Build-your-own abstraction layer |

**How it works**: Install each provider's SDK (`openai`, `anthropic`, `google-generativeai`, `boto3`, etc.) and write a custom abstraction layer with a common interface (`complete(messages, tools)`) that dispatches to the appropriate SDK based on configuration.

**Models supported**: Any model from any provider you integrate.

**Tool calling support**: Depends on your implementation. Each provider has different tool calling formats that must be translated.

**Pricing**: Free to build; you pay each provider directly.

**Python SDK quality**: Depends on each provider's SDK individually.

**Pros**:
- Full control over behavior
- No third-party abstraction dependency
- Can optimize for specific providers
- Minimal dependency footprint per provider

**Cons**:
- **Significant development and maintenance effort**
- Must handle tool calling format differences across providers manually
- Must implement retries, fallbacks, error handling yourself
- Must track API changes across all providers
- Must handle streaming, token counting, cost tracking yourself
- Reinventing what LiteLLM already does with 500+ contributors
- High surface area for bugs
- Every new provider requires new integration code

**Verdict**: Not recommended. This is exactly what LiteLLM exists to solve. Building custom means maintaining format translations for tool calling across 4+ providers — a significant ongoing burden.

---

### 11. Instructor

| Attribute | Details |
|---|---|
| **URL** | https://github.com/jxnl/instructor |
| **Stars** | 10,000+ |
| **License** | MIT |

**How it works**: Library for structured output extraction from LLMs using Pydantic models. Patches the OpenAI client to return typed Pydantic objects instead of raw completions.

**Models supported**: Works with OpenAI, Anthropic, Google, Ollama, LiteLLM, and others.

**Tool calling support**: Uses tool calling internally to enforce structured output, but is focused on **output parsing**, not general tool invocation.

**Pros**:
- Excellent for structured output parsing
- Works with multiple providers
- Clean Pydantic integration

**Cons**:
- **Not a general-purpose tool calling abstraction** — designed for structured output, not for invoking external tools like MCP
- Doesn't solve the multi-provider routing/switching problem
- Complementary to LiteLLM, not a replacement

**Verdict**: Could be useful for parsing agent responses into structured formats, but does not address the core multi-model abstraction requirement.

---

### 12. Haystack (deepset)

| Attribute | Details |
|---|---|
| **URL** | https://github.com/deepset-ai/haystack |
| **Stars** | 19,000+ |
| **License** | Apache 2.0 |

**How it works**: Framework for building AI pipelines with provider abstractions. Component-based architecture with generators, retrievers, and converters.

**Models supported**: OpenAI, Anthropic, Google, Azure, Ollama, Hugging Face, Cohere, etc.

**Tool calling support**: Supported via ChatGenerator components.

**Pros**:
- Clean component-based architecture
- Good provider abstractions
- Pipeline composition

**Cons**:
- Full framework (like LangChain, but different philosophy)
- Overkill for just model abstraction
- Less adoption than LangChain
- Would conflict with our simple agent loop architecture

**Verdict**: Not recommended for the same reasons as LangChain — we need model abstraction, not a framework.

---

## Comparison Table

| Option | Tool Calling | Multi-Provider | Self-Hostable | Python SDK | Config-Driven | Fallbacks | Docker Ready | Complexity | Stars |
|---|---|---|---|---|---|---|---|---|---|
| **LiteLLM** | Full | 100+ providers | SDK + Proxy | Excellent | YAML | Built-in | Yes | Low-Medium | 39K |
| **GitHub Models** | Yes | ~15 models | No | openai SDK | Env vars | No | N/A (API) | Low | N/A |
| **OpenRouter** | Yes | Hundreds | No | openai SDK | Env vars | No | N/A (API) | Low | N/A |
| **Portkey** | Yes | 250+ | Yes (Node.js) | Good | YAML/JSON | Built-in | Yes (Node) | Medium | 11K |
| **Ollama** | Yes | Local only | Yes | Good | CLI/API | No | Yes | Low | 165K |
| **LangChain** | Full | All major | N/A (SDK) | Large | Code-based | Via agents | N/A | High | 100K |
| **Azure AI** | Yes | Azure models | No | Good | Code/env | No | N/A (API) | Medium | N/A |
| **Amazon Bedrock** | Yes (own format) | AWS models | No | boto3 | Code/env | No | N/A (API) | Medium-High | N/A |
| **Google Vertex** | Yes (own format) | GCP models | No | Good | Code/env | No | N/A (API) | Medium | N/A |
| **Custom SDKs** | Manual impl | Any | N/A | Per-provider | Custom | Custom | N/A | **Very High** | N/A |
| **Instructor** | Output only | Via patching | N/A | Excellent | N/A | N/A | N/A | Low | 10K |
| **Haystack** | Yes | All major | N/A (SDK) | Good | Pipeline config | Custom | N/A | High | 19K |

---

## Tiered Recommendations

### Tier 1: Recommended

#### LiteLLM (SDK mode) — PRIMARY RECOMMENDATION

**Why**: LiteLLM is the clear winner for this project. It checks every box:
- Native tool calling support with cross-provider format translation
- 100+ providers including native GitHub Models support (`github/gpt-4o`)
- Config-driven model switching via YAML
- Built-in fallback chains, retries, and load balancing
- Built-in MCP bridge for loading tools directly
- Battle-tested at scale (Stripe, Netflix)
- Pure Python, async-first, minimal overhead
- MIT licensed

**How to use it**: Import `litellm` in the agent core and call `litellm.acompletion()` instead of `openai.chat.completions.create()`. The model string determines the provider:
```python
# GitHub Models
response = await litellm.acompletion(model="github/gpt-4o", messages=messages, tools=tools)

# Direct OpenAI
response = await litellm.acompletion(model="gpt-4o", messages=messages, tools=tools)

# Anthropic
response = await litellm.acompletion(model="anthropic/claude-sonnet-4-20250514", messages=messages, tools=tools)

# Ollama (local)
response = await litellm.acompletion(model="ollama/llama3.1", messages=messages, tools=tools)
```

**Risk**: Adding a dependency. Mitigated by: MIT license, massive adoption, active maintenance, and the alternative (custom abstraction) is far worse.

---

### Tier 2: Viable Alternatives

#### GitHub Models API (as a provider, not abstraction)

Use as one provider behind LiteLLM for development/prototyping. Free tier is useful for testing but insufficient for production.

#### OpenRouter (as a provider)

Useful as a fallback provider or for accessing models not available through other providers. Can be configured in LiteLLM as a provider.

#### Portkey (if gateway architecture is preferred)

If the team prefers an external gateway over an embedded SDK, Portkey is a solid choice. The tradeoff is adding a Node.js service to the stack. Consider this if the project grows to need advanced features like semantic caching or guardrails at the gateway level.

---

### Tier 3: Niche / Complementary

#### Ollama (for local development)

Run as a Docker sidecar for local model testing without API costs. Configure in LiteLLM as `ollama/model-name`. Useful for development, CI/CD testing, and air-gapped environments.

#### Azure AI / Bedrock / Vertex AI (cloud-specific providers)

Use if the deployment target is a specific cloud. All can be accessed through LiteLLM without direct SDK integration. The natural upgrade path from GitHub Models is Azure AI (same underlying infrastructure).

#### Instructor (for structured output)

Could complement LiteLLM for parsing agent responses into Pydantic models if structured output becomes a requirement.

---

### Not Recommended

| Option | Reason |
|---|---|
| **LangChain** | Heavyweight framework; conflicts with our simple agent loop architecture |
| **Haystack** | Same as LangChain — framework overkill for model abstraction |
| **Custom SDKs** | High development/maintenance cost; reinvents what LiteLLM solves |

---

## Recommended Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    vault-operator-agent                         │
│                                                                │
│   ┌──────────┐    ┌──────────────────┐    ┌───────────────┐   │
│   │ FastAPI   │───▶│  Agent Core      │───▶│  MCP Client   │──┼──▶ vault-mcp-server
│   │ (mTLS)   │    │  (reasoning loop) │    └───────────────┘   │
│   └──────────┘    └────────┬─────────┘                         │
│                            │                                    │
│                   ┌────────▼─────────┐                         │
│                   │  LiteLLM SDK     │                         │
│                   │  (embedded)      │                         │
│                   └────────┬─────────┘                         │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       GitHub Models    OpenAI API    Anthropic API
       (dev/free)       (production)  (fallback)
              │
              ▼
         Ollama (local, optional)
```

### Configuration (config/models.yaml)

```yaml
# Primary model for production
model_list:
  - model_name: "default"
    litellm_params:
      model: "github/gpt-4o"
      api_key: "os.environ/GITHUB_TOKEN"

  - model_name: "default"  # fallback for same alias
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"
      api_key: "os.environ/ANTHROPIC_API_KEY"

  - model_name: "fast"
    litellm_params:
      model: "github/gpt-4o-mini"
      api_key: "os.environ/GITHUB_TOKEN"

  - model_name: "local"
    litellm_params:
      model: "ollama/llama3.1"
      api_base: "http://ollama:11434"

litellm_settings:
  fallbacks: [{"default": ["fast"]}]
  num_retries: 3
  request_timeout: 60
```

### Integration in Agent Core (src/llm/provider.py)

```python
import litellm
from typing import Any

async def complete(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "default",
) -> Any:
    """Unified LLM completion with tool calling support."""
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
    )
    return response
```

### Key Benefits of This Architecture

1. **Zero code changes to switch models** — change `config/models.yaml` or set `DEFAULT_MODEL` env var
2. **Automatic fallback** — if GitHub Models is rate-limited, falls back to Anthropic
3. **Local development** — point to Ollama for zero-cost testing
4. **Tool calling just works** — LiteLLM handles format translation per provider
5. **Future-proof** — adding a new provider is one YAML entry, not new code
6. **Preserves simple architecture** — LiteLLM is a thin SDK, not a framework

---

## Final Recommendation

**Use LiteLLM as an embedded SDK** (not proxy mode) for the `src/llm/` module.

**Rationale**:
1. It solves the exact problem statement — unified multi-model access with tool calling
2. It has the strongest tool calling support across providers, which is our P0 requirement
3. It natively supports GitHub Models (our free-tier development provider)
4. It's battle-tested at companies operating at far larger scale than ours
5. It's MIT licensed with no vendor lock-in
6. It's pure Python, async-first, and adds minimal complexity to our stack
7. Config-driven model switching aligns with our Docker/env-based configuration approach

**Combination strategy**:
- **LiteLLM** as the abstraction layer (replaces direct `openai` SDK usage)
- **GitHub Models** as the primary development/free-tier provider
- **OpenAI API** or **Anthropic API** as the production provider (when rate limits matter)
- **Ollama** as an optional local provider for offline development
- All accessed through LiteLLM's unified interface

**Impact on proposal**: The `src/llm/` module in the proposal should use `litellm` instead of the `openai` SDK directly. The `pyproject.toml` dependency changes from `openai` to `litellm` (which includes `openai` as a transitive dependency). Everything else in the proposal remains unchanged.
