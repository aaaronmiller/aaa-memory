# Hermes Config Audit — Cold-Start Tool-Use Reliability
**Date**: 2026-04-08  
**Source**: Multi-session analysis with community validation

---

## Misconfigurations Found

### 1. CRITICAL: Trinity Large Preview lacks OR tool-use endpoint verification
- OpenRouter free model listing shows Trinity Large Preview **without** "Tools" capability badge
- MiniMax M2.5, Trinity Mini, GPT-OSS-120B, GLM-4.5-Air **all have** the Tools badge
- When tool_calls model fails and Hermes falls back, it hits Trinity first → OR may reject with 404 for tool-use requests
- Community confirmed: "No endpoints found that support tool use" error for :free suffix + tool-calling
- **Fix**: Remove Trinity from fallback chain for tool-use scenarios

### 2. Secondary: Gemma 4 31B lacks Tools badge
- Carries only Vision capability on OR
- Auxiliary tasks involving tool use could silently fail
- **Fix**: Replace with Gemma 4 26B MoE variant (has Tools badge, 83.9% PinchBench vs 76.4% for dense)

### 3. Compression settings suboptimal
- threshold: 0.25 → too aggressive, breaks prompt caching during tool-heavy sessions
- target_ratio: 0.10 → retains too little context
- protect_last_n: 5 → not enough protection for recent tool results
- **Fix**: threshold 0.50, target_ratio 0.20, protect_last_n 10

### 4. Fallback cascade applies equally to all model slots
- fallback_providers list applies to primary model ONLY
- tool_calls model does NOT have its own fallback chain
- auxiliary model has separate auto-detection chain (OpenRouter → Nous Portal → Custom → API-key providers)
- **Three separate failure recovery paths for three separate model slots**

---

## Community Validation

### Confirmed Issues
- OpenRouter free tier: 50 requests/day without credits, 1000/day with $10 (20x multiplier)
- Free models rotate monthly — need 6+ fallback entries for insurance against rotation
- OR may silently re-route requests to different models on free tier
- OpenClaw Discord: "Picking a random free model often fails because many free models are not reliable at tool-calling"
- OpenClaw is the #1 user of Nemotron Nano on OpenRouter

### Config Recommendations from Community
- Use `openrouter/free` as meta-fallback (OR auto-selects best available free model with tool support)
- Set `require_parameters: true` on OR requests to hard-filter for tool-supporting endpoints
- Monitor via OR response headers: x-openrouter-processing-time and model/provider metadata
- hermes-plugins collection has model selection plugin for dynamic routing

---

## Final Optimized Config

```yaml
model:
  default: arcee-ai/trinity-large-preview:free
  tool_calls: minimax/minimax-m2.5:free
  auxiliary: google/gemma-4-26b-a4b-it:free
  base_url: http://127.0.0.1:8082/v1
  api_mode: chat_completions
  providers: {}
  fallback_providers:
    - provider: openrouter
      model: minimax/minimax-m2.5:free
    - provider: openrouter
      model: z-ai/glm-4.5-air:free
    - provider: openrouter
      model: stepfun/step-3.5-flash:free
    - provider: openrouter
      model: google/gemma-4-26b-a4b-it:free
    - provider: openrouter
      model: arcee-ai/trinity-large-preview:free
    - provider: openrouter
      model: nvidia/nemotron-3-super-120b-a12b:free

smart_model_routing:
  enabled: true
  max_simple_chars: 160
  max_simple_words: 28
  cheap_model:
    provider: openrouter
    model: nvidia/nemotron-3-nano-30b-a3b:free

compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20
  protect_last_n: 10

agent:
  max_turns: 240
  gateway_timeout: 1800
  tool_use_enforcement: auto
  reasoning_effort: medium
```

### What Was Removed
- `gemma-4-31b-it:free` — dense variant, beaten by MoE (76.4% vs 83.9%)
- `gpt-oss-120b:free` — worst at agent loops (67.1% PinchBench, 50.2% avg)

### What Was Added
- `glm-4.5-air:free` — 85.7% PinchBench, 76.4% BFCL-v3, perfect Galileo Tool Selection
- `gemma-4-26b-a4b-it:free` — MoE variant at 83.9% PinchBench
- `nemotron-3-nano-30b-a3b:free` — community-validated cheap model for subagent tasks

---

## Fallback Scope (Critical Understanding)

| Model Slot | Failure Recovery Path |
|-----------|---------------------|
| **default** (Trinity Preview) | Hermes fallback_providers list (6 models) |
| **tool_calls** (MiniMax M2.5) | Proxy layer cascade (same model list, separate order) |
| **auxiliary** (Gemma 4 MoE) | Auto-detection chain: OR → Nous Portal → Custom → API-key providers |

The fallback_providers list in Hermes config only protects the default model. Tool-calls failures are handled by the proxy. Auxiliary failures use the built-in auto-detection chain.
