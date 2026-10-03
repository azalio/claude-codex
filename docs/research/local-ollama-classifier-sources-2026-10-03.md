# Local Ollama command-classifier candidates

Verified against official Ollama library and API documentation on 2026-10-03. Sizes are library artifact/download sizes, **not runtime RAM or KV-cache allocations**. Context values are advertised tag maxima, **not the active server allocation**. No task-specific accuracy or latency is established by these sources; the local benchmark must decide.

## Recommended first benchmark set

| Exact tag | Library size | Advertised context | Role in comparison |
| --- | ---: | ---: | --- |
| `qwen3.5:0.8b` | 1.0GB | 256K | Smallest current Qwen candidate |
| `qwen3.5:2b` | 2.7GB | 256K | Middle current Qwen candidate |
| `qwen3.5:4b` | 3.4GB | 256K | Larger current Qwen candidate |
| `qwen3:1.7b` | 1.4GB | 40K | Smaller text-only generation baseline |
| `qwen3:4b` | 2.5GB | 256K | Text-only generation baseline |

These are research priorities, not measured quality rankings. Qwen3.5 tags are text/image models; Qwen3 tags are text models. Both family pages advertise thinking. [Qwen3.5 library](https://ollama.com/library/qwen3.5), [Qwen3 library](https://ollama.com/library/qwen3).

## Other officially available small tags

| Exact tag | Library size | Advertised context | Selection note |
| --- | ---: | ---: | --- |
| `qwen3:0.6b` | 523MB | 40K | Smallest text-only Qwen baseline |
| `gemma3:270m` | 292MB | 32K | Minimal text model; context below the proposed ~35K policy budget |
| `gemma3:1b` | 815MB | 32K | Text model; same context constraint |
| `gemma3:4b` | 3.3GB | 128K | Text/image comparison candidate |
| `llama3.2:1b` | 1.3GB | 128K | Text-only cross-family baseline |
| `llama3.2:3b` | 2.0GB | 128K | Larger text-only cross-family baseline |
| `gemma4:e2b` | 4.6–7.5GB | 128K | Newer edge variant, larger artifact than the proposed Qwen set |
| `gemma4:e4b` | 6.6–9.5GB | 128K | Larger edge variant |

Gemma3 requires Ollama 0.6+. Gemma4 advertises configurable thinking and native system-role support; “E” means effective parameters, not total parameters (E2B: 2.3B effective, 5.1B including embeddings). Llama3.2 officially supported languages do not include Russian. [Gemma3 library](https://ollama.com/library/gemma3), [Llama3.2 library](https://ollama.com/library/llama3.2), [Gemma4 library](https://ollama.com/library/gemma4).

## Request settings and evidence to collect

1. **Structured output:** local `/api/chat` accepts `format: "json"` or a JSON-schema object. Put the schema in the prompt as well, set temperature 0 for deterministic comparison, and validate the response. Schema conformance does not prove a correct approval decision. OpenAI-compatible chat uses `response_format`. [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [Chat endpoint](https://docs.ollama.com/api/chat).
2. **Thinking:** explicitly request `think: false` for the low-latency baseline; verify each loaded tag with `/api/show`. Current docs describe a `thinking` object containing supported `values` and `default`; metadata may be absent, and disabling is conditional on model support. Parse final `message.content`, separately from `message.thinking`. [Thinking](https://docs.ollama.com/capabilities/thinking).
3. **Actual context:** current defaults vary with VRAM (4K below 24GiB, 32K at 24–48GiB, 256K at 48GiB or more). Larger allocations require more memory. Server-wide `OLLAMA_CONTEXT_LENGTH` is documented; verify the actual `CONTEXT` and CPU/GPU split with `ollama ps`. Measure prompt tokens rather than assuming a fixed characters-to-tokens ratio. [Context length](https://docs.ollama.com/context-length).
4. **Responses route:** `/v1/responses` was added in Ollama 0.13.3; it is stateless. Current docs list `think` as an Ollama extension, `reasoning.effort`, and `max_output_tokens`. They do not list Responses structured-output fields; do not infer parity from chat. OpenAI compatibility does not offer context-size control; docs prescribe a model alias with `PARAMETER num_ctx <size>`. [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).
5. **Benchmark record:** collect tag/digest, Ollama version, `/api/show` context/thinking metadata, actual context, cold/warm latency, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, JSON validity, and approval confusion matrix. API durations are nanoseconds. Report false approvals separately from false refusals. [Chat response fields](https://docs.ollama.com/api/chat).

For native API testing, use `options.num_ctx` only after verifying the installed implementation or its runtime behavior: the current chat endpoint page exposes generic options without documenting that individual field. For the production Responses route, prefer the documented Modelfile alias and test `think: false` on the exact installed version.
