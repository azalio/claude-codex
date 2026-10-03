# Large session compaction recovery

Observed: a resumed neuro-vlad session repeatedly entered proxy compaction with a
character-based estimate of 542,637 input tokens against a 180,000 threshold.
The client reported HTTP 400: Context compaction exceeded its total deadline.
The transcript metadata retained new messages; no active background shell process
was found under the current CLI, only its caffeinate child. That does not establish
the state of every historical background task.

The proxy's four-minute recovery deadline applied to all segment waves and merging.
Restarting discarded in-memory replacement history, and completed segment summaries
were not persisted. Default segment inference also used the main GPT-6.1 Sol model.

A deterministic regression reproduced the lost work: segment 1 completed, segment 2
timed out, and a new proxy resent segment 1. With the fix, the new proxy reuses it even
when the request includes a new 'continue' message.

Fix:
- Internal checkpoint summaries default to GPT-6 Luna / low, independently configurable
  with CLAUDE_CODEX_COMPACTION_MODEL. Normal answer and auxiliary routing stay separate.
- Exact completed segment and merge summaries are cached by backend/session/agent/class,
  model, policy, and source content. Partial/failed summaries are not cached.
- Files are private, versioned, size-bounded, and expire for reads after 24 hours.
  Corrupt, symlinked, inaccessible, or insecure entries are treated as misses.
  CLAUDE_CODEX_CHECKPOINT_CACHE=0 disables reads and writes.
- Per-segment progress is logged without source or summary content.
- The four-minute deadline and oversized-input safeguards remain active.

Live verification used synthetic data, not the user's project or transcript.
The source had 2,183,393 serialized bytes (2,169,000 source characters), producing
19 bounded segments. GPT-6 Luna completed summary preparation and merging in
30.825 seconds. Recreating the backend with the same scope/source reused all
19 segments and the merge, returning the identical summary in 0.099 seconds.
These timings are not a guarantee for the user's real history or current server load.

Private measurement artifact:
~/.local/state/claude-codex/benchmarks/20261003-large-checkpoint-recovery/results.json

225 tests pass, including the restart regression, model override isolation, cache
permissions/invalidation/expiry/corruption/symlink checks, and the cache kill switch.
Ruff and diff checks pass. The user's original interactive session needs a launcher
restart to load the change; its successful recovery has not yet been observed.
