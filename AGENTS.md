# AGENTS.md

## Mandatory rules
- All script comments MUST be in English.
- `nogse_pipeline/bash_template` is the canonical source for pipeline scripts.
- Files under `nogse_pipeline/bash` MUST NOT be edited.

## Main decision rule
For any requested change, use this priority order:
1. reuse existing code,
2. refactor existing code,
3. extend existing code in a backward-compatible way,
4. parameterize an existing workflow,
5. add a small adapter layer,
6. create new code only as a last resort.

## The agent MUST
- search the repository for existing implementations before writing new code,
- reuse shared logic whenever possible,
- modify or generalize existing functionality instead of creating parallel implementations,
- keep equivalent tasks on a single coherent code path whenever possible,
- preserve existing output standards,
- keep changes small and targeted,
- briefly state what existing components are being reused, refactored, or extended,
- briefly justify any new code by explaining why reuse/refactor/extension was not enough.

## The agent MUST NOT
- create duplicate pipelines, duplicate scripts, near-identical implementations, or model-specific forks unless explicitly required,
- create parallel versions such as `v2`, `new`, `alternative`, or `fixed` unless explicitly requested,
- copy and paste logic across files when it can be centralized,
- duplicate output writers, schemas, naming conventions, metadata conventions, file organization, or postprocessing logic,
- introduce alternative names or formats for the same concept unless explicitly required,
- break existing interfaces without clear justification.

## File placement
- New logic SHOULD be added to existing shared modules whenever possible.
- New files MUST only be created when there is no appropriate existing location.
- Pipeline script changes MUST be made in `nogse_pipeline/bash_template`, never in `nogse_pipeline/bash`.
- Utility code MUST NOT be embedded in bash scripts if it belongs in reusable Python modules.
- Output-writing logic SHOULD remain centralized instead of being reimplemented in multiple scripts.

## Standardization
- Output formats MUST remain consistent across the repository.
- Column names, metadata fields, file naming, directory structure, and saved artifacts MUST follow existing repository conventions.
- If a standard exists, it MUST be followed.
- If no standard exists, the agent SHOULD introduce the smallest consistent standard needed and apply it consistently.

## CLI and output schema
- Existing CLI interfaces SHOULD be extended rather than duplicated.
- New CLI options MUST follow existing naming conventions in the repository.
- Output column names, metadata fields, and saved file structure MUST remain consistent with existing outputs.
- The agent MUST NOT introduce alternative column names or duplicate schema variants for the same concept unless explicitly required.
- Equivalent workflows MUST produce outputs in the same schema regardless of model or measurement naming.

## Implementation preferences
- Prefer small refactors that reduce duplication.
- Centralize shared logic into reusable functions or modules.
- Isolate model-specific behavior behind parameters or clearly bounded extension points.
- Preserve existing behavior unless there is a strong reason to change it.
- Maintain backward compatibility whenever possible.

## Quality target
- Optimize for auditability, debuggability, maintainability, and explicit modular code.
- Reduce parallel code paths.
- Equivalent inputs MUST produce outputs with the same schema and structure regardless of model or measurement naming.