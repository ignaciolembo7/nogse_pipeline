
### Stage 5: Canonical experiment-table construction

**What goes in**

- extracted Excel tables
- sequence-parameter spreadsheet
- filename-derived metadata

**What happens conceptually**

- the interleaved ROI tables are reshaped into long form: one row per ROI, direction, and `b_step`;
- the matching sequence-parameter row is attached;
- if the original acquisition was `b`-organized, gradient-amplitude surrogates are derived from the sequence timing;
- if the original acquisition was direct-`g`, the direct `g` axis is preserved and corresponding `bvalue_*` columns are derived where possible;
- `S0` and `value_norm` are added.

**What comes out**

- cleaned long-form signal tables (`*.long.parquet`, `*.xlsx`)

**Why this step is needed**

- later stages require a uniform representation no matter how the original sequence was named or organized;
- the long-form table is the central data model of the repository.

**Key physical or mathematical idea**

- the pipeline explicitly separates raw signal organization from physical metadata;
- the same signal can later be viewed either as a function of `g` or as a function of a derived `b`.

**Code**

- `scripts/process_one_results.py`
- `src/data_processing/reshape.py`
- `src/data_processing/schema.py`