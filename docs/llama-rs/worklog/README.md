# llama-rs worklog directory guide

This directory stores detailed worklog files that are rotated out of the
top-level `docs/llama-rs/WORKLOG.md`.

## Why this exists

- Keep `WORKLOG.md` compact and quickly readable.
- Preserve full benchmark/context history in dated files.
- Make long-running migration work auditable without turning the top-level file
  into a large append-only dump.

## Mandatory split criteria

When updating worklogs, rotate **automatically** if either condition is true:

- top-level `WORKLOG.md` line count is greater than `250`, or
- top-level `WORKLOG.md` size is greater than `16 KiB`.

Do not wait for manual instruction once a threshold is crossed.

## Rotation procedure

1. Create or append a dated file:
   - `docs/llama-rs/worklog/YYYY-MM-DD-<scope>.md`
2. Move detailed bullet blocks / benchmark-heavy sections from top-level into
   that dated file.
3. Update the index table in top-level `WORKLOG.md`.
4. Keep only a short snapshot in top-level `WORKLOG.md`:
   - concise status bullets (`<= 10`),
   - links to detailed files/artifacts.

## Naming convention

- Date prefix is required: `YYYY-MM-DD`.
- Use a short scope suffix, for example:
  - `migration-log`
  - `stepwise-hotspot-log`
  - `parity-rerun-log`

## Update policy

- Detailed progress belongs in dated files under this directory by default.
- Top-level `WORKLOG.md` should function as:
  - index,
  - policy entry point,
  - latest high-level status snapshot.
