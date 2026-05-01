# Contributing

Thanks for thinking about contributing — PRs and improvements are welcome.

## Quick Rules

- **Keep every tutorial folder self-contained.** A user should be able to `cd` into a tutorial directory and run it without depending on shared repo utilities.
- **Don’t commit secrets.** Use `.env.example` (placeholders only) and keep real keys in `.env` (gitignored).
- **Keep changes focused.** Avoid sweeping refactors across many days unless it’s a clear bugfix.
- **Update docs.** If you change behavior, update that tutorial’s `README.md`.

## Development Workflow

1. Fork the repo and create a branch:
   - `fix/01-readme-typo`
   - `feat/03-add-evaluator`
2. Make your change inside the relevant day folder.
3. Verify it runs locally from that folder:
   - `pip install -r requirements.txt`
   - follow that folder’s `README.md`
4. Open a PR with:
   - what changed
   - how you tested
   - which tutorial/day it affects

## Tutorial Checklist (Recommended)

When adding or improving a tutorial folder:

- `README.md` includes a **date** and a minimal “Run it” section.
- `requirements.txt` is present and reasonably minimal.
- Any needed environment variables are documented in a local `.env.example` (or clearly state “no env needed”).
- Outputs/artifacts (vector indexes, cache, `data/`, etc.) are gitignored.

