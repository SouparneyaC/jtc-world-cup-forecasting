# Contributing

Bug reports, documentation fixes and small pull requests are welcome.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
```

## Checks

Run both before opening a pull request. The CI workflow runs the same two commands.

```bash
venv/bin/ruff check .
venv/bin/pytest
```

The tests need no network and no account. They fit the models on simulated matches with known coefficients and check the results table against the published fit.

## What fits

- Fixes to the models, their tests and the docs.
- New models that follow the existing pattern: a pure function or a small frozen dataclass, a test on simulated data, and a note in `docs/methodology.md`.

## What does not

- Scraped data about other participants, or anything that identifies a person.
- Third-party datasets or papers. Link to them in `docs/data-sources.md` instead.
- Code that places orders or submits forecasts automatically.
