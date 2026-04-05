# Data Analysis Spectrum Tool

`fr_r_spectrum_tool_rebuild.py` is the GUI entry point.  
`spectrum_core.py` contains the shared parsing, PSD, cross-spectrum, and target
spectral logic.  
`smoke_check_data_pipeline.py` provides headless smoke checks.

## Install

```bash
python -m pip install -r requirements.txt
```

## Start GUI

```bash
python fr_r_spectrum_tool_rebuild.py
```

## Repository Fixtures

Minimal reproducible fixtures are bundled in:

- `tests/fixtures/smoke/ygas_h2o_1min.log`
- `tests/fixtures/smoke/toa5_h2o_1min.dat`

These two files are intentionally small and stable:

- each file contains one minute of data
- each file can be used by itself for single-file smoke
- together they provide a common time window for dual/compare smoke

More details are in `tests/fixtures/README.md`.

## Local Smoke

One command that works after clone:

```bash
python smoke_check_data_pipeline.py --mode repo-fixture-smoke
```

This bundled smoke covers:

- single ygas
- single dat
- dual ygas+dat
- single-vs-compare base spectrum consistency

## CI Command

The GitHub Actions workflow runs the same repository fixture smoke:

```bash
python smoke_check_data_pipeline.py --mode repo-fixture-smoke
```

Workflow file:

- `.github/workflows/repo-fixture-smoke.yml`

## Extra Checks

If you want the lower-level metadata smoke:

```bash
python smoke_check_data_pipeline.py --mode time-range-metadata-check
```

If you want direct compile checks:

```bash
python -m py_compile fr_r_spectrum_tool_rebuild.py
python -m py_compile spectrum_core.py
python -m py_compile smoke_check_data_pipeline.py
```
