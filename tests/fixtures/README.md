# Repository Smoke Fixtures

These fixtures are small, stable clips from real supported files. They are kept
in the repository so contributors can run the core smoke checks without any
private `D:\` samples.

Files:
- `smoke/ygas_h2o_1min.log`
  - YGAS MODE1 clip
  - 600 rows
  - starts at `2026-03-27 14:31:40`
- `smoke/toa5_h2o_1min.dat`
  - TOA5 dat clip with original 4-line header preserved
  - 600 data rows
  - starts at `2026-03-27 14:31:40`

Usage:
- single ygas smoke: use `smoke/ygas_h2o_1min.log`
- single dat smoke: use `smoke/toa5_h2o_1min.dat`
- dual/compare smoke: use the same two files together; they share a common
  one-minute time window.
