# Reverse Engineering Notes

This folder tracks the reverse-engineering context used while rebuilding the spectrum tool.

## Located artifacts

- Original executable:
  `C:\Users\A\Desktop\FRR_Sp_Alz_bete4.exe`
- Alternate copy:
  `C:\Users\A\Documents\xwechat_files\xuxin2023_9eb6\msg\file\2026-03\FRR_Sp_Alz_bete4.exe`
- Onefile payload dump:
  `C:\Users\A\Downloads\onefile_payload_decompressed.bin`

## Verified facts from the unpacked payload

- The app area around `FileViewerApp.cross_spectral_analysis` contains:
  - `data2`
  - `data1`
  - `csd`
  - `abs`
  - `plot_results`
  - `Frequency (Hz)`
  - `Density`
- The same area does not show:
  - `real`
  - `imag`
  - `np.real`
  - `np.imag`
  - `cospectrum`
  - `quadrature`

## Interpretation used in this rebuild

- The legacy FRR UI label "协谱分析" behaves closer to `abs(Pxy)` than `Re(Pxy)`.
- The current rebuild keeps generic dual-column analysis mathematically correct:
  - cross magnitude = `abs(Pxy)`
  - cospectrum = `real(Pxy)`
  - quadrature = `imag(Pxy)`
- The target-spectral canonical path maps the UI "协谱图" to `abs(Pxy)` to match observed FRR behavior.

## Current gap

The actual decompiled Python source file itself has not been located in the workspace yet.
Only the executable and payload artifact locations above were found automatically.
