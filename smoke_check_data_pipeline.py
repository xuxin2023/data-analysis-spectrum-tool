from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import spectrum_core as core


def parse_supported_file(path: Path) -> core.ParsedDataResult:
    profile = core.detect_file_profile(path, core.read_preview_lines(path))
    if profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
        return core.parse_ygas_mode1_file(path)
    if profile == "TOA5_DAT" or path.suffix.lower() == ".dat":
        return core.parse_toa5_file(path)
    raise ValueError(f"当前自检脚本仅支持 YGAS MODE1 或 TOA5 dat，未识别: {path.name}")


def format_timestamp_range(parsed: core.ParsedDataResult) -> tuple[str, str]:
    if not parsed.timestamp_col or parsed.timestamp_col not in parsed.dataframe.columns:
        return "无", "无"
    timestamps = core.parse_mixed_timestamp_series(parsed.dataframe[parsed.timestamp_col]).dropna()
    if timestamps.empty:
        return "无", "无"
    return (
        pd.Timestamp(timestamps.min()).strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."),
        pd.Timestamp(timestamps.max()).strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."),
    )


def resolve_fs(parsed: core.ParsedDataResult) -> float | None:
    estimated = core.estimate_fs_from_timestamp(parsed.dataframe, parsed.timestamp_col)
    if parsed.profile_name.startswith("YGAS_MODE1") or parsed.profile_name == "YGAS_MERGED":
        return estimated or core.DEFAULT_FS
    return estimated


def resolve_element_hit(parsed: core.ParsedDataResult, element: str | None, device_key: str) -> str | None:
    if not element:
        return None
    candidates = core.ELEMENT_PRESETS.get(element, {}).get(device_key, [])
    if not candidates:
        return None
    return core.find_element_hit(parsed.available_columns, candidates)


def choose_single_column(parsed: core.ParsedDataResult, element: str | None, device_key: str) -> str | None:
    hit = resolve_element_hit(parsed, element, device_key)
    if hit:
        return hit
    if parsed.suggested_columns:
        return parsed.suggested_columns[0]
    return None


def prepare_series(
    parsed: core.ParsedDataResult,
    column: str,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
        raise ValueError("未识别到时间列。")
    if column not in parsed.dataframe.columns:
        raise ValueError(f"目标列不存在: {column}")
    filtered = core.filter_by_time_range(parsed.dataframe, parsed.timestamp_col, start_dt, end_dt)
    frame = pd.DataFrame(
        {
            core.TIMESTAMP_COL: core.parse_mixed_timestamp_series(filtered[parsed.timestamp_col]),
            "value": pd.to_numeric(filtered[column], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        raise ValueError(f"列 {column} 在当前时间范围内没有可用数值。")
    return frame.sort_values(core.TIMESTAMP_COL).reset_index(drop=True)


def align_nearest(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    tolerance_seconds: float = 0.2,
) -> pd.DataFrame:
    left = frame_a.rename(columns={"value": "A值"}).sort_values(core.TIMESTAMP_COL)
    right = frame_b.rename(columns={"value": "B值"}).sort_values(core.TIMESTAMP_COL)
    aligned = pd.merge_asof(
        left,
        right,
        on=core.TIMESTAMP_COL,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    return aligned.dropna(subset=["A值", "B值"]).reset_index(drop=True)


def summarize_parsed(label: str, parsed: core.ParsedDataResult, element: str | None, device_key: str) -> None:
    start_text, end_text = format_timestamp_range(parsed)
    fs = resolve_fs(parsed)
    print(f"[{label}]")
    print(f"  profile: {parsed.profile_name}")
    print(f"  原始行数: {parsed.source_row_count or len(parsed.dataframe)}")
    print(
        "  时间戳有效率: "
        f"{parsed.timestamp_valid_ratio:.1%} ({parsed.timestamp_valid_count}/{parsed.source_row_count or len(parsed.dataframe)})"
    )
    print(f"  起始时间: {start_text}")
    print(f"  结束时间: {end_text}")
    print(f"  估算FS: {fs if fs is not None else '未知'}")
    print(f"  available_columns: {parsed.available_columns}")
    print(f"  suggested_columns: {parsed.suggested_columns}")
    if parsed.timestamp_warning:
        print(f"  时间戳警告: {parsed.timestamp_warning}")
    if element:
        hit = resolve_element_hit(parsed, element, device_key)
        print(f"  预设映射[{element}]命中: {hit if hit else '未命中'}")


def print_group_summary(group_records: list[dict[str, object]]) -> None:
    print("[组级QC摘要]")
    if not group_records:
        print("  无组记录")
        return
    for record in group_records:
        print(f"- group_key: {record.get('group_key')}")
        print(f"  group_label: {record.get('group_label')}")
        print(f"  ygas_label: {record.get('ygas_label')}")
        print(f"  dat_label: {record.get('dat_label')}")
        print(f"  ygas_window: {record.get('ygas_start')} -> {record.get('ygas_end')}")
        print(f"  dat_window: {record.get('dat_start')} -> {record.get('dat_end')}")
        print(f"  points: ygas={record.get('ygas_points')} dat={record.get('dat_points')}")
        print(f"  fs: ygas={record.get('ygas_fs')} dat={record.get('dat_fs')}")
        print(f"  freq_points: ygas={record.get('ygas_freq_points')} dat={record.get('dat_freq_points')}")
        print(
            "  coverage: "
            f"ygas={record.get('ygas_coverage_ratio')} dat={record.get('dat_coverage_ratio')}"
        )
        print(
            "  invalid_gaps: "
            f"ygas_leading={record.get('ygas_leading_invalid_gap_s')} "
            f"dat_leading={record.get('dat_leading_invalid_gap_s')} "
            f"ygas_trailing={record.get('ygas_trailing_invalid_gap_s')} "
            f"dat_trailing={record.get('dat_trailing_invalid_gap_s')}"
        )
        print(
            "  non_null_ratio: "
            f"ygas={record.get('ygas_non_null_ratio')} dat={record.get('dat_non_null_ratio')}"
        )
        print(
            "  spectrum_diag_ygas: "
            f"psd_kernel={record.get('ygas_psd_kernel')} "
            f"effective_fs={record.get('ygas_effective_fs', record.get('ygas_fs'))} "
            f"requested_nsegment={record.get('ygas_requested_nsegment')} "
            f"effective_nsegment={record.get('ygas_effective_nsegment')} "
            f"noverlap={record.get('ygas_noverlap')} "
            f"positive_freq_points={record.get('ygas_positive_freq_points', record.get('ygas_freq_points'))} "
            f"first_positive_freq={record.get('ygas_first_positive_freq')} "
            f"window_type={record.get('ygas_window_type')} "
            f"detrend={record.get('ygas_detrend')} "
            f"overlap={record.get('ygas_overlap')} "
            f"scaling_mode={record.get('ygas_scaling_mode')} "
            f"valid_freq_points={record.get('ygas_valid_freq_points', record.get('ygas_freq_points'))}"
        )
        print(
            "  spectrum_diag_dat: "
            f"psd_kernel={record.get('dat_psd_kernel')} "
            f"effective_fs={record.get('dat_effective_fs', record.get('dat_fs'))} "
            f"requested_nsegment={record.get('dat_requested_nsegment')} "
            f"effective_nsegment={record.get('dat_effective_nsegment')} "
            f"noverlap={record.get('dat_noverlap')} "
            f"positive_freq_points={record.get('dat_positive_freq_points', record.get('dat_freq_points'))} "
            f"first_positive_freq={record.get('dat_first_positive_freq')} "
            f"window_type={record.get('dat_window_type')} "
            f"detrend={record.get('dat_detrend')} "
            f"overlap={record.get('dat_overlap')} "
            f"scaling_mode={record.get('dat_scaling_mode')} "
            f"valid_freq_points={record.get('dat_valid_freq_points', record.get('dat_freq_points'))}"
        )
        print(
            "  decision: "
            f"status={record.get('status')} keep={record.get('keep')} "
            f"forced_include={record.get('forced_include')} reason={record.get('reason')}"
        )


def run_single_mode(args: argparse.Namespace) -> int:
    targets: list[tuple[str, Path, str]] = []
    for raw_path in args.ygas or []:
        targets.append(("YGAS", Path(raw_path), "A"))
    if args.dat:
        targets.append(("DAT", Path(args.dat), "B"))
    if not targets:
        raise ValueError("single 模式至少需要提供 --ygas 或 --dat。")

    for label, path, device_key in targets:
        parsed = parse_supported_file(path)
        summarize_parsed(label, parsed, args.element, device_key)
        chosen = choose_single_column(parsed, args.element, device_key)
        if not chosen:
            print("  单文件PSD: 无可用分析列")
            print("")
            continue
        fs = resolve_fs(parsed) or core.DEFAULT_FS
        frame = prepare_series(parsed, chosen)
        freq, density, _details = core.compute_psd_from_array_with_params(
            frame["value"].to_numpy(dtype=float),
            fs,
            args.nsegment,
            args.overlap_ratio,
        )
        mask = (freq > 0) & np.isfinite(freq) & np.isfinite(density) & (density > 0)
        print(f"  单文件PSD列: {chosen}")
        print(f"  单文件PSD有效频点数: {int(mask.sum())}")
        print("")
    return 0


def run_dual_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("dual 模式需要同时提供 --ygas 和 --dat。")

    if len(args.ygas) > 1:
        print(f"[提示] dual 模式仅使用第一个 ygas 文件: {args.ygas[0]}")
    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_path)
    parsed_b = parse_supported_file(dat_path)

    summarize_parsed("设备A", parsed_a, args.element, "A")
    summarize_parsed("设备B", parsed_b, args.element, "B")

    start_a, end_a = format_timestamp_range(parsed_a)
    start_b, end_b = format_timestamp_range(parsed_b)
    common_start = None
    common_end = None
    if parsed_a.timestamp_col and parsed_b.timestamp_col:
        ts_a = core.parse_mixed_timestamp_series(parsed_a.dataframe[parsed_a.timestamp_col]).dropna()
        ts_b = core.parse_mixed_timestamp_series(parsed_b.dataframe[parsed_b.timestamp_col]).dropna()
        if not ts_a.empty and not ts_b.empty:
            common_start = max(pd.Timestamp(ts_a.min()), pd.Timestamp(ts_b.min()))
            common_end = min(pd.Timestamp(ts_a.max()), pd.Timestamp(ts_b.max()))
    common_empty = common_start is None or common_end is None or common_start > common_end
    print("[共同时间范围]")
    print(f"  设备A时间范围: {start_a} -> {end_a}")
    print(f"  设备B时间范围: {start_b} -> {end_b}")
    print(f"  共同时间范围为空: {'是' if common_empty else '否'}")
    if not common_empty:
        print(f"  共同时间范围: {common_start} -> {common_end}")

    col_a = choose_single_column(parsed_a, args.element, "A")
    col_b = choose_single_column(parsed_b, args.element, "B")
    print(f"  设备A列: {col_a if col_a else '未命中'}")
    print(f"  设备B列: {col_b if col_b else '未命中'}")
    if common_empty or not col_a or not col_b:
        return 0

    frame_a = prepare_series(parsed_a, col_a, common_start, common_end)
    frame_b = prepare_series(parsed_b, col_b, common_start, common_end)
    aligned = align_nearest(frame_a, frame_b, tolerance_seconds=0.2)
    print("[双设备对齐]")
    print(f"  散点匹配点数: {len(aligned)}")
    if len(aligned) < 2:
        print("  双设备频域分析: 匹配点数不足")
        return 0

    fs_a = resolve_fs(parsed_a) or core.DEFAULT_FS
    fs_b = resolve_fs(parsed_b) or core.DEFAULT_FS
    freq_a, density_a, _details_a = core.compute_psd_from_array_with_params(
        frame_a["value"].to_numpy(dtype=float),
        fs_a,
        args.nsegment,
        args.overlap_ratio,
    )
    freq_b, density_b, _details_b = core.compute_psd_from_array_with_params(
        frame_b["value"].to_numpy(dtype=float),
        fs_b,
        args.nsegment,
        args.overlap_ratio,
    )
    psd_mask_a = (freq_a > 0) & np.isfinite(freq_a) & np.isfinite(density_a) & (density_a > 0)
    psd_mask_b = (freq_b > 0) & np.isfinite(freq_b) & np.isfinite(density_b) & (density_b > 0)
    print("[频域检查]")
    print(f"  设备A PSD有效频点数: {int(psd_mask_a.sum())}")
    print(f"  设备B PSD有效频点数: {int(psd_mask_b.sum())}")

    fs_aligned = core.estimate_fs_from_timestamp(aligned, core.TIMESTAMP_COL) or fs_a or fs_b or core.DEFAULT_FS
    for spectrum_type in (
        core.CROSS_SPECTRUM_MAGNITUDE,
        core.CROSS_SPECTRUM_REAL,
        core.CROSS_SPECTRUM_IMAG,
    ):
        freq_xy, values_xy, _details_xy = core.compute_cross_spectrum_from_arrays_with_params(
            aligned["A值"].to_numpy(dtype=float),
            aligned["B值"].to_numpy(dtype=float),
            fs_aligned,
            args.nsegment,
            args.overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message="当前数据不足以生成有效协谱图。",
        )
        mask_xy = core.build_spectrum_plot_mask(freq_xy, values_xy, spectrum_type)
        print(f"  {spectrum_type}有效频点数: {int(mask_xy.sum())}")

    print("")
    return 0


def run_legacy_target_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("目标谱图模式需要多个 --ygas 和 1 个 --dat。")

    payload = core.prepare_legacy_target_payload(
        ygas_paths=[Path(path) for path in args.ygas],
        dat_path=Path(args.dat),
        fs_ui=args.fs,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        target_element=args.element,
        time_range_strategy=args.time_range_strategy,
        start_raw=args.time_start,
        end_raw=args.time_end,
        grouping_mode="每个 ygas 文件视为一个组",
        parse_ygas=parse_supported_file,
        parse_dat=parse_supported_file,
        use_requested_nsegment=bool(args.legacy_use_requested_nsegment),
        legacy_psd_kernel=args.legacy_psd_kernel,
        forced_include_group_keys=set(args.forced_include_group_keys or []),
    )
    metadata = payload["target_metadata"]
    print("[目标谱图]")
    print(f"  target_element: {payload['target_element']}")
    print(f"  total_groups: {metadata.get('total_group_count')}")
    print(f"  kept_groups: {metadata.get('kept_group_count')}")
    print(f"  skipped_groups: {metadata.get('skipped_group_count')}")
    print(f"  actual_series_count: {metadata.get('actual_series_count')}")
    print(f"  ygas_column: {metadata.get('ygas_column')}")
    print(f"  dat_column: {metadata.get('dat_column')}")
    print(f"  time_range: {metadata.get('time_range_label')}")
    print(f"  target_use_requested_nsegment: {metadata.get('legacy_target_uses_requested_nsegment')}")
    print(f"  requested_nsegment: {metadata.get('requested_nsegment')}")
    print(f"  target_psd_kernel_requested: {metadata.get('legacy_psd_kernel_requested')}")
    print(f"  target_psd_kernel: {metadata.get('legacy_psd_kernel')}")
    print(f"  target_psd_kernel_selection_basis: {metadata.get('legacy_psd_kernel_selection_basis')}")
    print(f"  effective_nsegment_values: {metadata.get('effective_nsegment_values')}")
    print(
        "  first_positive_freq_range: "
        f"{metadata.get('first_positive_freq_min')} -> {metadata.get('first_positive_freq_max')}"
    )
    print(f"  manual_override_groups: {metadata.get('manual_override_groups')}")
    candidate_results = list(metadata.get("legacy_psd_candidate_results", []))
    if candidate_results:
        print("  target_psd_candidate_results:")
        for item in candidate_results:
            print(
                "    "
                f"{item.get('kernel_name')}: "
                f"score={item.get('score')} "
                f"median_positive_freq_points={item.get('median_positive_freq_points')} "
                f"median_first_positive_freq={item.get('median_first_positive_freq')} "
                f"summary={item.get('comparison_summary')}"
            )
    print_group_summary(list(metadata.get("group_records", [])))
    print("")
    return 0


def run_target_cospectrum_diagnose_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("target-cospectrum-diagnose 模式需要提供 --ygas 和 --dat。")

    output_root = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir = output_root / f"target_cospectrum_diag_{Path(args.dat).stem}_{args.element}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = core.diagnose_target_cospectrum_candidates(
        ygas_paths=[Path(path) for path in args.ygas],
        dat_path=Path(args.dat),
        target_element=args.element,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        time_range_strategy=args.time_range_strategy,
    )
    summary = result["candidate_summary"].copy()
    summary_path = output_dir / "candidate_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    spectra_dir = output_dir / "candidate_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    for candidate_name, frame in result["candidate_frames"].items():
        frame.to_csv(spectra_dir / f"{candidate_name}.csv", index=False, encoding="utf-8-sig")

    overlay_path = output_dir / "cospectrum_candidates_full.png"
    lowfreq_path = output_dir / "cospectrum_candidates_lowfreq_0_0.05hz.png"
    result["overlay_figure"].savefig(overlay_path, dpi=160)
    result["low_frequency_figure"].savefig(lowfreq_path, dpi=160)

    print("[目标协谱候选诊断]")
    for note in result["notes"]:
        print(f"- {note}")
    print(f"- 样例: ygas={len(args.ygas)} 个文件, dat={args.dat}, element={args.element}")
    print(f"- 时间范围: {result['time_range_label']}")
    print(f"- 目标列: {result['target_column']} | 参考列: {result['reference_column']}")
    print(f"- 当前产品协谱 kernel candidate: {result['selected_kernel_candidate_id']}")
    print(f"- 当前产品对齐策略仍保持: {result['selected_alignment_strategy']}")
    print("")

    print("[候选摘要]")
    print(
        summary[
            [
                "candidate_name",
                "alignment_strategy",
                "window",
                "detrend",
                "scaling",
                "matched_count",
                "integral_real",
                "first_positive_freq",
                "first_positive_real",
                "lowfreq_min_real",
                "lowfreq_max_real",
                "lowfreq_median_real",
                "diff_vs_current_l1",
                "diff_vs_current_l2",
                "diff_vs_current_lowfreq_l1",
            ]
        ].to_string(index=False)
    )
    print("")

    print("[低频段统计 <= 0.05 Hz]")
    print(
        summary[
            [
                "candidate_name",
                "lowfreq_min_real",
                "lowfreq_max_real",
                "lowfreq_median_real",
            ]
        ].to_string(index=False)
    )
    print("")

    print(f"[与当前实现差异摘要 baseline={result['baseline_candidate_name']}]")
    print(
        summary[
            [
                "candidate_name",
                "diff_vs_current_l1",
                "diff_vs_current_l2",
                "diff_vs_current_lowfreq_l1",
            ]
        ].to_string(index=False)
    )
    print("")

    print("[各 candidate 前 10 个正频点]")
    for candidate_name, frame in result["candidate_frames"].items():
        print(f"- {candidate_name}")
        print(
            frame[
                [
                    "frequency_hz",
                    "cross_spectrum_real",
                    "cross_spectrum_imag",
                    "cross_spectrum_abs",
                ]
            ].head(10).to_string(index=False)
        )
        print("")

    print("[输出文件]")
    print(f"  summary_csv: {summary_path}")
    print(f"  spectra_dir: {spectra_dir}")
    print(f"  overlay_full: {overlay_path}")
    print(f"  overlay_lowfreq: {lowfreq_path}")
    print("")
    return 0


def run_target_cospectrum_implementation_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("target-cospectrum-implementations 模式需要提供 --ygas 和 --dat。")

    output_root = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir = output_root / f"target_cospectrum_impl_{Path(args.dat).stem}_{args.element}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = core.diagnose_target_cospectrum_implementations(
        ygas_paths=[Path(path) for path in args.ygas],
        dat_path=Path(args.dat),
        target_element=args.element,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        time_range_strategy=args.time_range_strategy,
    )
    summary = result["implementation_summary"].copy()
    summary_path = output_dir / "implementation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    spectra_dir = output_dir / "implementation_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    for implementation_id, frame in result["implementation_frames"].items():
        frame.to_csv(spectra_dir / f"{implementation_id}.csv", index=False, encoding="utf-8-sig")

    overlay_path = output_dir / "cospectrum_implementations_full.png"
    lowfreq_path = output_dir / "cospectrum_implementations_lowfreq_0_0.05hz.png"
    result["overlay_figure"].savefig(overlay_path, dpi=160)
    result["low_frequency_figure"].savefig(lowfreq_path, dpi=160)

    print("[目标协谱实现硬比较]")
    for note in result["notes"]:
        print(f"- {note}")
    print(f"- 样例: ygas={len(args.ygas)} 个文件, dat={args.dat}, element={args.element}")
    print(f"- 时间范围: {result['time_range_label']}")
    print(f"- 目标列: {result['target_column']} | 参考列: {result['reference_column']}")
    print(f"- 对齐策略: {result['alignment_strategy']}")
    print("")

    print("[摘要]")
    for line in result["summary_lines"]:
        print(f"- {line}")
    print("")

    print("[实现摘要表]")
    print(
        summary[
            [
                "implementation_id",
                "implementation_label",
                "window",
                "detrend",
                "scaling",
                "return_onesided",
                "average",
                "matched_count",
                "first_positive_freq",
                "first_positive_real",
                "integral_real",
                "max_abs_diff_vs_current_helper",
                "lowfreq_mean_abs_diff_vs_current_helper",
            ]
        ].to_string(index=False)
    )
    print("")

    print("[当前三条 production path]")
    for path_name, info in result["runtime_paths"].items():
        print(
            f"- {path_name}: "
            f"cross_execution_path={info.get('cross_execution_path')} "
            f"cross_implementation_id={info.get('cross_implementation_id')} "
            f"cross_implementation_label={info.get('cross_implementation_label')}"
        )
    print("")

    largest = result.get("largest_diff_vs_current_helper") or {}
    lowfreq = result.get("most_lowfreq_diff_vs_current_helper") or {}
    print("[关键结论]")
    print(
        "- 当前 target helper vs generic default: "
        f"same={result.get('current_target_helper_equals_generic_default')} "
        f"max_abs_diff={result.get('max_abs_diff_vs_generic_default')} "
        f"first8_real_same={result.get('first8_real_same_as_generic_default')}"
    )
    print(
        "- 与当前实现差异最大: "
        f"{largest.get('implementation_id')} / {largest.get('implementation_label')} "
        f"(max_abs_diff={largest.get('max_abs_diff_vs_current_helper')})"
    )
    print(
        "- 低频段形状最不一样: "
        f"{lowfreq.get('implementation_id')} / {lowfreq.get('implementation_label')} "
        f"(lowfreq_mean_abs_diff={lowfreq.get('lowfreq_mean_abs_diff_vs_current_helper')})"
    )
    print("")

    print("[输出文件]")
    print(f"  summary_csv: {summary_path}")
    print(f"  spectra_dir: {spectra_dir}")
    print(f"  overlay_full: {overlay_path}")
    print(f"  overlay_lowfreq: {lowfreq_path}")
    print("")
    return 0


def run_cross_display_semantics_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("cross-display-semantics-check 模式需要提供 --ygas 和 --dat。")

    ygas_paths = [Path(path) for path in args.ygas]
    dat_path = Path(args.dat)
    target_element = args.element
    parsed_dat = parse_supported_file(dat_path)
    parsed_ygas = parse_supported_file(ygas_paths[0])

    reference_column = core.select_reference_uz_column(parsed_dat, device_key="B")
    dat_target_column = core.select_target_element_column(parsed_dat, device_key="B", target_element=target_element)
    ygas_target_column = core.select_target_element_column(parsed_ygas, device_key="A", target_element=target_element)
    if reference_column is None:
        raise ValueError("dat 文件中未找到 Uz 列，无法做 cross display semantics 验收。")
    if dat_target_column is None:
        raise ValueError(f"dat 文件中未找到 {target_element} 对应列。")
    if ygas_target_column is None:
        raise ValueError(f"ygas 文件中未找到 {target_element} 对应列。")

    ygas_start, ygas_end, _ = core.get_parsed_time_bounds(parsed_ygas)
    dat_start, dat_end, _ = core.get_parsed_time_bounds(parsed_dat)
    start_dt = max(pd.Timestamp(ygas_start), pd.Timestamp(dat_start))
    end_dt = min(pd.Timestamp(ygas_end), pd.Timestamp(dat_end))
    if start_dt >= end_dt:
        raise ValueError("首个 ygas 窗口与 dat 没有共同时间范围，无法验收 cross display semantics。")

    ygas_frame, _ygas_meta = core.build_target_window_series(parsed_ygas, ygas_target_column, start_dt, end_dt)
    dat_target_frame, _dat_target_meta = core.build_target_window_series(parsed_dat, dat_target_column, start_dt, end_dt)
    reference_frame, _reference_meta = core.build_target_window_series(parsed_dat, reference_column, start_dt, end_dt)
    ygas_fs = resolve_fs(parsed_ygas) or core.DEFAULT_FS
    dat_fs = resolve_fs(parsed_dat) or core.DEFAULT_FS

    target_context = core.build_target_spectral_context(
        target_element=target_element,
        reference_column=str(reference_column),
        ygas_target_column=str(ygas_target_column),
        dat_target_column=str(dat_target_column),
        spectrum_mode=core.LEGACY_TARGET_SPECTRUM_MODE_COSPECTRUM,
        display_target_label=target_element,
        comparison_is_target_spectral=True,
    )

    def _compute_target_pair(
        *,
        target_frame: pd.DataFrame,
        target_fs: float,
        target_column: str,
        series_role: str,
        device_kind: str,
        display_label: str,
        spectrum_type: str,
    ) -> dict[str, Any]:
        return core.compute_target_cross_spectrum_payload(
            target_frame=target_frame,
            reference_frame=reference_frame,
            target_fs=float(target_fs),
            reference_fs=float(dat_fs),
            target_element=target_element,
            reference_column=str(reference_column),
            target_column=str(target_column),
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            spectrum_type=spectrum_type,
            use_requested_nsegment=True,
            display_target_label=target_element,
            target_context=target_context,
            series_role=series_role,
            device_kind=device_kind,
            display_label=display_label,
        )

    advanced_ygas_cos = _compute_target_pair(
        target_frame=ygas_frame,
        target_fs=float(ygas_fs),
        target_column=str(ygas_target_column),
        series_role="ygas_target_vs_uz",
        device_kind="cross_ygas",
        display_label=f"{target_element}(ygas) vs {reference_column}",
        spectrum_type=core.CROSS_SPECTRUM_REAL,
    )
    advanced_ygas_mag = _compute_target_pair(
        target_frame=ygas_frame,
        target_fs=float(ygas_fs),
        target_column=str(ygas_target_column),
        series_role="ygas_target_vs_uz",
        device_kind="cross_ygas",
        display_label=f"{target_element}(ygas) vs {reference_column}",
        spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
    )
    advanced_dat_cos = _compute_target_pair(
        target_frame=dat_target_frame,
        target_fs=float(dat_fs),
        target_column=str(dat_target_column),
        series_role="dat_target_vs_uz",
        device_kind="cross_dat",
        display_label=f"{target_element}(dat) vs {reference_column}",
        spectrum_type=core.CROSS_SPECTRUM_REAL,
    )
    advanced_dat_mag = _compute_target_pair(
        target_frame=dat_target_frame,
        target_fs=float(dat_fs),
        target_column=str(dat_target_column),
        series_role="dat_target_vs_uz",
        device_kind="cross_dat",
        display_label=f"{target_element}(dat) vs {reference_column}",
        spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
    )
    advanced_ygas_quad = _compute_target_pair(
        target_frame=ygas_frame,
        target_fs=float(ygas_fs),
        target_column=str(ygas_target_column),
        series_role="ygas_target_vs_uz",
        device_kind="cross_ygas",
        display_label=f"{target_element}(ygas) vs {reference_column}",
        spectrum_type=core.CROSS_SPECTRUM_IMAG,
    )

    target_group_cos = core.prepare_legacy_target_payload(
        ygas_paths=ygas_paths,
        dat_path=dat_path,
        fs_ui=args.fs,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        target_element=target_element,
        time_range_strategy=args.time_range_strategy,
        start_raw=args.time_start,
        end_raw=args.time_end,
        grouping_mode="每个 ygas 文件视为一个组",
        parse_ygas=parse_supported_file,
        parse_dat=parse_supported_file,
        use_requested_nsegment=bool(args.legacy_use_requested_nsegment),
        legacy_target_spectrum_mode=core.LEGACY_TARGET_SPECTRUM_MODE_COSPECTRUM,
        legacy_psd_kernel=args.legacy_psd_kernel,
        forced_include_group_keys=set(args.forced_include_group_keys or []),
    )
    target_group_mag = core.prepare_legacy_target_payload(
        ygas_paths=ygas_paths,
        dat_path=dat_path,
        fs_ui=args.fs,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        target_element=target_element,
        time_range_strategy=args.time_range_strategy,
        start_raw=args.time_start,
        end_raw=args.time_end,
        grouping_mode="每个 ygas 文件视为一个组",
        parse_ygas=parse_supported_file,
        parse_dat=parse_supported_file,
        use_requested_nsegment=bool(args.legacy_use_requested_nsegment),
        legacy_target_spectrum_mode=core.LEGACY_TARGET_SPECTRUM_MODE_CROSS_MAGNITUDE,
        legacy_psd_kernel=args.legacy_psd_kernel,
        forced_include_group_keys=set(args.forced_include_group_keys or []),
    )

    generic_fs = core.estimate_fs_from_timestamp(dat_target_frame, core.TIMESTAMP_COL) or dat_fs
    generic_freq, _generic_raw_values, generic_details = core.compute_cross_spectrum_from_arrays_with_params(
        reference_frame["value"].to_numpy(dtype=float),
        dat_target_frame["value"].to_numpy(dtype=float),
        float(generic_fs),
        args.nsegment,
        args.overlap_ratio,
        spectrum_type=core.CROSS_SPECTRUM_REAL,
        insufficient_message="generic 双列分析未得到有效协谱图。",
    )
    generic_details = dict(generic_details)
    generic_details.update(core.describe_generic_default_cross_implementation())
    generic_details["alignment_strategy"] = core.GENERIC_SAME_FRAME_ALIGNMENT_STRATEGY
    generic_details["cross_reference_column"] = str(reference_column)
    generic_details["reference_column"] = str(reference_column)
    generic_details["target_column"] = str(dat_target_column)
    generic_details["cross_order"] = f"{reference_column} -> {dat_target_column}"
    generic_values, generic_mask, generic_details = core.resolve_cross_display_output(
        generic_freq,
        generic_details,
        analysis_context=generic_details.get("analysis_context"),
        cross_execution_path=generic_details.get("cross_execution_path"),
        spectrum_type=core.CROSS_SPECTRUM_REAL,
        insufficient_message="generic 双列分析未得到有效协谱图。",
    )

    def _masked_values(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(payload["mask"], dtype=bool)
        return np.asarray(payload["freq"], dtype=float)[mask], np.asarray(payload["values"], dtype=float)[mask]

    def _series_lookup(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
        return {
            (int(item.get("group_index", -1)), str(item.get("series_role") or "")): item
            for item in payload["series_results"]
            if str(item.get("device_kind", "")).startswith("cross")
        }

    def _allclose_payload(a: dict[str, Any], b: dict[str, Any]) -> bool:
        freq_a, values_a = _masked_values(a)
        freq_b, values_b = _masked_values(b)
        return bool(np.allclose(freq_a, freq_b) and np.allclose(values_a, values_b))

    group_cos_lookup = _series_lookup(target_group_cos)
    group_mag_lookup = _series_lookup(target_group_mag)
    group_pair_keys_same = set(group_cos_lookup) == set(group_mag_lookup)
    group_values_same = group_pair_keys_same
    if group_values_same:
        for key in sorted(group_cos_lookup):
            freq_cos = np.asarray(group_cos_lookup[key]["freq"], dtype=float)
            freq_mag = np.asarray(group_mag_lookup[key]["freq"], dtype=float)
            values_cos = np.asarray(group_cos_lookup[key]["values"], dtype=float)
            values_mag = np.asarray(group_mag_lookup[key]["values"], dtype=float)
            if not (np.allclose(freq_cos, freq_mag) and np.allclose(values_cos, values_mag)):
                group_values_same = False
                break

    generic_display_values = np.asarray(generic_values, dtype=float)[np.asarray(generic_mask, dtype=bool)]
    generic_real_values = np.asarray(generic_details["cross_spectrum_real"], dtype=float)[np.asarray(generic_mask, dtype=bool)]
    quad_mask = np.asarray(advanced_ygas_quad["mask"], dtype=bool)
    quad_display_values = np.asarray(advanced_ygas_quad["values"], dtype=float)[quad_mask]
    quad_imag_values = np.asarray(advanced_ygas_quad["details"]["cross_spectrum_imag"], dtype=float)[quad_mask]

    summary_pair_ygas = core.resolve_target_cross_pair(
        [str(target_context["summary_reference_column"]), str(target_context["summary_ygas_target_column"])],
        target_context,
    )
    summary_pair_dat = core.resolve_target_cross_pair(
        [str(target_context["summary_reference_column"]), str(target_context["summary_dat_target_column"])],
        target_context,
    )

    expected_group_series_count = int(target_group_cos["target_metadata"].get("kept_group_count", 0)) * 2
    actual_group_series_count = int(target_group_cos["target_metadata"].get("actual_series_count", 0))
    generated_roles = sorted(str(item) for item in (target_group_cos["target_metadata"].get("generated_cross_series_roles") or []))

    checks: list[tuple[str, bool]] = [
        ("advanced canonical ygas cos == magnitude", _allclose_payload(advanced_ygas_cos, advanced_ygas_mag)),
        ("advanced canonical dat cos == magnitude", _allclose_payload(advanced_dat_cos, advanced_dat_mag)),
        ("advanced canonical ygas display_semantics == abs_pxy", advanced_ygas_cos["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("advanced canonical dat display_semantics == abs_pxy", advanced_dat_cos["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("advanced canonical quadrature display_semantics == imag_pxy", advanced_ygas_quad["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_IMAG),
        ("advanced canonical quadrature uses imag source", np.allclose(quad_display_values, quad_imag_values)),
        ("target group canonical cos keys == magnitude keys", group_pair_keys_same),
        ("target group canonical cos values == magnitude values", group_values_same),
        ("target group actual_series_count == kept_groups * 2", actual_group_series_count == expected_group_series_count),
        ("target group generated_cross_series_roles == [dat_target_vs_uz, ygas_target_vs_uz]", generated_roles == ["dat_target_vs_uz", "ygas_target_vs_uz"]),
        ("target group display_semantics == abs_pxy", target_group_cos["target_metadata"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("target group reference_column == Uz", target_group_cos["target_metadata"].get("reference_column") == "Uz"),
        ("target group has ygas_target_column", bool(target_group_cos["target_metadata"].get("ygas_target_column"))),
        ("target group has dat_target_column", bool(target_group_cos["target_metadata"].get("dat_target_column"))),
        ("target group canonical_cross_pairs count == 2", len(target_group_cos["target_metadata"].get("canonical_cross_pairs") or []) == 2),
        ("summary ygas pair resolves as canonical", bool(summary_pair_ygas.get("uses_canonical_pair")) and summary_pair_ygas.get("resolved_pair", {}).get("series_role") == "ygas_target_vs_uz"),
        ("summary dat pair resolves as canonical", bool(summary_pair_dat.get("uses_canonical_pair")) and summary_pair_dat.get("resolved_pair", {}).get("series_role") == "dat_target_vs_uz"),
        ("generic cos display_semantics == real_pxy", generic_details.get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_REAL),
        ("generic cos display_value_source == cross_spectrum_real", generic_details.get("display_value_source") == "cross_spectrum_real"),
        ("generic cos values come from cross_spectrum_real", np.allclose(generic_display_values, generic_real_values)),
        ("canonical diagnostics.cross_execution_path == target_spectral_canonical", advanced_ygas_cos["details"].get("cross_execution_path") == "target_spectral_canonical"),
        ("canonical diagnostics.cross_implementation_id == A", advanced_ygas_cos["details"].get("cross_implementation_id") == "A"),
        ("canonical diagnostics.cross_reference_column == Uz", advanced_ygas_cos["details"].get("cross_reference_column") == "Uz"),
        (
            f"canonical diagnostics.cross_order == {reference_column} -> {ygas_target_column}",
            advanced_ygas_cos["details"].get("cross_order") == f"{reference_column} -> {ygas_target_column}",
        ),
    ]

    print("[cross display semantics check]")
    print(f"- element={target_element}")
    print(f"- ygas_count={len(ygas_paths)}")
    print(f"- dat={dat_path}")
    print(f"- ygas_target_column={ygas_target_column}")
    print(f"- dat_target_column={dat_target_column}")
    print(f"- reference_column={reference_column}")
    print("")

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("")
    print("[diagnostics]")
    diagnostics = target_group_cos["target_metadata"]
    for key in (
        "cross_execution_path",
        "cross_implementation_id",
        "reference_column",
        "ygas_target_column",
        "dat_target_column",
        "canonical_cross_pairs",
        "generated_cross_series_count",
        "generated_cross_series_roles",
    ):
        print(f"- {key}={diagnostics.get(key)}")

    failed = [label for label, ok in checks if not ok]
    if failed:
        print("")
        print("[failed checks]")
        for label in failed:
            print(f"- {label}")
        return 1

    print("")
    print("[summary]")
    print("- target_spectral canonical 下，协谱图现在按 abs(Pxy) 显示，因此与互谱幅值一致。")
    print("- target non-PSD 与高级页 canonical cross 现在都保留 ygas/dat 两条系列。")
    print("- comparison summary / main analysis 现在都能识别 B_Uz -> A_target 与 B_Uz -> B_target 两条 canonical pair。")
    print("- generic 双列分析仍保留数学定义：协谱=real(Pxy)，正交谱=imag(Pxy)。")
    return 0


def run_device_group_spectral_check_mode(args: argparse.Namespace) -> int:
    input_paths: list[Path] = [Path(value) for value in (args.ygas or [])]
    if args.dat:
        input_paths.append(Path(args.dat))
    if not input_paths:
        raise ValueError("device-group-spectral-check needs at least one --ygas or --dat input.")

    sample_path = input_paths[0]
    sample_device_key = "A" if sample_path.suffix.lower() in {".txt", ".log"} else "B"
    sample_parsed = parse_supported_file(sample_path)
    target_column = choose_single_column(sample_parsed, args.element, sample_device_key)
    if not target_column:
        raise ValueError(f"No analyzable target column found for {sample_path.name}.")

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import tkinter as tk
    import fr_r_spectrum_tool_rebuild as app_mod

    app_mod.FigureCanvasTkAgg = FigureCanvasTkAgg
    app_mod.NavigationToolbar2Tk = NavigationToolbar2Tk

    checks: list[tuple[str, bool, object]] = []
    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        payload_single = app.prepare_multi_spectral_compare_payload(
            [sample_path],
            target_column,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        checks.append(
            (
                "single_file_single_device",
                payload_single["device_count"] == 1 and len(payload_single["series_results"]) == 1,
                payload_single["device_groups"],
            )
        )

        app.plot_multi_spectral_results(
            target_column,
            list(payload_single["series_results"]),
            list(payload_single["skipped_files"]),
            payload=payload_single,
        )
        checks.append(
            (
                "single_device_render",
                len(app.current_compare_files) == 1 and app.current_plot_kind == "multi_spectral",
                {
                    "current_compare_files": list(app.current_compare_files),
                    "current_plot_kind": app.current_plot_kind,
                },
            )
        )

        with tempfile.TemporaryDirectory(prefix="device_group_check_") as temp_dir:
            temp_root = Path(temp_dir)
            same_device_copy = temp_root / f"{sample_path.stem}_copy{sample_path.suffix}"
            other_device_copy = temp_root / f"DEVICEB_{sample_path.stem}{sample_path.suffix}"
            shutil.copy2(sample_path, same_device_copy)
            shutil.copy2(sample_path, other_device_copy)

            same_device_paths = list(input_paths[:2]) if len(input_paths) >= 2 else [sample_path, same_device_copy]
            payload_same_device = app.prepare_multi_spectral_compare_payload(
                same_device_paths,
                target_column,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            same_groups = list(payload_same_device["device_groups"])
            checks.append(
                (
                    "single_device_multi_file_merge",
                    payload_same_device["device_count"] == 1
                    and len(same_groups) == 1
                    and int(same_groups[0]["file_count"]) >= 2,
                    same_groups,
                )
            )

            payload_multi_device = app.prepare_multi_spectral_compare_payload(
                [sample_path, other_device_copy],
                target_column,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            multi_groups = list(payload_multi_device["device_groups"])
            checks.append(
                (
                    "multi_device_grouping",
                    payload_multi_device["device_count"] >= 2 and len(multi_groups) >= 2,
                    multi_groups,
                )
            )
            checks.append(
                (
                    "filename_fallback",
                    any(str(item.get("device_source", "")) == "filename" for item in multi_groups),
                    multi_groups,
                )
            )
    finally:
        root.destroy()

    failed = False
    print("[device_group_spectral_check]")
    print(f"sample_path={sample_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_compare_base_spectrum_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-compare-base-spectrum-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_path)
    parsed_b = parse_supported_file(dat_path)
    col_a = choose_single_column(parsed_a, args.element, "A")
    col_b = choose_single_column(parsed_b, args.element, "B")
    if not col_a or not col_b:
        raise ValueError("Unable to resolve matching columns for the requested element.")

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import tkinter as tk
    import fr_r_spectrum_tool_rebuild as app_mod

    app_mod.FigureCanvasTkAgg = FigureCanvasTkAgg
    app_mod.NavigationToolbar2Tk = NavigationToolbar2Tk

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode="时间段内 PSD 对比",
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None

        single_ygas_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        single_dat_payload = app.prepare_multi_spectral_compare_payload(
            [dat_path],
            col_b,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode="时间段内 PSD 对比",
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy("时间段内 PSD 对比"),
            plot_style=app.resolve_plot_style("时间段内 PSD 对比"),
            plot_layout=app.resolve_plot_layout("时间段内 PSD 对比"),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
        )
    finally:
        root.destroy()

    if len(single_ygas_payload["series_results"]) != 1 or len(single_dat_payload["series_results"]) != 1:
        raise ValueError("Single-type payload did not produce exactly one spectrum series per side.")

    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
    )
    dual_dat = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "dat" and str(item["column"]) == col_b
    )
    single_ygas = single_ygas_payload["series_results"][0]
    single_dat = single_dat_payload["series_results"][0]

    checks = [
        (
            "ygas_base_freq_equal",
            np.array_equal(single_ygas["freq"], dual_ygas["freq"]),
            {"single_points": len(single_ygas["freq"]), "compare_points": len(dual_ygas["freq"])},
        ),
        (
            "ygas_base_density_allclose",
            np.allclose(single_ygas["density"], dual_ygas["density"], rtol=1e-12, atol=1e-12),
            {"single_first5": single_ygas["density"][:5].tolist(), "compare_first5": dual_ygas["density"][:5].tolist()},
        ),
        (
            "dat_base_freq_equal",
            np.array_equal(single_dat["freq"], dual_dat["freq"]),
            {"single_points": len(single_dat["freq"]), "compare_points": len(dual_dat["freq"])},
        ),
        (
            "dat_base_density_allclose",
            np.allclose(single_dat["density"], dual_dat["density"], rtol=1e-12, atol=1e-12),
            {"single_first5": single_dat["density"][:5].tolist(), "compare_first5": dual_dat["density"][:5].tolist()},
        ),
        (
            "shared_base_builder",
            single_ygas["details"].get("base_spectrum_builder") == "shared_profile_based"
            and dual_ygas["details"].get("base_spectrum_builder") == "shared_profile_based"
            and single_dat["details"].get("base_spectrum_builder") == "shared_profile_based"
            and dual_dat["details"].get("base_spectrum_builder") == "shared_profile_based",
            {
                "single_ygas_builder": single_ygas["details"].get("base_spectrum_builder"),
                "dual_ygas_builder": dual_ygas["details"].get("base_spectrum_builder"),
                "single_dat_builder": single_dat["details"].get("base_spectrum_builder"),
                "dual_dat_builder": dual_dat["details"].get("base_spectrum_builder"),
            },
        ),
    ]

    failed = False
    print("[single_compare_base_spectrum_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"time_range={start_dt} ~ {end_dt}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    example_log = r"C:\Users\A\Desktop\SAMPLE202603270700-202603270730.log"
    return argparse.ArgumentParser(
        description="不打开 GUI，检查 YGAS/dat 解析、时间戳、FS、共同时间范围和目标谱图组级数据链。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            f"  python smoke_check_data_pipeline.py --mode single --ygas \"{example_log}\"\n"
            "  python smoke_check_data_pipeline.py --mode single --dat \"你的_TOA5_文件.dat\"\n"
            "  python smoke_check_data_pipeline.py --mode dual --ygas \"你的_ygas.log\" --dat \"你的_TOA5_文件.dat\" --element CO2\n"
            "  python smoke_check_data_pipeline.py --mode legacy-target --ygas a.log b.log --dat c.dat --element CO2  # 目标谱图模式\n"
        ),
    )


def main() -> int:
    parser = build_parser()
    parser.add_argument("--ygas", nargs="+", help="一个或多个 YGAS 高频 txt/log 文件路径")
    parser.add_argument("--dat", help="TOA5 dat 文件路径")
    parser.add_argument(
        "--element",
        choices=["CO2", "H2O", "温度", "压力"],
        default="CO2",
        help="按预设要素检查映射命中",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "dual",
            "legacy-target",
            "target-group",
            "target-cospectrum-diagnose",
            "target-cospectrum-implementations",
            "cross-display-semantics-check",
            "device-group-spectral-check",
            "single-compare-base-spectrum-check",
        ],
        default=None,
        help="single=单文件检查，dual=双设备检查，legacy-target/target-group=组驱动目标谱图自检，target-cospectrum-diagnose=目标协谱候选排查",
    )
    parser.add_argument("--fs", type=float, default=core.DEFAULT_FS, help="频谱分析使用的 FS 提示值")
    parser.add_argument("--nsegment", type=int, default=core.DEFAULT_NSEGMENT, help="Welch nsegment")
    parser.add_argument("--overlap-ratio", type=float, default=core.DEFAULT_OVERLAP_RATIO, help="Welch overlap ratio")
    parser.add_argument(
        "--legacy-psd-kernel",
        choices=list(core.LEGACY_TARGET_PSD_KERNEL_CHOICES),
        default=core.LEGACY_TARGET_PSD_KERNEL_DEFAULT,
        help="目标谱图专用频谱核；默认使用 exact_legacy_welch，legacy_candidate_best 仅保留作显式诊断",
    )
    parser.add_argument(
        "--legacy-use-requested-nsegment",
        action="store_true",
        help="目标谱图模式沿用当前 --nsegment；默认使用目标谱图预设",
    )
    parser.add_argument(
        "--time-range-strategy",
        default="使用 txt+dat 共同时间范围",
        help="目标谱图模式时间范围策略",
    )
    parser.add_argument("--time-start", default="", help="目标谱图手动起始时间")
    parser.add_argument("--time-end", default="", help="目标谱图手动结束时间")
    parser.add_argument(
        "--forced-include-group-key",
        dest="forced_include_group_keys",
        action="append",
        default=[],
        help="目标谱图模式下手动放开的 group_key，可重复传入",
    )
    parser.add_argument("--output-dir", default="", help="headless 诊断输出目录")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "dual" if args.ygas and args.dat else "single"

    try:
        if mode == "single":
            return run_single_mode(args)
        if mode == "dual":
            return run_dual_mode(args)
        if mode == "target-cospectrum-diagnose":
            return run_target_cospectrum_diagnose_mode(args)
        if mode == "target-cospectrum-implementations":
            return run_target_cospectrum_implementation_mode(args)
        if mode == "cross-display-semantics-check":
            return run_cross_display_semantics_check_mode(args)
        if mode == "device-group-spectral-check":
            return run_device_group_spectral_check_mode(args)
        if mode == "single-compare-base-spectrum-check":
            return run_single_compare_base_spectrum_check_mode(args)
        return run_legacy_target_mode(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
