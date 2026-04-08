from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import spectrum_core as core

REPO_ROOT = Path(__file__).resolve().parent
REPO_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "smoke"
REPO_FIXTURE_YGAS = REPO_FIXTURE_DIR / "ygas_h2o_1min.log"
REPO_FIXTURE_DAT = REPO_FIXTURE_DIR / "toa5_h2o_1min.dat"
REVERSE_ENGINEERING_README = REPO_ROOT / "reverse_engineering" / "README.md"


def resolve_repo_fixture_paths() -> dict[str, Path]:
    fixtures = {
        "ygas": REPO_FIXTURE_YGAS,
        "dat": REPO_FIXTURE_DAT,
    }
    missing = [str(path) for path in fixtures.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "repo fixture files are missing: " + ", ".join(missing)
        )
    return fixtures


def clone_args(args: argparse.Namespace, **overrides: object) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def configure_auto_compare_context(app: object, ygas_path: Path, dat_path: Path) -> dict[str, object]:
    auto_payload = app.prepare_folder_auto_selection_payload([ygas_path, dat_path])
    selected_dat_path = Path(auto_payload["selected_dat_path"])
    if hasattr(app, "cache_auto_compare_context_payload"):
        app.cache_auto_compare_context_payload(
            auto_payload,
            source="smoke_manual_configure",
            selected_dat_path=selected_dat_path,
        )
        selected_dat_label = str(app.selected_dat_var.get())
        app.refresh_single_compare_style_preview_state()
    else:
        app.auto_prepare_payload = auto_payload
        app.auto_dat_options = {app.build_dat_option_label(info): Path(info["path"]) for info in auto_payload["dat_infos"]}
        selected_dat_label = next(
            label for label, path in app.auto_dat_options.items() if Path(path) == selected_dat_path
        )
        app.selected_dat_var.set(selected_dat_label)
        app.refresh_single_compare_style_preview_state()
    return {
        "auto_payload": auto_payload,
        "selected_dat_path": selected_dat_path,
        "selected_dat_label": selected_dat_label,
    }


def get_export_plot_execution_path(app: object) -> str:
    if getattr(app, "current_result_frame", None) is None or app.current_result_frame.empty:
        return ""
    if "plot_execution_path" not in app.current_result_frame.columns:
        return ""
    return str(app.current_result_frame.iloc[0].get("plot_execution_path") or "")


def get_export_metadata_value(app: object, key: str) -> str:
    if getattr(app, "current_result_frame", None) is None or app.current_result_frame.empty:
        return ""
    if key not in app.current_result_frame.columns:
        return ""
    return str(app.current_result_frame.iloc[0].get(key) or "")


def configure_compare_psd_render_test_state(app: object) -> None:
    app.plot_style_var.set("散点图")
    app.plot_layout_var.set("叠加同图")
    app.reference_slope_mode_var.set("两条都显示")
    app.use_separate_zoom_windows_var.set(False)


def extract_axis_render_state(ax: object) -> dict[str, object]:
    scatter_offsets = [
        np.asarray(collection.get_offsets(), dtype=float)
        for collection in list(getattr(ax, "collections", []))
    ]
    reference_slope_lines = []
    for line in list(getattr(ax, "lines", [])):
        if str(line.get_linestyle()) != "--":
            continue
        reference_slope_lines.append(
            {
                "label": str(line.get_label()),
                "x": np.asarray(line.get_xdata(), dtype=float),
                "y": np.asarray(line.get_ydata(), dtype=float),
            }
        )
    return {
        "xlim": tuple(float(value) for value in ax.get_xlim()),
        "ylim": tuple(float(value) for value in ax.get_ylim()),
        "scatter_offsets": scatter_offsets,
        "reference_slope_lines": reference_slope_lines,
        "legend_labels": list(ax.get_legend_handles_labels()[1]),
        "title": str(ax.get_title()),
    }


def render_prepared_multi_spectral_payload(app: object, payload: dict[str, object]) -> None:
    app.on_multi_spectral_compare_ready(payload)


def render_prepared_dual_plot_payload(app: object, payload: dict[str, object]) -> None:
    app.on_prepared_dual_plot_ready(payload)


def render_prepared_target_spectrum_payload(app: object, payload: dict[str, object]) -> None:
    app.on_target_spectrum_ready(payload)


def run_background_tasks_inline(app: object) -> None:
    def _start_background_task(*, status_text: str, worker: object, on_success: object, error_title: str) -> None:
        app.status_var.set(str(status_text))
        payload = worker(lambda _text: None)
        on_success(payload)

    app.start_background_task = _start_background_task


def build_headless_app() -> tuple[object, object]:
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
    app.refresh_canvas = lambda *args, **kwargs: None
    return root, app


def find_dat_option_label(app: object, dat_path: Path) -> str:
    target_key = str(Path(dat_path).resolve())
    for label, option_path in getattr(app, "auto_dat_options", {}).items():
        if str(Path(option_path).resolve()) == target_key:
            return str(label)
    raise ValueError(f"Unable to resolve dat option label for {dat_path}")


def write_truncated_dat_copy(source_path: Path, target_path: Path, *, keep_data_rows: int) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    header_lines = lines[:4]
    data_lines = lines[4 : 4 + max(int(keep_data_rows), 0)]
    if not data_lines:
        raise ValueError("Truncated dat copy must keep at least one data row.")
    target_path.write_text("\n".join([*header_lines, *data_lines]) + "\n", encoding="utf-8")


def format_smoke_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def write_extended_ygas_copy(
    source_path: Path,
    target_path: Path,
    *,
    row_count: int,
    start_offset_seconds: float = 0.0,
) -> None:
    source_lines = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_lines:
        raise ValueError("Extended ygas copy requires at least one source row.")

    base_timestamp = pd.Timestamp("2026-03-27 14:31:40") + pd.Timedelta(seconds=float(start_offset_seconds))
    output_lines: list[str] = []
    for index in range(max(int(row_count), 0)):
        line = source_lines[index % len(source_lines)]
        comma_index = line.find(",")
        if comma_index < 0:
            raise ValueError(f"Unexpected ygas row format: {line}")
        suffix = line[comma_index:]
        timestamp = format_smoke_timestamp(base_timestamp + pd.Timedelta(milliseconds=100 * index))
        output_lines.append(f"{timestamp}{suffix}")
    target_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def write_extended_dat_copy(
    source_path: Path,
    target_path: Path,
    *,
    row_count: int,
) -> None:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    header_lines = lines[:4]
    data_lines = [line for line in lines[4:] if line.strip()]
    if not data_lines:
        raise ValueError("Extended dat copy requires at least one source row.")

    base_timestamp = pd.Timestamp("2026-03-27 14:31:40")
    output_lines = list(header_lines)
    for index in range(max(int(row_count), 0)):
        parts = data_lines[index % len(data_lines)].split(",")
        if len(parts) < 2:
            raise ValueError(f"Unexpected dat row format: {data_lines[index % len(data_lines)]}")
        parts[0] = f"\"{format_smoke_timestamp(base_timestamp + pd.Timedelta(milliseconds=100 * index))}\""
        parts[1] = str(8000000 + index)
        output_lines.append(",".join(parts))
    target_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def prepare_target_spectrum_smoke_paths(
    temp_dir: Path,
    *,
    group_count: int,
) -> dict[str, object]:
    fixtures = resolve_repo_fixture_paths()
    dat_path = temp_dir / "target_reference.dat"
    write_extended_dat_copy(fixtures["dat"], dat_path, row_count=max(8192, 4096 + int(group_count) * 2048))

    ygas_paths: list[Path] = []
    for index in range(int(group_count)):
        ygas_path = temp_dir / f"target_group_{index + 1}.log"
        write_extended_ygas_copy(
            fixtures["ygas"],
            ygas_path,
            row_count=4096,
            start_offset_seconds=60.0 * index,
        )
        ygas_paths.append(ygas_path)
    return {"ygas_paths": ygas_paths, "dat_path": dat_path}


def prepare_target_spectrum_payload_for_smoke(
    app: object,
    *,
    ygas_paths: list[Path],
    dat_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    app.element_preset_var.set(str(args.element or "H2O"))
    app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
    app.legacy_target_spectrum_mode_var.set(core.LEGACY_TARGET_SPECTRUM_MODE_PSD)
    app.legacy_target_use_analysis_params_var.set(False)
    return app.prepare_target_spectrum_payload(
        ygas_paths=ygas_paths,
        dat_path=dat_path,
        fs_ui=args.fs,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        time_range_strategy=app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
        start_raw="",
        end_raw="",
        grouping_mode="每个 ygas 文件视为一个组",
        legacy_target_spectrum_mode=core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
        use_requested_nsegment=False,
        forced_include_group_keys=set(),
        reporter=None,
    )


def simulate_generate_plot_from_selected_files(
    app: object,
    root: object,
    *,
    selected_paths: list[Path],
    active_path: Path,
    parsed_active: core.ParsedDataResult,
    target_column: str,
) -> None:
    import tkinter as tk

    app.populate_file_list(selected_paths)
    app.select_paths_in_list(selected_paths, active_path=active_path)
    app.current_data_source_kind = "file"
    app.current_data_source_label = f"当前文件：{active_path.name}"
    app.current_file = active_path
    app.current_file_parsed = parsed_active
    app.current_layout_label = parsed_active.profile_name
    app.raw_data = parsed_active.dataframe.copy()
    app.column_vars = {str(target_column): tk.BooleanVar(master=root, value=True)}
    run_background_tasks_inline(app)
    app.generate_plot()


def prepare_compare_txt_side_reference(
    app: object,
    *,
    ygas_paths: list[Path],
    dat_path: Path,
    col_a: str,
    col_b: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    compare_mode = "时间段内 PSD 对比"
    dual_base = app.prepare_dual_compare_payload(
        ygas_paths=ygas_paths,
        dat_path=dat_path,
        selected_paths=[*ygas_paths, dat_path],
        compare_mode=compare_mode,
        start_dt=None,
        end_dt=None,
    )
    selection_meta = dict(dual_base["selection_meta"])
    if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
        start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
    else:
        start_dt, end_dt = None, None
    dual_plot_payload = app.prepare_dual_plot_payload(
        parsed_a=dual_base["parsed_a"],
        label_a=str(dual_base["label_a"]),
        parsed_b=dual_base["parsed_b"],
        label_b=str(dual_base["label_b"]),
        pairs=[{"a_col": str(col_a), "b_col": str(col_b), "label": f"{col_a} vs {col_b}"}],
        selection_meta=selection_meta,
        compare_mode=compare_mode,
        compare_scope="单对单",
        start_dt=start_dt,
        end_dt=end_dt,
        mapping_name=str(args.element),
        scheme_name="smoke",
        alignment_strategy=app.get_alignment_strategy(compare_mode),
        plot_style=app.resolve_plot_style(compare_mode),
        plot_layout=app.resolve_plot_layout(compare_mode),
        fs_ui=args.fs,
        requested_nsegment=args.nsegment,
        overlap_ratio=args.overlap_ratio,
        match_tolerance=0.2,
        spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
        time_range_context={
            "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
            "has_txt_dat_context": bool(
                selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
            ),
        },
    )
    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == str(col_a)
    )
    return {
        "dual_base": dual_base,
        "selection_meta": selection_meta,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "dual_plot_payload": dual_plot_payload,
        "dual_ygas": dual_ygas,
    }


def capture_single_device_and_dual_compare_psd_render_states(
    app: object,
    *,
    ygas_paths: list[Path],
    selected_files: list[Path],
    dat_path: Path,
    col_a: str,
    col_b: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    single_payload = app.prepare_multi_spectral_compare_payload(
        selected_files,
        str(col_a),
        args.fs,
        args.nsegment,
        args.overlap_ratio,
    )
    render_prepared_multi_spectral_payload(app, single_payload)
    if not app.figure.axes:
        raise ValueError("Single-device compare-style render did not produce a visible axis.")
    single_axis_state = extract_axis_render_state(app.figure.axes[0])
    single_status_text = app.status_var.get()
    single_diag_text = app.diagnostic_var.get()
    single_plot_kind = app.current_plot_kind
    single_plot_style = app.current_plot_style_label
    single_plot_layout = app.current_plot_layout_label
    single_export_plot_execution_path = get_export_plot_execution_path(app)
    single_details = dict(single_payload["series_results"][0]["details"])

    compare_reference = prepare_compare_txt_side_reference(
        app,
        ygas_paths=ygas_paths,
        dat_path=dat_path,
        col_a=str(col_a),
        col_b=str(col_b),
        args=args,
    )
    render_prepared_dual_plot_payload(app, compare_reference["dual_plot_payload"])
    if not app.figure.axes:
        raise ValueError("Dual compare PSD render did not produce a visible axis.")
    dual_axis_state = extract_axis_render_state(app.figure.axes[0])

    return {
        "single_payload": single_payload,
        "single_details": single_details,
        "single_axis_state": single_axis_state,
        "single_status_text": single_status_text,
        "single_diag_text": single_diag_text,
        "single_plot_kind": single_plot_kind,
        "single_plot_style": single_plot_style,
        "single_plot_layout": single_plot_layout,
        "single_export_plot_execution_path": single_export_plot_execution_path,
        "compare_reference": compare_reference,
        "dual_axis_state": dual_axis_state,
        "dual_plot_style": app.current_plot_style_label,
        "dual_plot_layout": app.current_plot_layout_label,
    }


def capture_single_and_dual_target_spectrum_render_states(
    app: object,
    *,
    single_ygas_path: Path,
    dual_ygas_paths: list[Path],
    dat_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    single_payload = prepare_target_spectrum_payload_for_smoke(
        app,
        ygas_paths=[single_ygas_path],
        dat_path=dat_path,
        args=args,
    )
    render_prepared_target_spectrum_payload(app, single_payload)
    if not app.figure.axes:
        raise ValueError("Single target-spectrum render did not produce a visible axis.")
    single_axis_state = extract_axis_render_state(app.figure.axes[0])
    single_status_text = app.status_var.get()
    single_diag_text = app.diagnostic_var.get()
    single_plot_kind = app.current_plot_kind
    single_plot_style = app.current_plot_style_label
    single_plot_layout = app.current_plot_layout_label
    single_export_plot_execution_path = get_export_plot_execution_path(app)
    single_export_render_semantics = get_export_metadata_value(app, "render_semantics")

    dual_payload = prepare_target_spectrum_payload_for_smoke(
        app,
        ygas_paths=dual_ygas_paths,
        dat_path=dat_path,
        args=args,
    )
    render_prepared_target_spectrum_payload(app, dual_payload)
    if not app.figure.axes:
        raise ValueError("Dual target-spectrum render did not produce a visible axis.")
    dual_axis_state = extract_axis_render_state(app.figure.axes[0])

    return {
        "single_payload": single_payload,
        "single_axis_state": single_axis_state,
        "single_status_text": single_status_text,
        "single_diag_text": single_diag_text,
        "single_plot_kind": single_plot_kind,
        "single_plot_style": single_plot_style,
        "single_plot_layout": single_plot_layout,
        "single_export_plot_execution_path": single_export_plot_execution_path,
        "single_export_render_semantics": single_export_render_semantics,
        "dual_payload": dual_payload,
        "dual_axis_state": dual_axis_state,
        "dual_plot_kind": app.current_plot_kind,
        "dual_plot_style": app.current_plot_style_label,
        "dual_plot_layout": app.current_plot_layout_label,
        "dual_export_plot_execution_path": get_export_plot_execution_path(app),
        "dual_export_render_semantics": get_export_metadata_value(app, "render_semantics"),
    }


def simulate_single_file_open(app: object, source_path: Path) -> None:
    if hasattr(app, "reset_auto_compare_context"):
        app.reset_auto_compare_context()
    app.current_folder = source_path.parent
    files = app.get_supported_files(app.current_folder)
    app.populate_file_list(files)
    if not app.select_file_in_list(source_path):
        raise ValueError(f"Failed to select {source_path.name} in the current file list.")
    app.apply_file_read_defaults(source_path)
    load_result = app.load_file_with_file_settings(source_path)
    app.on_file_loaded(source_path, load_result)


def extract_markdown_section_items(path: Path, heading: str) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    target_heading = f"## {heading}"
    collected: list[str] = []
    collecting = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            if collecting:
                break
            collecting = line == target_heading
            continue
        if not collecting:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            collected.append(stripped[2:].strip())
        else:
            collected.append(stripped)
    return collected


def load_reverse_engineering_semantics_context() -> dict[str, list[str]]:
    return {
        "verified_facts": extract_markdown_section_items(
            REVERSE_ENGINEERING_README,
            "Verified facts from the unpacked payload",
        ),
        "interpretation": extract_markdown_section_items(
            REVERSE_ENGINEERING_README,
            "Interpretation used in this rebuild",
        ),
        "current_gap": extract_markdown_section_items(
            REVERSE_ENGINEERING_README,
            "Current gap",
        ),
    }


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


def run_repo_fixture_smoke_mode(args: argparse.Namespace) -> int:
    fixtures = resolve_repo_fixture_paths()
    print("[repo_fixture_smoke]")
    print(f"fixture_ygas={fixtures['ygas']}")
    print(f"fixture_dat={fixtures['dat']}")

    checks: list[tuple[str, object, argparse.Namespace]] = [
        (
            "single_ygas",
            run_single_mode,
            clone_args(
                args,
                mode="single",
                ygas=[str(fixtures["ygas"])],
                dat=None,
                element="H2O",
            ),
        ),
        (
            "single_dat",
            run_single_mode,
            clone_args(
                args,
                mode="single",
                ygas=None,
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "dual_ygas_dat",
            run_dual_mode,
            clone_args(
                args,
                mode="dual",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "device_group_spectral",
            run_device_group_spectral_check_mode,
            clone_args(
                args,
                mode="device-group-spectral-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "single_compare_base_spectrum",
            run_single_compare_base_spectrum_core_metadata_check_mode,
            clone_args(
                args,
                mode="single-compare-base-spectrum-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "single_device_selection_scope",
            run_single_device_selection_scope_check_mode,
            clone_args(
                args,
                mode="single-device-selection-scope-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "device_count_default_single_compare_style",
            run_device_count_default_single_compare_style_check_mode,
            clone_args(
                args,
                mode="device-count-default-single-compare-style-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "device_count_default_single_fallback",
            run_device_count_default_single_fallback_check_mode,
            clone_args(
                args,
                mode="device-count-default-single-fallback-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "device_count_default_dual",
            run_device_count_default_dual_check_mode,
            clone_args(
                args,
                mode="device-count-default-dual-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "device_count_default_multi",
            run_device_count_default_multi_check_mode,
            clone_args(
                args,
                mode="device-count-default-multi-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "selected_files_direct_generate",
            run_selected_files_direct_generate_check_mode,
            clone_args(
                args,
                mode="selected-files-direct-generate-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "selected_files_direct_target_spectrum_single",
            run_selected_files_direct_target_spectrum_single_check_mode,
            clone_args(
                args,
                mode="selected-files-direct-target-spectrum-single-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "selected_files_direct_target_spectrum_dual",
            run_selected_files_direct_target_spectrum_dual_check_mode,
            clone_args(
                args,
                mode="selected-files-direct-target-spectrum-dual-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "single_vs_dual_target_spectrum_visible_subset",
            run_single_vs_dual_target_spectrum_visible_subset_check_mode,
            clone_args(
                args,
                mode="single-vs-dual-target-spectrum-visible-subset-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "single_vs_dual_target_spectrum_style",
            run_single_vs_dual_target_spectrum_style_check_mode,
            clone_args(
                args,
                mode="single-vs-dual-target-spectrum-style-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "txt_dat_not_equals_device_count",
            run_txt_dat_not_equals_device_count_check_mode,
            clone_args(
                args,
                mode="txt-dat-not-equals-device-count-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
        (
            "cross_display_semantics",
            run_cross_display_semantics_check_mode,
            clone_args(
                args,
                mode="cross-display-semantics-check",
                ygas=[str(fixtures["ygas"])],
                dat=str(fixtures["dat"]),
                element="H2O",
            ),
        ),
    ]

    failed = False
    for name, func, mode_args in checks:
        try:
            exit_code = int(func(mode_args))
            ok = exit_code == 0
            detail = f"exit_code={exit_code}"
        except Exception as exc:
            ok = False
            detail = str(exc)
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


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


def run_cross_display_semantics_check_mode(
    args: argparse.Namespace,
    *,
    header: str = "cross display semantics check",
    include_readme_context: bool = False,
) -> int:
    if args.ygas and args.dat:
        ygas_paths = [Path(path) for path in args.ygas]
        dat_path = Path(args.dat)
        input_source = "cli_inputs"
    else:
        fixtures = resolve_repo_fixture_paths()
        ygas_paths = [fixtures["ygas"]]
        dat_path = fixtures["dat"]
        input_source = "repo_fixture_default"
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
    generic_mag_freq, _generic_mag_raw_values, generic_mag_details = core.compute_cross_spectrum_from_arrays_with_params(
        reference_frame["value"].to_numpy(dtype=float),
        dat_target_frame["value"].to_numpy(dtype=float),
        float(generic_fs),
        args.nsegment,
        args.overlap_ratio,
        spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
        insufficient_message="generic 双列分析未得到有效互谱幅值。",
    )
    generic_mag_details = dict(generic_mag_details)
    generic_mag_details.update(core.describe_generic_default_cross_implementation())
    generic_mag_details["alignment_strategy"] = core.GENERIC_SAME_FRAME_ALIGNMENT_STRATEGY
    generic_mag_details["cross_reference_column"] = str(reference_column)
    generic_mag_details["reference_column"] = str(reference_column)
    generic_mag_details["target_column"] = str(dat_target_column)
    generic_mag_details["cross_order"] = f"{reference_column} -> {dat_target_column}"
    generic_mag_values, generic_mag_mask, generic_mag_details = core.resolve_cross_display_output(
        generic_mag_freq,
        generic_mag_details,
        analysis_context=generic_mag_details.get("analysis_context"),
        cross_execution_path=generic_mag_details.get("cross_execution_path"),
        spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
        insufficient_message="generic 双列分析未得到有效互谱幅值。",
    )
    generic_quad_freq, _generic_quad_raw_values, generic_quad_details = core.compute_cross_spectrum_from_arrays_with_params(
        reference_frame["value"].to_numpy(dtype=float),
        dat_target_frame["value"].to_numpy(dtype=float),
        float(generic_fs),
        args.nsegment,
        args.overlap_ratio,
        spectrum_type=core.CROSS_SPECTRUM_IMAG,
        insufficient_message="generic 双列分析未得到有效正交谱。",
    )
    generic_quad_details = dict(generic_quad_details)
    generic_quad_details.update(core.describe_generic_default_cross_implementation())
    generic_quad_details["alignment_strategy"] = core.GENERIC_SAME_FRAME_ALIGNMENT_STRATEGY
    generic_quad_details["cross_reference_column"] = str(reference_column)
    generic_quad_details["reference_column"] = str(reference_column)
    generic_quad_details["target_column"] = str(dat_target_column)
    generic_quad_details["cross_order"] = f"{reference_column} -> {dat_target_column}"
    generic_quad_values, generic_quad_mask, generic_quad_details = core.resolve_cross_display_output(
        generic_quad_freq,
        generic_quad_details,
        analysis_context=generic_quad_details.get("analysis_context"),
        cross_execution_path=generic_quad_details.get("cross_execution_path"),
        spectrum_type=core.CROSS_SPECTRUM_IMAG,
        insufficient_message="generic 双列分析未得到有效正交谱。",
    )

    def _masked_values(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(payload["mask"], dtype=bool)
        return np.asarray(payload["freq"], dtype=float)[mask], np.asarray(payload["values"], dtype=float)[mask]

    def _masked_detail_values(mask: np.ndarray, details: dict[str, Any], detail_key: str) -> np.ndarray:
        return np.asarray(details[detail_key], dtype=float)[np.asarray(mask, dtype=bool)]

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
    generic_real_values = _masked_detail_values(generic_mask, generic_details, "cross_spectrum_real")
    generic_mag_display_values = np.asarray(generic_mag_values, dtype=float)[np.asarray(generic_mag_mask, dtype=bool)]
    generic_mag_abs_values = _masked_detail_values(generic_mag_mask, generic_mag_details, "cross_spectrum_abs")
    generic_quad_display_values = np.asarray(generic_quad_values, dtype=float)[np.asarray(generic_quad_mask, dtype=bool)]
    generic_quad_imag_values = _masked_detail_values(generic_quad_mask, generic_quad_details, "cross_spectrum_imag")
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

    math_checks: list[tuple[str, bool]] = [
        ("generic cross magnitude display_semantics == abs_pxy", generic_mag_details.get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("generic cross magnitude display_value_source == cross_spectrum_abs", generic_mag_details.get("display_value_source") == "cross_spectrum_abs"),
        ("generic cross magnitude values come from abs(Pxy)", np.allclose(generic_mag_display_values, generic_mag_abs_values)),
        ("generic cospectrum display_semantics == real_pxy", generic_details.get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_REAL),
        ("generic cospectrum display_value_source == cross_spectrum_real", generic_details.get("display_value_source") == "cross_spectrum_real"),
        ("generic cospectrum values come from real(Pxy)", np.allclose(generic_display_values, generic_real_values)),
        ("generic quadrature display_semantics == imag_pxy", generic_quad_details.get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_IMAG),
        ("generic quadrature display_value_source == cross_spectrum_imag", generic_quad_details.get("display_value_source") == "cross_spectrum_imag"),
        ("generic quadrature values come from imag(Pxy)", np.allclose(generic_quad_display_values, generic_quad_imag_values)),
    ]
    frr_checks: list[tuple[str, bool]] = [
        ("target canonical ygas cos == magnitude", _allclose_payload(advanced_ygas_cos, advanced_ygas_mag)),
        ("target canonical dat cos == magnitude", _allclose_payload(advanced_dat_cos, advanced_dat_mag)),
        ("target canonical ygas UI cospectrum display_semantics == abs_pxy", advanced_ygas_cos["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("target canonical dat UI cospectrum display_semantics == abs_pxy", advanced_dat_cos["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
        ("target canonical quadrature display_semantics == imag_pxy", advanced_ygas_quad["details"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_IMAG),
        ("target canonical quadrature values still come from imag(Pxy)", np.allclose(quad_display_values, quad_imag_values)),
        ("target group canonical cos keys == magnitude keys", group_pair_keys_same),
        ("target group canonical cos values == magnitude values", group_values_same),
        ("target group target-spectral metadata display_semantics == abs_pxy", target_group_cos["target_metadata"].get("display_semantics") == core.CROSS_DISPLAY_SEMANTICS_ABS),
    ]
    diagnostic_checks: list[tuple[str, bool]] = [
        ("target group actual_series_count == kept_groups * 2", actual_group_series_count == expected_group_series_count),
        ("target group generated_cross_series_roles == [dat_target_vs_uz, ygas_target_vs_uz]", generated_roles == ["dat_target_vs_uz", "ygas_target_vs_uz"]),
        ("target group reference_column == Uz", target_group_cos["target_metadata"].get("reference_column") == "Uz"),
        ("target group has ygas_target_column", bool(target_group_cos["target_metadata"].get("ygas_target_column"))),
        ("target group has dat_target_column", bool(target_group_cos["target_metadata"].get("dat_target_column"))),
        ("target group canonical_cross_pairs count == 2", len(target_group_cos["target_metadata"].get("canonical_cross_pairs") or []) == 2),
        ("summary ygas pair resolves as canonical", bool(summary_pair_ygas.get("uses_canonical_pair")) and summary_pair_ygas.get("resolved_pair", {}).get("series_role") == "ygas_target_vs_uz"),
        ("summary dat pair resolves as canonical", bool(summary_pair_dat.get("uses_canonical_pair")) and summary_pair_dat.get("resolved_pair", {}).get("series_role") == "dat_target_vs_uz"),
        ("canonical diagnostics.cross_execution_path == target_spectral_canonical", advanced_ygas_cos["details"].get("cross_execution_path") == "target_spectral_canonical"),
        ("canonical diagnostics.cross_implementation_id == A", advanced_ygas_cos["details"].get("cross_implementation_id") == "A"),
        ("canonical diagnostics.cross_reference_column == Uz", advanced_ygas_cos["details"].get("cross_reference_column") == "Uz"),
        (
            f"canonical diagnostics.cross_order == {reference_column} -> {ygas_target_column}",
            advanced_ygas_cos["details"].get("cross_order") == f"{reference_column} -> {ygas_target_column}",
        ),
    ]

    print(f"[{header}]")
    print(f"- input_source={input_source}")
    print(f"- element={target_element}")
    print(f"- ygas_count={len(ygas_paths)}")
    print(f"- dat={dat_path}")
    print(f"- ygas_target_column={ygas_target_column}")
    print(f"- dat_target_column={dat_target_column}")
    print(f"- reference_column={reference_column}")
    print("")

    for section_title, section_checks in (
        ("数学正确语义", math_checks),
        ("FRR 兼容语义", frr_checks),
        ("诊断字段", diagnostic_checks),
    ):
        print(f"[{section_title}]")
        for label, ok in section_checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print("")

    if include_readme_context:
        print("[reverse_engineering README 对照]")
        for section_title, key in (
            ("已确认的 payload 事实", "verified_facts"),
            ("当前 rebuild 采用的解释", "interpretation"),
            ("当前仍然存在的 gap", "current_gap"),
        ):
            print(f"- {section_title}:")
            items = load_reverse_engineering_semantics_context().get(key, [])
            if not items:
                print("  - evidence unavailable")
                continue
            for item in items:
                print(f"  - {item}")
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
        "display_semantics",
        "display_value_source",
    ):
        print(f"- {key}={diagnostics.get(key)}")

    failed = [label for label, ok in [*math_checks, *frr_checks, *diagnostic_checks] if not ok]
    if failed:
        print("")
        print("[failed checks]")
        for label in failed:
            print(f"- {label}")
        return 1

    print("")
    print("[summary]")
    print("- 数学正确语义：generic dual-column analysis 保持 cross magnitude=abs(Pxy), cospectrum=real(Pxy), quadrature=imag(Pxy)。")
    print("- FRR 兼容语义：target-spectral canonical path 下，UI“协谱图”仍映射到 abs(Pxy)，因此会与互谱幅值一致。")
    print("- README 对照说明：当前仅基于 payload 线索与仓库实现做回归验证，没有声称已完全还原原始源码。")
    print("- reverse_engineering/README.md 已明确：当前仍未在仓库中找到实际反编译 Python 源文件。")
    return 0


def run_frr_compat_semantics_check_mode(args: argparse.Namespace) -> int:
    return run_cross_display_semantics_check_mode(
        args,
        header="frr compat semantics check",
        include_readme_context=True,
    )


def run_device_group_spectral_check_mode(args: argparse.Namespace) -> int:
    input_paths: list[Path] = [Path(value) for value in (args.ygas or [])]
    if args.dat:
        input_paths.append(Path(args.dat))
    if not input_paths:
        raise ValueError("device-group-spectral-check needs at least one --ygas or --dat input.")

    sample_path = Path(args.ygas[0]) if args.ygas else input_paths[0]
    sample_device_key = "A" if sample_path.suffix.lower() in {".txt", ".log"} else "B"
    sample_parsed = parse_supported_file(sample_path)
    target_column = choose_single_column(sample_parsed, args.element, sample_device_key)
    if not target_column:
        raise ValueError(f"No analyzable target column found for {sample_path.name}.")

    checks: list[tuple[str, bool, object]] = []
    root, app = build_headless_app()
    try:
        default_single_plot_execution_path = (
            "single_device_compare_psd_render"
            if bool(args.dat) and sample_path.suffix.lower() in {".txt", ".log"}
            else "single_device_spectrum"
        )
        default_single_plot_kind = (
            "single_device_compare_psd"
            if default_single_plot_execution_path == "single_device_compare_psd_render"
            else "single_device_spectrum"
        )
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
                payload_single["effective_device_count"] == 1
                and len(payload_single["series_results"]) == 1
                and str(payload_single.get("plot_execution_path")) == default_single_plot_execution_path,
                {
                    "device_groups": payload_single["device_groups"],
                    "plot_execution_path": payload_single.get("plot_execution_path"),
                    "default_single_plot_execution_path": default_single_plot_execution_path,
                },
            )
        )

        render_prepared_multi_spectral_payload(app, payload_single)
        checks.append(
            (
                "single_device_render",
                len(app.current_compare_files) == 1
                and app.current_plot_kind == default_single_plot_kind
                and get_export_plot_execution_path(app) == default_single_plot_execution_path,
                {
                    "current_compare_files": list(app.current_compare_files),
                    "current_plot_kind": app.current_plot_kind,
                    "export_plot_execution_path": get_export_plot_execution_path(app),
                    "default_single_plot_kind": default_single_plot_kind,
                    "default_single_plot_execution_path": default_single_plot_execution_path,
                },
            )
        )

        with tempfile.TemporaryDirectory(prefix="device_group_check_") as temp_dir:
            temp_root = Path(temp_dir)
            same_device_copy = temp_root / f"{sample_path.stem}_copy{sample_path.suffix}"
            shutil.copy2(sample_path, same_device_copy)
            same_device_paths = [sample_path, same_device_copy]
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
                    payload_same_device["effective_device_count"] == 1
                    and len(same_groups) == 1
                    and int(same_groups[0]["file_count"]) >= 2,
                    same_groups,
                )
            )

            if args.dat:
                dat_path = Path(args.dat)
                dat_copy_a = temp_root / "DEVB_toa5.dat"
                dat_copy_b = temp_root / "DEVC_toa5.dat"
                shutil.copy2(dat_path, dat_copy_a)
                shutil.copy2(dat_path, dat_copy_b)
                payload_multi_device = app.prepare_multi_spectral_compare_payload(
                    [sample_path, dat_copy_a, dat_copy_b],
                    target_column,
                    args.fs,
                    args.nsegment,
                    args.overlap_ratio,
                )
                multi_groups = list(payload_multi_device["device_groups"])
                checks.append(
                    (
                        "multi_device_grouping",
                        payload_multi_device["effective_device_count"] >= 2
                        and len(multi_groups) >= 2
                        and str(payload_multi_device.get("plot_execution_path")) in {"multi_device_compare", "multi_device_overlay"},
                        {
                            "device_groups": multi_groups,
                            "plot_execution_path": payload_multi_device.get("plot_execution_path"),
                        },
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


def run_single_vs_compare_point_display_contract_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-vs-compare-point-display-contract-check needs --ygas and --dat.")

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

    compare_mode = "时间段内 PSD 对比"
    single_tokens = [
        "single_series_valid_freq_points=",
        "single_series_frequency_point_count=",
        "series_count=",
    ]
    explicit_single_tokens = [
        "visible_single_side_valid_freq_points=",
        "visible_single_side_frequency_point_count=",
        "compare_geometry_total_valid_freq_points=",
        "compare_geometry_total_frequency_points=",
    ]
    single_overlay_tokens = [
        "txt_single_series_valid_freq_points=",
        "txt_single_series_frequency_point_count=",
        "series_count=",
    ]
    compare_tokens = [
        "txt_single_series_valid_freq_points=",
        "txt_single_series_frequency_point_count=",
        "dat_single_series_valid_freq_points=",
        "dat_single_series_frequency_point_count=",
        "series_count=",
        "total_valid_freq_points_across_series=",
        "total_frequency_points_across_series=",
    ]

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None

        single_scope_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if len(single_scope_payload["series_results"]) != 1:
            raise ValueError("single-vs-compare-point-display-contract-check expects exactly one synced single series.")
        single_scope_series = single_scope_payload["series_results"][0]
        single_scope_details = dict(single_scope_series["details"])

        app.current_file = ygas_path
        app.plot_results(
            "spectral",
            np.asarray(single_scope_series["freq"], dtype=float),
            np.asarray(single_scope_series["density"], dtype=float),
            [col_a],
            dict(single_scope_details),
        )
        single_plot_diag_text = app.diagnostic_var.get()
        single_plot_status_text = app.status_var.get()
        single_plot_contract = dict(app.current_point_count_contract)

        render_prepared_multi_spectral_payload(app, single_scope_payload)
        single_overlay_diag_text = app.diagnostic_var.get()
        single_overlay_status_text = app.status_var.get()
        single_overlay_contract = dict(app.current_point_count_contract)

        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": bool(
                    selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                ),
            },
        )
        if dual_plot_payload.get("kind") != "psd_compare":
            raise ValueError(f"expected psd_compare payload, got {dual_plot_payload.get('kind')}")
        app.on_prepared_dual_plot_ready(dual_plot_payload)
        compare_diag_text = app.diagnostic_var.get()
        compare_status_text = app.status_var.get()
        compare_contract = dict(app.current_point_count_contract)
    finally:
        root.destroy()

    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
    )
    dual_dat = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "dat" and str(item["column"]) == col_b
    )
    expected_compare_total_valid = sum(
        int(item["details"].get("valid_freq_points", len(item["freq"])))
        for item in dual_plot_payload["series_results"]
    )
    expected_compare_total_frequency = sum(
        int(item["details"].get("frequency_point_count", len(item["freq"])))
        for item in dual_plot_payload["series_results"]
    )
    expected_compare_rendered = sum(len(item["freq"]) for item in dual_plot_payload["series_results"])
    expected_single_visible_valid = int(dual_ygas["details"].get("valid_freq_points", 0))
    expected_single_visible_frequency = int(dual_ygas["details"].get("frequency_point_count", 0))

    checks = [
        (
            "single_plot_contract_tokens_visible_in_diag",
            all(token in single_plot_diag_text for token in single_tokens),
            {"diag": single_plot_diag_text},
        ),
        (
            "single_plot_contract_tokens_visible_in_status",
            all(token in single_plot_status_text for token in single_tokens),
            {"status": single_plot_status_text},
        ),
        (
            "single_plot_explicit_visible_vs_geometry_tokens_visible",
            all(token in single_plot_diag_text for token in explicit_single_tokens)
            and all(token in single_plot_status_text for token in explicit_single_tokens),
            {
                "single_plot_status": single_plot_status_text,
                "single_plot_diag": single_plot_diag_text,
            },
        ),
        (
            "single_overlay_contract_tokens_visible_in_diag",
            all(token in single_overlay_diag_text for token in single_overlay_tokens),
            {"diag": single_overlay_diag_text},
        ),
        (
            "single_overlay_contract_tokens_visible_in_status",
            all(token in single_overlay_status_text for token in single_overlay_tokens),
            {"status": single_overlay_status_text},
        ),
        (
            "single_overlay_explicit_visible_vs_geometry_tokens_visible",
            all(token in single_overlay_diag_text for token in explicit_single_tokens)
            and all(token in single_overlay_status_text for token in explicit_single_tokens),
            {
                "single_overlay_status": single_overlay_status_text,
                "single_overlay_diag": single_overlay_diag_text,
            },
        ),
        (
            "compare_contract_tokens_visible_in_diag",
            all(token in compare_diag_text for token in compare_tokens),
            {"diag": compare_diag_text},
        ),
        (
            "compare_contract_tokens_visible_in_status",
            all(token in compare_status_text for token in compare_tokens),
            {"status": compare_status_text},
        ),
        (
            "compare_diag_no_ambiguous_valid_freq_label",
            "有效频点数=" not in compare_diag_text,
            {"diag": compare_diag_text},
        ),
        (
            "single_status_no_legacy_current_freq_label",
            "当前频点数=" not in single_plot_status_text and "当前频点数=" not in single_overlay_status_text,
            {
                "single_plot_status": single_plot_status_text,
                "single_overlay_status": single_overlay_status_text,
            },
        ),
        (
            "single_visible_and_compare_geometry_labels_are_distinct",
            "visible_single_side_valid_freq_points=" in single_plot_status_text
            and "compare_geometry_total_valid_freq_points=" in single_plot_status_text
            and "visible_single_side_valid_freq_points=" in single_overlay_status_text
            and "compare_geometry_total_valid_freq_points=" in single_overlay_status_text
            and int(single_scope_details.get("visible_single_side_valid_freq_points", 0))
            == int(expected_single_visible_valid)
            and int(single_scope_details.get("visible_single_side_frequency_point_count", 0))
            == int(expected_single_visible_frequency)
            and int(single_scope_details.get("compare_geometry_total_valid_freq_points", 0))
            == int(expected_compare_total_valid)
            and int(single_scope_details.get("compare_geometry_total_frequency_points", 0))
            == int(expected_compare_total_frequency)
            and int(single_scope_details.get("compare_geometry_total_valid_freq_points", 0))
            > int(single_scope_details.get("visible_single_side_valid_freq_points", 0)),
            {
                "single_scope_details": single_scope_details,
                "expected_single_visible_valid": expected_single_visible_valid,
                "expected_single_visible_frequency": expected_single_visible_frequency,
                "expected_compare_total_valid": expected_compare_total_valid,
                "expected_compare_total_frequency": expected_compare_total_frequency,
            },
        ),
        (
            "single_plot_single_series_matches_compare_txt",
            int(single_plot_contract["global"]["single_series_valid_freq_points"])
            == int(dual_ygas["details"].get("valid_freq_points", 0))
            and int(single_plot_contract["global"]["single_series_frequency_point_count"])
            == int(dual_ygas["details"].get("frequency_point_count", 0)),
            {
                "single_plot_contract": single_plot_contract,
                "compare_txt_details": dual_ygas["details"],
            },
        ),
        (
            "single_overlay_single_series_matches_compare_txt",
            int(single_overlay_contract["global"]["single_series_valid_freq_points"])
            == int(dual_ygas["details"].get("valid_freq_points", 0))
            and int(single_overlay_contract["global"]["single_series_frequency_point_count"])
            == int(dual_ygas["details"].get("frequency_point_count", 0)),
            {
                "single_overlay_contract": single_overlay_contract,
                "compare_txt_details": dual_ygas["details"],
            },
        ),
        (
            "compare_txt_side_contract_matches_payload",
            int(compare_contract["txt"]["single_series_valid_freq_points"])
            == int(dual_ygas["details"].get("valid_freq_points", 0))
            and int(compare_contract["txt"]["single_series_frequency_point_count"])
            == int(dual_ygas["details"].get("frequency_point_count", 0)),
            {
                "compare_txt_contract": compare_contract.get("txt"),
                "compare_txt_details": dual_ygas["details"],
            },
        ),
        (
            "compare_dat_side_contract_matches_payload",
            int(compare_contract["dat"]["single_series_valid_freq_points"])
            == int(dual_dat["details"].get("valid_freq_points", 0))
            and int(compare_contract["dat"]["single_series_frequency_point_count"])
            == int(dual_dat["details"].get("frequency_point_count", 0)),
            {
                "compare_dat_contract": compare_contract.get("dat"),
                "compare_dat_details": dual_dat["details"],
            },
        ),
        (
            "compare_total_valid_matches_series_sum",
            int(compare_contract["global"]["total_valid_freq_points_across_series"]) == int(expected_compare_total_valid),
            {
                "compare_global_contract": compare_contract.get("global"),
                "expected_total_valid": expected_compare_total_valid,
            },
        ),
        (
            "compare_total_frequency_matches_series_sum",
            int(compare_contract["global"]["total_frequency_points_across_series"]) == int(expected_compare_total_frequency),
            {
                "compare_global_contract": compare_contract.get("global"),
                "expected_total_frequency": expected_compare_total_frequency,
            },
        ),
        (
            "compare_series_count_matches_series_len",
            int(compare_contract["global"]["series_count"]) == int(len(dual_plot_payload["series_results"])),
            {
                "compare_global_contract": compare_contract.get("global"),
                "payload_series_count": len(dual_plot_payload["series_results"]),
            },
        ),
        (
            "compare_side_totals_sum_to_global",
            int(compare_contract["txt"]["total_valid_freq_points_across_series"])
            + int(compare_contract["dat"]["total_valid_freq_points_across_series"])
            == int(compare_contract["global"]["total_valid_freq_points_across_series"])
            and int(compare_contract["txt"]["total_frequency_points_across_series"])
            + int(compare_contract["dat"]["total_frequency_points_across_series"])
            == int(compare_contract["global"]["total_frequency_points_across_series"]),
            {"compare_contract": compare_contract},
        ),
        (
            "rendered_points_equal_payload_lengths_no_decimation",
            int(compare_contract["global"]["total_rendered_point_count_across_series"]) == int(expected_compare_rendered)
            and int(expected_compare_rendered) == int(expected_compare_total_valid),
            {
                "compare_global_contract": compare_contract.get("global"),
                "expected_rendered": expected_compare_rendered,
                "expected_total_valid": expected_compare_total_valid,
            },
        ),
    ]

    failed = False
    print("[single_vs_compare_point_display_contract_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"same_window_time_range={start_dt} ~ {end_dt}")
    print(f"single_plot_contract={single_plot_contract}")
    print(f"single_overlay_contract={single_overlay_contract}")
    print(f"compare_contract={compare_contract}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_compare_style_preview_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_default_single_compare_style_check_mode(args)

    if not args.ygas or not args.dat:
        raise ValueError("single-compare-style-preview-check needs --ygas and --dat.")

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

    compare_mode = "时间段内 PSD 对比"
    compare_style_status_tag = "单图按对比图方式显示=是"
    compare_style_diag_tag = "single_compare_style_preview=enabled"

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        def configure_single_source(source_path: Path, parsed_source: ParsedFileResult, selected_column: str) -> None:
            app.current_data_source_kind = "file"
            app.current_data_source_label = f"当前文件：{source_path.name}"
            app.current_file = source_path
            app.current_file_parsed = parsed_source
            app.current_layout_label = parsed_source.profile_name
            app.raw_data = parsed_source.dataframe.copy()
            app.column_vars = {selected_column: tk.BooleanVar(master=root, value=True)}

        configure_single_source(ygas_path, parsed_a, str(col_a))

        app.auto_prepare_payload = None
        app.auto_dat_options = {}
        app.selected_dat_var.set("")
        app.refresh_single_compare_style_preview_state()
        preview_unavailable_without_pair = not bool(app.single_compare_style_preview_available)

        auto_payload = app.prepare_folder_auto_selection_payload([ygas_path, dat_path])
        app.auto_prepare_payload = auto_payload
        app.auto_dat_options = {app.build_dat_option_label(info): Path(info["path"]) for info in auto_payload["dat_infos"]}
        selected_dat_path = Path(auto_payload["selected_dat_path"])
        selected_dat_label = next(
            label for label, path in app.auto_dat_options.items() if Path(path) == selected_dat_path
        )
        app.selected_dat_var.set(selected_dat_label)
        app.refresh_single_compare_style_preview_state()
        preview_available_with_pair = bool(app.single_compare_style_preview_available)

        app.single_compare_style_preview_var.set(False)
        app.perform_analysis("spectral", quiet=True)
        plain_contract = dict(app.current_point_count_contract)
        plain_status_text = app.status_var.get()
        plain_diag_text = app.diagnostic_var.get()
        plain_plot_kind = app.current_plot_kind

        app.single_compare_style_preview_var.set(True)
        app.perform_analysis("spectral", quiet=True)
        preview_contract = dict(app.current_point_count_contract)
        preview_status_text = app.status_var.get()
        preview_diag_text = app.diagnostic_var.get()
        preview_plot_kind = app.current_plot_kind

        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None

        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": bool(
                    selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                ),
            },
        )
        app.on_prepared_dual_plot_ready(dual_plot_payload)
        compare_contract = dict(app.current_point_count_contract)

        dual_ygas = next(
            item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
        )
        dual_dat = next(
            item for item in dual_plot_payload["series_results"] if str(item["side"]) == "dat" and str(item["column"]) == col_b
        )

        configure_single_source(dat_path, parsed_b, str(col_b))
        app.refresh_single_compare_style_preview_state()
        preview_available_with_pair_from_dat = bool(app.single_compare_style_preview_available)

        app.single_compare_style_preview_var.set(False)
        app.perform_analysis("spectral", quiet=True)
        dat_plain_contract = dict(app.current_point_count_contract)
        dat_plain_status_text = app.status_var.get()
        dat_plain_diag_text = app.diagnostic_var.get()
        dat_plain_plot_kind = app.current_plot_kind

        app.single_compare_style_preview_var.set(True)
        app.perform_analysis("spectral", quiet=True)
        dat_preview_contract = dict(app.current_point_count_contract)
        dat_preview_status_text = app.status_var.get()
        dat_preview_diag_text = app.diagnostic_var.get()
        dat_preview_plot_kind = app.current_plot_kind
    finally:
        root.destroy()

    checks = [
        (
            "preview_requires_pairable_txt_dat_inputs",
            preview_unavailable_without_pair and preview_available_with_pair and preview_available_with_pair_from_dat,
            {
                "preview_unavailable_without_pair": preview_unavailable_without_pair,
                "preview_available_with_pair": preview_available_with_pair,
                "preview_available_with_pair_from_dat": preview_available_with_pair_from_dat,
            },
        ),
        (
            "preview_off_still_single_series_from_ygas",
            plain_plot_kind == "spectral"
            and int(plain_contract["global"]["series_count"]) == 1
            and "txt" not in plain_contract
            and "dat" not in plain_contract,
            {
                "plain_plot_kind": plain_plot_kind,
                "plain_contract": plain_contract,
            },
        ),
        (
            "preview_off_status_has_no_compare_style_tag_from_ygas",
            compare_style_status_tag not in plain_status_text and compare_style_diag_tag not in plain_diag_text,
            {
                "plain_status": plain_status_text,
                "plain_diag": plain_diag_text,
            },
        ),
        (
            "preview_on_uses_compare_render_kind_from_ygas",
            preview_plot_kind == "dual_psd_compare",
            {"preview_plot_kind": preview_plot_kind},
        ),
        (
            "preview_on_status_marks_compare_style_from_ygas",
            compare_style_status_tag in preview_status_text and compare_style_diag_tag in preview_diag_text,
            {
                "preview_status": preview_status_text,
                "preview_diag": preview_diag_text,
            },
        ),
        (
            "preview_off_still_single_series_from_dat",
            dat_plain_plot_kind == "spectral"
            and int(dat_plain_contract["global"]["series_count"]) == 1
            and "txt" not in dat_plain_contract
            and "dat" not in dat_plain_contract,
            {
                "dat_plain_plot_kind": dat_plain_plot_kind,
                "dat_plain_contract": dat_plain_contract,
            },
        ),
        (
            "preview_off_status_has_no_compare_style_tag_from_dat",
            compare_style_status_tag not in dat_plain_status_text and compare_style_diag_tag not in dat_plain_diag_text,
            {
                "dat_plain_status": dat_plain_status_text,
                "dat_plain_diag": dat_plain_diag_text,
            },
        ),
        (
            "preview_on_uses_compare_render_kind_from_dat",
            dat_preview_plot_kind == "dual_psd_compare",
            {"dat_preview_plot_kind": dat_preview_plot_kind},
        ),
        (
            "preview_on_status_marks_compare_style_from_dat",
            compare_style_status_tag in dat_preview_status_text and compare_style_diag_tag in dat_preview_diag_text,
            {
                "dat_preview_status": dat_preview_status_text,
                "dat_preview_diag": dat_preview_diag_text,
            },
        ),
        (
            "preview_on_total_rendered_matches_compare_from_ygas",
            int(preview_contract["global"]["total_rendered_point_count_across_series"])
            == int(compare_contract["global"]["total_rendered_point_count_across_series"]),
            {
                "preview_global": preview_contract.get("global"),
                "compare_global": compare_contract.get("global"),
            },
        ),
        (
            "preview_on_total_rendered_matches_compare_from_dat",
            int(dat_preview_contract["global"]["total_rendered_point_count_across_series"])
            == int(compare_contract["global"]["total_rendered_point_count_across_series"]),
            {
                "dat_preview_global": dat_preview_contract.get("global"),
                "compare_global": compare_contract.get("global"),
            },
        ),
        (
            "preview_on_total_valid_matches_compare_from_ygas",
            int(preview_contract["global"]["total_valid_freq_points_across_series"])
            == int(compare_contract["global"]["total_valid_freq_points_across_series"])
            and int(preview_contract["global"]["total_frequency_points_across_series"])
            == int(compare_contract["global"]["total_frequency_points_across_series"]),
            {
                "preview_global": preview_contract.get("global"),
                "compare_global": compare_contract.get("global"),
            },
        ),
        (
            "preview_on_total_valid_matches_compare_from_dat",
            int(dat_preview_contract["global"]["total_valid_freq_points_across_series"])
            == int(compare_contract["global"]["total_valid_freq_points_across_series"])
            and int(dat_preview_contract["global"]["total_frequency_points_across_series"])
            == int(compare_contract["global"]["total_frequency_points_across_series"]),
            {
                "dat_preview_global": dat_preview_contract.get("global"),
                "compare_global": compare_contract.get("global"),
            },
        ),
        (
            "preview_on_txt_side_matches_compare_from_ygas",
            int(preview_contract["txt"]["total_rendered_point_count_across_series"]) == int(len(dual_ygas["freq"]))
            and int(preview_contract["txt"]["total_valid_freq_points_across_series"])
            == int(compare_contract["txt"]["total_valid_freq_points_across_series"])
            and int(preview_contract["txt"]["total_frequency_points_across_series"])
            == int(compare_contract["txt"]["total_frequency_points_across_series"]),
            {
                "preview_txt": preview_contract.get("txt"),
                "compare_txt": compare_contract.get("txt"),
                "expected_txt_rendered": len(dual_ygas["freq"]),
            },
        ),
        (
            "preview_on_txt_side_matches_compare_from_dat",
            int(dat_preview_contract["txt"]["total_rendered_point_count_across_series"]) == int(len(dual_ygas["freq"]))
            and int(dat_preview_contract["txt"]["total_valid_freq_points_across_series"])
            == int(compare_contract["txt"]["total_valid_freq_points_across_series"])
            and int(dat_preview_contract["txt"]["total_frequency_points_across_series"])
            == int(compare_contract["txt"]["total_frequency_points_across_series"]),
            {
                "dat_preview_txt": dat_preview_contract.get("txt"),
                "compare_txt": compare_contract.get("txt"),
                "expected_txt_rendered": len(dual_ygas["freq"]),
            },
        ),
        (
            "preview_on_dat_side_matches_compare_from_ygas",
            int(preview_contract["dat"]["total_rendered_point_count_across_series"]) == int(len(dual_dat["freq"]))
            and int(preview_contract["dat"]["total_valid_freq_points_across_series"])
            == int(compare_contract["dat"]["total_valid_freq_points_across_series"])
            and int(preview_contract["dat"]["total_frequency_points_across_series"])
            == int(compare_contract["dat"]["total_frequency_points_across_series"]),
            {
                "preview_dat": preview_contract.get("dat"),
                "compare_dat": compare_contract.get("dat"),
                "expected_dat_rendered": len(dual_dat["freq"]),
            },
        ),
        (
            "preview_on_dat_side_matches_compare_from_dat",
            int(dat_preview_contract["dat"]["total_rendered_point_count_across_series"]) == int(len(dual_dat["freq"]))
            and int(dat_preview_contract["dat"]["total_valid_freq_points_across_series"])
            == int(compare_contract["dat"]["total_valid_freq_points_across_series"])
            and int(dat_preview_contract["dat"]["total_frequency_points_across_series"])
            == int(compare_contract["dat"]["total_frequency_points_across_series"]),
            {
                "dat_preview_dat": dat_preview_contract.get("dat"),
                "compare_dat": compare_contract.get("dat"),
                "expected_dat_rendered": len(dual_dat["freq"]),
            },
        ),
    ]

    failed = False
    print("[single_compare_style_preview_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"plain_contract={plain_contract}")
    print(f"preview_contract={preview_contract}")
    print(f"dat_plain_contract={dat_plain_contract}")
    print(f"dat_preview_contract={dat_preview_contract}")
    print(f"compare_contract={compare_contract}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_txt_compare_side_equivalence_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-txt-compare-side-equivalence-check needs --ygas and --dat.")

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

    compare_mode = "时间段内 PSD 对比"
    required_status_tokens = [
        "single_txt_execution_path=compare_side_equivalent",
        "single_txt_selection_scope=merged_ygas_compare_scope",
        "single_txt_time_range_policy=txt_dat_common_window",
    ]
    required_diag_tokens = [*required_status_tokens, "plot_execution_path=single_txt_compare_side_equivalent"]

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")

        def configure_single_source(source_path: Path, parsed_source: ParsedFileResult, selected_column: str) -> None:
            app.current_data_source_kind = "file"
            app.current_data_source_label = f"当前文件：{source_path.name}"
            app.current_file = source_path
            app.current_file_parsed = parsed_source
            app.current_layout_label = parsed_source.profile_name
            app.raw_data = parsed_source.dataframe.copy()
            app.column_vars = {selected_column: tk.BooleanVar(master=root, value=True)}

        def get_export_plot_execution_path() -> str:
            if app.current_result_frame.empty or "plot_execution_path" not in app.current_result_frame.columns:
                return ""
            return str(app.current_result_frame.iloc[0].get("plot_execution_path") or "")

        configure_single_source(ygas_path, parsed_a, str(col_a))
        app.auto_prepare_payload = None
        app.auto_dat_options = {}
        app.selected_dat_var.set("")
        app.refresh_single_compare_style_preview_state()
        preview_unavailable_without_pair = not bool(app.single_compare_style_preview_available)

        app.single_compare_style_preview_var.set(True)
        fallback_payload_without_pair = app.build_single_txt_compare_equivalent_payload(str(col_a))
        app.perform_analysis("spectral", quiet=True)
        fallback_without_pair_contract = dict(app.current_point_count_contract)
        fallback_without_pair_status_text = app.status_var.get()
        fallback_without_pair_diag_text = app.diagnostic_var.get()
        fallback_without_pair_plot_kind = app.current_plot_kind
        fallback_without_pair_export_path = get_export_plot_execution_path()

        auto_payload = app.prepare_folder_auto_selection_payload([ygas_path, dat_path])
        app.auto_prepare_payload = auto_payload
        app.auto_dat_options = {app.build_dat_option_label(info): Path(info["path"]) for info in auto_payload["dat_infos"]}
        selected_dat_path = Path(auto_payload["selected_dat_path"])
        selected_dat_label = next(
            label for label, path in app.auto_dat_options.items() if Path(path) == selected_dat_path
        )
        app.selected_dat_var.set(selected_dat_label)
        app.refresh_single_compare_style_preview_state()
        preview_available_with_pair = bool(app.single_compare_style_preview_available)

        app.single_compare_style_preview_var.set(True)
        single_payload = app.build_single_txt_compare_equivalent_payload(str(col_a))
        app.perform_analysis("spectral", quiet=True)
        single_contract = dict(app.current_point_count_contract)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        single_plot_kind = app.current_plot_kind
        single_rendered_freq = np.asarray(app.current_result_freq, dtype=float)
        single_rendered_density = np.asarray(app.current_result_values, dtype=float)
        single_export_plot_execution_path = get_export_plot_execution_path()

        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None

        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": bool(
                    selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                ),
            },
        )
        dual_ygas = next(
            item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
        )

        configure_single_source(dat_path, parsed_b, str(col_b))
        app.refresh_single_compare_style_preview_state()
        preview_available_from_dat = bool(app.single_compare_style_preview_available)
        app.single_compare_style_preview_var.set(True)
        dat_payload = app.build_single_txt_compare_equivalent_payload(str(col_b))
        app.perform_analysis("spectral", quiet=True)
        dat_fallback_contract = dict(app.current_point_count_contract)
        dat_fallback_status_text = app.status_var.get()
        dat_fallback_diag_text = app.diagnostic_var.get()
        dat_fallback_plot_kind = app.current_plot_kind
        dat_fallback_export_path = get_export_plot_execution_path()
    finally:
        root.destroy()

    if single_payload is None:
        raise ValueError("expected single txt compare-side equivalent payload under pairable context.")

    single_details = dict(single_payload["details"])
    compare_txt_details = dict(dual_ygas["details"])
    checks = [
        (
            "fallback_without_dat_context_uses_legacy_single_path",
            preview_unavailable_without_pair
            and fallback_payload_without_pair is None
            and fallback_without_pair_plot_kind == "spectral"
            and int(fallback_without_pair_contract["global"]["series_count"]) == 1
            and "single_txt_execution_path=" not in fallback_without_pair_status_text
            and "single_txt_execution_path=" not in fallback_without_pair_diag_text
            and fallback_without_pair_export_path == "single_file_base_spectrum",
            {
                "fallback_without_pair_contract": fallback_without_pair_contract,
                "fallback_without_pair_status": fallback_without_pair_status_text,
                "fallback_without_pair_diag": fallback_without_pair_diag_text,
                "fallback_without_pair_export_path": fallback_without_pair_export_path,
            },
        ),
        (
            "pairable_context_enables_equivalent_path",
            preview_available_with_pair,
            {"preview_available_with_pair": preview_available_with_pair},
        ),
        (
            "single_txt_equivalent_still_renders_single_series",
            single_plot_kind == "spectral"
            and int(single_contract["global"]["series_count"]) == 1
            and "txt" not in single_contract
            and "dat" not in single_contract,
            {"single_contract": single_contract, "single_plot_kind": single_plot_kind},
        ),
        (
            "single_txt_equivalent_status_and_diag_visible",
            all(token in single_status_text for token in required_status_tokens)
            and all(token in single_diag_text for token in required_diag_tokens),
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
        (
            "single_txt_equivalent_export_metadata_visible",
            single_export_plot_execution_path == "single_txt_compare_side_equivalent",
            {"single_export_plot_execution_path": single_export_plot_execution_path},
        ),
        (
            "single_txt_compare_side_valid_points_equal",
            int(single_details.get("valid_points", 0)) == int(compare_txt_details.get("valid_points", 0)),
            {"single_valid_points": single_details.get("valid_points"), "compare_valid_points": compare_txt_details.get("valid_points")},
        ),
        (
            "single_txt_compare_side_nperseg_equal",
            int(single_details.get("nperseg", 0)) == int(compare_txt_details.get("nperseg", 0)),
            {"single_nperseg": single_details.get("nperseg"), "compare_nperseg": compare_txt_details.get("nperseg")},
        ),
        (
            "single_txt_compare_side_valid_freq_points_equal",
            int(single_details.get("valid_freq_points", 0)) == int(compare_txt_details.get("valid_freq_points", 0)),
            {"single_valid_freq_points": single_details.get("valid_freq_points"), "compare_valid_freq_points": compare_txt_details.get("valid_freq_points")},
        ),
        (
            "single_txt_compare_side_frequency_point_count_equal",
            int(single_details.get("frequency_point_count", 0)) == int(compare_txt_details.get("frequency_point_count", 0)),
            {"single_frequency_point_count": single_details.get("frequency_point_count"), "compare_frequency_point_count": compare_txt_details.get("frequency_point_count")},
        ),
        (
            "single_txt_compare_side_freq_equal",
            np.array_equal(np.asarray(single_payload["freq"], dtype=float), np.asarray(dual_ygas["freq"], dtype=float)),
            {"single_points": len(single_payload["freq"]), "compare_points": len(dual_ygas["freq"])},
        ),
        (
            "single_txt_compare_side_density_allclose",
            np.allclose(
                np.asarray(single_payload["density"], dtype=float),
                np.asarray(dual_ygas["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            ),
            {
                "single_first5": np.asarray(single_payload["density"], dtype=float)[:5].tolist(),
                "compare_first5": np.asarray(dual_ygas["density"], dtype=float)[:5].tolist(),
            },
        ),
        (
            "single_txt_compare_side_rendered_point_count_equal",
            int(single_contract["global"]["total_rendered_point_count_across_series"]) == int(len(dual_ygas["freq"]))
            and np.array_equal(single_rendered_freq, np.asarray(dual_ygas["freq"], dtype=float))
            and np.allclose(single_rendered_density, np.asarray(dual_ygas["density"], dtype=float), rtol=1e-12, atol=1e-12),
            {
                "single_rendered_points": len(single_rendered_freq),
                "compare_rendered_points": len(dual_ygas["freq"]),
                "single_contract": single_contract,
            },
        ),
        (
            "single_txt_compare_side_policy_tags_match_compare_txt_side",
            str(single_details.get("single_txt_time_range_policy")) == str(compare_txt_details.get("time_range_policy"))
            and str(single_details.get("plot_execution_path")) == "single_txt_compare_side_equivalent",
            {"single_details": single_details, "compare_txt_details": compare_txt_details},
        ),
        (
            "current_dat_file_falls_back_to_legacy_single_path",
            not preview_available_from_dat
            and dat_payload is None
            and dat_fallback_plot_kind == "spectral"
            and int(dat_fallback_contract["global"]["series_count"]) == 1
            and "single_txt_execution_path=" not in dat_fallback_status_text
            and "single_txt_execution_path=" not in dat_fallback_diag_text
            and dat_fallback_export_path == "single_file_base_spectrum",
            {
                "preview_available_from_dat": preview_available_from_dat,
                "dat_fallback_contract": dat_fallback_contract,
                "dat_fallback_status": dat_fallback_status_text,
                "dat_fallback_diag": dat_fallback_diag_text,
                "dat_fallback_export_path": dat_fallback_export_path,
            },
        ),
    ]

    failed = False
    print("[single_txt_compare_side_equivalence_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"single_details={single_details}")
    print(f"compare_txt_details={compare_txt_details}")
    print(f"single_contract={single_contract}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_file_open_auto_bootstrap_compare_context_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-file-open-auto-bootstrap-compare-context-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_path)
    col_a = choose_single_column(parsed_a, args.element, "A")
    if not col_a:
        raise ValueError("Unable to resolve matching txt column for the requested element.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")

        pre_open_context_missing = app.auto_prepare_payload is None and not bool(app.auto_compare_context_available)
        simulate_single_file_open(app, ygas_path)
        snapshot = app.build_auto_compare_selection_snapshot()
        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        render_prepared_multi_spectral_payload(app, payload)
        plot_kind = app.current_plot_kind
        export_plot_execution_path = get_export_plot_execution_path(app)
        status_text = app.status_var.get()
        diag_text = app.diagnostic_var.get()
        preview_check_state = str(app.single_compare_style_preview_check.cget("state"))
    finally:
        root.destroy()

    if len(payload["series_results"]) != 1:
        raise ValueError("single-file-open-auto-bootstrap-compare-context-check expects exactly one single-device series.")
    details = dict(payload["series_results"][0]["details"])
    checks = [
        (
            "pre_open_context_missing",
            pre_open_context_missing,
            {"auto_prepare_payload_is_none": pre_open_context_missing},
        ),
        (
            "single_file_open_bootstraps_auto_compare_context",
            app.auto_prepare_payload is not None
            and bool(app.auto_compare_context_available)
            and str(app.auto_compare_context_source) == "single_file_auto_bootstrap"
            and snapshot is not None,
            {
                "auto_compare_context_available": app.auto_compare_context_available,
                "auto_compare_context_source": app.auto_compare_context_source,
                "snapshot": snapshot,
            },
        ),
        (
            "bootstrapped_context_points_to_expected_dat",
            snapshot is not None
            and str(Path(snapshot["dat_path"]).resolve()) == str(dat_path.resolve())
            and str(ygas_path.resolve()) in {str(Path(path).resolve()) for path in snapshot["ygas_paths"]},
            {"snapshot": snapshot, "expected_dat": str(dat_path)},
        ),
        (
            "single_device_path_uses_compare_txt_side_after_single_file_open",
            str(details.get("single_device_execution_path")) == "compare_txt_side_equivalent",
            {"details": details},
        ),
    ]

    failed = False
    print("[single_file_open_auto_bootstrap_compare_context_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"snapshot={snapshot}")
    print(f"details={details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_no_visible_change_regression_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-no-visible-change-regression-check needs --ygas and --dat.")

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

    compare_mode = "时间段内 PSD 对比"
    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        pre_open_context_missing = app.auto_prepare_payload is None and not bool(app.auto_compare_context_available)
        simulate_single_file_open(app, ygas_path)
        snapshot = app.build_auto_compare_selection_snapshot()
        single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        render_prepared_multi_spectral_payload(app, single_payload)
        single_contract = dict(app.current_point_count_contract)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()

        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None
        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": str(col_a), "b_col": str(col_b), "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": bool(
                    selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                ),
            },
        )
        dual_ygas = next(
            item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == str(col_a)
        )
    finally:
        root.destroy()

    if len(single_payload["series_results"]) != 1:
        raise ValueError("single-device-no-visible-change-regression-check expects exactly one single-device series.")
    single_details = dict(single_payload["series_results"][0]["details"])
    compare_txt_details = dict(dual_ygas["details"])
    checks = [
        (
            "pre_open_context_missing",
            pre_open_context_missing,
            {
                "auto_prepare_payload_is_none": pre_open_context_missing,
            },
        ),
        (
            "single_file_open_bootstraps_compare_context_before_single_device_plot",
            app.auto_prepare_payload is not None
            and bool(app.auto_compare_context_available)
            and str(app.auto_compare_context_source) == "single_file_auto_bootstrap"
            and snapshot is not None,
            {
                "auto_compare_context_available": app.auto_compare_context_available,
                "auto_compare_context_source": app.auto_compare_context_source,
                "snapshot": snapshot,
            },
        ),
        (
            "fixed_single_device_uses_compare_txt_side_equivalent",
            str(single_details.get("single_device_execution_path")) == "compare_txt_side_equivalent"
            and "single_device_render_semantics=compare_psd_single_side" in single_status_text
            and "plot_execution_path=single_device_compare_psd_render" in single_diag_text,
            {"single_details": single_details},
        ),
        (
            "single_device_valid_points_equal_compare_txt",
            int(single_details.get("valid_points", 0)) == int(compare_txt_details.get("valid_points", 0)),
            {"single_valid_points": single_details.get("valid_points"), "compare_valid_points": compare_txt_details.get("valid_points")},
        ),
        (
            "single_device_nperseg_equal_compare_txt",
            int(single_details.get("nperseg", 0)) == int(compare_txt_details.get("nperseg", 0)),
            {"single_nperseg": single_details.get("nperseg"), "compare_nperseg": compare_txt_details.get("nperseg")},
        ),
        (
            "single_device_valid_freq_points_equal_compare_txt",
            int(single_details.get("valid_freq_points", 0)) == int(compare_txt_details.get("valid_freq_points", 0)),
            {"single_valid_freq_points": single_details.get("valid_freq_points"), "compare_valid_freq_points": compare_txt_details.get("valid_freq_points")},
        ),
        (
            "single_device_freq_equal_compare_txt",
            np.array_equal(
                np.asarray(single_payload["series_results"][0]["freq"], dtype=float),
                np.asarray(dual_ygas["freq"], dtype=float),
            ),
            {"single_points": len(single_payload["series_results"][0]["freq"]), "compare_points": len(dual_ygas["freq"])},
        ),
        (
            "single_device_density_allclose_compare_txt",
            np.allclose(
                np.asarray(single_payload["series_results"][0]["density"], dtype=float),
                np.asarray(dual_ygas["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            ),
            {
                "single_first5": np.asarray(single_payload["series_results"][0]["density"], dtype=float)[:5].tolist(),
                "compare_first5": np.asarray(dual_ygas["density"], dtype=float)[:5].tolist(),
            },
        ),
        (
            "single_device_rendered_point_count_equal_compare_txt",
            int(single_details.get("rendered_point_count", 0)) == int(len(dual_ygas["freq"]))
            and int(single_contract["global"]["total_rendered_point_count_across_series"]) == int(len(dual_ygas["freq"])),
            {
                "single_rendered_point_count": single_details.get("rendered_point_count"),
                "compare_rendered_point_count": len(dual_ygas["freq"]),
                "single_contract": single_contract,
            },
        ),
        (
            "single_device_status_and_diag_report_auto_bootstrap",
            "auto_compare_context_built=true" in single_status_text
            and f"auto_compare_context_dat_file_name={dat_path.name}" in single_status_text
            and "auto_compare_context_source=single_file_auto_bootstrap" in single_diag_text,
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
    ]

    failed = False
    print("[single_device_no_visible_change_regression_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"snapshot={snapshot}")
    print(f"single_details={single_details}")
    print(f"compare_txt_details={compare_txt_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_txt_compare_side_equivalence_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-txt-compare-side-equivalence-check needs --ygas and --dat.")

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

    compare_mode = "时间段内 PSD 对比"
    required_status_tokens = [
        "single_device_execution_path=compare_txt_side_equivalent",
        "single_device_render_semantics=compare_psd_single_side",
        "single_device_selection_scope=merged_ygas_compare_scope",
        "single_device_time_range_policy=txt_dat_common_window",
    ]
    required_diag_tokens = [*required_status_tokens, "plot_execution_path=single_device_compare_psd_render"]

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")

        with tempfile.TemporaryDirectory(prefix="single_device_no_dat_context_") as temp_dir:
            isolated_ygas_path = Path(temp_dir) / ygas_path.name
            shutil.copy2(ygas_path, isolated_ygas_path)
            simulate_single_file_open(app, isolated_ygas_path)
            fallback_payload = app.prepare_multi_spectral_compare_payload(
                [isolated_ygas_path],
                str(col_a),
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
        if len(fallback_payload["series_results"]) != 1:
            raise ValueError("single-device fallback payload should contain exactly one series.")
        fallback_series = fallback_payload["series_results"][0]
        fallback_details = dict(fallback_series["details"])
        render_prepared_multi_spectral_payload(app, fallback_payload)
        fallback_status_text = app.status_var.get()
        fallback_diag_text = app.diagnostic_var.get()
        fallback_contract = dict(app.current_point_count_contract)
        fallback_export_plot_execution_path = get_export_plot_execution_path(app)

        configure_auto_compare_context(app, ygas_path, dat_path)
        default_toggle_value = bool(app.single_compare_style_preview_var.get())

        single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        if len(single_payload["series_results"]) != 1:
            raise ValueError("single-device compare-side equivalent payload should contain exactly one series.")
        single_series = single_payload["series_results"][0]
        single_details = dict(single_series["details"])
        render_prepared_multi_spectral_payload(app, single_payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        single_contract = dict(app.current_point_count_contract)
        single_export_plot_execution_path = get_export_plot_execution_path(app)
        single_plot_kind = app.current_plot_kind

        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None
        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": str(col_a), "b_col": str(col_b), "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": bool(
                    selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                ),
            },
        )
        dual_ygas = next(
            item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == str(col_a)
        )
    finally:
        root.destroy()

    compare_txt_details = dict(dual_ygas["details"])
    checks = [
        (
            "single_device_fallback_without_dat_context_uses_legacy_path",
            str(fallback_details.get("single_device_execution_path")) == "legacy_device_group_scope"
            and str(fallback_details.get("single_device_compare_side_fallback_reason")) == "no_compare_dat_context"
            and fallback_export_plot_execution_path == "single_device_grouped_spectrum"
            and "single_device_execution_path=legacy_device_group_scope" in fallback_status_text
            and "single_device_compare_side_fallback_reason=no_compare_dat_context" in fallback_diag_text
            and int(fallback_contract["global"]["series_count"]) == 1,
            {
                "fallback_details": fallback_details,
                "fallback_status": fallback_status_text,
                "fallback_diag": fallback_diag_text,
                "fallback_contract": fallback_contract,
                "fallback_export_plot_execution_path": fallback_export_plot_execution_path,
            },
        ),
        (
            "single_device_reuse_enabled_by_default_without_manual_toggle",
            default_toggle_value,
            {
                "default_toggle_value": default_toggle_value,
            },
        ),
        (
            "single_device_compare_side_equivalent_still_renders_single_series",
            single_plot_kind == "single_device_compare_psd" and int(single_contract["global"]["series_count"]) == 1,
            {"single_plot_kind": single_plot_kind, "single_contract": single_contract},
        ),
        (
            "single_device_compare_side_status_and_diag_visible",
            all(token in single_status_text for token in required_status_tokens)
            and all(token in single_diag_text for token in required_diag_tokens),
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
        (
            "single_device_compare_side_export_metadata_visible",
            single_export_plot_execution_path == "single_device_compare_psd_render",
            {"single_export_plot_execution_path": single_export_plot_execution_path},
        ),
        (
            "single_device_compare_side_valid_points_equal",
            int(single_details.get("valid_points", 0)) == int(compare_txt_details.get("valid_points", 0)),
            {"single_valid_points": single_details.get("valid_points"), "compare_valid_points": compare_txt_details.get("valid_points")},
        ),
        (
            "single_device_compare_side_nperseg_equal",
            int(single_details.get("nperseg", 0)) == int(compare_txt_details.get("nperseg", 0)),
            {"single_nperseg": single_details.get("nperseg"), "compare_nperseg": compare_txt_details.get("nperseg")},
        ),
        (
            "single_device_compare_side_valid_freq_points_equal",
            int(single_details.get("valid_freq_points", 0)) == int(compare_txt_details.get("valid_freq_points", 0)),
            {
                "single_valid_freq_points": single_details.get("valid_freq_points"),
                "compare_valid_freq_points": compare_txt_details.get("valid_freq_points"),
            },
        ),
        (
            "single_device_compare_side_frequency_point_count_equal",
            int(single_details.get("frequency_point_count", 0)) == int(compare_txt_details.get("frequency_point_count", 0)),
            {
                "single_frequency_point_count": single_details.get("frequency_point_count"),
                "compare_frequency_point_count": compare_txt_details.get("frequency_point_count"),
            },
        ),
        (
            "single_device_compare_side_freq_equal",
            np.array_equal(np.asarray(single_series["freq"], dtype=float), np.asarray(dual_ygas["freq"], dtype=float)),
            {"single_points": len(single_series["freq"]), "compare_points": len(dual_ygas["freq"])},
        ),
        (
            "single_device_compare_side_density_allclose",
            np.allclose(
                np.asarray(single_series["density"], dtype=float),
                np.asarray(dual_ygas["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            ),
            {
                "single_first5": np.asarray(single_series["density"], dtype=float)[:5].tolist(),
                "compare_first5": np.asarray(dual_ygas["density"], dtype=float)[:5].tolist(),
            },
        ),
        (
            "single_device_compare_side_rendered_point_count_equal",
            int(single_details.get("rendered_point_count", 0)) == int(len(dual_ygas["freq"]))
            and int(single_contract["global"]["total_rendered_point_count_across_series"]) == int(len(dual_ygas["freq"])),
            {
                "single_rendered_point_count": single_details.get("rendered_point_count"),
                "compare_rendered_point_count": len(dual_ygas["freq"]),
                "single_contract": single_contract,
            },
        ),
        (
            "single_device_compare_side_execution_tags_match_compare_scope",
            str(single_details.get("single_device_execution_path")) == "compare_txt_side_equivalent"
            and str(single_details.get("single_device_selection_scope")) == "merged_ygas_compare_scope"
            and str(single_details.get("single_device_time_range_policy")) == str(compare_txt_details.get("time_range_policy"))
            and "single_device_render_semantics=compare_psd_single_side" in single_status_text
            and "plot_execution_path=single_device_compare_psd_render" in single_diag_text,
            {
                "single_details": single_details,
                "compare_txt_details": compare_txt_details,
                "single_status": single_status_text,
                "single_diag": single_diag_text,
            },
        ),
    ]

    failed = False
    print("[single_device_txt_compare_side_equivalence_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"single_details={single_details}")
    print(f"compare_txt_details={compare_txt_details}")
    print(f"single_contract={single_contract}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_mixed_selection_txt_dat_reuse_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-mixed-selection-txt-dat-reuse-check needs --ygas and --dat.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        simulate_single_file_open(app, ygas_path)
        if not app.select_paths_in_list([ygas_path, dat_path], active_path=ygas_path):
            raise ValueError("Failed to create mixed txt+dat file selection in the current list.")

        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path, dat_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        series_results = list(payload.get("series_results") or [])
        if len(series_results) != 2:
            raise ValueError("single-device mixed txt+dat reuse check expects dual compare payload with exactly two series.")
        single_series = next((item for item in series_results if str(item.get("side")) == "txt"), series_results[0])
        compare_series = next((item for item in series_results if str(item.get("side")) == "dat"), None)
        if compare_series is None:
            raise ValueError("single-device mixed txt+dat reuse check expects a dat-side compare series.")
        single_details = dict(single_series["details"])
        compare_details = dict(compare_series["details"])
        render_prepared_multi_spectral_payload(app, payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        single_contract = dict(app.current_point_count_contract)
        single_plot_kind = app.current_plot_kind
        single_export_plot_execution_path = get_export_plot_execution_path(app)
    finally:
        root.destroy()

    checks = [
        (
            "mixed_selection_defaults_to_dual_compare_when_effective_device_count_is_two",
            int(payload.get("effective_device_count", 0)) == 2
            and str(payload.get("dispatch_render_semantics")) == "dual_psd_compare"
            and str(payload.get("plot_execution_path")) == "dual_psd_compare"
            and single_plot_kind == "dual_psd_compare"
            and single_export_plot_execution_path == "dual_psd_compare",
            {
                "effective_device_count": payload.get("effective_device_count"),
                "dispatch_render_semantics": payload.get("dispatch_render_semantics"),
                "plot_execution_path": payload.get("plot_execution_path"),
                "single_plot_kind": single_plot_kind,
                "single_export_plot_execution_path": single_export_plot_execution_path,
            },
        ),
        (
            "mixed_selection_keeps_txt_dat_source_counts_visible",
            int(payload.get("selection_file_count", 0)) == 2
            and int(payload.get("selected_txt_file_count", 0)) == 1
            and int(payload.get("selected_dat_file_count", 0)) == 1,
            {
                "selection_file_count": payload.get("selection_file_count"),
                "selected_txt_file_count": payload.get("selected_txt_file_count"),
                "selected_dat_file_count": payload.get("selected_dat_file_count"),
            },
        ),
        (
            "mixed_selection_produces_both_txt_and_dat_compare_series",
            str(single_series.get("side")) == "txt"
            and str(compare_series.get("side")) == "dat"
            and str(single_details.get("plot_execution_path")) == "dual_psd_compare"
            and str(compare_details.get("plot_execution_path")) == "dual_psd_compare",
            {
                "single_side": single_series.get("side"),
                "compare_side": compare_series.get("side"),
                "single_details": single_details,
                "compare_details": compare_details,
            },
        ),
        (
            "mixed_selection_status_and_diag_show_unified_dispatch",
            "effective_device_count=2" in single_status_text
            and "render_semantics=dual_psd_compare" in single_status_text
            and "selection_file_count=2" in single_diag_text
            and "selected_txt_file_count=1" in single_diag_text
            and "selected_dat_file_count=1" in single_diag_text
            and "plot_execution_path=dual_psd_compare" in single_diag_text,
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
        (
            "mixed_selection_render_contract_reflects_two_visible_series",
            int(single_contract["global"]["series_count"]) == 2
            and int(single_contract["global"]["total_rendered_point_count_across_series"]) == int(len(single_series["freq"]) + len(compare_series["freq"])),
            {
                "single_contract": single_contract,
                "single_points": len(single_series["freq"]),
                "compare_points": len(compare_series["freq"]),
                "single_contract": single_contract,
            },
        ),
    ]

    failed = False
    print("[single_device_mixed_selection_txt_dat_reuse_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"payload_plot_execution_path={payload.get('plot_execution_path')}")
    print(f"single_details={single_details}")
    print(f"compare_details={compare_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_mixed_selection_multi_txt_plus_dat_reuse_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-mixed-selection-multi-txt-plus-dat-reuse-check needs --ygas and --dat.")

    ygas_source = Path(args.ygas[0])
    dat_source = Path(args.dat)
    parsed_a = parse_supported_file(ygas_source)
    parsed_b = parse_supported_file(dat_source)
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

    with tempfile.TemporaryDirectory(prefix="single_device_mixed_multi_txt_") as temp_dir:
        temp_root = Path(temp_dir)
        ygas_path_a = temp_root / "partA_sensorx.log"
        ygas_path_b = temp_root / "partB_sensorx.log"
        dat_path = temp_root / dat_source.name
        shutil.copy2(ygas_source, ygas_path_a)
        shutil.copy2(ygas_source, ygas_path_b)
        shutil.copy2(dat_source, dat_path)

        root = tk.Tk()
        root.withdraw()
        app = app_mod.FileViewerApp(root)
        try:
            app.refresh_canvas = lambda *args, **kwargs: None
            app.element_preset_var.set(str(args.element or "H2O"))
            app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
            simulate_single_file_open(app, ygas_path_a)
            selected_paths = [ygas_path_a, ygas_path_b, dat_path]
            if not app.select_paths_in_list(selected_paths, active_path=ygas_path_a):
                raise ValueError("Failed to create mixed multi-txt + dat selection in the current list.")

            payload = app.prepare_multi_spectral_compare_payload(
                selected_paths,
                str(col_a),
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            series_results = list(payload.get("series_results") or [])
            if len(series_results) != 3:
                raise ValueError("single-device mixed multi-txt + dat reuse check expects three grouped series.")
            single_series = series_results[0]
            single_details = dict(single_series["details"])
            render_prepared_multi_spectral_payload(app, payload)
            single_status_text = app.status_var.get()
            single_diag_text = app.diagnostic_var.get()
            single_contract = dict(app.current_point_count_contract)
            single_plot_kind = app.current_plot_kind
            single_export_plot_execution_path = get_export_plot_execution_path(app)
        finally:
            root.destroy()

    checks = [
        (
            "mixed_multi_txt_defaults_to_multi_overlay_when_effective_device_count_is_three",
            int(payload.get("effective_device_count", 0)) == 3
            and str(payload.get("dispatch_render_semantics")) == "multi_device_overlay"
            and str(payload.get("plot_execution_path")) == "multi_device_overlay"
            and single_plot_kind == "multi_device_overlay"
            and single_export_plot_execution_path == "multi_device_overlay",
            {
                "effective_device_count": payload.get("effective_device_count"),
                "dispatch_render_semantics": payload.get("dispatch_render_semantics"),
                "plot_execution_path": payload.get("plot_execution_path"),
                "single_plot_kind": single_plot_kind,
                "single_export_plot_execution_path": single_export_plot_execution_path,
            },
        ),
        (
            "mixed_multi_txt_keeps_txt_dat_source_counts_visible",
            int(payload.get("selection_file_count", 0)) == 3
            and int(payload.get("selected_txt_file_count", 0)) == 2
            and int(payload.get("selected_dat_file_count", 0)) == 1,
            {
                "selection_file_count": payload.get("selection_file_count"),
                "selected_txt_file_count": payload.get("selected_txt_file_count"),
                "selected_dat_file_count": payload.get("selected_dat_file_count"),
            },
        ),
        (
            "mixed_multi_txt_status_and_diag_show_unified_dispatch",
            "effective_device_count=3" in single_status_text
            and "render_semantics=multi_device_overlay" in single_status_text
            and "selection_file_count=3" in single_diag_text
            and "selected_txt_file_count=2" in single_diag_text
            and "selected_dat_file_count=1" in single_diag_text
            and "plot_execution_path=multi_device_overlay" in single_diag_text,
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
        (
            "mixed_multi_txt_series_details_follow_multi_overlay_path",
            all(str(dict(item.get("details") or {}).get("plot_execution_path")) == "multi_device_overlay" for item in series_results),
            {"series_labels": [item.get("label") for item in series_results]},
        ),
        (
            "mixed_multi_txt_render_contract_reflects_three_visible_series",
            int(single_contract["global"]["series_count"]) == 3
            and int(single_contract["global"]["total_rendered_point_count_across_series"]) == int(
                sum(len(item["freq"]) for item in series_results)
            ),
            {
                "single_contract": single_contract,
                "series_point_counts": [len(item["freq"]) for item in series_results],
            },
        ),
    ]

    failed = False
    print("[single_device_mixed_selection_multi_txt_plus_dat_reuse_check]")
    print(f"payload_plot_execution_path={payload.get('plot_execution_path')}")
    print(f"single_details={single_details}")
    print(f"series_labels={[item.get('label') for item in series_results]}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_only_dat_still_fallback_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-only-dat-still-fallback-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_b = parse_supported_file(dat_path)
    col_b = choose_single_column(parsed_b, args.element, "B")
    if not col_b:
        raise ValueError("Unable to resolve matching dat column for the requested element.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        simulate_single_file_open(app, ygas_path)
        if not app.select_paths_in_list([dat_path], active_path=dat_path):
            raise ValueError("Failed to create only-dat selection in the current list.")

        payload = app.prepare_multi_spectral_compare_payload(
            [dat_path],
            str(col_b),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        if len(payload["series_results"]) != 1:
            raise ValueError("single-device only-dat fallback check expects exactly one series.")
        single_details = dict(payload["series_results"][0]["details"])
        render_prepared_multi_spectral_payload(app, payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
    finally:
        root.destroy()

    checks = [
        (
            "only_dat_still_uses_legacy_path",
            str(single_details.get("single_device_execution_path")) == "legacy_device_group_scope"
            and str(single_details.get("single_device_compare_side_fallback_reason")) == "selection_not_txt_side",
            {"single_details": single_details},
        ),
        (
            "only_dat_selection_counts_visible",
            int(single_details.get("selected_file_count", 0)) == 1
            and int(single_details.get("selected_txt_file_count", 0)) == 0
            and int(single_details.get("selected_dat_file_count", 0)) == 1
            and not bool(single_details.get("single_device_selection_filtered_to_txt_side")),
            {"single_details": single_details},
        ),
        (
            "only_dat_status_and_diag_explain_fallback",
            "single_device_compare_side_fallback_reason=selection_not_txt_side" in single_status_text
            and "selected_txt_file_count=0" in single_diag_text
            and "selected_dat_file_count=1" in single_diag_text
            and "当前 selection 不属于 txt/ygas 一侧，已回退旧单设备路径" in single_status_text,
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
    ]

    failed = False
    print("[single_device_only_dat_still_fallback_check]")
    print(f"dat_path={dat_path}")
    print(f"column_b={col_b}")
    print(f"single_details={single_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_active_dat_sync_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-active-dat-sync-check needs --ygas and --dat.")

    ygas_source = Path(args.ygas[0])
    dat_source = Path(args.dat)
    parsed_a = parse_supported_file(ygas_source)
    col_a = choose_single_column(parsed_a, args.element, "A")
    if not col_a:
        raise ValueError("Unable to resolve matching txt column for the requested element.")

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import tkinter as tk
    import fr_r_spectrum_tool_rebuild as app_mod

    app_mod.FigureCanvasTkAgg = FigureCanvasTkAgg
    app_mod.NavigationToolbar2Tk = NavigationToolbar2Tk

    with tempfile.TemporaryDirectory() as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        ygas_path = temp_dir / "ygas_h2o_1min.log"
        dat_a_path = temp_dir / "toa5_h2o_full.dat"
        dat_b_path = temp_dir / "toa5_h2o_short.dat"
        shutil.copy2(ygas_source, ygas_path)
        shutil.copy2(dat_source, dat_a_path)
        write_truncated_dat_copy(dat_source, dat_b_path, keep_data_rows=300)

        root = tk.Tk()
        root.withdraw()
        app = app_mod.FileViewerApp(root)
        try:
            app.refresh_canvas = lambda *args, **kwargs: None
            app.element_preset_var.set(str(args.element or "H2O"))
            app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
            simulate_single_file_open(app, ygas_path)

            default_dat_label = str(app.selected_dat_var.get())
            default_dat_path = app.auto_dat_options.get(default_dat_label)
            dat_b_label = find_dat_option_label(app, dat_b_path)
            app.selected_dat_var.set(dat_b_label)
            app.on_selected_dat_changed()

            payload = app.prepare_multi_spectral_compare_payload(
                [ygas_path],
                str(col_a),
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            if len(payload["series_results"]) != 1:
                raise ValueError("single-device active dat sync check expects exactly one series.")
            single_series = payload["series_results"][0]
            single_details = dict(single_series["details"])
            render_prepared_multi_spectral_payload(app, payload)
            single_status_text = app.status_var.get()
            single_diag_text = app.diagnostic_var.get()
            single_contract = dict(app.current_point_count_contract)
            compare_b = prepare_compare_txt_side_reference(
                app,
                ygas_paths=[ygas_path],
                dat_path=dat_b_path,
                col_a=str(col_a),
                col_b=str(choose_single_column(parse_supported_file(dat_b_path), args.element, "B")),
                args=args,
            )
            compare_b_txt = compare_b["dual_ygas"]
            compare_b_details = dict(compare_b_txt["details"])
            compare_a = prepare_compare_txt_side_reference(
                app,
                ygas_paths=[ygas_path],
                dat_path=dat_a_path,
                col_a=str(col_a),
                col_b=str(choose_single_column(parse_supported_file(dat_a_path), args.element, "B")),
                args=args,
            )
            compare_a_txt = compare_a["dual_ygas"]
        finally:
            root.destroy()

    checks = [
        (
            "active_compare_dat_switched_away_from_auto_default",
            default_dat_path is not None
            and Path(default_dat_path).resolve() == dat_a_path.resolve()
            and str(single_details.get("single_device_compare_context_dat_file_name")) == dat_b_path.name,
            {
                "default_dat_label": default_dat_label,
                "default_dat_path": str(default_dat_path) if default_dat_path is not None else "",
                "active_dat_label": dat_b_label,
                "single_details": single_details,
            },
        ),
        (
            "single_device_compare_context_bound_to_active_compare_ui_dat",
            str(single_details.get("single_device_compare_context_source")) == "active_compare_ui"
            and bool(single_details.get("single_device_compare_context_matches_current_compare_ui"))
            and bool(single_details.get("single_device_compare_context_dat_matches_selected_dat"))
            and str(single_details.get("single_device_compare_context_dat_file_name")) == dat_b_path.name,
            {"single_details": single_details},
        ),
        (
            "active_dat_sync_changes_reference_window",
            int(compare_b_details.get("valid_points", 0)) != int(compare_a_txt["details"].get("valid_points", 0)),
            {
                "compare_a_valid_points": compare_a_txt["details"].get("valid_points"),
                "compare_b_valid_points": compare_b_details.get("valid_points"),
            },
        ),
        (
            "single_device_compare_context_dat_matches_selected_dat",
            np.array_equal(np.asarray(single_series["freq"], dtype=float), np.asarray(compare_b_txt["freq"], dtype=float))
            and np.allclose(
                np.asarray(single_series["density"], dtype=float),
                np.asarray(compare_b_txt["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )
            and int(single_details.get("rendered_point_count", 0)) == int(len(compare_b_txt["freq"])),
            {
                "single_points": len(single_series["freq"]),
                "compare_points": len(compare_b_txt["freq"]),
                "single_rendered_point_count": single_details.get("rendered_point_count"),
                "compare_rendered_point_count": len(compare_b_txt["freq"]),
            },
        ),
        (
            "active_dat_sync_status_and_diag_visible",
            "single_device_compare_context_source=active_compare_ui" in single_status_text
            and "single_device_render_semantics=compare_psd_single_side" in single_status_text
            and f"single_device_compare_context_dat_file_name={dat_b_path.name}" in single_status_text
            and "single_device_compare_context_dat_matches_selected_dat=true" in single_diag_text
            and "single_device_compare_context_matches_current_compare_ui=true" in single_diag_text
            and "plot_execution_path=single_device_compare_psd_render" in single_diag_text,
            {
                "single_status": single_status_text,
                "single_diag": single_diag_text,
                "single_contract": single_contract,
            },
        ),
    ]

    failed = False
    print("[single_device_active_dat_sync_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_a_path={dat_a_path}")
    print(f"dat_b_path={dat_b_path}")
    print(f"single_details={single_details}")
    print(f"compare_b_details={compare_b_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_active_time_range_sync_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-active-time-range-sync-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_path)
    parsed_b = parse_supported_file(dat_path)
    col_a = choose_single_column(parsed_a, args.element, "A")
    col_b = choose_single_column(parsed_b, args.element, "B")
    if not col_a or not col_b:
        raise ValueError("Unable to resolve matching columns for the requested element.")

    timestamps = core.parse_mixed_timestamp_series(parsed_b.dataframe[parsed_b.timestamp_col]).dropna()
    if timestamps.empty:
        raise ValueError("Unable to resolve dat timestamps for manual range sync check.")
    manual_start = pd.Timestamp(timestamps.min()) + pd.Timedelta(seconds=10)
    manual_end = pd.Timestamp(timestamps.min()) + pd.Timedelta(seconds=29.9)

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        configure_auto_compare_context(app, ygas_path, dat_path)
        app.time_range_strategy_var.set("手动输入时间范围")
        app.time_start_var.set(manual_start.strftime("%Y-%m-%d %H:%M:%S"))
        app.time_end_var.set(manual_end.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."))

        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        if len(payload["series_results"]) != 1:
            raise ValueError("single-device active time-range sync check expects exactly one series.")
        single_series = payload["series_results"][0]
        single_details = dict(single_series["details"])
        render_prepared_multi_spectral_payload(app, payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        compare_reference = prepare_compare_txt_side_reference(
            app,
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
        compare_txt = compare_reference["dual_ygas"]
        compare_txt_details = dict(compare_txt["details"])

        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        full_reference = prepare_compare_txt_side_reference(
            app,
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
    finally:
        root.destroy()

    checks = [
        (
            "single_device_time_range_bound_to_active_compare_ui",
            str(single_details.get("single_device_compare_context_source")) == "active_compare_ui"
            and bool(single_details.get("single_device_compare_context_matches_current_compare_ui"))
            and str(single_details.get("single_device_compare_context_time_range_policy"))
            == str(compare_txt_details.get("time_range_policy"))
            and pd.Timestamp(single_details.get("single_device_compare_context_start")) == pd.Timestamp(compare_reference["start_dt"])
            and pd.Timestamp(single_details.get("single_device_compare_context_end")) == pd.Timestamp(compare_reference["end_dt"]),
            {
                "single_details": single_details,
                "compare_start": compare_reference["start_dt"],
                "compare_end": compare_reference["end_dt"],
            },
        ),
        (
            "single_device_active_time_range_matches_compare_txt_side",
            np.array_equal(np.asarray(single_series["freq"], dtype=float), np.asarray(compare_txt["freq"], dtype=float))
            and np.allclose(
                np.asarray(single_series["density"], dtype=float),
                np.asarray(compare_txt["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )
            and int(single_details.get("rendered_point_count", 0)) == int(len(compare_txt["freq"])),
            {
                "single_valid_points": single_details.get("valid_points"),
                "compare_valid_points": compare_txt_details.get("valid_points"),
                "single_rendered_point_count": single_details.get("rendered_point_count"),
                "compare_rendered_point_count": len(compare_txt["freq"]),
            },
        ),
        (
            "manual_range_is_meaningfully_different_from_common_window",
            int(compare_txt_details.get("valid_points", 0)) != int(full_reference["dual_ygas"]["details"].get("valid_points", 0)),
            {
                "manual_valid_points": compare_txt_details.get("valid_points"),
                "common_valid_points": full_reference["dual_ygas"]["details"].get("valid_points"),
            },
        ),
        (
            "active_time_range_status_and_diag_visible",
            "single_device_compare_context_source=active_compare_ui" in single_status_text
            and "single_device_render_semantics=compare_psd_single_side" in single_status_text
            and "single_device_compare_context_time_range_policy=" in single_status_text
            and "single_device_compare_context_start=" in single_diag_text
            and "single_device_compare_context_end=" in single_diag_text
            and "plot_execution_path=single_device_compare_psd_render" in single_diag_text,
            {
                "single_status": single_status_text,
                "single_diag": single_diag_text,
            },
        ),
    ]

    failed = False
    print("[single_device_active_time_range_sync_check]")
    print(f"single_details={single_details}")
    print(f"compare_txt_details={compare_txt_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_auto_bootstrap_fallback_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-auto-bootstrap-fallback-check needs --ygas and --dat.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        configure_auto_compare_context(app, ygas_path, dat_path)
        app.selected_dat_var.set("")
        app.select_paths_in_list([ygas_path], active_path=ygas_path)

        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        if len(payload["series_results"]) != 1:
            raise ValueError("single-device auto-bootstrap fallback check expects exactly one series.")
        single_series = payload["series_results"][0]
        single_details = dict(single_series["details"])
        render_prepared_multi_spectral_payload(app, payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        compare_reference = prepare_compare_txt_side_reference(
            app,
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
        compare_txt = compare_reference["dual_ygas"]
    finally:
        root.destroy()

    checks = [
        (
            "single_device_auto_bootstrap_fallback_tagged",
            str(single_details.get("single_device_execution_path")) == "compare_txt_side_equivalent"
            and str(single_details.get("single_device_compare_context_source")) == "auto_bootstrap_fallback"
            and not bool(single_details.get("single_device_compare_context_matches_current_compare_ui")),
            {"single_details": single_details},
        ),
        (
            "single_device_auto_bootstrap_fallback_arrays_match_compare_txt",
            np.array_equal(np.asarray(single_series["freq"], dtype=float), np.asarray(compare_txt["freq"], dtype=float))
            and np.allclose(
                np.asarray(single_series["density"], dtype=float),
                np.asarray(compare_txt["density"], dtype=float),
                rtol=1e-12,
                atol=1e-12,
            ),
            {
                "single_rendered_point_count": single_details.get("rendered_point_count"),
                "compare_rendered_point_count": len(compare_txt["freq"]),
            },
        ),
        (
            "single_device_auto_bootstrap_fallback_status_visible",
            "single_device_compare_context_source=auto_bootstrap_fallback" in single_status_text
            and "single_device_render_semantics=compare_psd_single_side" in single_status_text
            and "当前 compare UI 状态不完整，单设备暂用 auto bootstrap 上下文" in single_status_text
            and "single_device_compare_context_matches_current_compare_ui=false" in single_diag_text
            and "plot_execution_path=single_device_compare_psd_render" in single_diag_text,
            {"single_status": single_status_text, "single_diag": single_diag_text},
        ),
    ]

    failed = False
    print("[single_device_auto_bootstrap_fallback_check]")
    print(f"single_details={single_details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_single_compare_style_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("device-count-default-single-compare-style-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_path)
    col_a = choose_single_column(parsed_a, args.element, "A")
    if not col_a:
        raise ValueError("Unable to resolve matching txt column for the requested element.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        configure_auto_compare_context(app, ygas_path, dat_path)
        default_toggle_value = bool(app.single_compare_style_preview_var.get())
        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        render_prepared_multi_spectral_payload(app, payload)
        plot_kind = app.current_plot_kind
        export_plot_execution_path = get_export_plot_execution_path(app)
        status_text = app.status_var.get()
        diag_text = app.diagnostic_var.get()
        preview_check_state = str(app.single_compare_style_preview_check.cget("state"))
    finally:
        root.destroy()

    if len(payload["series_results"]) != 1:
        raise ValueError("single-device default reuse check expects exactly one series.")
    details = dict(payload["series_results"][0]["details"])
    checks = [
        (
            "default_indicator_is_not_manual_gate",
            preview_check_state == "disabled",
            {
                "single_compare_style_preview_default": default_toggle_value,
                "single_compare_style_preview_state": preview_check_state,
            },
        ),
        (
            "single_device_default_dispatch_path_active",
            int(details.get("effective_device_count", 0)) == 1
            and str(payload.get("plot_execution_path")) == "single_device_compare_psd_render"
            and str(payload.get("dispatch_render_semantics")) == "single_device_compare_psd",
            {"details": details},
        ),
        (
            "single_device_default_reuse_uses_compare_renderer",
            plot_kind == "single_device_compare_psd"
            and export_plot_execution_path == "single_device_compare_psd_render"
            and "single_device_execution_path=compare_txt_side_equivalent" in status_text
            and "current_plot_kind=single_device_compare_psd" in status_text
            and "plot_execution_path=single_device_compare_psd_render" in status_text
            and "single_device_compare_context_dat_file_name=" in diag_text
            and "single_device_compare_context_time_range_policy=" in diag_text,
            {
                "plot_kind": plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_default_single_compare_style_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"details={details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_render_semantic_equivalence_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-render-semantic-equivalence-check needs --ygas and --dat.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("?? txt+dat ??????")
        configure_auto_compare_context(app, ygas_path, dat_path)

        single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        if len(single_payload["series_results"]) != 1:
            raise ValueError("single-device render semantic equivalence check expects exactly one single-device series.")
        render_prepared_multi_spectral_payload(app, single_payload)
        single_status_text = app.status_var.get()
        single_diag_text = app.diagnostic_var.get()
        single_plot_kind = app.current_plot_kind
        single_plot_style = app.current_plot_style_label
        single_plot_layout = app.current_plot_layout_label
        single_export_plot_execution_path = get_export_plot_execution_path(app)
        single_titles = [axis.get_title() for axis in app.figure.axes]
        single_compare_files = list(app.current_compare_files)

        compare_reference = prepare_compare_txt_side_reference(
            app,
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
        render_prepared_dual_plot_payload(app, compare_reference["dual_plot_payload"])
        compare_plot_style = app.current_plot_style_label
        compare_plot_layout = app.current_plot_layout_label
        compare_titles = [axis.get_title() for axis in app.figure.axes]
    finally:
        root.destroy()

    compare_plot_name = "时间段内 PSD 对比"
    overlay_layout = single_plot_layout == "叠加同图"
    title_semantics_match = (
        bool(single_titles)
        and bool(compare_titles)
        and (
            (
                overlay_layout
                and single_titles[0].startswith("时间段内 PSD 对比（txt侧）")
                and compare_titles[0].startswith("时间段内 PSD 对比")
            )
            or (
                (not overlay_layout)
                and single_titles[0].startswith("设备A：")
                and any(title.startswith("设备A：") for title in compare_titles)
            )
        )
    )
    status_semantics_match = (
        "single_device_render_semantics=compare_psd_single_side" in single_status_text
        and "plot_execution_path=single_device_compare_psd_render" in single_diag_text
        and "图已生成：时间段内 PSD 对比 /" in single_status_text
    )
    style_layout_match = (
        single_plot_style == compare_plot_style
        and single_plot_layout == compare_plot_layout
        and single_plot_style == app.resolve_plot_style(compare_plot_name)
        and single_plot_layout == app.resolve_plot_layout(compare_plot_name)
    )
    checks = [
        (
            "single_device_compare_equivalent_uses_compare_render_path",
            single_plot_kind == "single_device_compare_psd"
            and single_export_plot_execution_path == "single_device_compare_psd_render",
            {
                "single_plot_kind": single_plot_kind,
                "single_export_plot_execution_path": single_export_plot_execution_path,
            },
        ),
        (
            "single_device_compare_equivalent_status_and_diag_expose_render_semantics",
            status_semantics_match,
            {
                "single_status": single_status_text,
                "single_diag": single_diag_text,
            },
        ),
        (
            "single_device_compare_equivalent_style_layout_match_compare_renderer",
            style_layout_match,
            {
                "single_plot_style": single_plot_style,
                "single_plot_layout": single_plot_layout,
                "compare_plot_style": compare_plot_style,
                "compare_plot_layout": compare_plot_layout,
            },
        ),
        (
            "single_device_compare_equivalent_title_semantics_match_compare",
            title_semantics_match and len(single_compare_files) == 1,
            {
                "single_titles": single_titles,
                "compare_titles": compare_titles,
                "single_compare_files": single_compare_files,
            },
        ),
    ]

    failed = False
    print("[single_device_render_semantic_equivalence_check]")
    print(f"single_titles={single_titles}")
    print(f"compare_titles={compare_titles}")
    print(f"single_status={single_status_text}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0
def run_single_device_vs_compare_render_style_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-vs-compare-render-style-check needs --ygas and --dat.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("浣跨敤 txt+dat 鍏卞悓鏃堕棿鑼冨洿")
        app.current_data_source_kind = "file"
        app.current_data_source_label = f"当前文件：{ygas_path.name}"
        app.current_file = ygas_path
        app.current_file_parsed = parsed_a
        app.current_layout_label = parsed_a.profile_name
        app.raw_data = parsed_a.dataframe.copy()
        app.current_data_source_kind = "file"
        app.current_data_source_label = f"当前文件：{ygas_path.name}"
        app.current_file = ygas_path
        app.current_file_parsed = parsed_a
        app.current_layout_label = parsed_a.profile_name
        app.raw_data = parsed_a.dataframe.copy()
        configure_auto_compare_context(app, ygas_path, dat_path)

        single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            str(col_a),
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        render_prepared_multi_spectral_payload(app, single_payload)
        single_style = app.current_plot_style_label
        single_layout = app.current_plot_layout_label
        single_titles = [axis.get_title() for axis in app.figure.axes]

        compare_reference = prepare_compare_txt_side_reference(
            app,
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
        render_prepared_dual_plot_payload(app, compare_reference["dual_plot_payload"])
        compare_style = app.current_plot_style_label
        compare_layout = app.current_plot_layout_label
        compare_titles = [axis.get_title() for axis in app.figure.axes]
        expected_short_label = str(compare_reference["dual_ygas"]["details"].get("time_range_short_label") or "")
    finally:
        root.destroy()

    checks = [
        (
            "single_device_compare_style_uses_same_plot_style",
            single_style == compare_style,
            {"single_style": single_style, "compare_style": compare_style},
        ),
        (
            "single_device_compare_style_uses_same_plot_layout_semantics",
            single_layout == compare_layout,
            {"single_layout": single_layout, "compare_layout": compare_layout},
        ),
        (
            "single_device_compare_style_uses_same_time_range_title_semantics",
            (
                not expected_short_label
                or (
                    expected_short_label in " ".join(single_titles)
                    and expected_short_label in " ".join(compare_titles)
                )
            ),
            {
                "expected_short_label": expected_short_label,
                "single_titles": single_titles,
                "compare_titles": compare_titles,
            },
        ),
    ]

    failed = False
    print("[single_device_vs_compare_render_style_check]")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_default_reuse_enabled_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_default_single_compare_style_check_mode(args)


def run_device_count_dispatch_single_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas:
        raise ValueError("device-count-dispatch-single-check needs --ygas.")

    ygas_path = Path(args.ygas[0])
    parsed = parse_supported_file(ygas_path)
    target_column = choose_single_column(parsed, args.element, "A")
    if not target_column:
        raise ValueError("Unable to resolve single-device target column.")

    root, app = build_headless_app()
    try:
        with tempfile.TemporaryDirectory(prefix="device_count_single_") as temp_dir:
            isolated_path = Path(temp_dir) / ygas_path.name
            shutil.copy2(ygas_path, isolated_path)
            payload = app.prepare_multi_spectral_compare_payload(
                [isolated_path],
                target_column,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            render_prepared_multi_spectral_payload(app, payload)
            details = dict(payload["series_results"][0]["details"])
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
            export_plot_execution_path = get_export_plot_execution_path(app)
    finally:
        root.destroy()

    checks = [
        (
            "payload_dispatches_to_single_device",
            int(payload.get("effective_device_count", 0)) == 1
            and str(payload.get("plot_execution_path")) == "single_device_spectrum",
            payload,
        ),
        (
            "single_device_details_tagged",
            int(details.get("effective_device_count", 0)) == 1
            and str(details.get("render_semantics")) == "single_device"
            and str(details.get("plot_execution_path")) == "single_device_spectrum",
            details,
        ),
        (
            "single_device_render_path_visible",
            app.current_plot_kind == "single_device_spectrum"
            and export_plot_execution_path == "single_device_spectrum"
            and "effective_device_count=1" in status_text
            and "render_semantics=single_device" in status_text
            and "plot_execution_path=single_device_spectrum" in diag_text,
            {
                "current_plot_kind": app.current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_dispatch_single_check]")
    print(f"ygas_path={ygas_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_dispatch_dual_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("device-count-dispatch-dual-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed = parse_supported_file(ygas_path)
    target_column = choose_single_column(parsed, args.element, "A")
    if not target_column:
        raise ValueError("Unable to resolve dual-device target column.")

    root, app = build_headless_app()
    try:
        payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path, dat_path],
            target_column,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
        )
        render_prepared_multi_spectral_payload(app, payload)
        details = dict(payload["series_results"][0]["details"])
        status_text = app.status_var.get()
        diag_text = app.diagnostic_var.get()
        export_plot_execution_path = get_export_plot_execution_path(app)
    finally:
        root.destroy()

    checks = [
        (
            "payload_dispatches_to_dual_compare_psd",
            int(payload.get("effective_device_count", 0)) == 2
            and str(payload.get("dispatch_render_semantics")) == "dual_psd_compare"
            and str(payload.get("plot_execution_path")) == "dual_psd_compare",
            payload,
        ),
        (
            "dual_device_render_path_visible",
            app.current_plot_kind == "dual_psd_compare"
            and export_plot_execution_path == "dual_psd_compare"
            and "effective_device_count=2" in status_text
            and "render_semantics=dual_psd_compare" in status_text
            and "plot_execution_path=dual_psd_compare" in diag_text,
            {
                "current_plot_kind": app.current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "status_text": status_text,
                "diag_text": diag_text,
                "details": details,
            },
        ),
    ]

    failed = False
    print("[device_count_dispatch_dual_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_dispatch_multi_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("device-count-dispatch-multi-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed = parse_supported_file(ygas_path)
    target_column = choose_single_column(parsed, args.element, "A")
    if not target_column:
        raise ValueError("Unable to resolve multi-device target column.")

    root, app = build_headless_app()
    try:
        with tempfile.TemporaryDirectory(prefix="device_count_multi_") as temp_dir:
            temp_root = Path(temp_dir)
            dat_copy_b = temp_root / "DEVB_toa5.dat"
            dat_copy_c = temp_root / "DEVC_toa5.dat"
            shutil.copy2(dat_path, dat_copy_b)
            shutil.copy2(dat_path, dat_copy_c)
            payload = app.prepare_multi_spectral_compare_payload(
                [ygas_path, dat_copy_b, dat_copy_c],
                target_column,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            render_prepared_multi_spectral_payload(app, payload)
            details = dict(payload["series_results"][0]["details"])
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
            export_plot_execution_path = get_export_plot_execution_path(app)
    finally:
        root.destroy()

    checks = [
        (
            "payload_dispatches_to_multi_device_overlay",
            int(payload.get("effective_device_count", 0)) >= 3
            and str(payload.get("dispatch_render_semantics")) == "multi_device_overlay"
            and str(payload.get("plot_execution_path")) == "multi_device_overlay",
            payload,
        ),
        (
            "multi_device_render_path_visible",
            app.current_plot_kind == "multi_device_overlay"
            and export_plot_execution_path == "multi_device_overlay"
            and f"effective_device_count={int(payload.get('effective_device_count', 0))}" in status_text
            and "render_semantics=multi_device_overlay" in status_text
            and "plot_execution_path=multi_device_overlay" in diag_text,
            {
                "current_plot_kind": app.current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "status_text": status_text,
                "diag_text": diag_text,
                "details": details,
            },
        ),
    ]

    failed = False
    print("[device_count_dispatch_multi_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_single_fallback_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_dispatch_single_check_mode(args)


def run_device_count_default_dual_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_dispatch_dual_check_mode(args)


def run_device_count_default_multi_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_dispatch_multi_check_mode(args)


def run_selected_files_direct_generate_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("selected-files-direct-generate-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed = parse_supported_file(ygas_path)
    target_column = choose_single_column(parsed, args.element, "A")
    if not target_column:
        raise ValueError("Unable to resolve direct-generate target column.")

    import tkinter as tk

    root, app = build_headless_app()
    try:
        app.element_preset_var.set(str(args.element or "H2O"))
        app.populate_file_list([ygas_path, dat_path])
        app.select_paths_in_list([ygas_path, dat_path], active_path=ygas_path)
        app.current_data_source_kind = "file"
        app.current_data_source_label = f"当前文件：{ygas_path.name}"
        app.current_file = ygas_path
        app.current_file_parsed = parsed
        app.current_layout_label = parsed.profile_name
        app.raw_data = parsed.dataframe.copy()
        app.column_vars = {str(target_column): tk.BooleanVar(master=root, value=True)}
        run_background_tasks_inline(app)
        app.generate_plot()
        status_text = app.status_var.get()
        diag_text = app.diagnostic_var.get()
        export_plot_execution_path = get_export_plot_execution_path(app)
    finally:
        root.destroy()

    checks = [
        (
            "direct_generate_uses_dual_compare_dispatch",
            app.current_plot_kind == "dual_psd_compare"
            and export_plot_execution_path == "dual_psd_compare",
            {
                "current_plot_kind": app.current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
            },
        ),
        (
            "direct_generate_exposes_unified_dispatch_metadata",
            "effective_device_count=2" in status_text
            and "render_semantics=dual_psd_compare" in status_text
            and "plot_execution_path=dual_psd_compare" in diag_text,
            {
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[selected_files_direct_generate_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_single_compare_style_check_mode(args: argparse.Namespace) -> int:
    root, app = build_headless_app()
    try:
        default_toggle_value = bool(app.single_compare_style_preview_var.get())
        preview_check_state = str(app.single_compare_style_preview_check.cget("state"))
        with tempfile.TemporaryDirectory(prefix="target_default_single_") as temp_dir:
            bundle = prepare_target_spectrum_smoke_paths(Path(temp_dir), group_count=1)
            ygas_path = Path(bundle["ygas_paths"][0])
            dat_path = Path(bundle["dat_path"])
            parsed_a = parse_supported_file(ygas_path)
            col_a = choose_single_column(parsed_a, args.element, "A")
            if not col_a:
                raise ValueError("Unable to resolve matching txt column for the requested element.")
            simulate_generate_plot_from_selected_files(
                app,
                root,
                selected_paths=[ygas_path, dat_path],
                active_path=ygas_path,
                parsed_active=parsed_a,
                target_column=str(col_a),
            )
            plot_kind = app.current_plot_kind
            export_plot_execution_path = get_export_plot_execution_path(app)
            export_render_semantics = get_export_metadata_value(app, "render_semantics")
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
    finally:
        root.destroy()

    checks = [
        (
            "default_indicator_is_not_manual_gate",
            default_toggle_value and preview_check_state == "disabled",
            {
                "single_compare_style_preview_default": default_toggle_value,
                "single_compare_style_preview_state": preview_check_state,
            },
        ),
        (
            "single_device_default_dispatch_path_active",
            plot_kind == "target_spectrum"
            and export_plot_execution_path == "target_spectrum_render"
            and export_render_semantics == "target_spectrum_single_group",
            {
                "plot_kind": plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "export_render_semantics": export_render_semantics,
            },
        ),
        (
            "single_device_default_reuse_uses_target_renderer",
            "effective_device_count=1" in status_text
            and "render_semantics=target_spectrum_single_group" in status_text
            and "current_plot_kind=target_spectrum" in status_text
            and "plot_execution_path=target_spectrum_render" in diag_text
            and "target_spectrum_group_count=1" in diag_text,
            {
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_default_single_compare_style_check]")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_single_fallback_check_mode(args: argparse.Namespace) -> int:
    fixtures = resolve_repo_fixture_paths()
    with tempfile.TemporaryDirectory(prefix="target_default_fallback_") as temp_dir:
        isolated_path = Path(temp_dir) / fixtures["ygas"].name
        shutil.copy2(fixtures["ygas"], isolated_path)
        parsed = parse_supported_file(isolated_path)
        target_column = choose_single_column(parsed, args.element, "A")
        if not target_column:
            raise ValueError("Unable to resolve single-device target column.")

        root, app = build_headless_app()
        try:
            simulate_generate_plot_from_selected_files(
                app,
                root,
                selected_paths=[isolated_path],
                active_path=isolated_path,
                parsed_active=parsed,
                target_column=target_column,
            )
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
            export_plot_execution_path = get_export_plot_execution_path(app)
            export_render_semantics = get_export_metadata_value(app, "render_semantics")
            current_plot_kind = app.current_plot_kind
            plot_title = str(app.figure.axes[0].get_title()) if app.figure.axes else ""
        finally:
            root.destroy()

    checks = [
        (
            "single_device_default_fallback_dispatch_path_active",
            current_plot_kind == "spectral"
            and export_plot_execution_path == "single_device_spectrum"
            and export_render_semantics == "single_device",
            {
                "current_plot_kind": current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "export_render_semantics": export_render_semantics,
            },
        ),
        (
            "single_device_default_fallback_visible",
            "effective_device_count=1" in status_text
            and "render_semantics=single_device" in status_text
            and "plot_execution_path=single_device_spectrum" in diag_text
            and "single_txt_execution_path=direct_generate_plain_spectral_fallback" in diag_text,
            {
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
        (
            "single_device_default_fallback_uses_plain_psd_title_semantics",
            plot_title.startswith("功率谱密度：")
            and "direct_generate_target_spectrum_fallback_reason=" in status_text
            and "single_txt_execution_path=direct_generate_plain_spectral_fallback" in diag_text,
            {
                "plot_title": plot_title,
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_default_single_fallback_check]")
    print(f"isolated_path={isolated_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_dual_check_mode(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="target_default_dual_") as temp_dir:
        bundle = prepare_target_spectrum_smoke_paths(Path(temp_dir), group_count=2)
        ygas_paths = [Path(path) for path in bundle["ygas_paths"]]
        dat_path = Path(bundle["dat_path"])
        parsed = parse_supported_file(ygas_paths[0])
        target_column = choose_single_column(parsed, args.element, "A")
        if not target_column:
            raise ValueError("Unable to resolve dual-device target column.")

        root, app = build_headless_app()
        try:
            simulate_generate_plot_from_selected_files(
                app,
                root,
                selected_paths=[*ygas_paths, dat_path],
                active_path=ygas_paths[0],
                parsed_active=parsed,
                target_column=target_column,
            )
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
            export_plot_execution_path = get_export_plot_execution_path(app)
            export_render_semantics = get_export_metadata_value(app, "render_semantics")
            current_plot_kind = app.current_plot_kind
        finally:
            root.destroy()

    checks = [
        (
            "dual_device_default_dispatch_path_active",
            current_plot_kind == "target_spectrum"
            and export_plot_execution_path == "target_spectrum_render"
            and export_render_semantics == "target_spectrum_dual_group",
            {
                "current_plot_kind": current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "export_render_semantics": export_render_semantics,
            },
        ),
        (
            "dual_device_default_status_visible",
            "effective_device_count=2" in status_text
            and "render_semantics=target_spectrum_dual_group" in status_text
            and "plot_execution_path=target_spectrum_render" in diag_text,
            {
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_default_dual_check]")
    print(f"ygas_paths={ygas_paths}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_device_count_default_multi_check_mode(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="target_default_multi_") as temp_dir:
        bundle = prepare_target_spectrum_smoke_paths(Path(temp_dir), group_count=3)
        ygas_paths = [Path(path) for path in bundle["ygas_paths"]]
        dat_path = Path(bundle["dat_path"])
        parsed = parse_supported_file(ygas_paths[0])
        target_column = choose_single_column(parsed, args.element, "A")
        if not target_column:
            raise ValueError("Unable to resolve multi-device target column.")

        root, app = build_headless_app()
        try:
            simulate_generate_plot_from_selected_files(
                app,
                root,
                selected_paths=[*ygas_paths, dat_path],
                active_path=ygas_paths[0],
                parsed_active=parsed,
                target_column=target_column,
            )
            status_text = app.status_var.get()
            diag_text = app.diagnostic_var.get()
            export_plot_execution_path = get_export_plot_execution_path(app)
            export_render_semantics = get_export_metadata_value(app, "render_semantics")
            current_plot_kind = app.current_plot_kind
        finally:
            root.destroy()

    checks = [
        (
            "multi_device_default_dispatch_path_active",
            current_plot_kind == "target_spectrum"
            and export_plot_execution_path == "target_spectrum_render"
            and export_render_semantics == "target_spectrum_multi_group",
            {
                "current_plot_kind": current_plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
                "export_render_semantics": export_render_semantics,
            },
        ),
        (
            "multi_device_default_status_visible",
            "effective_device_count=3" in status_text
            and "render_semantics=target_spectrum_multi_group" in status_text
            and "plot_execution_path=target_spectrum_render" in diag_text,
            {
                "status_text": status_text,
                "diag_text": diag_text,
            },
        ),
    ]

    failed = False
    print("[device_count_default_multi_check]")
    print(f"ygas_paths={ygas_paths}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_selected_files_direct_target_spectrum_single_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_default_single_compare_style_check_mode(args)


def run_selected_files_direct_target_spectrum_dual_check_mode(args: argparse.Namespace) -> int:
    return run_device_count_default_dual_check_mode(args)


def run_selected_files_direct_generate_check_mode(args: argparse.Namespace) -> int:
    checks = [
        (
            "single_group_direct_generate_prefers_target_spectrum",
            run_selected_files_direct_target_spectrum_single_check_mode,
        ),
        (
            "dual_group_direct_generate_prefers_target_spectrum",
            run_selected_files_direct_target_spectrum_dual_check_mode,
        ),
    ]
    failed = False
    print("[selected_files_direct_generate_check]")
    for name, func in checks:
        try:
            exit_code = int(func(args))
            ok = exit_code == 0
            detail = {"exit_code": exit_code}
        except Exception as exc:
            ok = False
            detail = {"error": str(exc)}
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_txt_dat_not_equals_device_count_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("txt-dat-not-equals-device-count-check needs --ygas and --dat.")

    ygas_path = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed = parse_supported_file(ygas_path)
    target_column = choose_single_column(parsed, args.element, "A")
    if not target_column:
        raise ValueError("Unable to resolve txt/dat device-count target column.")

    root, app = build_headless_app()
    try:
        with tempfile.TemporaryDirectory(prefix="txt_dat_device_count_") as temp_dir:
            temp_root = Path(temp_dir)
            dat_copy_b = temp_root / "DEVB_toa5.dat"
            dat_copy_c = temp_root / "DEVC_toa5.dat"
            shutil.copy2(dat_path, dat_copy_b)
            shutil.copy2(dat_path, dat_copy_c)
            payload = app.prepare_multi_spectral_compare_payload(
                [ygas_path, dat_copy_b, dat_copy_c],
                target_column,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
    finally:
        root.destroy()

    checks = [
        (
            "selection_type_counts_do_not_cap_device_count",
            int(payload.get("selected_txt_file_count", 0)) == 1
            and int(payload.get("selected_dat_file_count", 0)) == 2
            and int(payload.get("effective_device_count", 0)) == 3,
            payload,
        ),
        (
            "effective_device_ids_come_from_grouping_not_txt_dat_types",
            len(list(payload.get("effective_device_ids", []))) == 3
            and str(payload.get("dispatch_render_semantics")) == "multi_device_overlay",
            {
                "effective_device_ids": payload.get("effective_device_ids"),
                "dispatch_render_semantics": payload.get("dispatch_render_semantics"),
            },
        ),
    ]

    failed = False
    print("[txt_dat_not_equals_device_count_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"target_column={target_column}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_vs_dual_compare_axis_geometry_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-vs-dual-compare-axis-geometry-check needs --ygas and --dat.")

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
    preview_check_state = ""
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        configure_compare_psd_render_test_state(app)
        configure_auto_compare_context(app, ygas_path, dat_path)
        render_state = capture_single_device_and_dual_compare_psd_render_states(
            app,
            ygas_paths=[ygas_path],
            selected_files=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
    finally:
        root.destroy()

    single_axis_state = dict(render_state["single_axis_state"])
    dual_axis_state = dict(render_state["dual_axis_state"])
    single_details = dict(render_state["single_details"])
    checks = [
        (
            "single_device_compare_psd_axis_geometry_matches_dual_compare",
            np.allclose(np.asarray(single_axis_state["xlim"], dtype=float), np.asarray(dual_axis_state["xlim"], dtype=float), rtol=1e-12, atol=1e-12)
            and np.allclose(np.asarray(single_axis_state["ylim"], dtype=float), np.asarray(dual_axis_state["ylim"], dtype=float), rtol=1e-12, atol=1e-12),
            {
                "single_xlim": single_axis_state["xlim"],
                "dual_xlim": dual_axis_state["xlim"],
                "single_ylim": single_axis_state["ylim"],
                "dual_ylim": dual_axis_state["ylim"],
                "single_details": single_details,
            },
        ),
    ]

    failed = False
    print("[single_device_vs_dual_compare_axis_geometry_check]")
    print(f"single_axis_state={single_axis_state}")
    print(f"dual_axis_state={dual_axis_state}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_vs_dual_compare_reference_slope_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-vs-dual-compare-reference-slope-check needs --ygas and --dat.")

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
    preview_check_state = ""
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        configure_compare_psd_render_test_state(app)
        configure_auto_compare_context(app, ygas_path, dat_path)
        render_state = capture_single_device_and_dual_compare_psd_render_states(
            app,
            ygas_paths=[ygas_path],
            selected_files=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
    finally:
        root.destroy()

    single_lines = list(render_state["single_axis_state"]["reference_slope_lines"])
    dual_lines = list(render_state["dual_axis_state"]["reference_slope_lines"])
    same_line_count = len(single_lines) == len(dual_lines)
    same_line_data = same_line_count and all(
        str(single["label"]) == str(dual["label"])
        and np.array_equal(np.asarray(single["x"], dtype=float), np.asarray(dual["x"], dtype=float))
        and np.allclose(np.asarray(single["y"], dtype=float), np.asarray(dual["y"], dtype=float), rtol=1e-12, atol=1e-12)
        for single, dual in zip(single_lines, dual_lines)
    )
    checks = [
        (
            "single_device_compare_psd_reference_slope_matches_dual_compare",
            same_line_data,
            {
                "single_reference_slope_lines": single_lines,
                "dual_reference_slope_lines": dual_lines,
            },
        ),
    ]

    failed = False
    print("[single_device_vs_dual_compare_reference_slope_check]")
    print(f"single_reference_slope_lines={single_lines}")
    print(f"dual_reference_slope_lines={dual_lines}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_vs_dual_compare_visible_scatter_subset_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-vs-dual-compare-visible-scatter-subset-check needs --ygas and --dat.")

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
    preview_check_state = ""
    try:
        app.refresh_canvas = lambda *args, **kwargs: None
        app.element_preset_var.set(str(args.element or "H2O"))
        app.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        configure_compare_psd_render_test_state(app)
        configure_auto_compare_context(app, ygas_path, dat_path)
        render_state = capture_single_device_and_dual_compare_psd_render_states(
            app,
            ygas_paths=[ygas_path],
            selected_files=[ygas_path],
            dat_path=dat_path,
            col_a=str(col_a),
            col_b=str(col_b),
            args=args,
        )
    finally:
        root.destroy()

    single_offsets = list(render_state["single_axis_state"]["scatter_offsets"])
    dual_offsets = list(render_state["dual_axis_state"]["scatter_offsets"])
    txt_subset_match = (
        len(single_offsets) == 1
        and len(dual_offsets) == 2
        and np.array_equal(np.asarray(single_offsets[0], dtype=float), np.asarray(dual_offsets[0], dtype=float))
    )
    checks = [
        (
            "single_device_compare_psd_visible_scatter_is_dual_compare_minus_dat",
            len(single_offsets) == len(dual_offsets) - 1 and txt_subset_match,
            {
                "single_collection_count": len(single_offsets),
                "dual_collection_count": len(dual_offsets),
                "single_offsets": single_offsets,
                "dual_offsets": dual_offsets,
            },
        ),
    ]

    failed = False
    print("[single_device_vs_dual_compare_visible_scatter_subset_check]")
    print(f"single_offsets={single_offsets}")
    print(f"dual_offsets={dual_offsets}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_vs_dual_target_spectrum_visible_subset_check_mode(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="target_subset_check_") as temp_dir:
        bundle = prepare_target_spectrum_smoke_paths(Path(temp_dir), group_count=2)
        ygas_paths = [Path(path) for path in bundle["ygas_paths"]]
        dat_path = Path(bundle["dat_path"])

        root, app = build_headless_app()
        try:
            render_state = capture_single_and_dual_target_spectrum_render_states(
                app,
                single_ygas_path=ygas_paths[0],
                dual_ygas_paths=ygas_paths,
                dat_path=dat_path,
                args=args,
            )
        finally:
            root.destroy()

    single_offsets = list(render_state["single_axis_state"]["scatter_offsets"])
    dual_offsets = list(render_state["dual_axis_state"]["scatter_offsets"])
    subset_match = (
        len(single_offsets) == 2
        and len(dual_offsets) == 4
        and np.array_equal(np.asarray(single_offsets[0], dtype=float), np.asarray(dual_offsets[0], dtype=float))
        and np.array_equal(np.asarray(single_offsets[1], dtype=float), np.asarray(dual_offsets[2], dtype=float))
    )
    checks = [
        (
            "single_target_spectrum_visible_scatter_is_dual_subset",
            subset_match,
            {
                "single_collection_count": len(single_offsets),
                "dual_collection_count": len(dual_offsets),
                "single_offsets": single_offsets,
                "dual_offsets": dual_offsets,
            },
        ),
    ]

    failed = False
    print("[single_vs_dual_target_spectrum_visible_subset_check]")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_vs_dual_target_spectrum_style_check_mode(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="target_style_check_") as temp_dir:
        bundle = prepare_target_spectrum_smoke_paths(Path(temp_dir), group_count=2)
        ygas_paths = [Path(path) for path in bundle["ygas_paths"]]
        dat_path = Path(bundle["dat_path"])

        root, app = build_headless_app()
        try:
            render_state = capture_single_and_dual_target_spectrum_render_states(
                app,
                single_ygas_path=ygas_paths[0],
                dual_ygas_paths=ygas_paths,
                dat_path=dat_path,
                args=args,
            )
        finally:
            root.destroy()

    single_lines = list(render_state["single_axis_state"]["reference_slope_lines"])
    dual_lines = list(render_state["dual_axis_state"]["reference_slope_lines"])
    same_reference_lines = len(single_lines) == len(dual_lines) and all(
        str(single["label"]) == str(dual["label"])
        and np.array_equal(np.asarray(single["x"], dtype=float), np.asarray(dual["x"], dtype=float))
        and np.allclose(
            np.asarray(single["y"], dtype=float),
            np.asarray(dual["y"], dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        for single, dual in zip(single_lines, dual_lines)
    )
    checks = [
        (
            "single_and_dual_target_spectrum_title_semantics_match",
            str(render_state["single_axis_state"]["title"]) == str(render_state["dual_axis_state"]["title"]),
            {
                "single_title": render_state["single_axis_state"]["title"],
                "dual_title": render_state["dual_axis_state"]["title"],
            },
        ),
        (
            "single_and_dual_target_spectrum_style_layout_match",
            render_state["single_plot_kind"] == "target_spectrum"
            and render_state["dual_plot_kind"] == "target_spectrum"
            and str(render_state["single_plot_style"]) == str(render_state["dual_plot_style"])
            and str(render_state["single_plot_layout"]) == str(render_state["dual_plot_layout"]),
            {
                "single_plot_kind": render_state["single_plot_kind"],
                "dual_plot_kind": render_state["dual_plot_kind"],
                "single_plot_style": render_state["single_plot_style"],
                "dual_plot_style": render_state["dual_plot_style"],
                "single_plot_layout": render_state["single_plot_layout"],
                "dual_plot_layout": render_state["dual_plot_layout"],
                "single_export_plot_execution_path": render_state["single_export_plot_execution_path"],
                "dual_export_plot_execution_path": render_state["dual_export_plot_execution_path"],
                "single_export_render_semantics": render_state["single_export_render_semantics"],
                "dual_export_render_semantics": render_state["dual_export_render_semantics"],
            },
        ),
        (
            "single_and_dual_target_spectrum_reference_slope_semantics_match",
            same_reference_lines,
            {
                "single_reference_slope_lines": single_lines,
                "dual_reference_slope_lines": dual_lines,
            },
        ),
    ]

    failed = False
    print("[single_vs_dual_target_spectrum_style_check]")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_device_fallback_still_legacy_render_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas:
        raise ValueError("single-device-fallback-still-legacy-render-check needs --ygas.")

    ygas_path = Path(args.ygas[0])
    parsed_a = parse_supported_file(ygas_path)
    col_a = choose_single_column(parsed_a, args.element, "A")
    if not col_a:
        raise ValueError("Unable to resolve matching txt column for the requested element.")

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
        app.refresh_canvas = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory(prefix="single_device_legacy_render_") as temp_dir:
            isolated_ygas_path = Path(temp_dir) / ygas_path.name
            shutil.copy2(ygas_path, isolated_ygas_path)
            simulate_single_file_open(app, isolated_ygas_path)
            payload = app.prepare_multi_spectral_compare_payload(
                [isolated_ygas_path],
                str(col_a),
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            if len(payload["series_results"]) != 1:
                raise ValueError("single-device fallback legacy render check expects exactly one series.")
            details = dict(payload["series_results"][0]["details"])
            render_prepared_multi_spectral_payload(app, payload)
            plot_kind = app.current_plot_kind
            export_plot_execution_path = get_export_plot_execution_path(app)
            title = app.figure.axes[0].get_title() if app.figure.axes else ""
            status_text = app.status_var.get()
    finally:
        root.destroy()

    checks = [
        (
            "single_device_fallback_keeps_legacy_execution_path",
            str(details.get("single_device_execution_path")) == "legacy_device_group_scope"
            and str(details.get("single_device_compare_side_fallback_reason")) == "no_compare_dat_context",
            {"details": details},
        ),
        (
            "single_device_fallback_keeps_legacy_render_kind_and_export_path",
            plot_kind == "multi_spectral" and export_plot_execution_path == "single_device_grouped_spectrum",
            {
                "plot_kind": plot_kind,
                "export_plot_execution_path": export_plot_execution_path,
            },
        ),
        (
            "single_device_fallback_keeps_legacy_title_semantics",
            title.startswith("单台设备图谱 -") and "图已生成：单台设备图谱 /" in status_text,
            {"title": title, "status_text": status_text},
        ),
    ]

    failed = False
    print("[single_device_fallback_still_legacy_render_check]")
    print(f"details={details}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0
def run_single_device_selection_scope_check_mode(args: argparse.Namespace) -> int:
    if not args.ygas or not args.dat:
        raise ValueError("single-device-selection-scope-check needs --ygas and --dat.")

    ygas_source = Path(args.ygas[0])
    dat_path = Path(args.dat)
    parsed_a = parse_supported_file(ygas_source)
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

    compare_mode = "时间段内 PSD 对比"

    with tempfile.TemporaryDirectory(prefix="selection_scope_smoke_") as temp_dir:
        temp_root = Path(temp_dir)
        ygas_path_a = temp_root / "partA_sensorx.log"
        ygas_path_b = temp_root / "partB_sensorx.log"
        shutil.copy2(ygas_source, ygas_path_a)
        shutil.copy2(ygas_source, ygas_path_b)

        root = tk.Tk()
        root.withdraw()
        try:
            app = app_mod.FileViewerApp(root)
            single_full_payload = app.prepare_multi_spectral_compare_payload(
                [ygas_path_a, ygas_path_b],
                col_a,
                args.fs,
                args.nsegment,
                args.overlap_ratio,
            )
            dual_base = app.prepare_dual_compare_payload(
                ygas_paths=[ygas_path_a, ygas_path_b],
                dat_path=dat_path,
                selected_paths=[ygas_path_a, ygas_path_b, dat_path],
                compare_mode=compare_mode,
                start_dt=None,
                end_dt=None,
            )
            selection_meta = dict(dual_base["selection_meta"])
            if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
                start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
            else:
                start_dt, end_dt = None, None

            single_payload = app.prepare_multi_spectral_compare_payload(
                [ygas_path_a, ygas_path_b],
                col_a,
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
                compare_mode=compare_mode,
                compare_scope="单对单",
                start_dt=start_dt,
                end_dt=end_dt,
                mapping_name=str(args.element),
                scheme_name="smoke",
                alignment_strategy=app.get_alignment_strategy(compare_mode),
                plot_style=app.resolve_plot_style(compare_mode),
                plot_layout=app.resolve_plot_layout(compare_mode),
                fs_ui=args.fs,
                requested_nsegment=args.nsegment,
                overlap_ratio=args.overlap_ratio,
                match_tolerance=0.2,
                spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
                time_range_context={
                    "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                    "has_txt_dat_context": True,
                },
            )
        finally:
            root.destroy()

    if len(single_payload["series_results"]) != 1:
        raise ValueError("single-device selection scope payload did not collapse to exactly one series.")

    single_group = single_full_payload["device_groups"][0]
    single_series = single_payload["series_results"][0]
    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
    )

    checks = [
        (
            "filename_fallback_fragmentation_collapsed",
            single_full_payload["device_count"] == 1
            and len(single_full_payload["device_groups"]) == 1
            and int(single_group.get("file_count", 0)) == 2,
            {"device_groups": single_full_payload["device_groups"]},
        ),
        (
            "selection_merge_scope_flagged",
            str(single_group.get("selection_merge_scope")) == "selection_level"
            and "selection_scope" in str(single_group.get("group_device_source", "")),
            {"selection_group": single_group},
        ),
        (
            "single_vs_compare_valid_points_equal",
            int(single_series["details"].get("valid_points", 0)) == int(dual_ygas["details"].get("valid_points", 0)),
            {
                "single_valid_points": single_series["details"].get("valid_points"),
                "compare_valid_points": dual_ygas["details"].get("valid_points"),
            },
        ),
        (
            "single_vs_compare_nperseg_equal",
            int(single_series["details"].get("nperseg", 0)) == int(dual_ygas["details"].get("nperseg", 0)),
            {
                "single_nperseg": single_series["details"].get("nperseg"),
                "compare_nperseg": dual_ygas["details"].get("nperseg"),
            },
        ),
        (
            "single_vs_compare_valid_freq_points_equal",
            int(single_series["details"].get("valid_freq_points", 0))
            == int(dual_ygas["details"].get("valid_freq_points", 0)),
            {
                "single_valid_freq_points": single_series["details"].get("valid_freq_points"),
                "compare_valid_freq_points": dual_ygas["details"].get("valid_freq_points"),
            },
        ),
        (
            "single_vs_compare_frequency_point_count_equal",
            int(single_series["details"].get("frequency_point_count", 0))
            == int(dual_ygas["details"].get("frequency_point_count", 0)),
            {
                "single_frequency_point_count": single_series["details"].get("frequency_point_count"),
                "compare_frequency_point_count": dual_ygas["details"].get("frequency_point_count"),
            },
        ),
        (
            "single_vs_compare_freq_equal",
            np.array_equal(single_series["freq"], dual_ygas["freq"]),
            {"single_points": len(single_series["freq"]), "compare_points": len(dual_ygas["freq"])},
        ),
        (
            "single_vs_compare_density_allclose",
            np.allclose(single_series["density"], dual_ygas["density"], rtol=1e-12, atol=1e-12),
            {
                "single_first5": single_series["density"][:5].tolist(),
                "compare_first5": dual_ygas["density"][:5].tolist(),
            },
        ),
    ]

    failed = False
    print("[single_device_selection_scope_check]")
    print(f"ygas_source={ygas_source}")
    print(f"dat_path={dat_path}")
    print(f"selection_paths={[str(ygas_path_a), str(ygas_path_b)]}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"time_range={start_dt} ~ {end_dt}")
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
        ygas_loader_result = app.load_file_with_file_settings(ygas_path)
        app.on_file_loaded(ygas_path, ygas_loader_result)
        single_plot_column = col_a
        if single_plot_column not in app.column_vars:
            loaded_parsed = app.current_file_parsed
            if loaded_parsed is not None:
                resolved_loaded_column = choose_single_column(loaded_parsed, args.element, "single")
                if resolved_loaded_column:
                    single_plot_column = resolved_loaded_column
            if single_plot_column not in app.column_vars:
                element_token = str(args.element or "").strip().lower()
                fallback_column = next(
                    (
                        name
                        for name in app.column_vars
                        if element_token and element_token in str(name).strip().lower()
                    ),
                    None,
                )
                if fallback_column is not None:
                    single_plot_column = str(fallback_column)
        if single_plot_column not in app.column_vars:
            raise ValueError(
                f"Loaded single-file UI does not expose target column: {col_a}; available={list(app.column_vars.keys())}"
            )
        for column_name, column_var in app.column_vars.items():
            column_var.set(column_name == single_plot_column)
        single_plot_freq, single_plot_density, single_plot_details = app.spectral_analysis()
        app.plot_results("spectral", single_plot_freq, single_plot_density, [single_plot_column], single_plot_details)
        single_plot_title = app.figure.axes[0].get_title() if app.figure.axes else ""
        single_plot_diag_text = app.diagnostic_var.get()
        single_plot_status_text = app.status_var.get()
        single_plot_export_frame = app.current_result_frame.copy() if app.current_result_frame is not None else pd.DataFrame()
        single_plot_export_columns = list(single_plot_export_frame.columns)
        single_plot_default_filename = app.build_default_filename()
        single_plot_default_data_filename = app.build_default_data_filename()

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


def run_single_compare_base_spectrum_hardened_check_mode(args: argparse.Namespace) -> int:
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

    compare_mode = "时间段内 PSD 对比"
    required_metadata_keys = list(app_mod.get_required_export_metadata_fields())
    required_detail_metadata_keys = [key for key in required_metadata_keys if key != "time_range_policy_note"]
    required_detail_metadata_keys = [key for key in required_metadata_keys if key != "time_range_policy_note"]
    export_metadata_columns = required_metadata_keys + ["time_range_policy_note"]

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None
        compare_time_range_meta = app.build_compare_time_range_meta(
            strategy_label=app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
            start_dt=start_dt,
            end_dt=end_dt,
            has_txt_dat_context=bool(
                selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
            ),
        )

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
        single_full_ygas_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=None,
            end_dt=None,
        )
        single_plot_series = single_full_ygas_payload["series_results"][0]
        single_plot_details = dict(single_plot_series["details"])
        app.current_file = ygas_path
        app.refresh_canvas = lambda *args, **kwargs: None
        app.plot_results(
            "spectral",
            np.asarray(single_plot_series["freq"], dtype=float),
            np.asarray(single_plot_series["density"], dtype=float),
            [col_a],
            single_plot_details,
        )
        single_plot_title = app.figure.axes[0].get_title() if app.figure.axes else ""
        single_plot_diag_text = app.diagnostic_var.get()
        single_plot_status_text = app.status_var.get()
        single_plot_export_frame = app.current_result_frame.copy() if app.current_result_frame is not None else pd.DataFrame()
        single_plot_export_columns = list(single_plot_export_frame.columns)
        single_plot_default_filename = app.build_default_filename()
        single_plot_default_data_filename = app.build_default_data_filename()
        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_meta=compare_time_range_meta,
        )

        render_prepared_multi_spectral_payload(app, single_full_ygas_payload)
        single_title = app.figure.axes[0].get_title() if app.figure.axes else ""
        single_diag_text = app.diagnostic_var.get()
        single_status_text = app.status_var.get()
        single_export_columns = list(app.current_result_frame.columns)
        single_default_filename = app.build_default_filename()
        single_default_data_filename = app.build_default_data_filename()

        app.on_prepared_dual_plot_ready(dual_plot_payload)
        compare_titles = [axis.get_title() for axis in app.figure.axes]
        compare_diag_text = app.diagnostic_var.get()
        compare_status_text = app.status_var.get()
        compare_export_columns = list(app.current_result_frame.columns)
        compare_default_filename = app.build_default_filename()
        compare_default_data_filename = app.build_default_data_filename()

        synced_range = app.sync_compare_common_time_range_to_single_analysis(selection_meta)
        synced_start_dt, synced_end_dt = app.resolve_optional_time_range_inputs()
        synced_single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=synced_start_dt,
            end_dt=synced_end_dt,
        )
    finally:
        root.destroy()

    if (
        len(single_ygas_payload["series_results"]) != 1
        or len(single_dat_payload["series_results"]) != 1
        or len(single_full_ygas_payload["series_results"]) != 1
    ):
        raise ValueError("Single-type payload did not produce exactly one spectrum series per side.")

    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
    )
    dual_dat = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "dat" and str(item["column"]) == col_b
    )
    single_ygas = single_ygas_payload["series_results"][0]
    single_dat = single_dat_payload["series_results"][0]
    single_full_ygas = single_full_ygas_payload["series_results"][0]
    synced_single_ygas = synced_single_payload["series_results"][0]
    single_short_label = app.resolve_time_range_policy_short_label(
        single_full_ygas["details"].get("time_range_policy"),
        single_full_ygas["details"].get("time_range_policy_label"),
    )
    compare_short_label = app.resolve_time_range_policy_short_label(
        dual_ygas["details"].get("time_range_policy"),
        dual_ygas["details"].get("time_range_policy_label"),
    )
    single_file_tag = app.resolve_time_range_policy_file_tag(
        single_full_ygas["details"].get("time_range_policy"),
        single_full_ygas["details"].get("time_range_policy_label"),
    )
    compare_file_tag = app.resolve_time_range_policy_file_tag(
        dual_ygas["details"].get("time_range_policy"),
        dual_ygas["details"].get("time_range_policy_label"),
    )

    checks = [
        (
            "single_same_window_payload_respects_requested_range",
            single_ygas["details"].get("base_requested_start") == start_dt
            and single_ygas["details"].get("base_requested_end") == end_dt
            and single_ygas["details"].get("render_semantics") == "single_device_compare_psd"
            and single_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render",
            {
                "requested_range": (start_dt, end_dt),
                "single_requested": (
                    single_ygas["details"].get("base_requested_start"),
                    single_ygas["details"].get("base_requested_end"),
                ),
                "render_semantics": single_ygas["details"].get("render_semantics"),
                "plot_execution_path": single_ygas["details"].get("plot_execution_path"),
            },
        ),
        (
            "single_default_policy_reuses_compare_window_when_context_available",
            single_full_ygas["details"].get("time_range_policy") == "txt_dat_common_window"
            and single_full_ygas["details"].get("render_semantics") == "single_device_compare_psd"
            and single_full_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render",
            {
                "single_policy": single_full_ygas["details"].get("time_range_policy"),
                "single_label": single_full_ygas["details"].get("time_range_policy_label"),
                "render_semantics": single_full_ygas["details"].get("render_semantics"),
                "plot_execution_path": single_full_ygas["details"].get("plot_execution_path"),
            },
        ),
        (
            "single_default_matches_compare_window_when_context_available",
            single_full_ygas["details"].get("time_range_policy") == dual_ygas["details"].get("time_range_policy")
            and single_full_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render"
            and dual_ygas["details"].get("time_range_policy") == "txt_dat_common_window",
            {
                "single_policy": single_full_ygas["details"].get("time_range_policy"),
                "compare_policy": dual_ygas["details"].get("time_range_policy"),
                "single_actual": (
                    single_full_ygas["details"].get("base_actual_start"),
                    single_full_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
        (
            "single_dat_same_window_payload_respects_requested_range",
            single_dat["details"].get("base_requested_start") == start_dt
            and single_dat["details"].get("base_requested_end") == end_dt
            and single_dat["details"].get("render_semantics") == "single_device"
            and single_dat["details"].get("plot_execution_path") == "single_device_spectrum",
            {
                "requested_range": (start_dt, end_dt),
                "single_requested": (
                    single_dat["details"].get("base_requested_start"),
                    single_dat["details"].get("base_requested_end"),
                ),
                "render_semantics": single_dat["details"].get("render_semantics"),
                "plot_execution_path": single_dat["details"].get("plot_execution_path"),
            },
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
        (
            "same_window_metadata_keys_present",
            all(single_ygas["details"].get(key) is not None for key in required_metadata_keys)
            and all(single_dat["details"].get(key) is not None for key in required_metadata_keys)
            and all(dual_ygas["details"].get(key) is not None for key in required_metadata_keys)
            and all(dual_dat["details"].get(key) is not None for key in required_metadata_keys),
            {
                "required_keys": required_metadata_keys,
                "single_ygas_keys": sorted(key for key in required_metadata_keys if single_ygas["details"].get(key) is not None),
                "single_dat_keys": sorted(key for key in required_metadata_keys if single_dat["details"].get(key) is not None),
                "dual_ygas_keys": sorted(key for key in required_metadata_keys if dual_ygas["details"].get(key) is not None),
                "dual_dat_keys": sorted(key for key in required_metadata_keys if dual_dat["details"].get(key) is not None),
            },
        ),
        (
            "compare_common_window_policy",
            dual_ygas["details"].get("time_range_policy") == "txt_dat_common_window"
            and dual_dat["details"].get("time_range_policy") == "txt_dat_common_window",
            {
                "dual_ygas_policy": dual_ygas["details"].get("time_range_policy"),
                "dual_dat_policy": dual_dat["details"].get("time_range_policy"),
                "compare_policy_label": compare_time_range_meta.get("time_range_policy_label"),
            },
        ),
        (
            "compare_manual_sync_matches_compare_window",
            synced_single_ygas["details"].get("base_requested_start") == dual_ygas["details"].get("base_requested_start")
            and synced_single_ygas["details"].get("base_requested_end") == dual_ygas["details"].get("base_requested_end")
            and synced_single_ygas["details"].get("base_actual_start") == dual_ygas["details"].get("base_actual_start")
            and synced_single_ygas["details"].get("base_actual_end") == dual_ygas["details"].get("base_actual_end"),
            {
                "synced_single_policy": synced_single_ygas["details"].get("time_range_policy"),
                "compare_policy": dual_ygas["details"].get("time_range_policy"),
                "synced_single_actual": (
                    synced_single_ygas["details"].get("base_actual_start"),
                    synced_single_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
        (
            "single_auto_reuse_note_visible",
            ("双设备 PSD 对比使用txt+dat 共同时间范围" in single_diag_text)
            or ("时间窗=txt+dat 共同时间范围" in single_status_text),
            {
                "diagnostic_contains_note": "双设备 PSD 对比使用txt+dat 共同时间范围" in single_diag_text,
                "status_contains_label": "时间窗=txt+dat 共同时间范围" in single_status_text,
                "single_status": single_status_text,
            },
        ),
        (
            "compare_window_note_visible",
            ("双设备 PSD 对比使用txt+dat 共同时间范围" in compare_diag_text)
            or ("时间窗=txt+dat 共同时间范围" in compare_status_text),
            {
                "diagnostic_contains_note": "双设备 PSD 对比使用txt+dat 共同时间范围" in compare_diag_text,
                "status_contains_label": "时间窗=txt+dat 共同时间范围" in compare_status_text,
                "compare_status": compare_status_text,
            },
        ),
        (
            "single_export_metadata_columns",
            all(column in single_export_columns for column in export_metadata_columns),
            {
                "required_columns": export_metadata_columns,
                "single_export_columns": single_export_columns,
            },
        ),
        (
            "compare_export_metadata_columns",
            all(column in compare_export_columns for column in export_metadata_columns),
            {
                "required_columns": export_metadata_columns,
                "compare_export_columns": compare_export_columns,
            },
        ),
        (
            "single_plot_title_contains_policy_short_label",
            bool(single_plot_short_label) and single_plot_short_label in single_plot_title,
            {"title": single_plot_title, "expected_label": single_plot_short_label},
        ),
        (
            "single_overlay_title_contains_policy_short_label",
            bool(single_overlay_short_label) and single_overlay_short_label in single_overlay_title,
            {"title": single_overlay_title, "expected_label": single_overlay_short_label},
        ),
        (
            "compare_title_contains_policy_short_label",
            bool(compare_short_label) and any(compare_short_label in title for title in compare_titles),
            {"titles": compare_titles, "expected_label": compare_short_label},
        ),
        (
            "single_plot_default_filename_contains_policy_tag",
            bool(single_plot_file_tag)
            and f"_{single_plot_file_tag}_" in single_plot_default_filename
            and f"_{single_plot_file_tag}_" in single_plot_default_data_filename,
            {
                "plot_filename": single_plot_default_filename,
                "data_filename": single_plot_default_data_filename,
                "expected_tag": single_plot_file_tag,
            },
        ),
        (
            "single_overlay_default_filename_contains_policy_tag",
            bool(single_overlay_file_tag)
            and f"_{single_overlay_file_tag}_" in single_overlay_default_filename
            and f"_{single_overlay_file_tag}_" in single_overlay_default_data_filename,
            {
                "plot_filename": single_overlay_default_filename,
                "data_filename": single_overlay_default_data_filename,
                "expected_tag": single_overlay_file_tag,
            },
        ),
        (
            "compare_default_filename_contains_policy_tag",
            bool(compare_file_tag)
            and f"_{compare_file_tag}_" in compare_default_filename
            and f"_{compare_file_tag}_" in compare_default_data_filename,
            {
                "plot_filename": compare_default_filename,
                "data_filename": compare_default_data_filename,
                "expected_tag": compare_file_tag,
            },
        ),
        (
            "single_and_compare_status_show_unified_hint",
            "time_range_policy 与 base_actual_time_range" in single_status_text
            and "time_range_policy 与 base_actual_time_range" in compare_status_text,
            {
                "single_status": single_status_text,
                "compare_status": compare_status_text,
            },
        ),
        (
            "single_and_compare_diag_show_unified_hint",
            "time_range_policy 与 base_actual_time_range" in single_diag_text
            and "time_range_policy 与 base_actual_time_range" in compare_diag_text,
            {
                "single_diag": single_diag_text,
                "compare_diag": compare_diag_text,
            },
        ),
        (
            "compare_to_single_sync_returns_common_window",
            synced_range == (start_dt, end_dt),
            {"synced_range": synced_range, "compare_range": (start_dt, end_dt)},
        ),
        (
            "compare_to_single_sync_requested_range_equal",
            synced_single_ygas["details"].get("base_requested_start") == dual_ygas["details"].get("base_requested_start")
            and synced_single_ygas["details"].get("base_requested_end") == dual_ygas["details"].get("base_requested_end"),
            {
                "synced_single_requested": (
                    synced_single_ygas["details"].get("base_requested_start"),
                    synced_single_ygas["details"].get("base_requested_end"),
                ),
                "compare_requested": (
                    dual_ygas["details"].get("base_requested_start"),
                    dual_ygas["details"].get("base_requested_end"),
                ),
            },
        ),
        (
            "compare_to_single_sync_actual_range_equal",
            synced_single_ygas["details"].get("base_actual_start") == dual_ygas["details"].get("base_actual_start")
            and synced_single_ygas["details"].get("base_actual_end") == dual_ygas["details"].get("base_actual_end"),
            {
                "synced_single_actual": (
                    synced_single_ygas["details"].get("base_actual_start"),
                    synced_single_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
    ]

    failed = False
    print("[single_compare_base_spectrum_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"same_window_time_range={start_dt} ~ {end_dt}")
    print(
        "single_full_time_range="
        f"{single_full_ygas['details'].get('base_actual_start')} ~ {single_full_ygas['details'].get('base_actual_end')}"
    )
    print(
        "compare_time_range="
        f"{dual_ygas['details'].get('base_actual_start')} ~ {dual_ygas['details'].get('base_actual_end')}"
    )
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_time_range_metadata_check_mode(args: argparse.Namespace) -> int:
    requested_start = pd.Timestamp("2026-03-27 14:30:00")
    requested_end = pd.Timestamp("2026-03-27 14:59:59.900000")
    actual_start = pd.Timestamp("2026-03-27 14:30:47.600000")
    actual_end = pd.Timestamp("2026-03-27 14:59:59.900000")

    single_meta = core.build_single_time_range_metadata(
        start_dt=requested_start,
        end_dt=requested_end,
        has_timestamp=True,
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    compare_meta = core.build_compare_time_range_metadata(
        strategy_label="手动输入时间范围",
        start_dt=requested_start,
        end_dt=requested_end,
        has_txt_dat_context=True,
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
    )

    checks = [
        (
            "time_range_metadata_keys_equal",
            sorted(single_meta.keys()) == sorted(compare_meta.keys()),
            {"single_keys": sorted(single_meta.keys()), "compare_keys": sorted(compare_meta.keys())},
        ),
        (
            "time_range_export_keys_present",
            all(key in single_meta for key in core.TIME_RANGE_METADATA_EXPORT_KEYS)
            and all(key in compare_meta for key in core.TIME_RANGE_METADATA_EXPORT_KEYS),
            {"export_keys": list(core.TIME_RANGE_METADATA_EXPORT_KEYS)},
        ),
        (
            "requested_time_range_text_equal",
            single_meta.get("base_requested_time_range") == compare_meta.get("base_requested_time_range"),
            {
                "single_requested": single_meta.get("base_requested_time_range"),
                "compare_requested": compare_meta.get("base_requested_time_range"),
            },
        ),
        (
            "actual_time_range_text_equal",
            single_meta.get("base_actual_time_range") == compare_meta.get("base_actual_time_range"),
            {
                "single_actual": single_meta.get("base_actual_time_range"),
                "compare_actual": compare_meta.get("base_actual_time_range"),
            },
        ),
        (
            "display_suffix_equal_for_same_window",
            core.build_time_range_display_suffix(single_meta) == core.build_time_range_display_suffix(compare_meta),
            {
                "single_suffix": core.build_time_range_display_suffix(single_meta),
                "compare_suffix": core.build_time_range_display_suffix(compare_meta),
            },
        ),
        (
            "filename_suffix_equal_for_same_window",
            core.build_time_range_filename_suffix(single_meta) == core.build_time_range_filename_suffix(compare_meta),
            {
                "single_suffix": core.build_time_range_filename_suffix(single_meta),
                "compare_suffix": core.build_time_range_filename_suffix(compare_meta),
            },
        ),
        (
            "difference_hint_equal",
            single_meta.get("time_range_difference_hint") == compare_meta.get("time_range_difference_hint"),
            {
                "single_hint": single_meta.get("time_range_difference_hint"),
                "compare_hint": compare_meta.get("time_range_difference_hint"),
            },
        ),
    ]

    failed = False
    print("[time_range_metadata_check]")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status}")
        print(f"  detail={detail}")
        failed = failed or (not ok)
    return 1 if failed else 0


def run_single_compare_base_spectrum_core_metadata_check_mode(args: argparse.Namespace) -> int:
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

    compare_mode = "时间段内 PSD 对比"
    required_metadata_keys = list(app_mod.get_required_export_metadata_fields())
    required_detail_metadata_keys = [key for key in required_metadata_keys if key != "time_range_policy_note"]
    required_export_columns = [app_mod.EXPORT_SCHEMA_VERSION_FIELD, *required_metadata_keys]
    full_export_contract_columns = list(app_mod.get_export_metadata_field_order(include_additional=True))
    legacy_time_range_export_columns = list(core.TIME_RANGE_METADATA_EXPORT_KEYS)

    root = tk.Tk()
    root.withdraw()
    app = app_mod.FileViewerApp(root)
    try:
        dual_base = app.prepare_dual_compare_payload(
            ygas_paths=[ygas_path],
            dat_path=dat_path,
            selected_paths=[ygas_path, dat_path],
            compare_mode=compare_mode,
            start_dt=None,
            end_dt=None,
        )
        selection_meta = dict(dual_base["selection_meta"])
        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = app.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])
        else:
            start_dt, end_dt = None, None
        compare_policy_meta = core.resolve_compare_time_range_policy(
            strategy_label=app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
            start_dt=start_dt,
            end_dt=end_dt,
            has_txt_dat_context=bool(
                selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
            ),
        )

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
        single_full_ygas_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=None,
            end_dt=None,
        )
        single_plot_series = single_full_ygas_payload["series_results"][0]
        single_plot_details = dict(single_plot_series["details"])
        app.current_file = ygas_path
        app.refresh_canvas = lambda *args, **kwargs: None
        app.plot_results(
            "spectral",
            np.asarray(single_plot_series["freq"], dtype=float),
            np.asarray(single_plot_series["density"], dtype=float),
            [col_a],
            single_plot_details,
        )
        single_plot_title = app.figure.axes[0].get_title() if app.figure.axes else ""
        single_plot_diag_text = app.diagnostic_var.get()
        single_plot_status_text = app.status_var.get()
        single_plot_export_frame = app.current_result_frame.copy() if app.current_result_frame is not None else pd.DataFrame()
        single_plot_export_columns = list(single_plot_export_frame.columns)
        single_plot_default_filename = app.build_default_filename()
        single_plot_default_data_filename = app.build_default_data_filename()
        dual_plot_payload = app.prepare_dual_plot_payload(
            parsed_a=dual_base["parsed_a"],
            label_a=str(dual_base["label_a"]),
            parsed_b=dual_base["parsed_b"],
            label_b=str(dual_base["label_b"]),
            pairs=[{"a_col": col_a, "b_col": col_b, "label": f"{col_a} vs {col_b}"}],
            selection_meta=selection_meta,
            compare_mode=compare_mode,
            compare_scope="单对单",
            start_dt=start_dt,
            end_dt=end_dt,
            mapping_name=str(args.element),
            scheme_name="smoke",
            alignment_strategy=app.get_alignment_strategy(compare_mode),
            plot_style=app.resolve_plot_style(compare_mode),
            plot_layout=app.resolve_plot_layout(compare_mode),
            fs_ui=args.fs,
            requested_nsegment=args.nsegment,
            overlap_ratio=args.overlap_ratio,
            match_tolerance=0.2,
            spectrum_type=core.CROSS_SPECTRUM_MAGNITUDE,
            time_range_context={
                "strategy_label": app.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                "has_txt_dat_context": True,
            },
        )

        render_prepared_multi_spectral_payload(app, single_full_ygas_payload)
        single_overlay_title = app.figure.axes[0].get_title() if app.figure.axes else ""
        single_overlay_diag_text = app.diagnostic_var.get()
        single_overlay_status_text = app.status_var.get()
        single_overlay_export_frame = (
            app.current_result_frame.copy() if app.current_result_frame is not None else pd.DataFrame()
        )
        single_overlay_export_columns = list(single_overlay_export_frame.columns)
        single_overlay_default_filename = app.build_default_filename()
        single_overlay_default_data_filename = app.build_default_data_filename()

        app.on_prepared_dual_plot_ready(dual_plot_payload)
        compare_titles = [axis.get_title() for axis in app.figure.axes]
        compare_diag_text = app.diagnostic_var.get()
        compare_status_text = app.status_var.get()
        compare_export_frame = app.current_result_frame.copy() if app.current_result_frame is not None else pd.DataFrame()
        compare_export_columns = list(compare_export_frame.columns)
        compare_default_filename = app.build_default_filename()
        compare_default_data_filename = app.build_default_data_filename()

        synced_range = app.sync_compare_common_time_range_to_single_analysis(selection_meta)
        synced_start_dt, synced_end_dt = app.resolve_optional_time_range_inputs()
        synced_single_payload = app.prepare_multi_spectral_compare_payload(
            [ygas_path],
            col_a,
            args.fs,
            args.nsegment,
            args.overlap_ratio,
            start_dt=synced_start_dt,
            end_dt=synced_end_dt,
        )
    finally:
        root.destroy()

    if (
        len(single_ygas_payload["series_results"]) != 1
        or len(single_dat_payload["series_results"]) != 1
        or len(single_full_ygas_payload["series_results"]) != 1
    ):
        raise ValueError("Single-type payload did not produce exactly one spectrum series per side.")

    dual_ygas = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "txt" and str(item["column"]) == col_a
    )
    dual_dat = next(
        item for item in dual_plot_payload["series_results"] if str(item["side"]) == "dat" and str(item["column"]) == col_b
    )
    single_ygas = single_ygas_payload["series_results"][0]
    single_dat = single_dat_payload["series_results"][0]
    single_full_ygas = single_full_ygas_payload["series_results"][0]
    synced_single_ygas = synced_single_payload["series_results"][0]

    single_plot_export_row = single_plot_export_frame.iloc[0].to_dict() if not single_plot_export_frame.empty else {}
    single_overlay_export_row = (
        single_overlay_export_frame.iloc[0].to_dict() if not single_overlay_export_frame.empty else {}
    )
    compare_export_row = compare_export_frame.iloc[0].to_dict() if not compare_export_frame.empty else {}
    single_plot_short_label = core.resolve_time_range_policy_short_label(
        single_plot_details.get("time_range_policy"),
        single_plot_details.get("time_range_policy_label"),
    )
    single_overlay_short_label = core.resolve_time_range_policy_short_label(
        single_full_ygas["details"].get("time_range_policy"),
        single_full_ygas["details"].get("time_range_policy_label"),
    )
    compare_short_label = core.resolve_time_range_policy_short_label(
        dual_ygas["details"].get("time_range_policy"),
        dual_ygas["details"].get("time_range_policy_label"),
    )
    single_plot_file_tag = core.resolve_time_range_policy_file_tag(
        single_plot_details.get("time_range_policy"),
        single_plot_details.get("time_range_policy_label"),
    )
    single_overlay_file_tag = core.resolve_time_range_policy_file_tag(
        single_full_ygas["details"].get("time_range_policy"),
        single_full_ygas["details"].get("time_range_policy_label"),
    )
    compare_file_tag = core.resolve_time_range_policy_file_tag(
        dual_ygas["details"].get("time_range_policy"),
        dual_ygas["details"].get("time_range_policy_label"),
    )

    def export_value_matches(row: dict[str, object], details: dict[str, object], key: str) -> bool:
        expected = details.get(key)
        if expected is None:
            return True
        actual = row.get(key)
        expected_text = core.format_metadata_timestamp(expected) if isinstance(expected, pd.Timestamp) else str(expected)
        return str(actual) == expected_text

    def overlay_export_value_matches(row: dict[str, object], series_items: list[dict[str, object]], key: str) -> bool:
        expected_values: list[str] = []
        for item in series_items:
            details = dict(item.get("details", {}))
            value = details.get(key)
            if value is None:
                continue
            text_value = core.format_metadata_timestamp(value) if isinstance(value, pd.Timestamp) else str(value)
            if text_value and text_value not in expected_values:
                expected_values.append(text_value)
        if not expected_values:
            return True
        expected_text = expected_values[0] if len(expected_values) == 1 else " | ".join(expected_values)
        return str(row.get(key)) == expected_text

    def contract_columns_in_order(columns: list[str]) -> list[str]:
        return [column for column in columns if column in full_export_contract_columns]

    def contract_order_is_stable(columns: list[str]) -> bool:
        present = contract_columns_in_order(columns)
        expected = [column for column in full_export_contract_columns if column in columns]
        return present == expected

    def schema_version_matches(row: dict[str, object]) -> bool:
        return str(row.get(app_mod.EXPORT_SCHEMA_VERSION_FIELD)) == app_mod.EXPORT_SCHEMA_VERSION

    def required_metadata_values_match(row: dict[str, object], details: dict[str, object]) -> bool:
        return schema_version_matches(row) and all(
            export_value_matches(row, details, key) for key in required_metadata_keys
        )

    def required_overlay_values_match(row: dict[str, object], series_items: list[dict[str, object]]) -> bool:
        return schema_version_matches(row) and all(
            overlay_export_value_matches(row, series_items, key) for key in required_metadata_keys
        )

    checks = [
        (
            "single_same_window_payload_respects_requested_range",
            single_ygas["details"].get("base_requested_start") == start_dt
            and single_ygas["details"].get("base_requested_end") == end_dt
            and single_ygas["details"].get("render_semantics") == "single_device_compare_psd"
            and single_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render",
            {
                "requested_range": (start_dt, end_dt),
                "single_requested": (
                    single_ygas["details"].get("base_requested_start"),
                    single_ygas["details"].get("base_requested_end"),
                ),
                "render_semantics": single_ygas["details"].get("render_semantics"),
                "plot_execution_path": single_ygas["details"].get("plot_execution_path"),
            },
        ),
        (
            "single_default_policy_reuses_compare_window_when_context_available",
            single_full_ygas["details"].get("time_range_policy") == "txt_dat_common_window"
            and single_full_ygas["details"].get("render_semantics") == "single_device_compare_psd"
            and single_full_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render",
            {
                "single_policy": single_full_ygas["details"].get("time_range_policy"),
                "single_label": single_full_ygas["details"].get("time_range_policy_label"),
                "render_semantics": single_full_ygas["details"].get("render_semantics"),
                "plot_execution_path": single_full_ygas["details"].get("plot_execution_path"),
            },
        ),
        (
            "single_default_matches_compare_window_when_context_available",
            single_full_ygas["details"].get("time_range_policy") == dual_ygas["details"].get("time_range_policy")
            and single_full_ygas["details"].get("plot_execution_path") == "single_device_compare_psd_render"
            and dual_ygas["details"].get("time_range_policy") == "txt_dat_common_window",
            {
                "single_policy": single_full_ygas["details"].get("time_range_policy"),
                "compare_policy": dual_ygas["details"].get("time_range_policy"),
                "single_actual": (
                    single_full_ygas["details"].get("base_actual_start"),
                    single_full_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
        (
            "single_dat_same_window_payload_respects_requested_range",
            single_dat["details"].get("base_requested_start") == start_dt
            and single_dat["details"].get("base_requested_end") == end_dt
            and single_dat["details"].get("render_semantics") == "single_device"
            and single_dat["details"].get("plot_execution_path") == "single_device_spectrum",
            {
                "requested_range": (start_dt, end_dt),
                "single_requested": (
                    single_dat["details"].get("base_requested_start"),
                    single_dat["details"].get("base_requested_end"),
                ),
                "render_semantics": single_dat["details"].get("render_semantics"),
                "plot_execution_path": single_dat["details"].get("plot_execution_path"),
            },
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
        (
            "same_window_metadata_keys_present",
            all(single_ygas["details"].get(key) is not None for key in required_detail_metadata_keys)
            and all(single_dat["details"].get(key) is not None for key in required_detail_metadata_keys)
            and all(dual_ygas["details"].get(key) is not None for key in required_detail_metadata_keys)
            and all(dual_dat["details"].get(key) is not None for key in required_detail_metadata_keys),
            {
                "required_keys": required_detail_metadata_keys,
                "export_only_keys": [key for key in required_metadata_keys if key not in required_detail_metadata_keys],
            },
        ),
        (
            "compare_common_window_policy",
            dual_ygas["details"].get("time_range_policy") == "txt_dat_common_window"
            and dual_dat["details"].get("time_range_policy") == "txt_dat_common_window",
            {
                "dual_ygas_policy": dual_ygas["details"].get("time_range_policy"),
                "dual_dat_policy": dual_dat["details"].get("time_range_policy"),
                "compare_policy_label": compare_policy_meta.get("time_range_policy_label"),
            },
        ),
        (
            "compare_manual_sync_matches_compare_window",
            synced_single_ygas["details"].get("base_requested_start") == dual_ygas["details"].get("base_requested_start")
            and synced_single_ygas["details"].get("base_requested_end") == dual_ygas["details"].get("base_requested_end")
            and synced_single_ygas["details"].get("base_actual_start") == dual_ygas["details"].get("base_actual_start")
            and synced_single_ygas["details"].get("base_actual_end") == dual_ygas["details"].get("base_actual_end"),
            {
                "synced_single_policy": synced_single_ygas["details"].get("time_range_policy"),
                "compare_policy": dual_ygas["details"].get("time_range_policy"),
                "synced_single_actual": (
                    synced_single_ygas["details"].get("base_actual_start"),
                    synced_single_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
        (
            "single_plot_title_contains_policy_short_label",
            bool(single_plot_short_label) and single_plot_short_label in single_plot_title,
            {"title": single_plot_title, "expected_label": single_plot_short_label},
        ),
        (
            "single_overlay_title_contains_policy_short_label",
            bool(single_overlay_short_label) and single_overlay_short_label in single_overlay_title,
            {"title": single_overlay_title, "expected_label": single_overlay_short_label},
        ),
        (
            "compare_title_contains_policy_short_label",
            bool(compare_short_label) and any(compare_short_label in title for title in compare_titles),
            {"titles": compare_titles, "expected_label": compare_short_label},
        ),
        (
            "single_plot_default_filename_contains_policy_tag",
            bool(single_plot_file_tag)
            and f"_{single_plot_file_tag}_" in single_plot_default_filename
            and f"_{single_plot_file_tag}_" in single_plot_default_data_filename,
            {
                "plot_filename": single_plot_default_filename,
                "data_filename": single_plot_default_data_filename,
                "expected_tag": single_plot_file_tag,
            },
        ),
        (
            "single_overlay_default_filename_contains_policy_tag",
            bool(single_overlay_file_tag)
            and f"_{single_overlay_file_tag}_" in single_overlay_default_filename
            and f"_{single_overlay_file_tag}_" in single_overlay_default_data_filename,
            {
                "plot_filename": single_overlay_default_filename,
                "data_filename": single_overlay_default_data_filename,
                "expected_tag": single_overlay_file_tag,
            },
        ),
        (
            "compare_default_filename_contains_policy_tag",
            bool(compare_file_tag)
            and f"_{compare_file_tag}_" in compare_default_filename
            and f"_{compare_file_tag}_" in compare_default_data_filename,
            {
                "plot_filename": compare_default_filename,
                "data_filename": compare_default_data_filename,
                "expected_tag": compare_file_tag,
            },
        ),
        (
            "single_and_compare_status_show_unified_hint",
            "time_range_policy 与 base_actual_time_range" in single_plot_status_text
            and "time_range_policy 与 base_actual_time_range" in single_overlay_status_text
            and "time_range_policy 与 base_actual_time_range" in compare_status_text,
            {
                "single_plot_status": single_plot_status_text,
                "single_overlay_status": single_overlay_status_text,
                "compare_status": compare_status_text,
            },
        ),
        (
            "single_and_compare_diag_show_unified_hint",
            "time_range_policy 与 base_actual_time_range" in single_plot_diag_text
            and "time_range_policy 与 base_actual_time_range" in single_overlay_diag_text
            and "time_range_policy 与 base_actual_time_range" in compare_diag_text,
            {
                "single_plot_diag": single_plot_diag_text,
                "single_overlay_diag": single_overlay_diag_text,
                "compare_diag": compare_diag_text,
            },
        ),
        (
            "single_plot_diag_items_from_same_metadata",
            all(item in single_plot_diag_text for item in core.build_time_range_diagnostic_items(single_plot_details)),
            {"expected_items": core.build_time_range_diagnostic_items(single_plot_details)},
        ),
        (
            "single_overlay_diag_items_from_same_metadata",
            all(
                item in single_overlay_diag_text
                for item in core.build_time_range_diagnostic_items(single_full_ygas["details"])
            ),
            {"expected_items": core.build_time_range_diagnostic_items(single_full_ygas["details"])},
        ),
        (
            "single_plot_status_from_same_metadata",
            str(single_plot_details.get("time_range_policy_label") or "") in single_plot_status_text
            and str(single_plot_details.get("time_range_difference_hint") or "") in single_plot_status_text,
            {
                "single_status": single_plot_status_text,
                "expected_label": single_plot_details.get("time_range_policy_label"),
                "expected_hint": single_plot_details.get("time_range_difference_hint"),
            },
        ),
        (
            "single_overlay_status_from_same_metadata",
            str(single_full_ygas["details"].get("time_range_policy_label") or "") in single_overlay_status_text
            and str(single_full_ygas["details"].get("time_range_difference_hint") or "") in single_overlay_status_text,
            {
                "single_status": single_overlay_status_text,
                "expected_label": single_full_ygas["details"].get("time_range_policy_label"),
                "expected_hint": single_full_ygas["details"].get("time_range_difference_hint"),
            },
        ),
        (
            "compare_status_from_same_metadata",
            str(dual_ygas["details"].get("time_range_policy_label") or "") in compare_status_text
            and str(dual_ygas["details"].get("time_range_difference_hint") or "") in compare_status_text,
            {
                "compare_status": compare_status_text,
                "expected_label": dual_ygas["details"].get("time_range_policy_label"),
                "expected_hint": dual_ygas["details"].get("time_range_difference_hint"),
            },
        ),
        (
            "compare_diag_items_from_same_metadata",
            all(item in compare_diag_text for item in core.build_time_range_diagnostic_items(dual_ygas["details"])),
            {"expected_items": core.build_time_range_diagnostic_items(dual_ygas["details"])},
        ),
        (
            "single_plot_export_required_columns",
            all(column in single_plot_export_columns for column in required_export_columns),
            {"required_columns": required_export_columns, "single_export_columns": single_plot_export_columns},
        ),
        (
            "single_overlay_export_required_columns",
            all(column in single_overlay_export_columns for column in required_export_columns),
            {"required_columns": required_export_columns, "single_export_columns": single_overlay_export_columns},
        ),
        (
            "compare_export_required_columns",
            all(column in compare_export_columns for column in required_export_columns),
            {"required_columns": required_export_columns, "compare_export_columns": compare_export_columns},
        ),
        (
            "single_plot_export_contract_order_stable",
            contract_order_is_stable(single_plot_export_columns),
            {
                "actual_contract_columns": contract_columns_in_order(single_plot_export_columns),
                "expected_contract_columns": [
                    column for column in full_export_contract_columns if column in single_plot_export_columns
                ],
            },
        ),
        (
            "single_overlay_export_contract_order_stable",
            contract_order_is_stable(single_overlay_export_columns),
            {
                "actual_contract_columns": contract_columns_in_order(single_overlay_export_columns),
                "expected_contract_columns": [
                    column for column in full_export_contract_columns if column in single_overlay_export_columns
                ],
            },
        ),
        (
            "compare_export_contract_order_stable",
            contract_order_is_stable(compare_export_columns),
            {
                "actual_contract_columns": contract_columns_in_order(compare_export_columns),
                "expected_contract_columns": [
                    column for column in full_export_contract_columns if column in compare_export_columns
                ],
            },
        ),
        (
            "single_plot_export_schema_version_correct",
            schema_version_matches(single_plot_export_row),
            {"schema_version": single_plot_export_row.get(app_mod.EXPORT_SCHEMA_VERSION_FIELD)},
        ),
        (
            "single_overlay_export_schema_version_correct",
            schema_version_matches(single_overlay_export_row),
            {"schema_version": single_overlay_export_row.get(app_mod.EXPORT_SCHEMA_VERSION_FIELD)},
        ),
        (
            "compare_export_schema_version_correct",
            schema_version_matches(compare_export_row),
            {"schema_version": compare_export_row.get(app_mod.EXPORT_SCHEMA_VERSION_FIELD)},
        ),
        (
            "single_plot_export_required_values_match_metadata",
            required_metadata_values_match(single_plot_export_row, single_plot_details),
            {"export_row": single_plot_export_row},
        ),
        (
            "single_overlay_export_required_values_match_metadata",
            required_overlay_values_match(single_overlay_export_row, [single_full_ygas]),
            {"export_row": single_overlay_export_row},
        ),
        (
            "compare_export_required_values_match_metadata",
            required_overlay_values_match(compare_export_row, [dual_ygas, dual_dat]),
            {"export_row": compare_export_row},
        ),
        (
            "single_plot_legacy_time_range_columns_preserved",
            all(column in single_plot_export_columns for column in legacy_time_range_export_columns),
            {
                "legacy_columns": legacy_time_range_export_columns,
                "single_export_columns": single_plot_export_columns,
            },
        ),
        (
            "single_overlay_legacy_time_range_columns_preserved",
            all(column in single_overlay_export_columns for column in legacy_time_range_export_columns),
            {
                "legacy_columns": legacy_time_range_export_columns,
                "single_export_columns": single_overlay_export_columns,
            },
        ),
        (
            "compare_legacy_time_range_columns_preserved",
            all(column in compare_export_columns for column in legacy_time_range_export_columns),
            {
                "legacy_columns": legacy_time_range_export_columns,
                "compare_export_columns": compare_export_columns,
            },
        ),
        (
            "single_plot_and_overlay_contract_columns_same",
            contract_columns_in_order(single_plot_export_columns) == contract_columns_in_order(single_overlay_export_columns),
            {
                "single_plot_contract_columns": contract_columns_in_order(single_plot_export_columns),
                "single_overlay_contract_columns": contract_columns_in_order(single_overlay_export_columns),
            },
        ),
        (
            "compare_to_single_sync_returns_common_window",
            synced_range == (start_dt, end_dt),
            {"synced_range": synced_range, "compare_range": (start_dt, end_dt)},
        ),
        (
            "compare_to_single_sync_requested_range_equal",
            synced_single_ygas["details"].get("base_requested_start") == dual_ygas["details"].get("base_requested_start")
            and synced_single_ygas["details"].get("base_requested_end") == dual_ygas["details"].get("base_requested_end"),
            {
                "synced_single_requested": (
                    synced_single_ygas["details"].get("base_requested_start"),
                    synced_single_ygas["details"].get("base_requested_end"),
                ),
                "compare_requested": (
                    dual_ygas["details"].get("base_requested_start"),
                    dual_ygas["details"].get("base_requested_end"),
                ),
            },
        ),
        (
            "compare_to_single_sync_actual_range_equal",
            synced_single_ygas["details"].get("base_actual_start") == dual_ygas["details"].get("base_actual_start")
            and synced_single_ygas["details"].get("base_actual_end") == dual_ygas["details"].get("base_actual_end"),
            {
                "synced_single_actual": (
                    synced_single_ygas["details"].get("base_actual_start"),
                    synced_single_ygas["details"].get("base_actual_end"),
                ),
                "compare_actual": (
                    dual_ygas["details"].get("base_actual_start"),
                    dual_ygas["details"].get("base_actual_end"),
                ),
            },
        ),
    ]

    failed = False
    print("[single_compare_base_spectrum_check]")
    print(f"ygas_path={ygas_path}")
    print(f"dat_path={dat_path}")
    print(f"column_a={col_a}")
    print(f"column_b={col_b}")
    print(f"same_window_time_range={start_dt} ~ {end_dt}")
    print(
        "single_full_time_range="
        f"{single_full_ygas['details'].get('base_actual_start')} ~ {single_full_ygas['details'].get('base_actual_end')}"
    )
    print(
        "compare_time_range="
        f"{dual_ygas['details'].get('base_actual_start')} ~ {dual_ygas['details'].get('base_actual_end')}"
    )
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
            "frr-compat-semantics-check",
            "device-group-spectral-check",
            "device-count-dispatch-single-check",
            "device-count-dispatch-dual-check",
            "device-count-dispatch-multi-check",
            "device-count-default-single-compare-style-check",
            "device-count-default-single-fallback-check",
            "device-count-default-dual-check",
            "device-count-default-multi-check",
            "selected-files-direct-generate-check",
            "selected-files-direct-target-spectrum-single-check",
            "selected-files-direct-target-spectrum-dual-check",
            "txt-dat-not-equals-device-count-check",
            "single-compare-base-spectrum-check",
            "single-device-selection-scope-check",
            "single-file-open-auto-bootstrap-compare-context-check",
            "single-device-no-visible-change-regression-check",
            "single-device-txt-compare-side-equivalence-check",
            "single-device-mixed-selection-txt-dat-reuse-check",
            "single-device-mixed-selection-multi-txt-plus-dat-reuse-check",
            "single-device-only-dat-still-fallback-check",
            "single-device-default-reuse-enabled-check",
            "single-device-render-semantic-equivalence-check",
            "single-device-vs-compare-render-style-check",
            "single-device-vs-dual-compare-axis-geometry-check",
            "single-device-vs-dual-compare-reference-slope-check",
            "single-device-vs-dual-compare-visible-scatter-subset-check",
            "single-vs-dual-target-spectrum-visible-subset-check",
            "single-vs-dual-target-spectrum-style-check",
            "single-device-fallback-still-legacy-render-check",
            "single-device-active-dat-sync-check",
            "single-device-active-time-range-sync-check",
            "single-device-auto-bootstrap-fallback-check",
            "single-txt-compare-side-equivalence-check",
            "single-compare-style-preview-check",
            "single-vs-compare-point-display-contract-check",
            "time-range-metadata-check",
            "repo-fixture-smoke",
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
        if mode == "frr-compat-semantics-check":
            return run_frr_compat_semantics_check_mode(args)
        if mode == "device-group-spectral-check":
            return run_device_group_spectral_check_mode(args)
        if mode == "device-count-dispatch-single-check":
            return run_device_count_dispatch_single_check_mode(args)
        if mode == "device-count-dispatch-dual-check":
            return run_device_count_dispatch_dual_check_mode(args)
        if mode == "device-count-dispatch-multi-check":
            return run_device_count_dispatch_multi_check_mode(args)
        if mode == "device-count-default-single-compare-style-check":
            return run_device_count_default_single_compare_style_check_mode(args)
        if mode == "device-count-default-single-fallback-check":
            return run_device_count_default_single_fallback_check_mode(args)
        if mode == "device-count-default-dual-check":
            return run_device_count_default_dual_check_mode(args)
        if mode == "device-count-default-multi-check":
            return run_device_count_default_multi_check_mode(args)
        if mode == "selected-files-direct-generate-check":
            return run_selected_files_direct_generate_check_mode(args)
        if mode == "selected-files-direct-target-spectrum-single-check":
            return run_selected_files_direct_target_spectrum_single_check_mode(args)
        if mode == "selected-files-direct-target-spectrum-dual-check":
            return run_selected_files_direct_target_spectrum_dual_check_mode(args)
        if mode == "txt-dat-not-equals-device-count-check":
            return run_txt_dat_not_equals_device_count_check_mode(args)
        if mode == "single-compare-base-spectrum-check":
            return run_single_compare_base_spectrum_core_metadata_check_mode(args)
        if mode == "single-device-selection-scope-check":
            return run_single_device_selection_scope_check_mode(args)
        if mode == "single-file-open-auto-bootstrap-compare-context-check":
            return run_single_file_open_auto_bootstrap_compare_context_check_mode(args)
        if mode == "single-device-no-visible-change-regression-check":
            return run_single_device_no_visible_change_regression_check_mode(args)
        if mode == "single-device-txt-compare-side-equivalence-check":
            return run_single_device_txt_compare_side_equivalence_check_mode(args)
        if mode == "single-device-mixed-selection-txt-dat-reuse-check":
            return run_single_device_mixed_selection_txt_dat_reuse_check_mode(args)
        if mode == "single-device-mixed-selection-multi-txt-plus-dat-reuse-check":
            return run_single_device_mixed_selection_multi_txt_plus_dat_reuse_check_mode(args)
        if mode == "single-device-only-dat-still-fallback-check":
            return run_single_device_only_dat_still_fallback_check_mode(args)
        if mode == "single-device-default-reuse-enabled-check":
            return run_single_device_default_reuse_enabled_check_mode(args)
        if mode == "single-device-render-semantic-equivalence-check":
            return run_single_device_render_semantic_equivalence_check_mode(args)
        if mode == "single-device-vs-compare-render-style-check":
            return run_single_device_vs_compare_render_style_check_mode(args)
        if mode == "single-device-vs-dual-compare-axis-geometry-check":
            return run_single_device_vs_dual_compare_axis_geometry_check_mode(args)
        if mode == "single-device-vs-dual-compare-reference-slope-check":
            return run_single_device_vs_dual_compare_reference_slope_check_mode(args)
        if mode == "single-device-vs-dual-compare-visible-scatter-subset-check":
            return run_single_device_vs_dual_compare_visible_scatter_subset_check_mode(args)
        if mode == "single-vs-dual-target-spectrum-visible-subset-check":
            return run_single_vs_dual_target_spectrum_visible_subset_check_mode(args)
        if mode == "single-vs-dual-target-spectrum-style-check":
            return run_single_vs_dual_target_spectrum_style_check_mode(args)
        if mode == "single-device-fallback-still-legacy-render-check":
            return run_single_device_fallback_still_legacy_render_check_mode(args)
        if mode == "single-device-active-dat-sync-check":
            return run_single_device_active_dat_sync_check_mode(args)
        if mode == "single-device-active-time-range-sync-check":
            return run_single_device_active_time_range_sync_check_mode(args)
        if mode == "single-device-auto-bootstrap-fallback-check":
            return run_single_device_auto_bootstrap_fallback_check_mode(args)
        if mode == "single-txt-compare-side-equivalence-check":
            return run_single_txt_compare_side_equivalence_check_mode(args)
        if mode == "single-compare-style-preview-check":
            return run_single_compare_style_preview_check_mode(args)
        if mode == "single-vs-compare-point-display-contract-check":
            return run_single_vs_compare_point_display_contract_check_mode(args)
        if mode == "time-range-metadata-check":
            return run_time_range_metadata_check_mode(args)
        if mode == "repo-fixture-smoke":
            return run_repo_fixture_smoke_mode(args)
        return run_legacy_target_mode(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
