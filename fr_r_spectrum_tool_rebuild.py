from __future__ import annotations

import json
import math
import queue
import re
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import spectrum_core as core
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib import font_manager
from tkinter import colorchooser, filedialog, font as tkfont, messagebox, ttk


FigureCanvasTkAgg: Any | None = None
NavigationToolbar2Tk: Any | None = None
MATPLOTLIB_CJK_FONT: str | None = None


ALLOWED_SUFFIXES = {".txt", ".log", ".csv", ".dat"}
ROWS_PER_PAGE = 100
DEFAULT_FS = core.DEFAULT_FS
DEFAULT_NSEGMENT = core.DEFAULT_NSEGMENT
DEFAULT_OVERLAP_RATIO = core.DEFAULT_OVERLAP_RATIO
TEXT_LIKE_SUFFIXES = {".txt", ".log"}
TIMESTAMP_PATTERN = core.TIMESTAMP_PATTERN
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
HEADER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .-]{0,63}$")
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
LEGACY_TARGET_COLOR_PAIRS = [
    {"ygas": "#1f77b4", "dat": "#ff7f0e"},
    {"ygas": "#d62728", "dat": "#2ca02c"},
    {"ygas": "#9467bd", "dat": "#8c564b"},
    {"ygas": "#17becf", "dat": "#e377c2"},
    {"ygas": "#7f7f7f", "dat": "#bcbd22"},
    {"ygas": "#393b79", "dat": "#637939"},
    {"ygas": "#843c39", "dat": "#3182bd"},
    {"ygas": "#5254a3", "dat": "#31a354"},
]
CROSS_SPECTRUM_MAGNITUDE = core.CROSS_SPECTRUM_MAGNITUDE
CROSS_SPECTRUM_REAL = core.CROSS_SPECTRUM_REAL
CROSS_SPECTRUM_IMAG = core.CROSS_SPECTRUM_IMAG
CROSS_SPECTRUM_OPTIONS = list(core.CROSS_SPECTRUM_OPTIONS)
LEGACY_TARGET_SPECTRUM_MODE_CHOICES = list(core.LEGACY_TARGET_SPECTRUM_MODE_CHOICES)
REFERENCE_SLOPE_MODE_AUTO = "自动"
REFERENCE_SLOPE_MODE_MINUS_2_3 = "仅 -2/3"
REFERENCE_SLOPE_MODE_MINUS_4_3 = "仅 -4/3"
REFERENCE_SLOPE_MODE_BOTH = "两条都显示"
REFERENCE_SLOPE_MODE_NONE = "不显示"
REFERENCE_SLOPE_MODE_OPTIONS = [
    REFERENCE_SLOPE_MODE_AUTO,
    REFERENCE_SLOPE_MODE_MINUS_2_3,
    REFERENCE_SLOPE_MODE_MINUS_4_3,
    REFERENCE_SLOPE_MODE_BOTH,
    REFERENCE_SLOPE_MODE_NONE,
]
REFERENCE_SLOPE_LIBRARY: dict[str, dict[str, float | str]] = {
    REFERENCE_SLOPE_MODE_MINUS_2_3: {
        "slope": -2 / 3,
        "label": "Reference slope (-2/3)",
        "short_label": "(-2/3)",
        "color": "#d65f5f",
    },
    REFERENCE_SLOPE_MODE_MINUS_4_3: {
        "slope": -4 / 3,
        "label": "Reference slope (-4/3)",
        "short_label": "(-4/3)",
        "color": "#4c78a8",
    },
}
LEGACY_TARGET_COLOR_MODE_BY_DEVICE = "按设备两色"
LEGACY_TARGET_COLOR_MODE_BY_GROUP = "按组配对着色"
LEGACY_TARGET_COLOR_MODE_CHOICES = [LEGACY_TARGET_COLOR_MODE_BY_DEVICE, LEGACY_TARGET_COLOR_MODE_BY_GROUP]
LEGACY_TARGET_DEVICE_COLORS = {"ygas": "#1f77b4", "dat": "#ff7f0e"}
LEGACY_TARGET_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LEGACY_TARGET_LINESTYLES = ["-", "--", "-.", ":"]
MODE1_COLUMN_TEMPLATE = list(core.MODE1_COLUMN_TEMPLATE)
ELEMENT_PRESETS: dict[str, dict[str, list[str]]] = dict(core.ELEMENT_PRESETS)
EXPORT_SCHEMA_VERSION = "psd_export_v2"
EXPORT_SCHEMA_VERSION_FIELD = "export_schema_version"
EXPORT_REQUIRED_METADATA_FIELDS = (
    "base_spectrum_builder",
    "base_fs_source",
    "base_requested_start",
    "base_requested_end",
    "base_actual_start",
    "base_actual_end",
    "time_range_policy",
    "time_range_policy_label",
    "time_range_policy_note",
)
EXPORT_ADDITIONAL_METADATA_FIELDS = (
    *core.TIME_RANGE_METADATA_EXPORT_KEYS,
    "plot_execution_path",
    "render_semantics",
    "effective_device_count",
    "effective_device_ids",
    "selection_file_count",
    "selected_txt_file_count",
    "selected_dat_file_count",
    "data_context_source",
    "cross_execution_path",
    "cross_implementation_id",
    "cross_reference_column",
    "reference_column",
    "ygas_target_column",
    "dat_target_column",
    "target_column",
    "cross_order",
    "canonical_cross_pairs",
    "generated_cross_series_count",
    "generated_cross_series_roles",
    "display_semantics",
    "display_value_source",
)


def get_required_export_metadata_fields() -> tuple[str, ...]:
    return EXPORT_REQUIRED_METADATA_FIELDS


def get_export_metadata_field_order(*, include_additional: bool = False) -> tuple[str, ...]:
    ordered_fields: list[str] = [EXPORT_SCHEMA_VERSION_FIELD, *EXPORT_REQUIRED_METADATA_FIELDS]
    if include_additional:
        for field in EXPORT_ADDITIONAL_METADATA_FIELDS:
            if field not in ordered_fields:
                ordered_fields.append(field)
    return tuple(ordered_fields)


def build_export_metadata_from_details(
    details: dict[str, Any],
    *,
    include_additional: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {EXPORT_SCHEMA_VERSION_FIELD: EXPORT_SCHEMA_VERSION}
    for key in get_export_metadata_field_order(include_additional=include_additional):
        if key == EXPORT_SCHEMA_VERSION_FIELD:
            continue
        value = details.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def build_aggregated_export_metadata(
    series_results: list[dict[str, Any]],
    *,
    formatter: Any,
    include_additional: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {EXPORT_SCHEMA_VERSION_FIELD: EXPORT_SCHEMA_VERSION}
    for key in get_export_metadata_field_order(include_additional=include_additional):
        if key == EXPORT_SCHEMA_VERSION_FIELD:
            continue
        values: list[str] = []
        for item in series_results:
            details = dict(item.get("details", {}))
            value = details.get(key)
            if value is None:
                continue
            text_value = formatter(value)
            if text_value and text_value not in values:
                values.append(text_value)
        if values:
            metadata[key] = values[0] if len(values) == 1 else " | ".join(values)
    return metadata


def _coerce_optional_int(value: Any, fallback: int = 0) -> int:
    try:
        if value is None:
            raise TypeError
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _format_series_metric_values(values: list[int]) -> str:
    unique_values = list(dict.fromkeys(int(value) for value in values))
    if not unique_values:
        return "0"
    return "/".join(str(value) for value in unique_values)


def build_series_point_count_contract(series_results: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_series: list[dict[str, Any]] = []
    for item in series_results:
        details = dict(item.get("details", {}))
        freq_values = np.asarray(item.get("freq", []), dtype=float)
        valid_freq_points = _coerce_optional_int(details.get("valid_freq_points"), len(freq_values))
        frequency_point_count = _coerce_optional_int(details.get("frequency_point_count"), len(freq_values))
        normalized_series.append(
            {
                "label": str(item.get("label") or item.get("column") or f"series_{len(normalized_series) + 1}"),
                "valid_freq_points": valid_freq_points,
                "frequency_point_count": frequency_point_count,
                "rendered_point_count": int(len(freq_values)),
            }
        )

    valid_freq_points_values = [item["valid_freq_points"] for item in normalized_series]
    frequency_point_count_values = [item["frequency_point_count"] for item in normalized_series]
    rendered_point_count_values = [item["rendered_point_count"] for item in normalized_series]
    return {
        "series_count": int(len(normalized_series)),
        "single_series_valid_freq_points": int(valid_freq_points_values[0]) if valid_freq_points_values else 0,
        "single_series_valid_freq_points_display": _format_series_metric_values(valid_freq_points_values),
        "single_series_frequency_point_count": int(frequency_point_count_values[0]) if frequency_point_count_values else 0,
        "single_series_frequency_point_count_display": _format_series_metric_values(frequency_point_count_values),
        "total_valid_freq_points_across_series": int(sum(valid_freq_points_values)),
        "total_frequency_points_across_series": int(sum(frequency_point_count_values)),
        "total_rendered_point_count_across_series": int(sum(rendered_point_count_values)),
        "valid_freq_points_values": valid_freq_points_values,
        "frequency_point_count_values": frequency_point_count_values,
        "rendered_point_count_values": rendered_point_count_values,
        "series_summaries": [
            f"{item['label']}:{item['valid_freq_points']}/{item['frequency_point_count']}"
            for item in normalized_series
        ],
    }


def build_series_point_count_items(contract: dict[str, Any], *, prefix: str = "") -> list[str]:
    normalized_prefix = str(prefix or "")
    if normalized_prefix and not normalized_prefix.endswith("_"):
        normalized_prefix = f"{normalized_prefix}_"
    items = [
        f"{normalized_prefix}single_series_valid_freq_points={contract.get('single_series_valid_freq_points_display', '0')}",
        f"{normalized_prefix}single_series_frequency_point_count={contract.get('single_series_frequency_point_count_display', '0')}",
        f"{normalized_prefix}series_count={int(contract.get('series_count', 0))}",
        f"{normalized_prefix}total_valid_freq_points_across_series={int(contract.get('total_valid_freq_points_across_series', 0))}",
        f"{normalized_prefix}total_frequency_points_across_series={int(contract.get('total_frequency_points_across_series', 0))}",
    ]
    series_summaries = list(contract.get("series_summaries", []))
    if int(contract.get("series_count", 0)) > 1 and series_summaries:
        items.append(f"{normalized_prefix}series_point_breakdown={' | '.join(series_summaries)}")
    return items


def build_series_point_count_status_items(
    contract: dict[str, Any],
    *,
    prefix: str = "",
    include_totals: bool = True,
) -> list[str]:
    normalized_prefix = str(prefix or "")
    if normalized_prefix and not normalized_prefix.endswith("_"):
        normalized_prefix = f"{normalized_prefix}_"
    items = [
        f"{normalized_prefix}single_series_valid_freq_points={contract.get('single_series_valid_freq_points_display', '0')}",
        f"{normalized_prefix}single_series_frequency_point_count={contract.get('single_series_frequency_point_count_display', '0')}",
    ]
    if include_totals:
        items.extend(
            [
                f"{normalized_prefix}series_count={int(contract.get('series_count', 0))}",
                f"{normalized_prefix}total_valid_freq_points_across_series={int(contract.get('total_valid_freq_points_across_series', 0))}",
                f"{normalized_prefix}total_frequency_points_across_series={int(contract.get('total_frequency_points_across_series', 0))}",
            ]
        )
    return items


def annotate_compare_geometry_point_count_fields(
    details: dict[str, Any],
    *,
    visible_contract: dict[str, Any],
    geometry_contract: dict[str, Any],
) -> None:
    details["visible_single_side_valid_freq_points"] = int(
        visible_contract.get("single_series_valid_freq_points", 0)
    )
    details["visible_single_side_frequency_point_count"] = int(
        visible_contract.get("single_series_frequency_point_count", 0)
    )
    details["compare_geometry_total_valid_freq_points"] = int(
        geometry_contract.get("total_valid_freq_points_across_series", 0)
    )
    details["compare_geometry_total_frequency_points"] = int(
        geometry_contract.get("total_frequency_points_across_series", 0)
    )


def build_compare_geometry_point_count_items(details: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if "visible_single_side_valid_freq_points" in details:
        items.append(
            f"visible_single_side_valid_freq_points={_coerce_optional_int(details.get('visible_single_side_valid_freq_points'))}"
        )
    if "visible_single_side_frequency_point_count" in details:
        items.append(
            "visible_single_side_frequency_point_count="
            f"{_coerce_optional_int(details.get('visible_single_side_frequency_point_count'))}"
        )
    if "compare_geometry_total_valid_freq_points" in details:
        items.append(
            "compare_geometry_total_valid_freq_points="
            f"{_coerce_optional_int(details.get('compare_geometry_total_valid_freq_points'))}"
        )
    if "compare_geometry_total_frequency_points" in details:
        items.append(
            "compare_geometry_total_frequency_points="
            f"{_coerce_optional_int(details.get('compare_geometry_total_frequency_points'))}"
        )
    return items


def normalize_effective_device_ids(value: Any) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized: list[str] = []
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def annotate_device_dispatch_details(
    details: dict[str, Any],
    *,
    effective_device_ids: list[str] | tuple[str, ...] | set[str],
    plot_execution_path: str,
    render_semantics: str,
    selection_file_count: int,
    selected_txt_file_count: int,
    selected_dat_file_count: int,
    data_context_source: str,
) -> None:
    normalized_device_ids = normalize_effective_device_ids(effective_device_ids)
    details["effective_device_count"] = int(len(normalized_device_ids))
    details["effective_device_ids"] = normalized_device_ids
    details["plot_execution_path"] = str(plot_execution_path)
    details["render_semantics"] = str(render_semantics)
    details["selection_file_count"] = int(selection_file_count)
    details["selected_file_count"] = int(selection_file_count)
    details["selected_txt_file_count"] = int(selected_txt_file_count)
    details["selected_dat_file_count"] = int(selected_dat_file_count)
    details["data_context_source"] = str(data_context_source or "")
    if details.get("time_range_policy") is not None:
        details["time_range_policy"] = str(details.get("time_range_policy") or "")


def build_device_dispatch_items(details: dict[str, Any], *, include_plot_execution_path: bool = False) -> list[str]:
    items: list[str] = []
    normalized_device_ids = normalize_effective_device_ids(details.get("effective_device_ids"))
    if "effective_device_count" in details or normalized_device_ids:
        items.append(
            f"effective_device_count={int(details.get('effective_device_count', len(normalized_device_ids)))}"
        )
    if "effective_device_ids" in details or normalized_device_ids:
        items.append(f"effective_device_ids={','.join(normalized_device_ids)}")
    for key in (
        "render_semantics",
        "selection_file_count",
        "selected_txt_file_count",
        "selected_dat_file_count",
        "data_context_source",
        "time_range_policy",
    ):
        value = details.get(key)
        if value is not None and value != "":
            items.append(f"{key}={value}")
    if include_plot_execution_path:
        plot_execution_path = details.get("plot_execution_path")
        if plot_execution_path is not None and plot_execution_path != "":
            items.append(f"plot_execution_path={plot_execution_path}")
    return items


def build_execution_items(
    details: dict[str, Any],
    keys: tuple[str, ...],
    *,
    include_plot_execution_path: bool = False,
) -> list[str]:
    items: list[str] = []
    for key in keys:
        value = details.get(key)
        if value is not None:
            items.append(f"{key}={value}")
    if include_plot_execution_path:
        plot_execution_path = details.get("plot_execution_path")
        if plot_execution_path is not None:
            items.append(f"plot_execution_path={plot_execution_path}")
    return items


def build_auto_compare_context_items(details: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if "auto_compare_context_built" in details:
        built_value = bool(details.get("auto_compare_context_built"))
        items.append(f"auto_compare_context_built={'true' if built_value else 'false'}")
    if "auto_compare_context_available" in details:
        available_value = bool(details.get("auto_compare_context_available"))
        items.append(f"auto_compare_context_available={'true' if available_value else 'false'}")
    for key in ("auto_compare_context_source", "auto_compare_context_dat_file_name", "auto_compare_context_txt_count"):
        value = details.get(key)
        if value is not None and value != "":
            items.append(f"{key}={value}")
    return items


def build_single_txt_execution_items(details: dict[str, Any], *, include_plot_execution_path: bool = False) -> list[str]:
    items = build_device_dispatch_items(details, include_plot_execution_path=include_plot_execution_path)
    items.extend(
        build_execution_items(
        details,
        (
            "single_txt_execution_path",
            "single_txt_selection_scope",
            "single_txt_time_range_policy",
            "direct_generate_target_spectrum_fallback_reason",
        ),
        include_plot_execution_path=False,
    ))
    items.extend(build_auto_compare_context_items(details))
    return items


def build_single_device_execution_items(details: dict[str, Any], *, include_plot_execution_path: bool = False) -> list[str]:
    items = build_device_dispatch_items(details, include_plot_execution_path=include_plot_execution_path)
    items.extend(build_execution_items(
        details,
        (
            "single_device_execution_path",
            "single_device_render_semantics",
            "current_plot_kind",
            "single_device_selection_scope",
            "single_device_time_range_policy",
            "single_device_compare_side_fallback_reason",
        ),
        include_plot_execution_path=False,
    ))
    for key in (
        "single_device_compare_context_source",
        "single_device_compare_context_dat_file_name",
        "single_device_compare_context_time_range_policy",
        "single_device_compare_context_start",
        "single_device_compare_context_end",
    ):
        value = details.get(key)
        if value is not None and value != "":
            items.append(f"{key}={value}")
    for key in (
        "single_device_compare_context_matches_current_compare_ui",
        "single_device_compare_context_dat_matches_selected_dat",
    ):
        if key in details:
            items.append(f"{key}={'true' if bool(details.get(key)) else 'false'}")
    if "single_device_selection_filtered_to_txt_side" in details:
        filtered_value = bool(details.get("single_device_selection_filtered_to_txt_side"))
        items.append(f"single_device_selection_filtered_to_txt_side={'true' if filtered_value else 'false'}")
    items.extend(build_auto_compare_context_items(details))
    return items


def get_resource_path(*parts: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        base_dir = Path(getattr(sys, "_MEIPASS"))
    else:
        base_dir = Path(__file__).resolve().parent
    return base_dir.joinpath(*parts)


def apply_window_icon(window: tk.Misc) -> tk.PhotoImage | None:
    icon_image: tk.PhotoImage | None = None
    icon_ico = get_resource_path("assets", "app_icon.ico")
    icon_png = get_resource_path("assets", "app_icon.png")
    if icon_ico.exists():
        try:
            window.iconbitmap(default=str(icon_ico))
        except tk.TclError:
            pass
    if icon_png.exists():
        try:
            icon_image = tk.PhotoImage(file=str(icon_png))
            window.iconphoto(True, icon_image)
        except tk.TclError:
            icon_image = None
    return icon_image


def configure_matplotlib_fonts() -> None:
    global MATPLOTLIB_CJK_FONT
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    MATPLOTLIB_CJK_FONT = None
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            MATPLOTLIB_CJK_FONT = name
            break
    if MATPLOTLIB_CJK_FONT is None:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


def build_legacy_target_plot_title(target_element: str) -> str:
    chinese_prefix = "\u65f6\u95f4\u5e8f\u5217\u8c31\u5206\u6790"
    english_prefix = "Time Series Spectrum"
    prefix = chinese_prefix if MATPLOTLIB_CJK_FONT else english_prefix
    return f"{prefix} - {target_element}"


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", text.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("._") or "column"


def float_to_fraction_str(value: float) -> str:
    fraction = Fraction(value).limit_denominator(24)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def normalize_reference_slope_mode(reference_mode: str | None) -> str:
    mode = str(reference_mode or "").strip() or REFERENCE_SLOPE_MODE_AUTO
    if mode not in REFERENCE_SLOPE_MODE_OPTIONS:
        return REFERENCE_SLOPE_MODE_AUTO
    return mode


def resolve_reference_slope_specs(
    reference_mode: str | None,
    *,
    is_psd: bool,
    spectrum_type: str | None = None,
) -> list[dict[str, float | str]]:
    mode = normalize_reference_slope_mode(reference_mode)
    if mode == REFERENCE_SLOPE_MODE_AUTO:
        if is_psd:
            keys = [REFERENCE_SLOPE_MODE_MINUS_2_3]
        elif spectrum_type in {CROSS_SPECTRUM_MAGNITUDE, CROSS_SPECTRUM_REAL}:
            keys = [REFERENCE_SLOPE_MODE_MINUS_4_3]
        else:
            keys = []
    elif mode == REFERENCE_SLOPE_MODE_BOTH:
        keys = [REFERENCE_SLOPE_MODE_MINUS_2_3, REFERENCE_SLOPE_MODE_MINUS_4_3]
    elif mode == REFERENCE_SLOPE_MODE_NONE:
        keys = []
    else:
        keys = [mode]
    return [dict(REFERENCE_SLOPE_LIBRARY[key]) for key in keys]


def format_reference_slope_selection(specs: list[dict[str, float | str]]) -> str:
    labels = [str(spec.get("short_label", "")).strip() for spec in specs if str(spec.get("short_label", "")).strip()]
    return "none" if not labels else ",".join(labels)


def parse_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def build_generated_column_names(count: int) -> list[str]:
    return core.build_generated_column_names(count)


def build_mode1_column_names(count: int) -> list[str]:
    if count == len(MODE1_COLUMN_TEMPLATE):
        return list(MODE1_COLUMN_TEMPLATE)
    return build_generated_column_names(count)


def parse_mixed_timestamp_series(series: pd.Series) -> pd.Series:
    return core.parse_mixed_timestamp_series(series)


def build_timestamp_parse_stats(source: pd.Series, parsed: pd.Series) -> dict[str, Any]:
    return core.build_timestamp_parse_stats(source, parsed)


def looks_like_timestamp_series(series: pd.Series) -> bool:
    return core.looks_like_timestamp_series(series)


def looks_like_incremental_index(series: pd.Series) -> bool:
    return core.looks_like_incremental_index(series)


def detect_mode1_layout(df: pd.DataFrame) -> dict[str, Any]:
    excluded_cols = {"时间戳", "状态字符", "状态寄存器", "校验和"}
    if len(df.columns) == len(MODE1_COLUMN_TEMPLATE) and looks_like_timestamp_series(df.iloc[:, 0]):
        return {
            "matched": True,
            "variant": "mode1-15",
            "display_columns": list(MODE1_COLUMN_TEMPLATE),
            "excluded_cols": excluded_cols,
        }

    if len(df.columns) == len(MODE1_COLUMN_TEMPLATE) + 1:
        first_col = df.iloc[:, 0]
        second_col = df.iloc[:, 1]
        if looks_like_incremental_index(first_col) and looks_like_timestamp_series(second_col):
            return {
                "matched": True,
                "variant": "mode1-16-with-index",
                "display_columns": ["索引列", *MODE1_COLUMN_TEMPLATE],
                "excluded_cols": {"索引列", *excluded_cols},
            }

    return {
        "matched": False,
        "variant": None,
        "display_columns": build_generated_column_names(len(df.columns)),
        "excluded_cols": set(),
    }


def detect_file_profile(path: Path, preview_lines: list[str]) -> str:
    return core.detect_file_profile(path, preview_lines)


def filter_by_time_range(
    df: pd.DataFrame,
    timestamp_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    return core.filter_by_time_range(df, timestamp_col, start_dt, end_dt)


def estimate_fs_from_timestamp(df: pd.DataFrame, timestamp_col: str) -> float | None:
    return core.estimate_fs_from_timestamp(df, timestamp_col)


@dataclass
class LoaderResult:
    dataframe: pd.DataFrame
    columns: list[str]
    available_columns: list[str] | None = None
    suggested_columns: list[str] | None = None
    used_mode1_template: bool = False
    mode1_layout: dict[str, Any] | None = None
    profile_name: str | None = None
    excluded_columns: set[str] | None = None


@dataclass
class ParsedFileResult:
    dataframe: pd.DataFrame
    profile_name: str
    timestamp_col: str | None
    suggested_columns: list[str]
    available_columns: list[str]
    source_row_count: int = 0
    timestamp_valid_count: int = 0
    timestamp_valid_ratio: float = 0.0
    timestamp_warning: str | None = None


def read_preview_lines(path: Path, max_lines: int = 8) -> list[str]:
    return core.read_preview_lines(path, max_lines=max_lines)


class AsyncFileLoader:
    def __init__(self, file_path: str, delimiter: str, start_row: int, header_row: int | None) -> None:
        self.file_path = file_path
        self.delimiter = delimiter
        self.start_row = start_row
        self.header_row = header_row

    def load(self) -> LoaderResult:
        # Confirmed from v4: the EXE used pandas.read_csv with sep/skiprows/header/engine/on_bad_lines/dtype.
        df = pd.read_csv(
            self.file_path,
            sep=self.delimiter,
            skiprows=self.start_row,
            header=self.header_row,
            engine="python",
            on_bad_lines="skip",
            dtype=str,
        )

        # Practical completion: header=None 时生成更适合高频 txt 数据的显示列名。
        used_mode1_template = False
        mode1_layout: dict[str, Any] | None = None
        if self.header_row is None:
            if Path(self.file_path).suffix.lower() in TEXT_LIKE_SUFFIXES:
                mode1_layout = detect_mode1_layout(df)
                df.columns = mode1_layout["display_columns"]
                used_mode1_template = bool(mode1_layout["matched"])
            else:
                df.columns = build_generated_column_names(len(df.columns))
        else:
            # Practical completion: normalize display/selection keys to strings for Tk widgets.
            df.columns = [str(col) for col in df.columns]
        return LoaderResult(
            dataframe=df,
            columns=list(df.columns),
            available_columns=list(df.columns),
            suggested_columns=None,
            used_mode1_template=used_mode1_template,
            mode1_layout=mode1_layout,
            profile_name=None,
            excluded_columns=set(mode1_layout.get("excluded_cols", set())) if mode1_layout is not None else set(),
        )


class FileViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("数据分析")
        self.root.geometry("1400x900")
        self.window_icon_image = apply_window_icon(self.root)

        self.current_folder: Path | None = None
        self.current_file: Path | None = None
        self.current_file_parsed: ParsedFileResult | None = None
        self.raw_data: pd.DataFrame | None = None
        self.preview_data: pd.DataFrame | None = None
        self.current_comparison_frame: pd.DataFrame | None = None
        self.current_comparison_metadata: dict[str, Any] = {}
        self.current_target_plot_metadata: dict[str, Any] = {}
        self.current_data_source_kind = "file"
        self.current_data_source_label = "当前文件"
        self.column_data: dict[str, np.ndarray] = {}
        self.non_numeric_cols: set[str] = set()
        self.unsuitable_spectrum_cols: set[str] = set()
        self.excluded_analysis_cols: set[str] = set()
        self.column_vars: dict[str, tk.BooleanVar] = {}
        self.saved_selections: dict[str, list[str]] = {}
        self.file_paths: list[Path] = []
        self.loading_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.active_load_token = 0
        self.check_queue_after_id: str | None = None
        self.task_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.active_task_token = 0
        self.check_task_after_id: str | None = None
        self.reload_after_id: str | None = None
        self.auto_analysis_after_id: str | None = None
        self.loading_in_progress = False
        self.background_task_in_progress = False
        self.page_index = 0
        self.current_plot_kind: str | None = None
        self.current_plot_columns: list[str] = []
        self.current_result_freq: np.ndarray | None = None
        self.current_result_values: np.ndarray | None = None
        self.current_result_frame: pd.DataFrame | None = None
        self.current_target_group_preview_frame: pd.DataFrame | None = None
        self.current_aligned_frame: pd.DataFrame | None = None
        self.current_lazy_aligned_frames: list[pd.DataFrame] = []
        self.current_aligned_metadata: dict[str, Any] = {}
        self.current_point_count_contract: dict[str, Any] = {}
        self.current_compare_files: list[str] = []
        self.current_plot_style_label = ""
        self.current_plot_layout_label = ""
        self.separate_plot_entries: list[dict[str, Any]] = []
        self.separate_plot_windows: list[tk.Toplevel] = []
        self.auto_prepare_payload: dict[str, Any] | None = None
        self.auto_dat_options: dict[str, Path] = {}
        self.auto_compare_context_available = False
        self.auto_compare_context_source = ""
        self.auto_compare_context_dat_file_name = ""
        self.auto_compare_context_txt_count = 0
        self.target_group_forceable_map: dict[str, bool] = {}
        self.pending_target_group_override_keys: set[str] = set()
        self.no_header_prompt_state: dict[str, str] = {}
        self.saved_read_settings: dict[str, dict[str, str | bool]] = {}
        self.target_parsed_file_cache: dict[tuple[Any, ...], ParsedFileResult] = {}
        self.last_param_error_message: str | None = None
        self.pending_direct_generate_target_spectrum_fallback_reason: str | None = None
        self.pending_selected_files_default_render_quiet = False
        self.suspend_setting_reload = False
        self.current_used_mode1_template = False
        self.current_layout_label = "未加载"

        self.delimiter_var = tk.StringVar(value="逗号 (,)")
        self.custom_delimiter_var = tk.StringVar(value="")
        self.start_row_var = tk.StringVar(value="0")
        self.header_row_var = tk.StringVar(value="0")
        self.no_header_var = tk.BooleanVar(value=False)
        self.fs_var = tk.StringVar(value=str(DEFAULT_FS))
        self.nsegment_var = tk.StringVar(value=str(DEFAULT_NSEGMENT))
        self.overlap_ratio_var = tk.StringVar(value=str(DEFAULT_OVERLAP_RATIO))
        self.cross_spectrum_type_var = tk.StringVar(value=CROSS_SPECTRUM_MAGNITUDE)
        self.reference_slope_mode_var = tk.StringVar(value=REFERENCE_SLOPE_MODE_AUTO)
        self.compare_mode_var = tk.StringVar(value="时间序列对比")
        self.mapping_mode_var = tk.StringVar(value="预设映射")
        self.element_preset_var = tk.StringVar(value="CO2")
        self.compare_scope_var = tk.StringVar(value="单对单")
        self.alignment_strategy_var = tk.StringVar(value="最近邻 + 容差")
        self.plot_style_var = tk.StringVar(value="自动")
        self.plot_layout_var = tk.StringVar(value="叠加同图")
        self.use_separate_zoom_windows_var = tk.BooleanVar(value=False)
        self.scheme_name_var = tk.StringVar(value="默认方案")
        self.time_range_strategy_var = tk.StringVar(value="使用 txt+dat 共同时间范围")
        self.time_start_var = tk.StringVar(value="")
        self.time_end_var = tk.StringVar(value="")
        self.match_tolerance_var = tk.StringVar(value="0.2")
        self.device_a_column_var = tk.StringVar(value="")
        self.device_b_column_var = tk.StringVar(value="")
        self.compare_file_info_var = tk.StringVar(value="自动准备：请先选择包含 txt/log 和 dat 的文件夹")
        self.txt_merge_summary_var = tk.StringVar(value="txt 合并摘要：等待选择 txt/log 文件")
        self.dat_summary_var = tk.StringVar(value="dat 摘要：等待选择 dat 文件")
        self.folder_prepare_summary_var = tk.StringVar(value="自动准备摘要：等待选择文件夹")
        self.selected_dat_var = tk.StringVar(value="")
        self.single_compare_style_preview_var = tk.BooleanVar(value=True)
        self.single_compare_style_preview_info_var = tk.StringVar(
            value="单设备优先复用对比图语义：等待可复用的 compare 上下文；直接生成图会优先尝试目标谱图。"
        )
        self.legacy_target_use_analysis_params_var = tk.BooleanVar(value=False)
        self.legacy_target_spectrum_mode_var = tk.StringVar(value=core.LEGACY_TARGET_SPECTRUM_MODE_PSD)
        self.legacy_target_color_mode_var = tk.StringVar(value=LEGACY_TARGET_COLOR_MODE_BY_DEVICE)
        self.target_cross_ygas_color_var = tk.StringVar(value="")
        self.target_cross_dat_color_var = tk.StringVar(value="")
        self.target_group_qc_summary_var = tk.StringVar(value="目标谱图组质控：等待生成目标谱图")
        self.preserve_selection_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.page_info_var = tk.StringVar(value="0 / 0")
        self.diagnostic_var = tk.StringVar(value="绘图诊断信息：等待加载文件")
        self.plot_title_var = tk.StringVar(value="图形结果")

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas: FigureCanvasTkAgg | None = None
        self.plot_toolbar: NavigationToolbar2Tk | None = None
        self.plot_container: ttk.Frame | None = None
        self.plot_toolbar_frame: ttk.Frame | None = None
        self.right_notebook: ttk.Notebook | None = None
        self.table_tab: ttk.Frame | None = None
        self.plot_tab: ttk.Frame | None = None
        self.left_workflow_notebook: ttk.Notebook | None = None
        self.single_analysis_tab: ttk.Frame | None = None
        self.dual_compare_tab: ttk.Frame | None = None
        self.advanced_param_tab: ttk.Frame | None = None
        self.single_compare_style_preview_check: ttk.Checkbutton | None = None
        self.single_compare_style_preview_available = False
        self.table_font = tkfont.nametofont("TkDefaultFont")

        self._build_ui()
        self._bind_events()
        self._update_delimiter_state()
        self.render_plot_message("请选择文件夹，程序会自动识别文件并准备目标谱图。")

    def _build_ui(self) -> None:
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_pane, width=360)
        right_panel = ttk.Frame(main_pane)
        main_pane.add(left_panel, weight=0)
        main_pane.add(right_panel, weight=1)

        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        left_scroll_container = ttk.Frame(parent)
        left_scroll_container.pack(fill=tk.BOTH, expand=True)

        self.left_panel_canvas = tk.Canvas(left_scroll_container, highlightthickness=0)
        self.left_panel_scrollbar = ttk.Scrollbar(
            left_scroll_container,
            orient=tk.VERTICAL,
            command=self.left_panel_canvas.yview,
        )
        self.left_panel_inner = ttk.Frame(self.left_panel_canvas)
        self.left_panel_inner.bind(
            "<Configure>",
            lambda _event: self.left_panel_canvas.configure(scrollregion=self.left_panel_canvas.bbox("all")),
        )
        self.left_panel_window = self.left_panel_canvas.create_window((0, 0), window=self.left_panel_inner, anchor="nw")
        self.left_panel_canvas.bind(
            "<Configure>",
            lambda event: self.left_panel_canvas.itemconfigure(self.left_panel_window, width=event.width),
        )
        self.left_panel_canvas.configure(yscrollcommand=self.left_panel_scrollbar.set)
        self.left_panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_panel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        parent = self.left_panel_inner

        source_button_frame = ttk.Frame(parent)
        source_button_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        folder_button = ttk.Button(source_button_frame, text="选择文件夹", command=self.async_open_directory)
        folder_button.pack(fill=tk.X, expand=True)

        ttk.Label(
            parent,
            text="使用方式：选择文件夹 → 自动识别文件 → 选择要素 → 生成目标谱图",
            foreground="#666666",
            wraplength=320,
            justify="left",
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        file_list_frame = ttk.LabelFrame(parent, text="文件列表", height=160)
        file_list_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)
        file_list_frame.pack_propagate(False)

        self.file_listbox = tk.Listbox(file_list_frame, exportselection=False, selectmode=tk.EXTENDED)
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.left_workflow_notebook = ttk.Notebook(parent)
        self.left_workflow_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.single_analysis_tab = ttk.Frame(self.left_workflow_notebook)
        self.dual_compare_tab = ttk.Frame(self.left_workflow_notebook)
        self.advanced_param_tab = ttk.Frame(self.left_workflow_notebook)
        self.left_workflow_notebook.add(self.dual_compare_tab, text="主路径")
        self.left_workflow_notebook.add(self.single_analysis_tab, text="汇总表分析")
        self.left_workflow_notebook.add(self.advanced_param_tab, text="高级功能")
        self.left_workflow_notebook.select(self.dual_compare_tab)

        single_parent = self.single_analysis_tab
        dual_parent = self.dual_compare_tab
        advanced_parent = self.advanced_param_tab

        quick_start_frame = ttk.LabelFrame(dual_parent, text="目标谱图主路径")
        quick_start_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(quick_start_frame, text="目标要素").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.quick_target_element_combo = ttk.Combobox(
            quick_start_frame,
            textvariable=self.element_preset_var,
            state="readonly",
            values=["CO2", "H2O", "温度", "压力"],
        )
        self.quick_target_element_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(quick_start_frame, text="自动选择 dat").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.selected_dat_combo = ttk.Combobox(
            quick_start_frame,
            textvariable=self.selected_dat_var,
            state="readonly",
            values=[],
        )
        self.selected_dat_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(
            quick_start_frame,
            textvariable=self.compare_file_info_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            quick_start_frame,
            textvariable=self.txt_merge_summary_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            quick_start_frame,
            textvariable=self.dat_summary_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            quick_start_frame,
            textvariable=self.folder_prepare_summary_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))

        ttk.Label(quick_start_frame, text="谱类型").grid(row=6, column=0, sticky="w", padx=6, pady=4)
        self.legacy_target_spectrum_mode_combo = ttk.Combobox(
            quick_start_frame,
            textvariable=self.legacy_target_spectrum_mode_var,
            state="readonly",
            values=LEGACY_TARGET_SPECTRUM_MODE_CHOICES,
        )
        self.legacy_target_spectrum_mode_combo.grid(row=6, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(quick_start_frame, text="颜色模式").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        self.legacy_target_color_mode_combo = ttk.Combobox(
            quick_start_frame,
            textvariable=self.legacy_target_color_mode_var,
            state="readonly",
            values=LEGACY_TARGET_COLOR_MODE_CHOICES,
        )
        self.legacy_target_color_mode_combo.grid(row=7, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(quick_start_frame, text="ygas 颜色").grid(row=8, column=0, sticky="w", padx=6, pady=4)
        ygas_color_frame = ttk.Frame(quick_start_frame)
        ygas_color_frame.grid(row=8, column=1, sticky="ew", padx=6, pady=4)
        ttk.Entry(ygas_color_frame, textvariable=self.target_cross_ygas_color_var, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            ygas_color_frame,
            text="选择",
            command=lambda: self.choose_target_cross_color("cross_ygas"),
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(quick_start_frame, text="dat 颜色").grid(row=9, column=0, sticky="w", padx=6, pady=4)
        dat_color_frame = ttk.Frame(quick_start_frame)
        dat_color_frame.grid(row=9, column=1, sticky="ew", padx=6, pady=4)
        ttk.Entry(dat_color_frame, textvariable=self.target_cross_dat_color_var, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            dat_color_frame,
            text="选择",
            command=lambda: self.choose_target_cross_color("cross_dat"),
            width=5,
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Checkbutton(
            quick_start_frame,
            text="沿用当前分析参数（默认关闭）",
            variable=self.legacy_target_use_analysis_params_var,
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        ttk.Label(
            quick_start_frame,
            text="默认按目标谱图预设自动解析 NSEGMENT，避免历史方案里的 256 覆盖当前链路。",
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).grid(row=11, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))

        quick_button_frame = ttk.Frame(quick_start_frame)
        quick_button_frame.grid(row=12, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        self.btn_generate_target_main = ttk.Button(
            quick_button_frame,
            text="生成目标谱图",
            command=self.generate_legacy_compatible_target_plot,
        )
        self.btn_generate_target_main.pack(fill=tk.X, pady=(0, 4))
        self.btn_save_main = ttk.Button(quick_button_frame, text="导出图片", command=self.save_figure)
        self.btn_save_main.pack(fill=tk.X, pady=(0, 4))
        self.btn_export_summary_main = ttk.Button(
            quick_button_frame,
            text="导出汇总表",
            command=self.export_comparison_summary,
        )
        self.btn_export_summary_main.pack(fill=tk.X)
        quick_start_frame.columnconfigure(1, weight=1)

        settings_frame = ttk.LabelFrame(single_parent, text="读取设置")
        settings_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(settings_frame, text="分隔符").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.delim_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.delimiter_var,
            state="readonly",
            values=["逗号 (,)", "制表符", "空格", "分号 (;)", "自定义"],
        )
        self.delim_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(settings_frame, text="自定义").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.custom_delim_entry = ttk.Entry(settings_frame, textvariable=self.custom_delimiter_var)
        self.custom_delim_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        spinbox_validator = (self.root.register(self.validate_spinbox), "%P")

        ttk.Label(settings_frame, text="跳过前几行").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.start_spin = tk.Spinbox(
            settings_frame,
            from_=0,
            to=999999,
            textvariable=self.start_row_var,
            validate="key",
            validatecommand=spinbox_validator,
        )
        self.start_spin.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(
            settings_frame,
            text="跳过前几行：文件前面如果有说明文字，可在这里跳过",
            foreground="#666666",
            wraplength=300,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        ttk.Label(settings_frame, text="列名所在行").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.header_spin = tk.Spinbox(
            settings_frame,
            from_=0,
            to=999999,
            textvariable=self.header_row_var,
            validate="key",
            validatecommand=spinbox_validator,
        )
        self.header_spin.grid(row=4, column=1, sticky="ew", padx=6, pady=4)

        self.no_header_check = ttk.Checkbutton(settings_frame, text="文件无表头", variable=self.no_header_var)
        self.no_header_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            settings_frame,
            text="文件无表头：勾选后表示第一行就是数据，不是列名",
            foreground="#666666",
            wraplength=300,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        ttk.Label(
            settings_frame,
            text="txt 高频数据建议使用：逗号分隔 + 无表头",
            foreground="#666666",
            wraplength=300,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))
        settings_frame.columnconfigure(1, weight=1)

        analysis_frame = ttk.LabelFrame(advanced_parent, text="分析参数")
        analysis_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(analysis_frame, text="采样频率 FS（Hz）").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.fs_entry = ttk.Entry(analysis_frame, textvariable=self.fs_var)
        self.fs_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(
            analysis_frame,
            text="表示每秒采样点数，10 表示每秒 10 个点",
            foreground="#666666",
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        ttk.Label(analysis_frame, text="每段点数 NSEGMENT").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.nsegment_spin = tk.Spinbox(
            analysis_frame,
            from_=2,
            to=65536,
            textvariable=self.nsegment_var,
            validate="key",
            validatecommand=spinbox_validator,
        )
        self.nsegment_spin.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(
            analysis_frame,
            text="每次参与频谱计算的点数，常用 256；数据较短时可改 128 或 64",
            foreground="#666666",
            wraplength=300,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        ttk.Label(analysis_frame, text="重叠比例 OVERLAP_RATIO").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.overlap_entry = ttk.Entry(analysis_frame, textvariable=self.overlap_ratio_var)
        self.overlap_entry.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(
            analysis_frame,
            text="相邻两段数据重叠比例，常用 0.5",
            foreground="#666666",
            wraplength=300,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

        ttk.Label(
            analysis_frame,
            text="当前高频 txt 数据建议：FS=10，NSEGMENT=256，OVERLAP_RATIO=0.5",
            foreground="#555555",
            wraplength=300,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 4))

        note = (
            "FS 默认 10.0 为实用补齐，"
            "不是字节级逆向已确认值。"
        )
        ttk.Label(analysis_frame, text=note, foreground="#555555", wraplength=300).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6)
        )
        analysis_frame.columnconfigure(1, weight=1)

        utility_frame = ttk.LabelFrame(advanced_parent, text="界面与文件工具")
        utility_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(utility_frame, text="选择文件", command=self.async_open_file).pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Button(utility_frame, text="恢复默认布局", command=self.restore_default_layout).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(utility_frame, text="参数说明", command=self.show_parameter_help).pack(fill=tk.X, padx=6, pady=(2, 6))

        compare_frame = ttk.LabelFrame(advanced_parent, text="高级功能：双设备与汇总表分析")
        compare_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(compare_frame, text="方案名").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.scheme_name_entry = ttk.Entry(compare_frame, textvariable=self.scheme_name_var)
        self.scheme_name_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        scheme_button_frame = ttk.Frame(compare_frame)
        scheme_button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        ttk.Button(scheme_button_frame, text="保存方案", command=self.save_compare_scheme).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(scheme_button_frame, text="加载方案", command=self.load_compare_scheme).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        mapping_frame = ttk.LabelFrame(compare_frame, text="要素映射")
        mapping_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        ttk.Label(mapping_frame, text="映射方式").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.mapping_mode_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.mapping_mode_var,
            state="readonly",
            values=["预设映射", "手动映射"],
        )
        self.mapping_mode_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(mapping_frame, text="预设要素").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.element_preset_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.element_preset_var,
            state="readonly",
            values=list(ELEMENT_PRESETS.keys()),
        )
        self.element_preset_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(mapping_frame, text="对比范围").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.compare_scope_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.compare_scope_var,
            state="readonly",
            values=["单对单", "设备A一列对设备B多列", "设备A多列对设备B多列"],
        )
        self.compare_scope_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(mapping_frame, text="对齐策略").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.alignment_strategy_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.alignment_strategy_var,
            state="readonly",
            values=["最近邻匹配", "最近邻 + 容差", "重采样后对齐"],
        )
        self.alignment_strategy_combo.grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(mapping_frame, text="出图样式").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.plot_style_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.plot_style_var,
            state="readonly",
            values=["自动", "连线图", "散点图", "连线+散点"],
        )
        self.plot_style_combo.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(mapping_frame, text="出图布局").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.plot_layout_combo = ttk.Combobox(
            mapping_frame,
            textvariable=self.plot_layout_var,
            state="readonly",
            values=["叠加同图", "上下分图", "左右分图", "分别生成两张图"],
        )
        self.plot_layout_combo.grid(row=5, column=1, sticky="ew", padx=6, pady=4)
        ttk.Checkbutton(
            mapping_frame,
            text="独立窗口仅用于放大查看",
            variable=self.use_separate_zoom_windows_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 4))
        mapping_frame.columnconfigure(1, weight=1)

        ttk.Label(compare_frame, text="比对模式").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.compare_mode_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.compare_mode_var,
            state="readonly",
            values=["时间序列对比", "散点一致性对比", "时间段内 PSD 对比", "互谱幅值", "协谱图", "正交谱图", "差值时间序列", "比值时间序列"],
        )
        self.compare_mode_combo.grid(row=3, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="时间范围策略").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.time_range_strategy_combo = ttk.Combobox(
            compare_frame,
            textvariable=self.time_range_strategy_var,
            state="readonly",
            values=["手动输入时间范围", "使用 dat 时间范围", "使用 txt+dat 共同时间范围", "最近10分钟", "最近30分钟", "最近1小时"],
        )
        self.time_range_strategy_combo.grid(row=4, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="开始时间").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.time_start_entry = ttk.Entry(compare_frame, textvariable=self.time_start_var)
        self.time_start_entry.grid(row=5, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="结束时间").grid(row=6, column=0, sticky="w", padx=6, pady=4)
        self.time_end_entry = ttk.Entry(compare_frame, textvariable=self.time_end_var)
        self.time_end_entry.grid(row=6, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="匹配容差（秒）").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        self.match_tolerance_entry = ttk.Entry(compare_frame, textvariable=self.match_tolerance_var)
        self.match_tolerance_entry.grid(row=7, column=1, sticky="ew", padx=6, pady=4)

        ttk.Button(compare_frame, text="使用当前文件时间范围", command=self.use_current_file_time_range).grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2)
        )
        ttk.Button(compare_frame, text="使用 dat 时间范围", command=self.use_selected_dat_time_range).grid(
            row=9, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2)
        )
        ttk.Button(compare_frame, text="使用所有选中文件共同时间范围", command=self.use_common_selected_time_range).grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2)
        )
        ttk.Button(compare_frame, text="自动补齐txt覆盖范围", command=self.auto_fill_txt_covering_dat_range).grid(
            row=11, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2)
        )
        time_shortcut_frame = ttk.Frame(compare_frame)
        time_shortcut_frame.grid(row=12, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        quick_button_frame = ttk.Frame(time_shortcut_frame)
        quick_button_frame.pack(fill=tk.X)
        ttk.Button(quick_button_frame, text="最近10分钟", command=lambda: self.apply_recent_time_range(10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(quick_button_frame, text="最近30分钟", command=lambda: self.apply_recent_time_range(30)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(quick_button_frame, text="最近1小时", command=lambda: self.apply_recent_time_range(60)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(
            time_shortcut_frame,
            text="将当前 compare 共同时间窗同步到单图分析",
            command=self.sync_compare_common_time_range_to_single_analysis,
        ).pack(fill=tk.X, pady=(4, 0))

        ttk.Label(compare_frame, text="设备A分析列").grid(row=13, column=0, sticky="w", padx=6, pady=4)
        self.device_a_combo = ttk.Combobox(compare_frame, textvariable=self.device_a_column_var, state="readonly")
        self.device_a_combo.grid(row=13, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="设备A多列（可选）").grid(row=14, column=0, sticky="nw", padx=6, pady=4)
        a_multi_frame = ttk.Frame(compare_frame)
        a_multi_frame.grid(row=14, column=1, sticky="ew", padx=6, pady=4)
        self.device_a_multi_listbox = tk.Listbox(a_multi_frame, exportselection=False, selectmode=tk.EXTENDED, height=4)
        a_multi_scroll = ttk.Scrollbar(a_multi_frame, orient=tk.VERTICAL, command=self.device_a_multi_listbox.yview)
        self.device_a_multi_listbox.configure(yscrollcommand=a_multi_scroll.set)
        self.device_a_multi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        a_multi_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(compare_frame, text="设备B分析列").grid(row=15, column=0, sticky="w", padx=6, pady=4)
        self.device_b_combo = ttk.Combobox(compare_frame, textvariable=self.device_b_column_var, state="readonly")
        self.device_b_combo.grid(row=15, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(compare_frame, text="设备B多列（可选）").grid(row=16, column=0, sticky="nw", padx=6, pady=4)
        b_multi_frame = ttk.Frame(compare_frame)
        b_multi_frame.grid(row=16, column=1, sticky="ew", padx=6, pady=4)
        self.device_b_multi_listbox = tk.Listbox(b_multi_frame, exportselection=False, selectmode=tk.EXTENDED, height=5)
        b_multi_scroll = ttk.Scrollbar(b_multi_frame, orient=tk.VERTICAL, command=self.device_b_multi_listbox.yview)
        self.device_b_multi_listbox.configure(yscrollcommand=b_multi_scroll.set)
        self.device_b_multi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        b_multi_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            compare_frame,
            textvariable=self.compare_file_info_var,
            foreground="#666666",
            wraplength=300,
        ).grid(row=17, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            compare_frame,
            textvariable=self.txt_merge_summary_var,
            foreground="#666666",
            wraplength=300,
        ).grid(row=18, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 2))
        ttk.Label(
            compare_frame,
            textvariable=self.dat_summary_var,
            foreground="#666666",
            wraplength=300,
        ).grid(row=19, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
        ttk.Label(
            compare_frame,
            text="高级路径：如需互谱/协谱/正交谱，请先生成对比汇总表，再载入汇总表进行分析。",
            foreground="#666666",
            wraplength=300,
        ).grid(row=20, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        ttk.Button(compare_frame, text="生成对比汇总表", command=self.generate_comparison_summary).grid(
            row=21, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 2)
        )
        ttk.Button(compare_frame, text="生成目标谱图", command=self.generate_legacy_compatible_target_plot).grid(
            row=22, column=0, columnspan=2, sticky="ew", padx=6, pady=2
        )
        ttk.Button(compare_frame, text="使用汇总表进行分析", command=self.use_comparison_summary_for_analysis).grid(
            row=23, column=0, columnspan=2, sticky="ew", padx=6, pady=2
        )
        ttk.Button(compare_frame, text="导出汇总表", command=self.export_comparison_summary).grid(
            row=24, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 6)
        )
        compare_frame.columnconfigure(1, weight=1)

        advanced_action_frame = ttk.LabelFrame(advanced_parent, text="更多高级分析")
        advanced_action_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(advanced_action_frame, text="双设备直接出图", command=self.run_dual_device_compare).pack(
            fill=tk.X, padx=6, pady=(6, 2)
        )
        ttk.Button(advanced_action_frame, text="按设备生成图谱", command=self.perform_multi_spectral_compare).pack(
            fill=tk.X, padx=6, pady=2
        )
        ttk.Button(advanced_action_frame, text="导出对齐数据", command=self.export_aligned_data).pack(
            fill=tk.X, padx=6, pady=(2, 6)
        )

        target_qc_frame = ttk.LabelFrame(advanced_parent, text="目标谱图组级质控")
        target_qc_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 4))
        ttk.Label(
            target_qc_frame,
            text="默认会自动跳过覆盖不足、频点不足、时间窗过短或谱值明显离群的组。\n如需手动放开，请选中下面的组后重新点击“生成目标谱图”。",
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).pack(fill=tk.X, padx=6, pady=(6, 4))
        self.target_group_override_listbox = tk.Listbox(
            target_qc_frame,
            exportselection=False,
            selectmode=tk.EXTENDED,
            height=6,
        )
        target_qc_scroll = ttk.Scrollbar(target_qc_frame, orient=tk.VERTICAL, command=self.target_group_override_listbox.yview)
        self.target_group_override_listbox.configure(yscrollcommand=target_qc_scroll.set)
        self.target_group_override_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=(0, 4))
        target_qc_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=(0, 4))
        ttk.Label(
            target_qc_frame,
            textvariable=self.target_group_qc_summary_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        actions_frame = ttk.LabelFrame(single_parent, text="操作")
        actions_frame.pack(fill=tk.X, padx=8, pady=4)

        self.preserve_button = ttk.Checkbutton(
            actions_frame,
            text="切换文件时保留已选列",
            variable=self.preserve_selection_var,
        )
        self.preserve_button.pack(anchor="w", padx=6, pady=(6, 2))

        self.single_compare_style_preview_check = ttk.Checkbutton(
            actions_frame,
            text="单设备优先复用对比图语义（自动）",
            variable=self.single_compare_style_preview_var,
            state="disabled",
        )
        self.single_compare_style_preview_check.pack(anchor="w", padx=6, pady=(0, 2))
        ttk.Label(
            actions_frame,
            textvariable=self.single_compare_style_preview_info_var,
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).pack(fill=tk.X, padx=6, pady=(0, 4))

        self.btn_spectral = ttk.Button(actions_frame, text="单列功率谱密度", command=lambda: self.perform_analysis("spectral"))
        self.btn_spectral.pack(fill=tk.X, padx=6, pady=2)

        self.btn_cross = ttk.Button(actions_frame, text="双列互谱分析", command=lambda: self.perform_analysis("cross"))
        self.btn_cross.pack(fill=tk.X, padx=6, pady=2)

        ttk.Label(actions_frame, text="谱类型").pack(anchor="w", padx=6, pady=(4, 0))
        self.cross_spectrum_type_combo = ttk.Combobox(
            actions_frame,
            textvariable=self.cross_spectrum_type_var,
            state="readonly",
            values=CROSS_SPECTRUM_OPTIONS,
        )
        self.cross_spectrum_type_combo.pack(fill=tk.X, padx=6, pady=(0, 2))

        ttk.Label(actions_frame, text="参考线").pack(anchor="w", padx=6, pady=(4, 0))
        self.reference_slope_mode_combo = ttk.Combobox(
            actions_frame,
            textvariable=self.reference_slope_mode_var,
            state="readonly",
            values=REFERENCE_SLOPE_MODE_OPTIONS,
        )
        self.reference_slope_mode_combo.pack(fill=tk.X, padx=6, pady=(0, 2))

        self.btn_generate = ttk.Button(actions_frame, text="生成图", command=self.generate_plot)
        self.btn_generate.pack(fill=tk.X, padx=6, pady=2)

        self.btn_save = ttk.Button(actions_frame, text="保存图片", command=self.save_figure)
        self.btn_save.pack(fill=tk.X, padx=6, pady=2)

        self.btn_export = ttk.Button(actions_frame, text="导出结果", command=self.export_results)
        self.btn_export.pack(fill=tk.X, padx=6, pady=(2, 6))

        column_frame = ttk.LabelFrame(single_parent, text="可分析列", height=320)
        column_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(4, 8))
        column_frame.pack_propagate(False)
        self.column_frame = column_frame

        self.column_canvas = tk.Canvas(column_frame, highlightthickness=0)
        self.column_scrollbar = ttk.Scrollbar(column_frame, orient=tk.VERTICAL, command=self.column_canvas.yview)
        self.column_inner = ttk.Frame(self.column_canvas)
        self.column_inner.bind(
            "<Configure>",
            lambda _event: self.column_canvas.configure(scrollregion=self.column_canvas.bbox("all")),
        )
        self.column_canvas_window = self.column_canvas.create_window((0, 0), window=self.column_inner, anchor="nw")
        self.column_canvas.configure(yscrollcommand=self.column_scrollbar.set)
        self.column_canvas.bind("<Configure>", self.on_column_canvas_configure)
        self.column_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.column_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(self.column_inner, text="加载文件后，在这里勾选要分析的列", foreground="#666666").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        self.right_notebook = ttk.Notebook(parent)
        self.right_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.table_tab = ttk.Frame(self.right_notebook)
        self.plot_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.table_tab, text="表格预览")
        self.right_notebook.add(self.plot_tab, text="图形结果")
        self.right_notebook.select(self.table_tab)

        table_frame = ttk.Frame(self.table_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)

        nav_frame = ttk.Frame(table_frame)
        nav_frame.pack(fill=tk.X, padx=6, pady=6)

        self.prev_page_button = ttk.Button(nav_frame, text="上一页", command=self.prev_page)
        self.prev_page_button.pack(side=tk.LEFT)

        self.next_page_button = ttk.Button(nav_frame, text="下一页", command=self.next_page)
        self.next_page_button.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(nav_frame, textvariable=self.page_info_var).pack(side=tk.LEFT, padx=12)

        table_container = ttk.Frame(table_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.tree = ttk.Treeview(table_container, show="headings")
        tree_y = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        tree_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_x.pack(side=tk.BOTTOM, fill=tk.X)

        plot_frame = ttk.Frame(self.plot_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            plot_frame,
            textvariable=self.plot_title_var,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(8, 4))
        self.plot_toolbar_frame = ttk.Frame(plot_frame)
        self.plot_toolbar_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.plot_container = ttk.Frame(plot_frame)
        self.plot_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_toolbar_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.pack(side=tk.LEFT, fill=tk.X)
        ttk.Label(
            plot_frame,
            textvariable=self.diagnostic_var,
            foreground="#555555",
            wraplength=900,
            justify="left",
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

    def show_plot_tab(self) -> None:
        if self.right_notebook is not None and self.plot_tab is not None:
            self.right_notebook.select(self.plot_tab)

    def show_table_tab(self) -> None:
        if self.right_notebook is not None and self.table_tab is not None:
            self.right_notebook.select(self.table_tab)

    def update_plot_title(self, title: str | None = None) -> None:
        if title:
            self.plot_title_var.set(f"图形结果：{title}")
            return
        for axis in self.figure.axes:
            axis_title = axis.get_title().strip()
            if axis_title:
                self.plot_title_var.set(f"图形结果：{axis_title}")
                return
        self.plot_title_var.set("图形结果")

    def refresh_canvas(self, *, switch_to_plot: bool = True, title: str | None = None) -> None:
        if self.canvas is None:
            return
        self.update_plot_title(title)
        self.canvas.draw_idle()
        self.root.update_idletasks()
        if switch_to_plot:
            self.show_plot_tab()

    def restore_default_layout(self) -> None:
        if self.left_workflow_notebook is not None and self.dual_compare_tab is not None:
            self.left_workflow_notebook.select(self.dual_compare_tab)
        self.show_table_tab()
        self.use_separate_zoom_windows_var.set(False)
        self.clear_separate_plot_windows()
        if self.plot_toolbar is not None:
            try:
                self.plot_toolbar.home()
            except Exception:
                pass
        self.status_var.set("已恢复默认布局")
        self.update_plot_title(None)
        self.root.update_idletasks()

    def _bind_events(self) -> None:
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        self.delim_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_settings())
        self.selected_dat_combo.bind("<<ComboboxSelected>>", self.on_selected_dat_changed)
        self.column_canvas.bind("<Enter>", self.bind_column_mousewheel)
        self.column_canvas.bind("<Leave>", self.unbind_column_mousewheel)
        self.column_inner.bind("<Enter>", self.bind_column_mousewheel)
        self.column_inner.bind("<Leave>", self.unbind_column_mousewheel)

        for variable in (
            self.custom_delimiter_var,
            self.start_row_var,
            self.header_row_var,
            self.no_header_var,
        ):
            variable.trace_add("write", lambda *_args: self.update_settings())

        for variable in (
            self.fs_var,
            self.nsegment_var,
            self.overlap_ratio_var,
            self.cross_spectrum_type_var,
            self.reference_slope_mode_var,
            self.single_compare_style_preview_var,
        ):
            variable.trace_add("write", lambda *_args: self.schedule_auto_analysis())

        for variable in (self.mapping_mode_var, self.element_preset_var, self.compare_scope_var):
            variable.trace_add("write", lambda *_args: self.refresh_compare_column_options())

    def on_column_canvas_configure(self, event: tk.Event[Any]) -> None:
        self.column_canvas.itemconfigure(self.column_canvas_window, width=event.width)

    def bind_column_mousewheel(self, _event: tk.Event[Any] | None = None) -> None:
        self.root.bind_all("<MouseWheel>", self.on_column_mousewheel)
        self.root.bind_all("<Button-4>", self.on_column_mousewheel)
        self.root.bind_all("<Button-5>", self.on_column_mousewheel)

    def unbind_column_mousewheel(self, _event: tk.Event[Any] | None = None) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def on_column_mousewheel(self, event: tk.Event[Any]) -> None:
        if getattr(event, "num", None) == 4:
            self.column_canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            self.column_canvas.yview_scroll(1, "units")
            return
        delta = getattr(event, "delta", 0)
        if delta:
            self.column_canvas.yview_scroll(int(-delta / 120), "units")

    def validate_spinbox(self, proposed: str) -> bool:
        return proposed == "" or proposed.isdigit()

    def show_parameter_help(self) -> None:
        help_text = (
            "分隔符：选择文件中各列之间的分隔方式，常见是逗号。\n\n"
            "跳过前几行：如果文件开头有说明文字、备注或设备信息，可以在这里跳过。\n\n"
            "文件无表头：勾选后表示第一行就是数据，不是列名。\n\n"
            "列名所在行：如果文件里有表头，可以填写列名所在的那一行编号。\n\n"
            "采样频率 FS（Hz）：表示每秒采样点数，10 表示每秒 10 个点。\n\n"
            "每段点数 NSEGMENT：每次参与频谱计算的数据点数，常用 256；数据较短时可改 128 或 64。\n\n"
            "重叠比例 OVERLAP_RATIO：相邻两段数据的重叠比例，常用 0.5。\n\n"
            "谱类型：双列分析时可在互谱幅值、协谱和正交谱之间切换。\n\n"
            "参考线：可选自动、仅 -2/3、仅 -4/3、两条都显示或不显示。\n\n"
            "单列功率谱密度：选中 1 列数值数据后，计算该列的频谱强度分布。\n\n"
            "双列互谱分析：选中 2 列数值数据后，分析两列在不同频率上的共同变化强度。"
        )
        messagebox.showinfo("参数说明", help_text)

    def parse_time_input(self, raw: str) -> pd.Timestamp | None:
        text = raw.strip()
        if not text:
            return None
        parsed_series = parse_mixed_timestamp_series(pd.Series([text]))
        parsed = parsed_series.iloc[0]
        if pd.isna(parsed):
            raise ValueError("时间格式错误，请使用 YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD HH:MM:SS.s")
        return pd.Timestamp(parsed)

    def resolve_time_range_inputs(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        start_dt = self.parse_time_input(self.time_start_var.get())
        end_dt = self.parse_time_input(self.time_end_var.get())
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("开始时间不能晚于结束时间。")
        return start_dt, end_dt

    def resolve_optional_time_range_inputs(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not self.time_start_var.get().strip() and not self.time_end_var.get().strip():
            return None, None
        return self.resolve_time_range_inputs()

    def format_metadata_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, pd.Timestamp):
            return core.format_metadata_timestamp(value)
        if isinstance(value, (list, tuple, set)):
            parts = [self.format_metadata_value(item) for item in value]
            return " | ".join(part for part in parts if part)
        return str(value)

    def attach_metadata_columns(self, frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        enriched = frame.copy()
        for key, value in metadata.items():
            if value is None or key in enriched.columns:
                continue
            enriched[key] = [self.format_metadata_value(value)] * len(enriched)
        return enriched

    def guess_timestamp_column(self, df: pd.DataFrame) -> str | None:
        preferred_names = ("时间戳", "TIMESTAMP", "timestamp", "Timestamp", "time", "Time", "日期时间")
        for name in preferred_names:
            if name in df.columns:
                parsed = parse_mixed_timestamp_series(df[name])
                if parsed.notna().sum() >= 3:
                    return name

        for col in df.columns[: min(len(df.columns), 3)]:
            parsed = parse_mixed_timestamp_series(df[col])
            if parsed.notna().sum() >= max(3, int(len(df) * 0.6)):
                return str(col)
        return None

    def get_profile_excluded_columns(
        self,
        profile_name: str,
        df: pd.DataFrame,
        timestamp_col: str | None,
    ) -> set[str]:
        return core.get_profile_excluded_columns(profile_name, df, timestamp_col)

    def prioritize_suggested_columns(self, columns: list[str]) -> list[str]:
        priority = ["CO2浓度", "H2O浓度"]
        ordered = [name for name in priority if name in columns]
        ordered.extend(name for name in columns if name not in ordered)
        return ordered

    def build_parsed_result_from_dataframe(
        self,
        df: pd.DataFrame,
        profile_name: str,
        timestamp_col: str | None,
        *,
        source_row_count: int | None = None,
    ) -> ParsedFileResult:
        return core.build_parsed_result_from_dataframe(
            df,
            profile_name,
            timestamp_col,
            source_row_count=source_row_count,
        )

    def parse_ygas_mode1_file(self, path: Path) -> ParsedFileResult:
        return core.parse_ygas_mode1_file(path)

    def parse_toa5_file(self, path: Path) -> ParsedFileResult:
        return core.parse_toa5_file(path)

    def parse_dat_file(self, path: Path) -> ParsedFileResult:
        return self.parse_toa5_file(path)

    def get_parsed_time_bounds(self, parsed: ParsedFileResult) -> tuple[pd.Timestamp, pd.Timestamp, int]:
        return core.get_parsed_time_bounds(parsed)

    def load_and_merge_ygas_files(self, paths: list[Path]) -> tuple[ParsedFileResult, dict[str, Any]]:
        return core.load_and_merge_ygas_files(paths)

    def classify_selected_compare_files(self) -> tuple[list[Path], Path | None, list[Path]]:
        selection_info = self.extract_single_device_txt_selection(self.get_selected_file_paths())
        ygas_paths = list(selection_info["selected_txt_paths"])
        dat_paths = list(selection_info["selected_dat_paths"])
        other_paths = list(selection_info["other_paths"])
        dat_path = dat_paths[0] if dat_paths else None
        if len(dat_paths) > 1:
            other_paths.extend(dat_paths[1:])
        return ygas_paths, dat_path, other_paths

    def extract_single_device_txt_selection(self, selected_files: list[Path]) -> dict[str, Any]:
        normalized_selection = [Path(path) for path in selected_files]
        selected_txt_paths: list[Path] = []
        selected_dat_paths: list[Path] = []
        other_paths: list[Path] = []

        for path in normalized_selection:
            profile = detect_file_profile(path, read_preview_lines(path))
            if profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"} or path.suffix.lower() in TEXT_LIKE_SUFFIXES:
                selected_txt_paths.append(path)
            elif profile == "TOA5_DAT" or path.suffix.lower() == ".dat":
                selected_dat_paths.append(path)
            else:
                other_paths.append(path)

        return {
            "selected_paths": normalized_selection,
            "selected_txt_paths": selected_txt_paths,
            "selected_dat_paths": selected_dat_paths,
            "other_paths": other_paths,
            "selected_file_count": int(len(normalized_selection)),
            "selected_txt_file_count": int(len(selected_txt_paths)),
            "selected_dat_file_count": int(len(selected_dat_paths)),
            "selected_other_file_count": int(len(other_paths)),
            "single_device_selection_filtered_to_txt_side": bool(
                selected_txt_paths and (selected_dat_paths or other_paths)
            ),
        }

    def ensure_legacy_target_selection_valid(
        self,
        ygas_paths: list[Path],
        dat_path: Path | None,
        other_paths: list[Path],
    ) -> None:
        if dat_path is None:
            raise ValueError("请先选择 1 个 dat 文件，再生成目标谱图。")
        if not ygas_paths:
            raise ValueError("请先选择 1 个或多个 ygas txt/log 文件，再生成目标谱图。")

        extra_dat_paths: list[Path] = []
        for path in other_paths:
            profile = detect_file_profile(path, read_preview_lines(path))
            if profile == "TOA5_DAT" or path.suffix.lower() == ".dat":
                extra_dat_paths.append(path)
        if extra_dat_paths:
            names = "、".join(path.name for path in extra_dat_paths[:3])
            suffix = " 等" if len(extra_dat_paths) > 3 else ""
            raise ValueError(
                f"目标谱图只支持 1 个 dat 文件，请取消多余 dat 选择：{names}{suffix}"
            )

    def validate_legacy_target_selection(
        self,
        ygas_paths: list[Path],
        dat_path: Path | None,
        other_paths: list[Path],
    ) -> None:
        self.ensure_legacy_target_selection_valid(ygas_paths, dat_path, other_paths)

    def build_compare_selection_from_paths(
        self,
        ygas_paths: list[Path],
        dat_path: Path,
        *,
        required_ygas_columns: list[str] | None = None,
        required_dat_columns: list[str] | None = None,
        target_context: dict[str, Any] | None = None,
        reporter: Any | None = None,
    ) -> tuple[ParsedFileResult, str, ParsedFileResult, str, dict[str, Any]]:
        if reporter is not None:
            reporter("正在解析设备A文件…")
        if required_ygas_columns:
            merged_ygas, txt_summary = core.load_and_merge_ygas_files_fast(ygas_paths, required_columns=required_ygas_columns)
        else:
            merged_ygas, txt_summary = self.load_and_merge_ygas_files(ygas_paths)
        if reporter is not None:
            reporter("正在解析设备B文件…")
        if required_dat_columns:
            parsed_dat = self.parse_target_profiled_file(dat_path, device_kind="dat", required_columns=required_dat_columns)
        else:
            parsed_dat = self.parse_dat_file(dat_path)
        dat_start, dat_end, dat_points = self.get_parsed_time_bounds(parsed_dat)
        dat_summary = {
            "start": dat_start,
            "end": dat_end,
            "total_points": dat_points,
            "raw_rows": int(parsed_dat.source_row_count or len(parsed_dat.dataframe)),
            "valid_timestamp_points": int(parsed_dat.timestamp_valid_count),
            "file_name": dat_path.name,
        }
        return merged_ygas, f"txt合并({len(ygas_paths)}个文件)", parsed_dat, dat_path.name, {
            "txt_paths": ygas_paths,
            "dat_path": dat_path,
            "txt_summary": txt_summary,
            "dat_summary": dat_summary,
            "target_spectral_context": dict(target_context) if target_context else None,
        }

    def build_compare_selection(self) -> tuple[ParsedFileResult, str, ParsedFileResult, str, dict[str, Any]]:
        ygas_paths, dat_path, _other = self.classify_selected_compare_files()
        if dat_path is None:
            raise ValueError("请先选择 1 个 dat 文件作为设备B。")
        if not ygas_paths:
            raise ValueError("请先选择 1 个或多个 txt/log 文件作为设备A。")
        return self.build_compare_selection_from_paths(ygas_paths, dat_path)

    def parse_generic_profiled_file(self, path: Path) -> ParsedFileResult:
        result = self.load_file_with_file_settings(path)
        df = result.dataframe.copy()
        preview_lines = read_preview_lines(path)
        detected_profile = detect_file_profile(path, preview_lines)
        if result.mode1_layout and result.mode1_layout.get("matched"):
            detected_profile = "YGAS_MODE1_16" if result.mode1_layout.get("variant") == "mode1-16-with-index" else "YGAS_MODE1_15"
        timestamp_col = self.guess_timestamp_column(df)
        return self.build_parsed_result_from_dataframe(df, detected_profile, timestamp_col)

    def parse_profiled_file(self, path: Path) -> ParsedFileResult:
        preview_lines = read_preview_lines(path)
        profile = detect_file_profile(path, preview_lines)
        if str(path) in self.saved_read_settings:
            generic = self.parse_generic_profiled_file(path)
            if generic.timestamp_col is not None and generic.suggested_columns:
                return generic
            if profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
                return self.parse_ygas_mode1_file(path)
            if profile == "TOA5_DAT":
                return self.parse_toa5_file(path)
            return generic

        if profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
            return self.parse_ygas_mode1_file(path)
        if profile == "TOA5_DAT":
            return self.parse_toa5_file(path)
        return self.parse_generic_profiled_file(path)

    def is_target_cross_compare_mode(self, compare_mode: str | None) -> bool:
        return str(compare_mode or "").strip() in {"互谱幅值", "协谱图", "正交谱图"}

    def build_target_required_columns(
        self,
        *,
        target_element: str,
        device_kind: str,
        include_reference: bool = False,
    ) -> list[str]:
        device_key = "A" if device_kind == "ygas" else "B"
        ordered: list[str] = []
        for value in ELEMENT_PRESETS.get(target_element, {}).get(device_key, []):
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)
        if include_reference and device_kind == "dat":
            for value in ["Uz", "uz", "W", "w", "通道3"]:
                if value not in ordered:
                    ordered.append(value)
        return ordered

    def resolve_target_spectral_context_for_parsed(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
        *,
        spectrum_mode: str | None,
        comparison_is_target_spectral: bool,
    ) -> dict[str, Any] | None:
        target_element = self.resolve_target_element_name()
        ygas_target_column = self.select_target_element_column(parsed_a, device_key="A", target_element=target_element)
        dat_target_column = self.select_target_element_column(parsed_b, device_key="B", target_element=target_element)
        reference_column = core.select_reference_uz_column(parsed_b, device_key="B")
        if not ygas_target_column or not dat_target_column or not reference_column:
            return None
        return core.build_target_spectral_context(
            target_element=target_element,
            reference_column=reference_column,
            ygas_target_column=ygas_target_column,
            dat_target_column=dat_target_column,
            display_target_label=target_element,
            spectrum_mode=spectrum_mode,
            comparison_is_target_spectral=comparison_is_target_spectral,
        )

    def get_current_target_spectral_context(self) -> dict[str, Any] | None:
        if self.current_data_source_kind in {"comparison_analysis", "comparison_preview"}:
            return core.get_target_spectral_context(self.current_comparison_metadata)
        return core.get_target_spectral_context(self.current_target_plot_metadata)

    def build_target_parsed_file_cache_key(
        self,
        path: Path,
        *,
        cache_scope: str,
        required_columns: list[str] | None = None,
    ) -> tuple[Any, ...]:
        stat = path.stat()
        setting_payload = self.saved_read_settings.get(str(path))
        setting_signature = json.dumps(setting_payload, sort_keys=True, ensure_ascii=False) if setting_payload else ""
        required_signature = json.dumps(sorted({str(value).strip() for value in (required_columns or []) if str(value).strip()}), ensure_ascii=False)
        return (
            cache_scope,
            str(path.resolve()),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            setting_signature,
            required_signature,
        )

    def parse_target_profiled_file(
        self,
        path: Path,
        *,
        device_kind: str,
        required_columns: list[str] | None = None,
    ) -> ParsedFileResult:
        cache_key = self.build_target_parsed_file_cache_key(
            path,
            cache_scope=f"target:{device_kind}",
            required_columns=required_columns,
        )
        cached = self.target_parsed_file_cache.get(cache_key)
        if cached is not None:
            return cached

        if str(path) in self.saved_read_settings:
            result = self.parse_profiled_file(path)
            self.target_parsed_file_cache[cache_key] = result
            return result

        preview_lines = read_preview_lines(path)
        profile = detect_file_profile(path, preview_lines)
        if profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
            result = core.parse_ygas_mode1_file_fast(path, required_columns=required_columns)
        elif profile == "TOA5_DAT":
            result = core.parse_toa5_file_fast(path, required_columns=required_columns)
        else:
            result = self.parse_profiled_file(path)
        self.target_parsed_file_cache[cache_key] = result
        return result

    def get_compare_file_paths(self) -> list[Path]:
        selected_files = self.get_selected_file_paths()
        return selected_files[:2]

    def get_listbox_selected_values(self, listbox: tk.Listbox) -> list[str]:
        return [str(listbox.get(index)) for index in listbox.curselection()]

    def get_selected_target_group_overrides(self) -> set[str]:
        selected_keys: set[str] = set(getattr(self, "pending_target_group_override_keys", set()))
        if not hasattr(self, "target_group_override_listbox") or self.target_group_override_listbox is None:
            return selected_keys
        for index in self.target_group_override_listbox.curselection():
            group_key = str(self.target_group_override_listbox.get(index)).split(" | ", 1)[0]
            if self.target_group_forceable_map.get(group_key, False):
                selected_keys.add(group_key)
        return selected_keys

    def update_target_group_qc_panel(self, target_metadata: dict[str, Any] | None = None) -> None:
        if not hasattr(self, "target_group_override_listbox") or self.target_group_override_listbox is None:
            return

        current_selection = self.get_selected_target_group_overrides()
        self.pending_target_group_override_keys = set(current_selection)
        self.target_group_override_listbox.delete(0, tk.END)
        self.target_group_forceable_map = {}
        if not target_metadata:
            self.pending_target_group_override_keys = set()
            self.target_group_qc_summary_var.set("目标谱图组质控：等待生成目标谱图")
            return

        group_records = list(target_metadata.get("group_records", []))
        for record in group_records:
            forceable = bool(record.get("forceable"))
            group_key = str(record.get("group_key", record.get("window_label", "")))
            group_label = str(record.get("group_label", group_key))
            status = str(record.get("status", "未知"))
            reason = str(record.get("reason", ""))
            manual_hint = " | 可手动保留" if forceable and status == "跳过" else ""
            display = f"{group_key} | {group_label} | {status} | {reason}{manual_hint}"
            self.target_group_override_listbox.insert(tk.END, display)
            self.target_group_forceable_map[group_key] = forceable
        for index, record in enumerate(group_records):
            if str(record.get("group_key", "")) in current_selection and bool(record.get("forceable")):
                self.target_group_override_listbox.selection_set(index)
        self.pending_target_group_override_keys = {
            str(record.get("group_key", ""))
            for record in group_records
            if str(record.get("group_key", "")) in current_selection and bool(record.get("forceable"))
        }

        kept = sum(1 for record in group_records if record.get("status") in {"保留", "手动保留"})
        skipped = sum(1 for record in group_records if record.get("status") == "跳过")
        forceable_count = sum(1 for record in group_records if bool(record.get("forceable")))
        effective_nsegment_values = sorted(
            {
                int(value)
                for record in group_records
                for value in (record.get("ygas_effective_nsegment"), record.get("dat_effective_nsegment"))
                if value is not None
            }
        )
        first_positive_freq_values = [
            float(value)
            for record in group_records
            for value in (record.get("ygas_first_positive_freq"), record.get("dat_first_positive_freq"))
            if value is not None
        ]
        psd_kernel_values = sorted(
            {
                str(value)
                for record in group_records
                for value in (record.get("ygas_psd_kernel"), record.get("dat_psd_kernel"))
                if value
            }
        )
        summary = f"目标谱图组质控：保留 {kept} 组 | 跳过 {skipped} 组 | 可手动放开 {forceable_count} 组"
        if effective_nsegment_values:
            summary += f" | effective_nsegment={ '/'.join(str(value) for value in effective_nsegment_values) }"
        if first_positive_freq_values:
            summary += f" | first_positive_freq≈{min(first_positive_freq_values):.12g} Hz"
        if psd_kernel_values:
            summary += f" | psd_kernel={ '/'.join(psd_kernel_values) }"
        self.target_group_qc_summary_var.set(summary)

    def set_listbox_values(self, listbox: tk.Listbox, columns: list[str], selected_values: list[str] | None = None) -> None:
        selected_values = selected_values or []
        listbox.delete(0, tk.END)
        for column in columns:
            listbox.insert(tk.END, column)
        for index, column in enumerate(columns):
            if column in selected_values:
                listbox.selection_set(index)

    def select_first_matching_column(self, columns: list[str], candidates: list[str]) -> str | None:
        lowered = {str(column).lower(): str(column) for column in columns}
        for candidate in candidates:
            direct = lowered.get(candidate.lower())
            if direct:
                return direct
        return None

    def apply_element_mapping(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
    ) -> tuple[str | None, str | None, str]:
        hit_a, hit_b, mapping_name = self.resolve_compare_mapping_hits(parsed_a, parsed_b)
        if hit_a:
            self.device_a_column_var.set(hit_a)
        if hit_b:
            self.device_b_column_var.set(hit_b)

        if not self.device_a_column_var.get() and parsed_a.suggested_columns:
            self.device_a_column_var.set(parsed_a.suggested_columns[0])
        if not self.device_b_column_var.get() and parsed_b.suggested_columns:
            self.device_b_column_var.set(parsed_b.suggested_columns[0])

        preferred_a = [self.device_a_column_var.get()] if self.device_a_column_var.get() else []
        preferred_b = [self.device_b_column_var.get()] if self.device_b_column_var.get() else []
        self.set_listbox_values(self.device_a_multi_listbox, parsed_a.suggested_columns, preferred_a)
        self.set_listbox_values(self.device_b_multi_listbox, parsed_b.suggested_columns, preferred_b)
        return hit_a, hit_b, mapping_name

    def resolve_compare_mapping_hits(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
    ) -> tuple[str | None, str | None, str]:
        mapping_mode = self.mapping_mode_var.get().strip() or "手动映射"
        mapping_name = "手动映射"
        hit_a: str | None = None
        hit_b: str | None = None
        if mapping_mode == "预设映射":
            mapping_name = self.element_preset_var.get().strip() or "CO2"
            preset = ELEMENT_PRESETS.get(mapping_name, {})
            hit_a = self.select_first_matching_column(parsed_a.suggested_columns, preset.get("A", []))
            hit_b = self.select_first_matching_column(parsed_b.suggested_columns, preset.get("B", []))
        return hit_a, hit_b, mapping_name

    def build_compare_pairs(self) -> list[dict[str, str]]:
        a_single = self.device_a_column_var.get().strip()
        b_single = self.device_b_column_var.get().strip()
        a_values = self.get_listbox_selected_values(self.device_a_multi_listbox) or ([a_single] if a_single else [])
        b_values = self.get_listbox_selected_values(self.device_b_multi_listbox) or ([b_single] if b_single else [])
        scope = self.compare_scope_var.get().strip() or "单对单"

        if scope == "单对单":
            if not a_single or not b_single:
                return []
            return [{"a_col": a_single, "b_col": b_single, "label": f"{a_single} vs {b_single}"}]

        if scope == "设备A一列对设备B多列":
            if not a_single or not b_values:
                return []
            return [{"a_col": a_single, "b_col": b_col, "label": f"{a_single} vs {b_col}"} for b_col in b_values]

        if not a_values or not b_values:
            return []
        if len(a_values) == len(b_values):
            return [{"a_col": a_col, "b_col": b_col, "label": f"{a_col} vs {b_col}"} for a_col, b_col in zip(a_values, b_values)]
        if len(a_values) == 1:
            return [{"a_col": a_values[0], "b_col": b_col, "label": f"{a_values[0]} vs {b_col}"} for b_col in b_values]
        if len(b_values) == 1:
            return [{"a_col": a_col, "b_col": b_values[0], "label": f"{a_col} vs {b_values[0]}"} for a_col in a_values]
        return [{"a_col": a_col, "b_col": b_col, "label": f"{a_col} vs {b_col}"} for a_col, b_col in zip(a_values, b_values)]

    def resolve_compare_pair_for_target_column(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
        target_column: str,
    ) -> dict[str, str] | None:
        normalized_target = str(target_column).strip()
        if not normalized_target:
            return None

        relevant_pairs = [
            dict(pair)
            for pair in self.build_compare_pairs()
            if str(pair.get("a_col") or "").strip() == normalized_target
        ]
        if relevant_pairs:
            return relevant_pairs[0]

        hit_a, hit_b, mapping_name = self.resolve_compare_mapping_hits(parsed_a, parsed_b)
        current_dat_column = self.device_b_column_var.get().strip() or str(hit_b or "").strip()
        current_txt_column = self.device_a_column_var.get().strip() or str(hit_a or "").strip()
        if current_txt_column and current_txt_column == normalized_target and current_dat_column:
            return {
                "a_col": normalized_target,
                "b_col": current_dat_column,
                "label": f"{normalized_target} vs {current_dat_column}",
            }

        normalized_target_lower = normalized_target.lower()
        for preset_name, preset in ELEMENT_PRESETS.items():
            a_candidates = [str(value).strip() for value in preset.get("A", []) if str(value).strip()]
            if normalized_target_lower not in {value.lower() for value in a_candidates}:
                continue
            preset_dat_column = self.select_first_matching_column(parsed_b.suggested_columns, preset.get("B", []))
            if preset_dat_column:
                return {
                    "a_col": normalized_target,
                    "b_col": preset_dat_column,
                    "label": f"{normalized_target} vs {preset_dat_column}",
                }
        return None

    def resolve_grouped_spectral_device_key(
        self,
        profile_name: str | None,
        path: Path,
    ) -> str | None:
        profile_family = self.resolve_selection_merge_profile_family(str(profile_name or ""))
        if profile_family == "ygas":
            return "A"
        if profile_family == "dat":
            return "B"
        suffix = path.suffix.lower()
        if suffix in TEXT_LIKE_SUFFIXES:
            return "A"
        if suffix == ".dat":
            return "B"
        return None

    def build_grouped_target_element_candidates(
        self,
        *,
        target_column: str,
        target_element: str,
    ) -> list[str]:
        candidates: list[str] = []
        normalized_target = str(target_column).strip().lower()

        def append_candidate(name: str) -> None:
            text = str(name).strip()
            if text and text in ELEMENT_PRESETS and text not in candidates:
                candidates.append(text)

        for preset_name, preset in ELEMENT_PRESETS.items():
            preset_values = {
                str(value).strip().lower()
                for key in ("A", "B")
                for value in preset.get(key, [])
                if str(value).strip()
            }
            if normalized_target and normalized_target in preset_values:
                append_candidate(str(preset_name))
        append_candidate(target_element)
        return candidates

    def build_grouped_spectral_required_columns(
        self,
        *,
        target_column: str,
        target_element: str,
        device_key: str | None,
    ) -> list[str]:
        ordered: list[str] = []

        def append_candidate(value: str) -> None:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

        append_candidate(target_column)
        for pair in self.build_compare_pairs():
            a_col = str(pair.get("a_col") or "").strip()
            b_col = str(pair.get("b_col") or "").strip()
            if device_key == "A":
                if a_col:
                    append_candidate(a_col)
                if b_col and b_col == str(target_column).strip():
                    append_candidate(a_col)
            elif device_key == "B":
                if b_col:
                    append_candidate(b_col)
                if a_col and a_col == str(target_column).strip():
                    append_candidate(b_col)

        if device_key in {"A", "B"}:
            for candidate_element in self.build_grouped_target_element_candidates(
                target_column=target_column,
                target_element=target_element,
            ):
                for value in ELEMENT_PRESETS.get(candidate_element, {}).get(device_key, []):
                    append_candidate(str(value))
        return ordered

    def resolve_grouped_spectral_target_column(
        self,
        parsed: ParsedFileResult,
        *,
        path: Path,
        target_column: str,
        target_element: str,
        device_key: str | None = None,
    ) -> dict[str, str] | None:
        normalized_target = str(target_column).strip()
        if not normalized_target:
            return None

        resolved_device_key = device_key or self.resolve_grouped_spectral_device_key(parsed.profile_name, path)
        direct_hit = self.select_first_matching_column(parsed.available_columns, [normalized_target])
        if direct_hit:
            return {
                "resolved_target_column": str(direct_hit),
                "resolution_source": "exact_column_match",
                "device_key": str(resolved_device_key or ""),
            }

        for pair in self.build_compare_pairs():
            a_col = str(pair.get("a_col") or "").strip()
            b_col = str(pair.get("b_col") or "").strip()
            if resolved_device_key == "A" and b_col and b_col == normalized_target:
                pair_hit = self.select_first_matching_column(parsed.available_columns, [a_col])
                if pair_hit:
                    return {
                        "resolved_target_column": str(pair_hit),
                        "resolution_source": "compare_pair_mapping",
                        "device_key": "A",
                    }
            if resolved_device_key == "B" and a_col and a_col == normalized_target:
                pair_hit = self.select_first_matching_column(parsed.available_columns, [b_col])
                if pair_hit:
                    return {
                        "resolved_target_column": str(pair_hit),
                        "resolution_source": "compare_pair_mapping",
                        "device_key": "B",
                    }

        if resolved_device_key in {"A", "B"}:
            for candidate_element in self.build_grouped_target_element_candidates(
                target_column=target_column,
                target_element=target_element,
            ):
                preset_hit = self.select_target_element_column(
                    parsed,
                    device_key=resolved_device_key,
                    target_element=candidate_element,
                )
                if preset_hit:
                    return {
                        "resolved_target_column": str(preset_hit),
                        "resolution_source": f"element_preset:{candidate_element}",
                        "device_key": str(resolved_device_key),
                    }
        return None

    def align_grouped_spectral_target_column(
        self,
        parsed: ParsedFileResult,
        *,
        target_column: str,
        resolved_target_column: str,
    ) -> ParsedFileResult:
        normalized_target = str(target_column).strip()
        normalized_resolved = str(resolved_target_column).strip()
        if not normalized_target or not normalized_resolved:
            raise ValueError("设备分组目标列对齐失败：目标列为空。")
        if normalized_resolved not in parsed.dataframe.columns:
            raise ValueError(f"设备分组目标列对齐失败：缺少列 {normalized_resolved}。")
        if normalized_resolved == normalized_target:
            return parsed

        aligned_frame = parsed.dataframe.rename(columns={normalized_resolved: normalized_target})
        aligned = self.build_parsed_result_from_dataframe(
            aligned_frame,
            parsed.profile_name,
            parsed.timestamp_col,
            source_row_count=parsed.source_row_count,
        )
        aligned.timestamp_valid_count = int(parsed.timestamp_valid_count)
        aligned.timestamp_valid_ratio = float(parsed.timestamp_valid_ratio)
        aligned.timestamp_warning = parsed.timestamp_warning
        return aligned

    def reset_auto_compare_context(self, *, clear_selected_dat: bool = True) -> None:
        self.auto_prepare_payload = None
        self.auto_dat_options = {}
        self.auto_compare_context_available = False
        self.auto_compare_context_source = ""
        self.auto_compare_context_dat_file_name = ""
        self.auto_compare_context_txt_count = 0
        if hasattr(self, "selected_dat_combo") and self.selected_dat_combo is not None:
            self.selected_dat_combo.configure(values=[])
        if clear_selected_dat:
            self.selected_dat_var.set("")

    def cache_auto_compare_context_payload(
        self,
        payload: dict[str, Any],
        *,
        source: str,
        selected_dat_path: Path | None = None,
    ) -> dict[str, Any] | None:
        self.auto_prepare_payload = payload
        ygas_infos = list(payload.get("ygas_infos", []))
        dat_infos = list(payload.get("dat_infos", []))
        self.auto_dat_options = {self.build_dat_option_label(info): Path(info["path"]) for info in dat_infos}
        if hasattr(self, "selected_dat_combo") and self.selected_dat_combo is not None:
            self.selected_dat_combo.configure(values=list(self.auto_dat_options.keys()))

        chosen_path = selected_dat_path or payload.get("selected_dat_path")
        if chosen_path is None:
            chosen_info = self.choose_best_dat_info(ygas_infos, dat_infos)
            chosen_path = Path(chosen_info["path"]) if chosen_info is not None else None
        if chosen_path is not None:
            chosen_path = Path(chosen_path)
            dat_label = next((label for label, path in self.auto_dat_options.items() if path == chosen_path), chosen_path.name)
            self.selected_dat_var.set(dat_label)
        else:
            self.selected_dat_var.set("")

        selection: dict[str, Any] | None = None
        if ygas_infos and dat_infos and chosen_path is not None:
            try:
                selection = self.build_auto_selection_for_dat(payload, chosen_path)
            except Exception:
                selection = None

        self.auto_compare_context_available = bool(selection is not None)
        self.auto_compare_context_source = str(source or "")
        self.auto_compare_context_dat_file_name = str(chosen_path.name if chosen_path is not None else "")
        self.auto_compare_context_txt_count = (
            int(len(selection.get("selected_ygas_paths", []))) if selection is not None else 0
        )
        return selection

    def build_auto_compare_bootstrap_files(self, selection_paths: list[Path]) -> list[Path]:
        normalized_selection = [Path(path) for path in selection_paths]
        def collect_supported_dat_files(folder: Path) -> list[Path]:
            dat_candidates: list[Path] = []
            try:
                supported_files = self.get_supported_files(folder)
            except Exception:
                return dat_candidates
            for path in supported_files:
                try:
                    profile = detect_file_profile(path, read_preview_lines(path))
                except Exception:
                    profile = None
                if profile == "TOA5_DAT" or path.suffix.lower() == ".dat":
                    dat_candidates.append(path)
            return dat_candidates

        if self.current_folder is not None and normalized_selection:
            current_folder_resolved = self.current_folder.resolve()
            selection_parents = {path.parent.resolve() for path in normalized_selection}
            if selection_parents == {current_folder_resolved}:
                current_files = list(self.file_paths) if self.file_paths else self.get_supported_files(self.current_folder)
                if any(path.suffix.lower() == ".dat" for path in current_files):
                    return current_files
                parent_dat_files = collect_supported_dat_files(self.current_folder.parent)
                if parent_dat_files:
                    return [*normalized_selection, *parent_dat_files]
                return current_files

        selection_parents = {path.parent.resolve() for path in normalized_selection}
        if len(selection_parents) == 1:
            parent = Path(next(iter(selection_parents)))
            current_files = self.get_supported_files(parent)
            if any(path.suffix.lower() == ".dat" for path in current_files):
                return current_files
            parent_dat_files = collect_supported_dat_files(parent.parent)
            if parent_dat_files:
                return [*normalized_selection, *parent_dat_files]
            return current_files
        return normalized_selection

    def ensure_auto_compare_context_for_selection(
        self,
        selection_paths: list[Path],
        *,
        source: str,
    ) -> bool:
        normalized_selection = [Path(path) for path in selection_paths]
        if not normalized_selection:
            return False

        existing_context = self.build_auto_compare_selection_snapshot()
        if existing_context is not None:
            ygas_keys = {str(Path(path).resolve()) for path in existing_context.get("ygas_paths", [])}
            selection_keys = {str(path.resolve()) for path in normalized_selection}
            if selection_keys.issubset(ygas_keys):
                return True

        candidate_files = self.build_auto_compare_bootstrap_files(normalized_selection)
        if len(candidate_files) < 2:
            self.auto_compare_context_available = False
            self.auto_compare_context_source = str(source or "")
            self.auto_compare_context_dat_file_name = ""
            self.auto_compare_context_txt_count = 0
            return False

        payload = self.prepare_folder_auto_selection_payload(candidate_files)
        selection = self.cache_auto_compare_context_payload(payload, source=source)
        self.refresh_single_compare_style_preview_state()
        return bool(selection is not None)

    def build_auto_compare_selection_snapshot(self) -> dict[str, Any] | None:
        payload = self.auto_prepare_payload
        if payload is None:
            return None
        ygas_infos = list(payload.get("ygas_infos", []))
        dat_infos = list(payload.get("dat_infos", []))
        if not ygas_infos or not dat_infos:
            return None

        dat_path: Path | None = None
        selected_dat_label = self.selected_dat_var.get().strip()
        if selected_dat_label:
            dat_path = self.auto_dat_options.get(selected_dat_label)
        if dat_path is None:
            selected_dat_path = payload.get("selected_dat_path")
            if selected_dat_path:
                dat_path = Path(selected_dat_path)
        if dat_path is None:
            chosen_info = self.choose_best_dat_info(ygas_infos, dat_infos)
            if chosen_info is not None:
                dat_path = Path(chosen_info["path"])
        if dat_path is None:
            return None

        try:
            selection = self.build_auto_selection_for_dat(payload, Path(dat_path))
        except Exception:
            return None

        selected_ygas_paths = [Path(path) for path in selection.get("selected_ygas_paths", [])]
        resolved_dat_path = Path(selection["dat_info"]["path"])
        return {
            "ygas_paths": selected_ygas_paths,
            "dat_path": resolved_dat_path,
            "selected_paths": [*selected_ygas_paths, resolved_dat_path],
            "auto_compare_context_available": bool(self.auto_compare_context_available),
            "auto_compare_context_source": str(self.auto_compare_context_source or ""),
            "auto_compare_context_dat_file_name": str(self.auto_compare_context_dat_file_name or resolved_dat_path.name),
            "auto_compare_context_txt_count": int(self.auto_compare_context_txt_count or len(selected_ygas_paths)),
        }

    def resolve_selected_dat_path_from_current_compare_ui(self) -> tuple[Path | None, str]:
        selected_dat_label = self.selected_dat_var.get().strip()
        if selected_dat_label:
            dat_path = self.auto_dat_options.get(selected_dat_label)
            if dat_path is not None:
                return Path(dat_path), "selected_dat_var"

        selection_info = self.extract_single_device_txt_selection(self.get_selected_file_paths())
        selected_dat_paths = list(selection_info["selected_dat_paths"])
        if selected_dat_paths:
            return Path(selected_dat_paths[0]), "selected_file_selection"

        return None, ""

    def build_compare_context_from_current_ui_state(self) -> dict[str, Any] | None:
        selection_info = self.extract_single_device_txt_selection(self.get_selected_file_paths())
        selected_txt_paths = [Path(path) for path in selection_info["selected_txt_paths"]]
        selected_dat_paths = [Path(path) for path in selection_info["selected_dat_paths"]]

        dat_path, dat_resolution = self.resolve_selected_dat_path_from_current_compare_ui()
        if dat_path is None:
            return None

        compare_txt_paths = list(selected_txt_paths)
        if not compare_txt_paths and self.auto_prepare_payload is not None:
            try:
                auto_selection = self.build_auto_selection_for_dat(self.auto_prepare_payload, Path(dat_path))
            except Exception:
                auto_selection = None
            if auto_selection is not None:
                compare_txt_paths = [Path(path) for path in auto_selection.get("selected_ygas_paths", [])]

        if not compare_txt_paths:
            return None

        selected_dat_label = self.selected_dat_var.get().strip()
        selected_dat_from_label = self.auto_dat_options.get(selected_dat_label) if selected_dat_label else None
        selected_dat_reference = (
            Path(selected_dat_from_label)
            if selected_dat_from_label is not None
            else (Path(selected_dat_paths[0]) if selected_dat_paths else None)
        )
        dat_matches_selected_dat = bool(
            selected_dat_reference is not None and Path(dat_path).resolve() == Path(selected_dat_reference).resolve()
        )

        return {
            "ygas_paths": compare_txt_paths,
            "dat_path": Path(dat_path),
            "selected_paths": [*compare_txt_paths, Path(dat_path)],
            "compare_context_source": "active_compare_ui",
            "compare_context_dat_resolution": str(dat_resolution or ""),
            "compare_context_selected_dat_label": str(selected_dat_label or ""),
            "compare_context_selected_dat_path": Path(selected_dat_reference) if selected_dat_reference is not None else None,
            "compare_context_matches_current_compare_ui": True,
            "compare_context_dat_matches_selected_dat": bool(dat_matches_selected_dat),
        }

    def build_compare_side_candidate_contexts(self) -> list[dict[str, Any]]:
        candidate_contexts: list[dict[str, Any]] = []
        seen_keys: set[tuple[tuple[str, ...], str]] = set()

        def append_context(context: dict[str, Any]) -> None:
            ygas_paths = [Path(path) for path in context.get("ygas_paths", [])]
            dat_path_value = context.get("dat_path")
            if not ygas_paths or dat_path_value is None:
                return
            dat_path = Path(dat_path_value)
            key = (
                tuple(sorted(str(path.resolve()) for path in ygas_paths)),
                str(dat_path.resolve()),
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            candidate_contexts.append(
                {
                    **context,
                    "ygas_paths": ygas_paths,
                    "dat_path": dat_path,
                    "selected_paths": [*ygas_paths, dat_path],
                }
            )

        active_context = self.build_compare_context_from_current_ui_state()
        if active_context is not None:
            append_context(active_context)

        selected_dat_path, _selected_dat_source = self.resolve_selected_dat_path_from_current_compare_ui()
        auto_context = self.build_auto_compare_selection_snapshot()
        if auto_context is not None:
            auto_dat_path = Path(auto_context["dat_path"])
            auto_context = {
                **auto_context,
                "compare_context_source": "auto_bootstrap_fallback",
                "compare_context_matches_current_compare_ui": False,
                "compare_context_dat_matches_selected_dat": bool(
                    selected_dat_path is not None and auto_dat_path.resolve() == Path(selected_dat_path).resolve()
                ),
            }
            append_context(auto_context)

        return candidate_contexts

    def resolve_active_compare_context(self, selection_paths: list[Path]) -> dict[str, Any]:
        normalized_selection = [Path(path) for path in selection_paths]
        if not normalized_selection:
            raise ValueError("当前没有选中的 txt/ygas 文件。")

        candidate_contexts = self.build_compare_side_candidate_contexts()
        if not candidate_contexts:
            raise ValueError("当前没有可复用的 dat / compare 上下文。")

        selection_keys = {str(path.resolve()) for path in normalized_selection}
        for context in candidate_contexts:
            ygas_keys = {str(Path(path).resolve()) for path in context.get("ygas_paths", [])}
            if selection_keys.issubset(ygas_keys):
                compare_selection = self.build_compare_selection_from_paths(
                    [Path(path) for path in context["ygas_paths"]],
                    Path(context["dat_path"]),
                )
                _merged_ygas, _txt_label, _parsed_dat, _dat_label, selection_meta = compare_selection
                txt_summary = selection_meta.get("txt_summary")
                dat_summary = selection_meta.get("dat_summary")
                if txt_summary is None or dat_summary is None:
                    raise ValueError("当前 compare 时间范围上下文不完整。")

                strategy_label = self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围"
                try:
                    start_dt, end_dt = self.resolve_compare_time_range_for_strategy(
                        strategy_label,
                        self.time_start_var.get().strip(),
                        self.time_end_var.get().strip(),
                        txt_summary,
                        dat_summary,
                    )
                except Exception as exc:
                    raise ValueError("当前 compare 时间范围无法解析。") from exc

                compare_time_range_meta = core.build_compare_time_range_metadata(
                    strategy_label=strategy_label,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    has_txt_dat_context=True,
                    requested_start=start_dt,
                    requested_end=end_dt,
                    actual_start=start_dt,
                    actual_end=end_dt,
                )
                return {
                    **context,
                    "compare_selection": compare_selection,
                    "selection_meta": selection_meta,
                    "txt_summary": txt_summary,
                    "dat_summary": dat_summary,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "compare_context_strategy_label": str(strategy_label),
                    "compare_context_time_range_policy": str(compare_time_range_meta.get("time_range_policy") or ""),
                    "compare_context_time_range_policy_label": str(
                        compare_time_range_meta.get("time_range_policy_label") or strategy_label
                    ),
                }

        raise ValueError("当前选择的文件不属于 compare 的 txt/ygas 一侧。")

    def resolve_txt_compare_side_equivalent_context(self, selection_paths: list[Path]) -> dict[str, Any]:
        return self.resolve_active_compare_context(selection_paths)

    def resolve_single_compare_style_preview_context(self) -> dict[str, Any]:
        if self.current_data_source_kind != "file" or self.current_file is None:
            raise ValueError("当前不是单文件视图，无法启用 txt 单图 compare-side 等效路径。")
        return self.resolve_txt_compare_side_equivalent_context([Path(self.current_file)])

    def refresh_single_compare_style_preview_state(self) -> None:
        available = False
        info_text = (
            "当前没有可复用的 compare 上下文；单文件预览会在上下文缺失时回退到普通单设备谱图，"
            "直接生成图仍会优先尝试目标谱图。"
        )
        try:
            context = self.resolve_single_compare_style_preview_context()
            available = True
            info_text = (
                f"已检测到可复用 compare 上下文（txt={len(context['ygas_paths'])} 个，dat=1 个）。"
                " 单文件预览仍可复用对比图语义；直接生成图会优先走目标谱图。"
            )
        except Exception:
            available = False

        self.single_compare_style_preview_available = available
        self.single_compare_style_preview_var.set(bool(available))
        if self.single_compare_style_preview_check is not None:
            self.single_compare_style_preview_check.configure(state="disabled")
        self.single_compare_style_preview_info_var.set(info_text)

    def build_txt_compare_side_equivalent_payload(
        self,
        *,
        selection_paths: list[Path],
        selected_column: str,
        execution_context: str,
        plot_execution_path: str,
        execution_path_key: str,
        execution_path_value: str,
        selection_scope_key: str,
        time_range_policy_key: str,
        fs_ui: float | None = None,
        requested_nsegment: int | None = None,
        overlap_ratio: float | None = None,
        compare_context_key_prefix: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            context = self.resolve_active_compare_context(selection_paths)
        except ValueError as exc:
            message = str(exc)
            if "dat / compare" in message:
                return None, "no_compare_dat_context"
            if "txt/ygas 一侧" in message:
                return None, "selection_not_txt_side"
            if "时间范围" in message:
                return None, "compare_time_range_unresolved"
            return None, "compare_context_unavailable"

        compare_selection = context["compare_selection"]
        merged_ygas, txt_label, _parsed_dat, dat_label, selection_meta = compare_selection
        valid_txt_columns = {str(column) for column in merged_ygas.dataframe.columns}
        if selected_column not in valid_txt_columns:
            return None, "merged_ygas_missing_target_column"

        txt_summary = selection_meta.get("txt_summary")
        dat_summary = selection_meta.get("dat_summary")
        if txt_summary is None or dat_summary is None:
            return None, "compare_time_range_context_missing"

        start_dt = context.get("start_dt")
        end_dt = context.get("end_dt")

        if fs_ui is None or requested_nsegment is None or overlap_ratio is None:
            fs_ui, requested_nsegment, overlap_ratio = self.get_analysis_params()
        payload = core.compute_base_spectrum_payload(
            merged_ygas,
            selected_column,
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            start_dt=start_dt,
            end_dt=end_dt,
            require_timestamp=True,
        )
        details = dict(payload["details"])
        details.update(
            core.build_compare_time_range_metadata(
                strategy_label=str(
                    context.get("compare_context_strategy_label")
                    or self.time_range_strategy_var.get().strip()
                    or "使用 txt+dat 共同时间范围"
                ),
                start_dt=start_dt,
                end_dt=end_dt,
                has_txt_dat_context=True,
                requested_start=details.get("base_requested_start"),
                requested_end=details.get("base_requested_end"),
                actual_start=details.get("base_actual_start"),
                actual_end=details.get("base_actual_end"),
            )
        )
        details["analysis_context"] = execution_context
        details["plot_execution_path"] = plot_execution_path
        details[execution_path_key] = execution_path_value
        details[selection_scope_key] = "merged_ygas_compare_scope"
        details[time_range_policy_key] = str(details.get("time_range_policy") or "")
        details["selection_merge_scope"] = "merged_ygas_compare_scope"
        details["group_device_source"] = "compare_txt_side"
        details["compare_txt_source_label"] = str(txt_label)
        details["compare_dat_source_label"] = str(dat_label)
        details["compare_txt_file_count"] = int(len(context["ygas_paths"]))
        details["compare_dat_file_name"] = str(Path(context["dat_path"]).name)
        details["auto_compare_context_built"] = True
        details["auto_compare_context_available"] = bool(context.get("auto_compare_context_available", True))
        details["auto_compare_context_source"] = str(
            context.get("auto_compare_context_source") or self.auto_compare_context_source or ""
        )
        details["auto_compare_context_dat_file_name"] = str(
            context.get("auto_compare_context_dat_file_name") or Path(context["dat_path"]).name
        )
        details["auto_compare_context_txt_count"] = int(
            context.get("auto_compare_context_txt_count") or len(context["ygas_paths"])
        )
        details["raw_source_rows"] = int(txt_summary.get("raw_rows", merged_ygas.source_row_count or len(merged_ygas.dataframe)))
        details["merged_rows"] = int(txt_summary.get("total_points", len(merged_ygas.dataframe)))
        details["rendered_point_count"] = int(len(payload["freq"]))
        annotate_device_dispatch_details(
            details,
            effective_device_ids=["compare_txt_side"],
            plot_execution_path=plot_execution_path,
            render_semantics="single_device_compare_compat",
            selection_file_count=len(selection_paths),
            selected_txt_file_count=len(selection_paths),
            selected_dat_file_count=0,
            data_context_source="compare_context_reuse",
        )
        visible_point_count_contract = build_series_point_count_contract(
            [
                {
                    "label": str(txt_label),
                    "column": str(selected_column),
                    "freq": np.asarray(payload["freq"], dtype=float),
                    "details": details,
                }
            ]
        )
        compare_geometry_series_results = self.build_compare_geometry_series_results(
            context=context,
            target_column=selected_column,
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
        )
        geometry_point_count_contract = build_series_point_count_contract(
            list(compare_geometry_series_results)
            or [
                {
                    "label": str(txt_label),
                    "column": str(selected_column),
                    "freq": np.asarray(payload["freq"], dtype=float),
                    "details": details,
                }
            ]
        )
        annotate_compare_geometry_point_count_fields(
            details,
            visible_contract=visible_point_count_contract,
            geometry_contract=geometry_point_count_contract,
        )
        if compare_context_key_prefix:
            details[f"{compare_context_key_prefix}_source"] = str(context.get("compare_context_source") or "")
            details[f"{compare_context_key_prefix}_dat_file_name"] = str(Path(context["dat_path"]).name)
            details[f"{compare_context_key_prefix}_time_range_policy"] = str(
                context.get("compare_context_time_range_policy") or details.get("time_range_policy") or ""
            )
            details[f"{compare_context_key_prefix}_start"] = start_dt
            details[f"{compare_context_key_prefix}_end"] = end_dt
            details[f"{compare_context_key_prefix}_matches_current_compare_ui"] = bool(
                context.get("compare_context_matches_current_compare_ui")
            )
            details[f"{compare_context_key_prefix}_dat_matches_selected_dat"] = bool(
                context.get("compare_context_dat_matches_selected_dat")
            )
        return {
            "freq": np.asarray(payload["freq"], dtype=float),
            "density": np.asarray(payload["density"], dtype=float),
            "details": details,
            "context": context,
            "selection_meta": selection_meta,
            "txt_label": str(txt_label),
            "dat_label": str(dat_label),
            "start_dt": start_dt,
            "end_dt": end_dt,
            "compare_geometry_series_results": compare_geometry_series_results,
        }, None

    def build_single_txt_compare_equivalent_payload(self, selected_column: str) -> dict[str, Any] | None:
        if self.current_data_source_kind != "file" or self.current_file is None:
            return None
        payload, _fallback_reason = self.build_txt_compare_side_equivalent_payload(
            selection_paths=[Path(self.current_file)],
            selected_column=selected_column,
            execution_context="single_txt_compare_side_equivalent",
            plot_execution_path="single_txt_compare_side_equivalent",
            execution_path_key="single_txt_execution_path",
            execution_path_value="compare_side_equivalent",
            selection_scope_key="single_txt_selection_scope",
            time_range_policy_key="single_txt_time_range_policy",
            compare_context_key_prefix="single_txt_compare_context",
        )
        return payload

    def build_single_device_txt_compare_equivalent_payload(
        self,
        selected_txt_files: list[Path],
        *,
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        selection_summary: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not selected_txt_files:
            return None, "selection_not_txt_side"

        payload, fallback_reason = self.build_txt_compare_side_equivalent_payload(
            selection_paths=selected_txt_files,
            selected_column=target_column,
            execution_context="single_device_compare_txt_side_equivalent",
            plot_execution_path="single_device_compare_txt_side_equivalent",
            execution_path_key="single_device_execution_path",
            execution_path_value="compare_txt_side_equivalent",
            selection_scope_key="single_device_selection_scope",
            time_range_policy_key="single_device_time_range_policy",
            fs_ui=fs,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            compare_context_key_prefix="single_device_compare_context",
        )
        if payload is None:
            return None, fallback_reason

        context = dict(payload.get("context") or {})
        selection_summary = dict(selection_summary or self.extract_single_device_txt_selection(selected_txt_files))
        source_paths = [Path(path) for path in context.get("ygas_paths", [])]
        device_label = str(payload.get("txt_label") or "compare txt 侧")
        device_id = "compare_txt_side"
        device_source = "compare_txt_side"
        file_count = int(len(source_paths))
        details = dict(payload["details"])
        details["device_id"] = device_id
        details["device_label"] = device_label
        details["device_source"] = device_source
        details["group_device_source"] = device_source
        details["device_file_count"] = file_count
        details["device_group_merge_strategy"] = "compare_txt_side_equivalent"
        details["source_paths"] = [str(path) for path in source_paths]
        details["selection_merge_scope"] = "merged_ygas_compare_scope"
        details["selected_file_count"] = int(selection_summary.get("selected_file_count", len(selected_txt_files)))
        details["selected_txt_file_count"] = int(
            selection_summary.get("selected_txt_file_count", len(selected_txt_files))
        )
        details["selected_dat_file_count"] = int(selection_summary.get("selected_dat_file_count", 0))
        details["single_device_selection_filtered_to_txt_side"] = bool(
            selection_summary.get("single_device_selection_filtered_to_txt_side", False)
        )
        annotate_device_dispatch_details(
            details,
            effective_device_ids=[device_id],
            plot_execution_path="single_device_compare_txt_side_equivalent",
            render_semantics="single_device_compare_compat",
            selection_file_count=int(selection_summary.get("selected_file_count", len(selected_txt_files))),
            selected_txt_file_count=int(selection_summary.get("selected_txt_file_count", len(selected_txt_files))),
            selected_dat_file_count=int(selection_summary.get("selected_dat_file_count", 0)),
            data_context_source="compare_context_reuse",
        )
        compare_geometry_series_results = list(payload.get("compare_geometry_series_results", []))
        if not compare_geometry_series_results:
            compare_geometry_series_results = self.build_compare_geometry_series_results(
                context=context,
                target_column=target_column,
                fs_ui=fs,
                requested_nsegment=requested_nsegment,
                overlap_ratio=overlap_ratio,
            )

        series_result = {
            "label": device_label,
            "side": "txt",
            "column": str(target_column),
            "freq": np.asarray(payload["freq"], dtype=float),
            "density": np.asarray(payload["density"], dtype=float),
            "details": details,
            "device_id": device_id,
            "device_source": device_source,
            "file_count": file_count,
            "merge_strategy": "compare_txt_side_equivalent",
            "selection_merge_scope": "merged_ygas_compare_scope",
            "source_paths": [str(path) for path in source_paths],
        }
        return {
            "target_column": target_column,
            "series_results": [series_result],
            "skipped_files": [],
            "device_groups": [
                {
                    "device_id": device_id,
                    "device_label": device_label,
                    "device_source": device_source,
                    "group_device_source": device_source,
                    "file_count": file_count,
                    "merge_strategy": "compare_txt_side_equivalent",
                    "selection_merge_scope": "merged_ygas_compare_scope",
                    "raw_source_rows": int(details.get("raw_source_rows", 0) or 0),
                    "merged_rows": int(details.get("merged_rows", 0) or 0),
                }
            ],
            "device_count": 1,
            "file_count": int(selection_summary.get("selected_file_count", file_count)),
            "start_dt": payload.get("start_dt"),
            "end_dt": payload.get("end_dt"),
            "compare_geometry_series_results": compare_geometry_series_results,
        }, None

    def build_dual_device_compare_dispatch_payload(
        self,
        *,
        selected_files: list[Path],
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        active_compare_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        normalized_selected_files = [Path(path) for path in selected_files]
        compare_context = dict(active_compare_context or {})
        compare_selection = compare_context.get("compare_selection")
        if isinstance(compare_selection, tuple) and len(compare_selection) == 5:
            parsed_a, label_a, parsed_b, label_b, selection_meta = compare_selection
            resolved_start_dt = compare_context.get("start_dt")
            resolved_end_dt = compare_context.get("end_dt")
            strategy_label = str(
                compare_context.get("compare_context_strategy_label")
                or self.time_range_strategy_var.get().strip()
                or "使用 txt+dat 共同时间范围"
            )
            has_txt_dat_context = True
        else:
            if len(normalized_selected_files) < 2:
                return None, "insufficient_selected_files"
            parsed_a = self.parse_profiled_file(normalized_selected_files[0])
            parsed_b = self.parse_profiled_file(normalized_selected_files[1])
            label_a = self.format_source_label(normalized_selected_files[0])
            label_b = self.format_source_label(normalized_selected_files[1])
            selection_meta = {"txt_summary": None, "dat_summary": None}
            resolved_start_dt = start_dt
            resolved_end_dt = end_dt
            strategy_label = self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围"
            has_txt_dat_context = False

        resolved_pair = self.resolve_compare_pair_for_target_column(parsed_a, parsed_b, str(target_column))
        if resolved_pair is None:
            normalized_target = str(target_column).strip()
            columns_a = {str(value) for value in parsed_a.suggested_columns}
            columns_b = {str(value) for value in parsed_b.suggested_columns}
            if normalized_target and normalized_target in columns_a and normalized_target in columns_b:
                resolved_pair = {
                    "a_col": normalized_target,
                    "b_col": normalized_target,
                    "label": f"{normalized_target} vs {normalized_target}",
                }
        if resolved_pair is None:
            return None, "compare_pair_unresolved"

        payload = self.prepare_dual_plot_payload(
            parsed_a=parsed_a,
            label_a=str(label_a),
            parsed_b=parsed_b,
            label_b=str(label_b),
            pairs=[resolved_pair],
            selection_meta=dict(selection_meta or {}),
            compare_mode="时间段内 PSD 对比",
            compare_scope=self.compare_scope_var.get().strip() or "单对单",
            start_dt=resolved_start_dt,
            end_dt=resolved_end_dt,
            mapping_name=(
                self.element_preset_var.get().strip()
                if self.mapping_mode_var.get().strip() == "预设映射"
                else str(target_column)
            )
            or str(target_column),
            scheme_name=self.scheme_name_var.get().strip() or "默认方案",
            alignment_strategy=self.get_alignment_strategy("时间段内 PSD 对比"),
            plot_style=self.resolve_plot_style("时间段内 PSD 对比"),
            plot_layout=self.resolve_plot_layout("时间段内 PSD 对比"),
            fs_ui=fs,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            match_tolerance=self.get_match_tolerance_seconds(default_value=0.2),
            spectrum_type=self.get_selected_cross_spectrum_type("时间段内 PSD 对比"),
            time_range_context={
                "strategy_label": strategy_label,
                "has_txt_dat_context": has_txt_dat_context,
            },
        )
        if str(payload.get("kind") or "") != "psd_compare":
            return None, "dual_compare_payload_not_psd_compare"
        return payload, None

    def finalize_device_dispatch_payload(
        self,
        payload: dict[str, Any],
        *,
        device_groups: list[dict[str, Any]],
        effective_device_ids: list[str],
        effective_device_count: int,
        plot_execution_path: str,
        render_semantics: str,
        selection_file_count: int,
        selected_txt_file_count: int,
        selected_dat_file_count: int,
        data_context_source: str,
        time_range_policy: str | None = None,
        active_compare_context: dict[str, Any] | None = None,
        compare_geometry_series_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        finalized = dict(payload)
        finalized["device_groups"] = list(device_groups)
        finalized["device_count"] = int(effective_device_count)
        finalized["effective_device_count"] = int(effective_device_count)
        finalized["effective_device_ids"] = list(effective_device_ids)
        finalized["file_count"] = int(selection_file_count)
        finalized["selection_file_count"] = int(selection_file_count)
        finalized["selected_txt_file_count"] = int(selected_txt_file_count)
        finalized["selected_dat_file_count"] = int(selected_dat_file_count)
        finalized["data_context_source"] = str(data_context_source or "")
        finalized["dispatch_render_semantics"] = str(render_semantics)
        finalized["plot_execution_path"] = str(plot_execution_path)
        finalized["effective_device_source_hints"] = [
            str(group.get("device_source") or group.get("group_device_source") or "")
            for group in device_groups
        ]
        if compare_geometry_series_results is not None:
            finalized["compare_geometry_series_results"] = list(compare_geometry_series_results)
        if time_range_policy is not None and str(time_range_policy).strip():
            finalized["time_range_policy"] = str(time_range_policy)
        if active_compare_context is not None:
            finalized["active_compare_context_available"] = True
            finalized["active_compare_context_source"] = str(active_compare_context.get("compare_context_source") or "")
            finalized["active_compare_context_dat_file_name"] = str(Path(active_compare_context["dat_path"]).name)
            finalized["active_compare_context_time_range_policy"] = str(
                active_compare_context.get("compare_context_time_range_policy") or ""
            )
        else:
            finalized["active_compare_context_available"] = False

        series_results: list[dict[str, Any]] = []
        for raw_item in list(finalized.get("series_results", [])):
            item = dict(raw_item)
            details = dict(raw_item.get("details", {}))
            details["series_count"] = int(effective_device_count)
            details["analysis_context"] = str(plot_execution_path)
            if time_range_policy is not None and str(time_range_policy).strip():
                details["time_range_policy"] = str(time_range_policy)
            annotate_device_dispatch_details(
                details,
                effective_device_ids=effective_device_ids,
                plot_execution_path=plot_execution_path,
                render_semantics=render_semantics,
                selection_file_count=selection_file_count,
                selected_txt_file_count=selected_txt_file_count,
                selected_dat_file_count=selected_dat_file_count,
                data_context_source=data_context_source,
            )
            item["details"] = details
            series_results.append(item)
        finalized["series_results"] = series_results
        return finalized

    def build_compare_geometry_series_results(
        self,
        *,
        context: dict[str, Any],
        target_column: str,
        fs_ui: float,
        requested_nsegment: int,
        overlap_ratio: float,
    ) -> list[dict[str, Any]]:
        compare_selection = context.get("compare_selection")
        if not isinstance(compare_selection, tuple) or len(compare_selection) != 5:
            return []

        merged_ygas, txt_label, parsed_dat, dat_label, selection_meta = compare_selection
        compare_mode = "时间段内 PSD 对比"
        resolved_pair = self.resolve_compare_pair_for_target_column(
            merged_ygas,
            parsed_dat,
            str(target_column),
        )
        relevant_pairs = [resolved_pair] if resolved_pair is not None else []
        if not relevant_pairs:
            return []

        geometry_payload = self.prepare_dual_plot_payload(
            parsed_a=merged_ygas,
            label_a=str(txt_label),
            parsed_b=parsed_dat,
            label_b=str(dat_label),
            pairs=relevant_pairs,
            selection_meta=dict(selection_meta or {}),
            compare_mode=compare_mode,
            compare_scope=self.compare_scope_var.get().strip() or "单对单",
            start_dt=context.get("start_dt"),
            end_dt=context.get("end_dt"),
            mapping_name=(
                self.element_preset_var.get().strip()
                if self.mapping_mode_var.get().strip() == "预设映射"
                else str(target_column)
            )
            or str(target_column),
            scheme_name=self.scheme_name_var.get().strip() or "默认方案",
            alignment_strategy=self.get_alignment_strategy(compare_mode),
            plot_style=self.resolve_plot_style(compare_mode),
            plot_layout=self.resolve_plot_layout(compare_mode),
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            match_tolerance=self.get_match_tolerance_seconds(default_value=0.2),
            spectrum_type=self.get_selected_cross_spectrum_type(compare_mode),
            time_range_context={
                "strategy_label": str(
                    context.get("compare_context_strategy_label")
                    or self.time_range_strategy_var.get().strip()
                    or "使用 txt+dat 共同时间范围"
                ),
                "has_txt_dat_context": True,
            },
        )
        return list(geometry_payload.get("series_results", []))

    def apply_recent_time_range(self, minutes: int) -> None:
        try:
            _parsed_a, _label_a, _parsed_b, _label_b, meta = self.build_compare_selection()
            end_dt = meta["dat_summary"]["end"]
        except Exception:
            if self.current_file is None:
                self.render_plot_message("请先选择含时间戳的数据文件。", level="warning")
                return
            parsed = self.parse_profiled_file(self.current_file)
            start_bound, end_bound, _count = self.get_parsed_time_bounds(parsed)
            end_dt = end_bound
        start_dt = end_dt - pd.Timedelta(minutes=minutes)
        self.time_start_var.set(start_dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.time_end_var.set(end_dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.time_range_strategy_var.set(f"最近{minutes}分钟" if minutes < 60 else "最近1小时")
        self.status_var.set(f"已填入最近 {minutes} 分钟时间范围")

    def build_legacy_target_override_config_path(self, scheme_path: Path) -> Path:
        return scheme_path.with_name(f"{scheme_path.stem}_target_spectrum_overrides.json")

    def save_legacy_target_override_config(self, scheme_path: Path) -> Path:
        override_path = self.build_legacy_target_override_config_path(scheme_path)
        payload = {
            "target_element": self.resolve_target_element_name(),
            "selected_dat": self.selected_dat_var.get().strip(),
            "group_keys": sorted(self.get_selected_target_group_overrides()),
        }
        with override_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.pending_target_group_override_keys = set(str(value) for value in payload["group_keys"])
        return override_path

    def load_legacy_target_override_config(self, scheme_path: Path) -> set[str]:
        override_path = self.build_legacy_target_override_config_path(scheme_path)
        if not override_path.exists():
            legacy_fallback = scheme_path.with_name(f"{scheme_path.stem}_legacy_target_overrides.json")
            override_path = legacy_fallback if legacy_fallback.exists() else override_path
        if not override_path.exists():
            self.pending_target_group_override_keys = set()
            return set()
        with override_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        keys = {str(value) for value in payload.get("group_keys", []) if str(value).strip()}
        self.pending_target_group_override_keys = keys
        return keys

    def save_compare_scheme(self) -> None:
        scheme = {
            "scheme_name": self.scheme_name_var.get().strip() or "默认方案",
            "mapping_mode": self.mapping_mode_var.get(),
            "element_preset": self.element_preset_var.get(),
            "compare_scope": self.compare_scope_var.get(),
            "alignment_strategy": self.alignment_strategy_var.get(),
            "plot_style": self.plot_style_var.get(),
            "plot_layout": self.plot_layout_var.get(),
            "use_separate_zoom_windows": bool(self.use_separate_zoom_windows_var.get()),
            "cross_spectrum_type": self.cross_spectrum_type_var.get(),
            "reference_slope_mode": self.reference_slope_mode_var.get(),
            "compare_mode": self.compare_mode_var.get(),
            "time_range_strategy": self.time_range_strategy_var.get(),
            "time_start": self.time_start_var.get(),
            "time_end": self.time_end_var.get(),
            "match_tolerance": self.match_tolerance_var.get(),
            "device_a_column": self.device_a_column_var.get(),
            "device_b_column": self.device_b_column_var.get(),
            "device_a_multi": self.get_listbox_selected_values(self.device_a_multi_listbox),
            "device_b_multi": self.get_listbox_selected_values(self.device_b_multi_listbox),
            "selected_dat": self.selected_dat_var.get(),
            "legacy_target_use_analysis_params": bool(self.legacy_target_use_analysis_params_var.get()),
            "legacy_target_spectrum_mode": self.legacy_target_spectrum_mode_var.get(),
            "legacy_target_color_mode": self.legacy_target_color_mode_var.get(),
            "target_cross_ygas_color": self.target_cross_ygas_color_var.get(),
            "target_cross_dat_color": self.target_cross_dat_color_var.get(),
            "fs": self.fs_var.get(),
            "nsegment": self.nsegment_var.get(),
            "overlap_ratio": self.overlap_ratio_var.get(),
        }
        path = filedialog.asksaveasfilename(
            title="保存比对方案",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            initialfile=f"{sanitize_filename(scheme['scheme_name'])}_compare_scheme.json",
        )
        if not path:
            return
        scheme_path = Path(path)
        scheme["legacy_target_override_config"] = self.build_legacy_target_override_config_path(scheme_path).name
        with scheme_path.open("w", encoding="utf-8") as handle:
            json.dump(scheme, handle, ensure_ascii=False, indent=2)
        override_path = self.save_legacy_target_override_config(scheme_path)
        self.status_var.set(f"比对方案已保存：{path}")
        self.status_var.set(f"比对方案已保存：{path} | target overrides={override_path.name}")

    def load_compare_scheme(self) -> None:
        path = filedialog.askopenfilename(
            title="加载比对方案",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            scheme = json.load(handle)

        self.scheme_name_var.set(str(scheme.get("scheme_name", "默认方案")))
        self.mapping_mode_var.set(str(scheme.get("mapping_mode", "预设映射")))
        self.element_preset_var.set(str(scheme.get("element_preset", "CO2")))
        self.compare_scope_var.set(str(scheme.get("compare_scope", "单对单")))
        self.alignment_strategy_var.set(str(scheme.get("alignment_strategy", "最近邻 + 容差")))
        self.plot_style_var.set(str(scheme.get("plot_style", "自动")))
        self.plot_layout_var.set(str(scheme.get("plot_layout", "叠加同图")))
        self.use_separate_zoom_windows_var.set(bool(scheme.get("use_separate_zoom_windows", False)))
        self.cross_spectrum_type_var.set(str(scheme.get("cross_spectrum_type", CROSS_SPECTRUM_MAGNITUDE)))
        self.reference_slope_mode_var.set(str(scheme.get("reference_slope_mode", REFERENCE_SLOPE_MODE_AUTO)))
        self.compare_mode_var.set(str(scheme.get("compare_mode", "时间序列对比")))
        self.time_range_strategy_var.set(str(scheme.get("time_range_strategy", "使用 txt+dat 共同时间范围")))
        self.time_start_var.set(str(scheme.get("time_start", "")))
        self.time_end_var.set(str(scheme.get("time_end", "")))
        self.match_tolerance_var.set(str(scheme.get("match_tolerance", "0.2")))
        self.device_a_column_var.set(str(scheme.get("device_a_column", "")))
        self.device_b_column_var.set(str(scheme.get("device_b_column", "")))
        self.selected_dat_var.set(str(scheme.get("selected_dat", "")))
        self.legacy_target_use_analysis_params_var.set(bool(scheme.get("legacy_target_use_analysis_params", False)))
        self.legacy_target_spectrum_mode_var.set(
            str(scheme.get("legacy_target_spectrum_mode", core.LEGACY_TARGET_SPECTRUM_MODE_PSD))
        )
        self.legacy_target_color_mode_var.set(
            str(scheme.get("legacy_target_color_mode", LEGACY_TARGET_COLOR_MODE_BY_DEVICE))
        )
        self.target_cross_ygas_color_var.set(str(scheme.get("target_cross_ygas_color", "")))
        self.target_cross_dat_color_var.set(str(scheme.get("target_cross_dat_color", "")))
        self.fs_var.set(str(scheme.get("fs", self.fs_var.get())))
        self.nsegment_var.set(str(scheme.get("nsegment", self.nsegment_var.get())))
        self.overlap_ratio_var.set(str(scheme.get("overlap_ratio", self.overlap_ratio_var.get())))
        self.refresh_compare_column_options()
        self.set_listbox_values(
            self.device_a_multi_listbox,
            [str(self.device_a_multi_listbox.get(i)) for i in range(self.device_a_multi_listbox.size())],
            [str(value) for value in scheme.get("device_a_multi", [])],
        )
        self.set_listbox_values(
            self.device_b_multi_listbox,
            [str(self.device_b_multi_listbox.get(i)) for i in range(self.device_b_multi_listbox.size())],
            [str(value) for value in scheme.get("device_b_multi", [])],
        )
        scheme_path = Path(path)
        self.load_legacy_target_override_config(scheme_path)
        if self.current_target_plot_metadata:
            self.update_target_group_qc_panel(self.current_target_plot_metadata)
        self.status_var.set(f"已加载比对方案：{path}")

    def update_compare_summaries(
        self,
        txt_summary: dict[str, Any] | None = None,
        dat_summary: dict[str, Any] | None = None,
    ) -> None:
        if txt_summary is None:
            self.txt_merge_summary_var.set("txt 合并摘要：等待选择 txt/log 文件")
        else:
            self.txt_merge_summary_var.set(
                f"txt 合并摘要：文件数={txt_summary['file_count']} | 开始={txt_summary['start']} | "
                f"结束={txt_summary['end']} | 原始总行数={txt_summary.get('raw_rows', txt_summary['total_points'])} | "
                f"有效时间戳点数={txt_summary.get('valid_timestamp_points', txt_summary['total_points'])} | "
                f"合并后点数={txt_summary['total_points']}"
            )
        if dat_summary is None:
            self.dat_summary_var.set("dat 摘要：等待选择 dat 文件")
        else:
            self.dat_summary_var.set(
                f"dat 摘要：开始={dat_summary['start']} | 结束={dat_summary['end']} | "
                f"原始总行数={dat_summary.get('raw_rows', dat_summary['total_points'])} | "
                f"有效时间戳点数={dat_summary.get('valid_timestamp_points', dat_summary['total_points'])} | "
                f"总点数={dat_summary['total_points']}"
            )

    def build_timestamp_quality_items(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
        *,
        label_a: str = "设备A",
        label_b: str = "设备B",
    ) -> list[str]:
        items = [
            f"{label_a}原始点数={parsed_a.source_row_count or len(parsed_a.dataframe)}",
            f"{label_a}有效时间点数={parsed_a.timestamp_valid_count}",
            f"{label_a}时间戳有效率={parsed_a.timestamp_valid_ratio:.1%}",
            f"{label_b}原始点数={parsed_b.source_row_count or len(parsed_b.dataframe)}",
            f"{label_b}有效时间点数={parsed_b.timestamp_valid_count}",
            f"{label_b}时间戳有效率={parsed_b.timestamp_valid_ratio:.1%}",
        ]
        if parsed_a.timestamp_warning:
            items.append(f"{label_a}时间戳警告={parsed_a.timestamp_warning}")
        if parsed_b.timestamp_warning:
            items.append(f"{label_b}时间戳警告={parsed_b.timestamp_warning}")
        return items

    def update_compare_column_values(self, combo: ttk.Combobox, variable: tk.StringVar, columns: list[str]) -> None:
        combo.configure(values=columns)
        current = variable.get()
        if current in columns:
            return
        if "CO2浓度" in columns:
            variable.set("CO2浓度")
            return
        if "H2O浓度" in columns:
            variable.set("H2O浓度")
            return
        variable.set(columns[0] if columns else "")

    def refresh_compare_column_options(self) -> None:
        try:
            parsed_a, label_a, parsed_b, label_b, meta = self.build_compare_selection()
        except Exception as exc:
            selected_paths = self.get_selected_file_paths()
            ygas_paths, dat_path, _other = self.classify_selected_compare_files()
            if selected_paths and (dat_path is None or not ygas_paths):
                if ygas_paths and dat_path is None:
                    self.compare_file_info_var.set(
                        f"当前仅识别到 ygas 文件 {len(ygas_paths)} 个。可直接生成单台设备图谱；双设备对比需再补 1 个 dat 文件。"
                    )
                elif dat_path is not None and not ygas_paths:
                    dat_count = sum(1 for path in selected_paths if path.suffix.lower() == ".dat")
                    self.compare_file_info_var.set(
                        f"当前仅识别到 dat 文件 {max(dat_count, 1)} 个。可直接生成单台设备图谱；双设备对比需再补 1 个 txt/log 文件。"
                    )
                elif len(selected_paths) == 1:
                    self.compare_file_info_var.set(
                        f"当前仅选择 1 个文件：{selected_paths[0].name}。可直接生成单台设备图谱；双设备对比需再选择另一台设备文件。"
                    )
                else:
                    self.compare_file_info_var.set(
                        "当前所选文件尚未形成双设备配对。可先做单台设备图谱；如需对比，请同时选择 txt/log 和 dat。"
                    )
            elif len(selected_paths) < 2:
                self.compare_file_info_var.set("双设备比对：请先选择多个 txt/log 和 1 个 dat 文件")
            else:
                self.compare_file_info_var.set(f"双设备比对：文件解析失败 - {exc}")
            self.device_a_combo.configure(values=[])
            self.device_b_combo.configure(values=[])
            self.device_a_column_var.set("")
            self.device_b_column_var.set("")
            self.refresh_single_compare_style_preview_state()
            return

        self.update_compare_column_values(self.device_a_combo, self.device_a_column_var, parsed_a.suggested_columns)
        self.update_compare_column_values(self.device_b_combo, self.device_b_column_var, parsed_b.suggested_columns)
        hit_a, hit_b, mapping_name = self.apply_element_mapping(parsed_a, parsed_b)
        self.compare_file_info_var.set(
            f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 映射={mapping_name}"
        )
        if self.mapping_mode_var.get() == "预设映射" and (hit_a is None or hit_b is None):
            self.compare_file_info_var.set(
                f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 预设“{mapping_name}”未完全命中"
            )
        self.update_compare_summaries(meta["txt_summary"], meta["dat_summary"])
        self.refresh_single_compare_style_preview_state()

    def build_device_grouped_spectral_selection(
        self,
        selected_files: list[Path],
        *,
        target_column: str,
        reporter: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        grouped: dict[str, dict[str, Any]] = {}
        skipped_files: list[str] = []
        parsed_entries: list[dict[str, Any]] = []
        total_files = len(selected_files)
        target_element = self.resolve_target_element_name()

        for index, path in enumerate(selected_files, start=1):
            if reporter is not None:
                reporter(f"正在解析文件并识别设备… {index}/{total_files}：{path.name}")
            preview_lines = read_preview_lines(path)
            detected_profile = detect_file_profile(path, preview_lines)
            device_key = self.resolve_grouped_spectral_device_key(detected_profile, path)
            required_columns = self.build_grouped_spectral_required_columns(
                target_column=target_column,
                target_element=target_element,
                device_key=device_key,
            )
            try:
                parsed = self.parse_target_profiled_file(
                    path,
                    device_kind="grouped_spectral",
                    required_columns=required_columns or [target_column],
                )
            except Exception as exc:
                skipped_files.append(f"{path.name}(解析失败: {exc})")
                continue

            resolved_target = self.resolve_grouped_spectral_target_column(
                parsed,
                path=path,
                target_column=target_column,
                target_element=target_element,
                device_key=device_key,
            )
            if resolved_target is None:
                skipped_files.append(f"{path.name}(缺少目标列映射)")
                continue
            aligned_parsed = self.align_grouped_spectral_target_column(
                parsed,
                target_column=target_column,
                resolved_target_column=str(resolved_target["resolved_target_column"]),
            )

            device_info = core.resolve_device_identifier(parsed, path)
            parsed_entries.append(
                {
                    "path": path,
                    "parsed": parsed,
                    "aligned_parsed": aligned_parsed,
                    "device_info": dict(device_info),
                    "resolved_target_column": str(resolved_target["resolved_target_column"]),
                    "target_resolution_source": str(resolved_target["resolution_source"]),
                    "device_key": str(resolved_target.get("device_key") or device_key or ""),
                }
            )
            group = grouped.setdefault(
                str(device_info["device_id"]),
                {
                    "device_id": str(device_info["device_id"]),
                    "device_label": str(device_info["device_label"]),
                    "device_source": str(device_info["device_source"]),
                    "entries": [],
                },
            )
            group["entries"].append(
                {
                    "path": path,
                    "parsed": parsed,
                    "aligned_parsed": aligned_parsed,
                    "resolved_target_column": str(resolved_target["resolved_target_column"]),
                    "target_resolution_source": str(resolved_target["resolution_source"]),
                    "device_key": str(resolved_target.get("device_key") or device_key or ""),
                }
            )

        selection_override = self.build_selection_scope_merged_device_group(
            selected_files=selected_files,
            parsed_entries=parsed_entries,
            target_column=target_column,
        )
        if selection_override is not None:
            return [selection_override], skipped_files

        device_groups: list[dict[str, Any]] = []
        for group in grouped.values():
            entries = list(group["entries"])
            merged_parsed, merge_metadata = core.merge_parsed_results_for_device_group(
                [item["aligned_parsed"] for item in entries],
                source_paths=[item["path"] for item in entries],
                device_id=group["device_id"],
            )
            column_values = pd.to_numeric(merged_parsed.dataframe[target_column], errors="coerce").to_numpy(dtype=float)
            valid_count = int(np.count_nonzero(np.isfinite(column_values)))
            if valid_count < 2:
                skipped_files.append(f"{group['device_label']}(有效点数不足)")
                continue
            group["merged_parsed"] = merged_parsed
            group["merge_metadata"] = merge_metadata
            group["file_count"] = len(entries)
            group["source_paths"] = [item["path"] for item in entries]
            group["valid_value_count"] = valid_count
            group["selection_merge_scope"] = "device_group"
            group["resolved_target_columns"] = sorted({
                str(item["resolved_target_column"])
                for item in entries
            })
            group["target_resolution_sources"] = [
                str(item["target_resolution_source"])
                for item in entries
            ]
            device_groups.append(group)

        device_groups.sort(key=lambda item: str(item["device_label"]).lower())
        return device_groups, skipped_files

    def resolve_selection_merge_profile_family(self, profile_name: str) -> str:
        normalized = str(profile_name or "").strip()
        if normalized.startswith("YGAS_MODE1") or normalized == "YGAS_MERGED":
            return "ygas"
        if normalized in {"TOA5_DAT", "TOA5_MERGED"}:
            return "dat"
        return "generic"

    def build_selection_scope_merged_device_group(
        self,
        *,
        selected_files: list[Path],
        parsed_entries: list[dict[str, Any]],
        target_column: str,
    ) -> dict[str, Any] | None:
        if len(selected_files) < 2 or len(parsed_entries) != len(selected_files):
            return None

        selection_hint = core.resolve_selection_scope_device_hint(selected_files)
        if not bool(selection_hint.get("is_consistent")):
            return None

        profile_families = {
            self.resolve_selection_merge_profile_family(str(entry["parsed"].profile_name))
            for entry in parsed_entries
        }
        if len(profile_families) != 1:
            return None
        profile_family = next(iter(profile_families))
        device_sources = {str(entry["device_info"].get("device_source", "")) for entry in parsed_entries}
        unique_device_ids = {str(entry["device_info"].get("device_id", "")) for entry in parsed_entries}

        if profile_family == "ygas":
            should_override = len(unique_device_ids) > 1 or "filename" in device_sources
        else:
            should_override = len(unique_device_ids) > 1 and device_sources == {"filename"}
        if not should_override:
            return None

        selection_device_id = str(selection_hint.get("device_id") or "selection_scope")
        selection_device_label = str(selection_hint.get("device_label") or selection_device_id)
        source_paths = list(selected_files)
        resolved_target_columns = {
            str(entry.get("resolved_target_column") or "").strip()
            for entry in parsed_entries
            if str(entry.get("resolved_target_column") or "").strip()
        }
        resolved_target_column_list = sorted(resolved_target_columns)
        source_target_column = resolved_target_column_list[0] if resolved_target_column_list else str(target_column)

        if profile_family == "ygas" and len(resolved_target_columns) == 1:
            merged_parsed, summary = core.load_and_merge_ygas_files_fast(
                source_paths,
                required_columns=[source_target_column],
            )
            merged_parsed = self.align_grouped_spectral_target_column(
                merged_parsed,
                target_column=target_column,
                resolved_target_column=source_target_column,
            )
            merge_metadata: dict[str, Any] = {
                "device_id": selection_device_id,
                "file_count": len(source_paths),
                "source_paths": [str(path) for path in source_paths],
                "merge_strategy": "selection_scope_ygas_fast_merge",
                "profile_name": merged_parsed.profile_name,
                "timestamp_col": merged_parsed.timestamp_col,
                "raw_rows": int(summary.get("raw_rows", merged_parsed.source_row_count or len(merged_parsed.dataframe))),
                "merged_points": int(summary.get("total_points", len(merged_parsed.dataframe))),
                "selection_merge_scope": "selection_level",
                "group_device_source": "selection_scope:filename_hint",
                "normalized_hints": list(selection_hint.get("normalized_hints", [])),
            }
        else:
            merged_parsed, merge_metadata = core.merge_parsed_results_for_device_group(
                [entry.get("aligned_parsed", entry["parsed"]) for entry in parsed_entries],
                source_paths=source_paths,
                device_id=selection_device_id,
            )
            merge_metadata = dict(merge_metadata)
            merge_metadata["merge_strategy"] = (
                "selection_scope_profile_merge"
                if profile_family != "ygas"
                else "selection_scope_aligned_profile_merge"
            )
            merge_metadata["selection_merge_scope"] = "selection_level"
            merge_metadata["group_device_source"] = "selection_scope:filename_hint"
            merge_metadata["normalized_hints"] = list(selection_hint.get("normalized_hints", []))

        if target_column not in merged_parsed.dataframe.columns:
            return None
        column_values = pd.to_numeric(merged_parsed.dataframe[target_column], errors="coerce").to_numpy(dtype=float)
        valid_count = int(np.count_nonzero(np.isfinite(column_values)))
        if valid_count < 2:
            return None

        return {
            "device_id": selection_device_id,
            "device_label": selection_device_label,
            "device_source": str(merge_metadata.get("group_device_source") or "selection_scope:filename_hint"),
            "entries": list(parsed_entries),
            "merged_parsed": merged_parsed,
            "merge_metadata": merge_metadata,
            "file_count": len(parsed_entries),
            "source_paths": source_paths,
            "valid_value_count": valid_count,
            "selection_merge_scope": str(merge_metadata.get("selection_merge_scope") or "selection_level"),
            "resolved_target_columns": list(resolved_target_column_list),
            "target_resolution_sources": [
                str(entry.get("target_resolution_source") or "")
                for entry in parsed_entries
            ],
        }

    def resolve_effective_device_groups(
        self,
        selected_files: list[Path],
        *,
        target_column: str,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        normalized_selected_files = [Path(path) for path in selected_files]
        selection_summary = self.extract_single_device_txt_selection(normalized_selected_files)
        selected_txt_paths = list(selection_summary["selected_txt_paths"])
        compare_context_paths = selected_txt_paths or list(normalized_selected_files)
        if compare_context_paths:
            self.ensure_auto_compare_context_for_selection(
                compare_context_paths,
                source="single_device_button_bootstrap",
            )

        device_groups, skipped_files = self.build_device_grouped_spectral_selection(
            normalized_selected_files,
            target_column=target_column,
            reporter=reporter,
        )
        effective_device_ids = [str(group["device_id"]) for group in device_groups]
        return {
            "selection_summary": selection_summary,
            "device_groups": device_groups,
            "skipped_files": skipped_files,
            "effective_device_count": int(len(device_groups)),
            "effective_device_ids": effective_device_ids,
            "data_context_source": "effective_device_grouping",
        }

    def resolve_target_spectrum_render_semantics(self, effective_device_count: int) -> str:
        normalized_count = int(max(effective_device_count, 0))
        if normalized_count <= 0:
            return "target_spectrum_empty"
        if normalized_count == 1:
            return "target_spectrum_single_group"
        if normalized_count == 2:
            return "target_spectrum_dual_group"
        return "target_spectrum_multi_group"

    def resolve_target_spectrum_visible_series_scope(
        self,
        target_metadata: dict[str, Any],
        *,
        is_psd_mode: bool,
    ) -> str:
        if not is_psd_mode:
            return "all_series"
        selection_file_count = int(target_metadata.get("selection_file_count", 0) or 0)
        selected_txt_file_count = int(target_metadata.get("selected_txt_file_count", 0) or 0)
        selected_dat_file_count = int(target_metadata.get("selected_dat_file_count", 0) or 0)
        if selection_file_count == 1 and selected_txt_file_count == 1 and selected_dat_file_count == 0:
            return "ygas_only"
        if selection_file_count == 1 and selected_dat_file_count == 1 and selected_txt_file_count == 0:
            return "dat_only"
        return "all_series"

    def is_target_spectrum_series_visible(
        self,
        item: dict[str, Any],
        *,
        visible_series_scope: str,
    ) -> bool:
        if visible_series_scope == "all_series":
            return True
        device_kind = str(item.get("device_kind", "")).strip()
        if visible_series_scope == "ygas_only":
            return device_kind == "ygas"
        if visible_series_scope == "dat_only":
            return device_kind == "dat"
        return True

    def compute_positive_log_axis_limits(self, values: np.ndarray) -> tuple[float, float] | None:
        value_array = np.asarray(values, dtype=float)
        mask = np.isfinite(value_array) & (value_array > 0)
        if not np.any(mask):
            return None
        positive_values = value_array[mask]
        vmin = float(np.min(positive_values))
        vmax = float(np.max(positive_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
            return None
        if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-18):
            return vmin / 1.5, vmax * 1.5
        log_min = float(np.log10(vmin))
        log_max = float(np.log10(vmax))
        padding = max((log_max - log_min) * 0.05, 0.05)
        return 10 ** (log_min - padding), 10 ** (log_max + padding)

    def build_target_spectrum_dispatch_groups(
        self,
        target_metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        group_records = list(target_metadata.get("group_records", []))
        kept_records = [
            dict(record)
            for record in group_records
            if record.get("status") in {"保留", "手动保留"}
        ]

        device_groups: list[dict[str, Any]] = []
        effective_device_ids: list[str] = []
        for index, record in enumerate(kept_records, start=1):
            raw_device_id = str(
                record.get("group_key")
                or record.get("group_label")
                or record.get("ygas_label")
                or f"target_group_{index}"
            ).strip()
            device_id = raw_device_id or f"target_group_{index}"
            if device_id in effective_device_ids:
                device_id = f"{device_id}#{index}"
            effective_device_ids.append(device_id)

            device_groups.append(
                {
                    "device_id": device_id,
                    "device_label": str(record.get("group_label") or record.get("ygas_label") or device_id),
                    "device_source": str(record.get("group_source") or "target_spectrum_grouping"),
                    "group_device_source": str(record.get("group_source") or "target_spectrum_grouping"),
                    "file_count": 1,
                    "merge_strategy": "target_spectrum_grouping",
                    "selection_merge_scope": "target_spectrum_group",
                    "raw_source_rows": int(record.get("ygas_points", 0) or 0),
                    "merged_rows": int(record.get("ygas_points", 0) or 0),
                    "resolved_target_columns": [
                        str(value)
                        for value in (
                            record.get("ygas_column"),
                            record.get("dat_column"),
                        )
                        if str(value or "").strip()
                    ],
                    "target_resolution_sources": ["target_spectrum_grouping"],
                }
            )
        return device_groups, effective_device_ids

    def finalize_target_spectrum_dispatch_payload(
        self,
        payload: dict[str, Any],
        *,
        selection_file_count: int,
        selected_txt_file_count: int,
        selected_dat_file_count: int,
        data_context_source: str,
        active_compare_context: dict[str, Any] | None = None,
        dat_context_file_name: str = "",
    ) -> dict[str, Any]:
        target_metadata = dict(payload.get("target_metadata", {}))
        device_groups, effective_device_ids = self.build_target_spectrum_dispatch_groups(target_metadata)
        effective_device_count = int(target_metadata.get("kept_group_count", len(device_groups)))
        series_results = list(payload.get("series_results", []))
        if effective_device_count <= 0 or not series_results:
            raise ValueError("目标谱图 payload 未生成有效分组。")

        first_details = dict(series_results[0].get("details", {}))
        render_semantics = self.resolve_target_spectrum_render_semantics(effective_device_count)
        time_range_policy = str(
            target_metadata.get("time_range_policy")
            or first_details.get("time_range_policy")
            or (
                active_compare_context.get("compare_context_time_range_policy")
                if active_compare_context is not None
                else ""
            )
            or ""
        )
        resolved_dat_context_file_name = str(dat_context_file_name or "").strip()
        if not resolved_dat_context_file_name:
            group_records = list(target_metadata.get("group_records", []))
            if group_records:
                resolved_dat_context_file_name = str(group_records[0].get("dat_label") or "").strip()

        finalized = self.finalize_device_dispatch_payload(
            payload,
            device_groups=device_groups,
            effective_device_ids=effective_device_ids,
            effective_device_count=effective_device_count,
            plot_execution_path="target_spectrum_render",
            render_semantics=render_semantics,
            selection_file_count=int(selection_file_count),
            selected_txt_file_count=int(selected_txt_file_count),
            selected_dat_file_count=int(selected_dat_file_count),
            data_context_source=data_context_source,
            time_range_policy=time_range_policy,
            active_compare_context=active_compare_context,
        )

        target_metadata.update(
            {
                "effective_device_count": int(effective_device_count),
                "effective_device_ids": list(effective_device_ids),
                "target_spectrum_group_count": int(effective_device_count),
                "plot_execution_path": "target_spectrum_render",
                "render_semantics": render_semantics,
                "selection_file_count": int(selection_file_count),
                "selected_txt_file_count": int(selected_txt_file_count),
                "selected_dat_file_count": int(selected_dat_file_count),
                "data_context_source": data_context_source,
                "time_range_policy": time_range_policy,
                "current_plot_kind": "target_spectrum",
                "default_dispatch_family": "target_spectrum",
            }
        )
        if resolved_dat_context_file_name:
            target_metadata["target_spectrum_context_dat_file_name"] = resolved_dat_context_file_name
        if active_compare_context is not None:
            target_metadata["active_compare_context_available"] = True
            target_metadata["active_compare_context_dat_file_name"] = str(Path(active_compare_context["dat_path"]).name)
            target_metadata["active_compare_context_time_range_policy"] = str(
                active_compare_context.get("compare_context_time_range_policy") or ""
            )
        else:
            target_metadata["active_compare_context_available"] = False

        total_series_count = int(len(finalized.get("series_results", [])))
        annotated_series_results: list[dict[str, Any]] = []
        for raw_item in list(finalized.get("series_results", [])):
            item = dict(raw_item)
            details = dict(item.get("details", {}))
            details["current_plot_kind"] = "target_spectrum"
            details["target_spectrum_group_count"] = int(effective_device_count)
            details["target_spectrum_series_count"] = int(total_series_count)
            details["series_count"] = int(total_series_count)
            if resolved_dat_context_file_name:
                details["target_spectrum_context_dat_file_name"] = resolved_dat_context_file_name
            item["details"] = details
            annotated_series_results.append(item)

        finalized["series_results"] = annotated_series_results
        finalized["target_metadata"] = target_metadata
        finalized["default_dispatch_family"] = "target_spectrum"
        finalized["target_spectrum_group_count"] = int(effective_device_count)
        return finalized

    def ensure_target_spectrum_dispatch_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        series_results = list(payload.get("series_results", []))
        if not series_results:
            return payload
        first_details = dict(series_results[0].get("details", {}))
        if str(payload.get("plot_execution_path") or first_details.get("plot_execution_path") or "").strip():
            return payload

        target_metadata = dict(payload.get("target_metadata", {}))
        total_group_count = int(target_metadata.get("total_group_count", target_metadata.get("kept_group_count", 0)) or 0)
        selected_txt_file_count = max(total_group_count, 1)
        selected_dat_file_count = 1 if total_group_count > 0 else 0
        dat_context_file_name = ""
        group_records = list(target_metadata.get("group_records", []))
        if group_records:
            dat_context_file_name = str(group_records[0].get("dat_label") or "").strip()
        return self.finalize_target_spectrum_dispatch_payload(
            payload,
            selection_file_count=int(selected_txt_file_count + selected_dat_file_count),
            selected_txt_file_count=int(selected_txt_file_count),
            selected_dat_file_count=int(selected_dat_file_count),
            data_context_source="target_spectrum_builder",
            active_compare_context=None,
            dat_context_file_name=dat_context_file_name,
        )

    def prepare_ygas_only_target_spectrum_dispatch_payload(
        self,
        selected_txt_paths: list[Path],
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        *,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        target_element = self.resolve_target_element_name()
        ygas_required_columns = self.build_target_required_columns(
            target_element=target_element,
            device_kind="ygas",
        )
        use_requested_nsegment = bool(self.legacy_target_use_analysis_params_var.get())
        selected_psd_kernel = core.LEGACY_TARGET_PSD_KERNEL_DEFAULT
        skipped_windows: list[str] = []
        group_records: list[dict[str, Any]] = []
        wrapped_series_results: list[dict[str, Any]] = []
        effective_nsegment_values: list[int] = []
        positive_freq_values: list[int] = []
        first_positive_freq_values: list[float] = []
        ygas_windows: list[dict[str, Any]] = []
        total_paths = len(selected_txt_paths)

        def _resolve_ygas_fs(parsed: ParsedFileResult, frame: pd.DataFrame, ui_fs: float) -> float:
            manual_override = not math.isclose(ui_fs, DEFAULT_FS, rel_tol=0.0, abs_tol=1e-9)
            if manual_override:
                return float(ui_fs)
            estimated = core.estimate_fs_from_timestamp(frame, core.TIMESTAMP_COL)
            if estimated:
                return float(estimated)
            return float(DEFAULT_FS)

        def _build_skip_record(
            *,
            group_key: str,
            labels: dict[str, str],
            window_label: str,
            window_start: pd.Timestamp | None,
            window_end: pd.Timestamp | None,
            reason: str,
            ygas_column_name: str | None,
        ) -> dict[str, Any]:
            return {
                "group_key": group_key,
                "group_label": labels["group_label"],
                "window_label": window_label,
                "ygas_start": window_start,
                "ygas_end": window_end,
                "dat_start": None,
                "dat_end": None,
                "ygas_points": 0,
                "dat_points": 0,
                "ygas_fs": None,
                "dat_fs": None,
                "ygas_coverage_ratio": None,
                "dat_coverage_ratio": None,
                "ygas_leading_invalid_gap_s": None,
                "dat_leading_invalid_gap_s": None,
                "ygas_trailing_invalid_gap_s": None,
                "dat_trailing_invalid_gap_s": None,
                "ygas_non_null_ratio": None,
                "dat_non_null_ratio": None,
                "ygas_freq_points": 0,
                "dat_freq_points": 0,
                "ygas_column": ygas_column_name,
                "dat_column": "",
                "ygas_label": labels["ygas_label"],
                "dat_label": "",
                "keep": False,
                "forced_include": False,
                "reason": str(reason),
                "status": "跳过",
                "forceable": False,
                "group_source": "ygas_only_target_spectrum",
                "spectrum_mode": core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
            }

        for index, path in enumerate(selected_txt_paths, start=1):
            if reporter is not None:
                reporter(f"正在解析 ygas 文件 {index}/{total_paths}: {path.name}")
            try:
                parsed_ygas = self.parse_target_profiled_file(
                    path,
                    device_kind="ygas",
                    required_columns=ygas_required_columns,
                )
                window_start, window_end, _window_points = core.get_parsed_time_bounds(parsed_ygas)
                labels = core.build_legacy_group_labels(
                    target_element=target_element,
                    window_start=pd.Timestamp(window_start),
                    ygas_path=path,
                )
                group_key = core.build_target_group_key(path, pd.Timestamp(window_start))
                window_label = f"{pd.Timestamp(window_start):%Y-%m-%d %H:%M} ~ {pd.Timestamp(window_end):%H:%M}"
                resolved_target_column = str(target_column or "").strip()
                if resolved_target_column not in parsed_ygas.dataframe.columns:
                    resolved_target_column = str(
                        self.select_target_element_column(
                            parsed_ygas,
                            device_key="A",
                            target_element=target_element,
                        )
                        or ""
                    ).strip()
                if not resolved_target_column:
                    skipped_windows.append(f"{group_key}(未找到{target_element}列)")
                    group_records.append(
                        _build_skip_record(
                            group_key=group_key,
                            labels=labels,
                            window_label=window_label,
                            window_start=pd.Timestamp(window_start),
                            window_end=pd.Timestamp(window_end),
                            reason=f"未找到{target_element}列",
                            ygas_column_name=None,
                        )
                    )
                    continue
                ygas_windows.append(
                    {
                        "path": path,
                        "parsed": parsed_ygas,
                        "column": resolved_target_column,
                        "start": pd.Timestamp(window_start),
                        "end": pd.Timestamp(window_end),
                        "labels": labels,
                        "group_key": group_key,
                    }
                )
            except Exception as exc:
                fallback_start = pd.Timestamp(start_dt) if start_dt is not None else None
                fallback_end = pd.Timestamp(end_dt) if end_dt is not None else None
                labels = {
                    "group_label": path.name,
                    "ygas_label": path.name,
                    "dat_label": "",
                }
                group_records.append(
                    _build_skip_record(
                        group_key=path.name,
                        labels=labels,
                        window_label=path.name,
                        window_start=fallback_start,
                        window_end=fallback_end,
                        reason=str(exc),
                        ygas_column_name=str(target_column or "").strip() or None,
                    )
                )
                skipped_windows.append(f"{path.name}({exc})")

        if not ygas_windows:
            raise ValueError(f"没有找到可用于生成“{target_element}”目标谱图的 ygas 时间窗口。")

        txt_summary = core.build_range_summary_from_entries(
            [
                {
                    "path": item["path"],
                    "start": item["start"],
                    "end": item["end"],
                    "points": int(item["parsed"].timestamp_valid_count),
                    "raw_rows": int(item["parsed"].source_row_count or len(item["parsed"].dataframe)),
                    "valid_timestamp_points": int(item["parsed"].timestamp_valid_count),
                }
                for item in ygas_windows
            ]
        )
        resolved_start, resolved_end, resolved_strategy_label = self.resolve_target_time_range_for_strategy(
            self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
            self.time_start_var.get().strip(),
            self.time_end_var.get().strip(),
            txt_summary,
            None,
        )
        effective_start_override = pd.Timestamp(start_dt) if start_dt is not None else (
            pd.Timestamp(resolved_start) if resolved_start is not None else None
        )
        effective_end_override = pd.Timestamp(end_dt) if end_dt is not None else (
            pd.Timestamp(resolved_end) if resolved_end is not None else None
        )
        time_range_policy = (
            "full_file"
            if effective_start_override is None and effective_end_override is None
            else "manual_time_range"
        )
        time_range_policy_label = "全文件" if time_range_policy == "full_file" else (
            str(resolved_strategy_label or "手动时间范围")
        )

        for group_index, item in enumerate(sorted(ygas_windows, key=lambda value: pd.Timestamp(value["start"]))):
            window_start = pd.Timestamp(item["start"])
            window_end = pd.Timestamp(item["end"])
            effective_start = max([window_start] + ([effective_start_override] if effective_start_override is not None else []))
            effective_end = min([window_end] + ([effective_end_override] if effective_end_override is not None else []))
            window_label = f"{effective_start:%Y-%m-%d %H:%M} ~ {effective_end:%H:%M}"
            if effective_start >= effective_end:
                reason = "不在当前目标时间范围内"
                skipped_windows.append(f"{item['group_key']}({reason})")
                group_records.append(
                    _build_skip_record(
                        group_key=str(item["group_key"]),
                        labels=dict(item["labels"]),
                        window_label=window_label,
                        window_start=effective_start,
                        window_end=effective_end,
                        reason=reason,
                        ygas_column_name=str(item["column"]),
                    )
                )
                continue

            try:
                ygas_frame, ygas_meta = core.build_target_window_series(
                    item["parsed"],
                    str(item["column"]),
                    effective_start,
                    effective_end,
                )
                group_requested_nsegment = int(
                    requested_nsegment
                    if use_requested_nsegment
                    else core.legacy_target_nsegment_resolver(len(ygas_frame))
                )
                ygas_fs = _resolve_ygas_fs(item["parsed"], ygas_frame, fs)
                freq, density, details = core.compute_legacy_target_psd_from_array(
                    ygas_frame["value"].to_numpy(dtype=float),
                    float(ygas_fs),
                    int(group_requested_nsegment),
                    overlap_ratio,
                    psd_kernel=selected_psd_kernel,
                )
                mask = core.build_spectrum_plot_mask(freq, density, CROSS_SPECTRUM_MAGNITUDE)
                valid_freq_points = int(np.count_nonzero(mask))
                if valid_freq_points <= 0:
                    raise ValueError("目标谱图没有生成有效正频点。")
                details = dict(details)
                details.update(
                    {
                        "device_kind": "ygas",
                        "data_context_source": "ygas_only_target_spectrum_no_dat",
                        "time_range_policy": time_range_policy,
                        "time_range_policy_label": time_range_policy_label,
                        "time_range_policy_note": str(resolved_strategy_label or time_range_policy_label),
                        "base_source_start": ygas_meta.get("source_start"),
                        "base_source_end": ygas_meta.get("source_end"),
                        "base_requested_start": ygas_meta.get("requested_start"),
                        "base_requested_end": ygas_meta.get("requested_end"),
                        "base_actual_start": ygas_meta.get("actual_start"),
                        "base_actual_end": ygas_meta.get("actual_end"),
                        "base_requested_duration_s": ygas_meta.get("requested_duration_s"),
                        "base_actual_duration_s": ygas_meta.get("actual_duration_s"),
                        "base_leading_invalid_gap_s": ygas_meta.get("leading_invalid_gap_s"),
                        "base_trailing_invalid_gap_s": ygas_meta.get("trailing_invalid_gap_s"),
                        "base_non_null_ratio": ygas_meta.get("non_null_ratio"),
                        "base_timestamp_valid_count": ygas_meta.get("timestamp_valid_count"),
                        "base_timestamp_valid_ratio": ygas_meta.get("timestamp_valid_ratio"),
                        "base_timestamp_warning": ygas_meta.get("timestamp_warning"),
                        "coverage_ratio": ygas_meta.get("coverage_ratio"),
                        "window_start": effective_start,
                        "window_end": effective_end,
                        "selected_target_column": str(item["column"]),
                    }
                )
                wrapped_series_results.append(
                    {
                        "label": str(item["labels"]["ygas_label"]),
                        "device_kind": "ygas",
                        "group_index": group_index,
                        "freq": freq,
                        "density": density,
                        "valid_points": int(ygas_meta.get("valid_points", len(ygas_frame))),
                        "valid_freq_points": valid_freq_points,
                        "details": details,
                    }
                )
                effective_nsegment = int(details.get("effective_nsegment", details.get("nsegment", 0)) or 0)
                if effective_nsegment > 0:
                    effective_nsegment_values.append(effective_nsegment)
                positive_freq_values.append(valid_freq_points)
                if details.get("first_positive_freq") is not None:
                    first_positive_freq_values.append(float(details["first_positive_freq"]))
                group_records.append(
                    {
                        "group_key": str(item["group_key"]),
                        "group_label": str(item["labels"]["group_label"]),
                        "window_label": window_label,
                        "ygas_start": effective_start,
                        "ygas_end": effective_end,
                        "dat_start": None,
                        "dat_end": None,
                        "ygas_points": int(ygas_meta.get("valid_points", len(ygas_frame))),
                        "dat_points": 0,
                        "ygas_fs": float(ygas_fs),
                        "dat_fs": None,
                        "ygas_coverage_ratio": ygas_meta.get("coverage_ratio"),
                        "dat_coverage_ratio": None,
                        "ygas_leading_invalid_gap_s": ygas_meta.get("leading_invalid_gap_s"),
                        "dat_leading_invalid_gap_s": None,
                        "ygas_trailing_invalid_gap_s": ygas_meta.get("trailing_invalid_gap_s"),
                        "dat_trailing_invalid_gap_s": None,
                        "ygas_non_null_ratio": ygas_meta.get("non_null_ratio"),
                        "dat_non_null_ratio": None,
                        "ygas_freq_points": valid_freq_points,
                        "dat_freq_points": 0,
                        "ygas_column": str(item["column"]),
                        "dat_column": "",
                        "ygas_label": str(item["labels"]["ygas_label"]),
                        "dat_label": "",
                        "keep": True,
                        "forced_include": False,
                        "reason": "未提供dat，按单设备目标谱图语义生成",
                        "status": "保留",
                        "forceable": False,
                        "group_source": "ygas_only_target_spectrum",
                        "spectrum_mode": core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
                        **core.build_legacy_target_spectrum_fields(
                            "ygas",
                            details,
                            valid_freq_points=valid_freq_points,
                        ),
                        "dat_effective_fs": None,
                        "dat_requested_nsegment": None,
                        "dat_effective_nsegment": None,
                        "dat_noverlap": None,
                        "dat_nsegment_source": None,
                        "dat_psd_kernel": None,
                        "dat_positive_freq_points": None,
                        "dat_first_positive_freq": None,
                        "dat_window_type": None,
                        "dat_detrend": None,
                        "dat_overlap": None,
                        "dat_scaling_mode": None,
                        "dat_valid_freq_points": None,
                    }
                )
            except Exception as exc:
                skipped_windows.append(f"{item['group_key']}({exc})")
                group_records.append(
                    _build_skip_record(
                        group_key=str(item["group_key"]),
                        labels=dict(item["labels"]),
                        window_label=window_label,
                        window_start=effective_start,
                        window_end=effective_end,
                        reason=str(exc),
                        ygas_column_name=str(item["column"]),
                    )
                )

        if not wrapped_series_results:
            raise ValueError("当前选择没有可用于目标谱图语义渲染的有效 ygas 谱值。")

        group_count = int(len(wrapped_series_results))
        render_semantics = self.resolve_target_spectrum_render_semantics(group_count)
        for item in wrapped_series_results:
            details = dict(item.get("details", {}))
            details["plot_execution_path"] = "target_spectrum_render"
            details["render_semantics"] = render_semantics
            details["current_plot_kind"] = "target_spectrum"
            details["selection_file_count"] = int(len(selected_txt_paths))
            details["selected_file_count"] = int(len(selected_txt_paths))
            details["selected_txt_file_count"] = int(len(selected_txt_paths))
            details["selected_dat_file_count"] = 0
            details["target_spectrum_group_count"] = group_count
            details["target_spectrum_series_count"] = group_count
            item["details"] = details

        preview_frame = self.build_target_group_preview_frame(group_records)
        qc_frame = self.build_target_group_qc_export_frame(group_records)
        kept_group_records = [
            dict(record)
            for record in group_records
            if record.get("status") in {"保留", "手动保留"}
        ]
        first_kept_record = kept_group_records[0]
        requested_nsegment_summary = int(
            requested_nsegment
            if use_requested_nsegment
            else max(
                int(record.get("ygas_requested_nsegment", 0) or 0)
                for record in kept_group_records
            )
        )
        target_metadata = {
            "mode_label": "目标谱图",
            "spectrum_mode": core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
            "ygas_column": str(first_kept_record.get("ygas_column") or target_column),
            "dat_column": None,
            "ygas_target_column": str(first_kept_record.get("ygas_column") or target_column),
            "dat_target_column": None,
            "time_range_label": (
                f"{time_range_policy_label}: "
                f"{(effective_start_override if effective_start_override is not None else pd.Timestamp(txt_summary['start'])):%Y-%m-%d %H:%M:%S}"
                " ~ "
                f"{(effective_end_override if effective_end_override is not None else pd.Timestamp(txt_summary['end'])):%Y-%m-%d %H:%M:%S}"
            ),
            "group_records": group_records,
            "group_preview_frame": preview_frame,
            "group_qc_export_frame": qc_frame,
            "total_group_count": int(len(group_records)),
            "kept_group_count": group_count,
            "skipped_group_count": int(sum(1 for record in group_records if record.get("status") == "跳过")),
            "expected_series_count": group_count,
            "actual_series_count": group_count,
            "requested_nsegment": requested_nsegment_summary,
            "legacy_target_uses_requested_nsegment": bool(use_requested_nsegment),
            "legacy_psd_kernel_requested": selected_psd_kernel,
            "legacy_psd_kernel": selected_psd_kernel,
            "legacy_psd_kernel_selection_basis": "未提供 dat，沿用目标谱图默认 PSD 核。",
            "legacy_psd_candidate_results": [],
            "reference_line_mode": "fixed_f_pow_-2_3",
            "reference_line_at_1hz": 1.0,
            "effective_nsegment_values": effective_nsegment_values,
            "positive_freq_points_min": min(positive_freq_values) if positive_freq_values else None,
            "positive_freq_points_max": max(positive_freq_values) if positive_freq_values else None,
            "first_positive_freq_min": min(first_positive_freq_values) if first_positive_freq_values else None,
            "first_positive_freq_max": max(first_positive_freq_values) if first_positive_freq_values else None,
            "matched_count_min": None,
            "matched_count_max": None,
            "tolerance_seconds_min": None,
            "tolerance_seconds_max": None,
            "generated_cross_series_count": None,
            "generated_cross_series_roles": None,
            "skipped_windows": skipped_windows,
            "time_range_policy": time_range_policy,
            "time_range_policy_label": time_range_policy_label,
            "target_spectrum_context_mode": "ygas_only_no_dat",
        }
        payload = {
            "series_results": wrapped_series_results,
            "target_element": str(target_element),
            "target_column": str(target_column),
            "target_metadata": target_metadata,
        }
        finalized = self.finalize_target_spectrum_dispatch_payload(
            payload,
            selection_file_count=int(len(selected_txt_paths)),
            selected_txt_file_count=int(len(selected_txt_paths)),
            selected_dat_file_count=0,
            data_context_source="ygas_only_target_spectrum_no_dat",
            active_compare_context=None,
            dat_context_file_name="",
        )
        finalized_target_metadata = dict(finalized.get("target_metadata", {}))
        finalized_target_metadata["target_spectrum_context_mode"] = "ygas_only_no_dat"
        finalized["target_metadata"] = finalized_target_metadata
        return finalized

    def prepare_default_target_spectrum_dispatch_payload(
        self,
        selected_files: list[Path],
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        normalized_selected_files = [Path(path) for path in selected_files]
        selection_summary = self.extract_single_device_txt_selection(normalized_selected_files)
        selected_txt_paths = [Path(path) for path in selection_summary.get("selected_txt_paths", [])]
        selected_dat_paths = [Path(path) for path in selection_summary.get("selected_dat_paths", [])]
        if not selected_txt_paths:
            raise ValueError("当前选择中没有可用于目标谱图的 txt/ygas 文件。")

        self.ensure_auto_compare_context_for_selection(
            selected_txt_paths,
            source="direct_generate_target_spectrum",
        )
        active_compare_context: dict[str, Any] | None = None
        try:
            active_compare_context = self.resolve_active_compare_context(selected_txt_paths)
        except ValueError:
            active_compare_context = None

        dat_path = selected_dat_paths[0] if selected_dat_paths else None
        data_context_source = "selected_txt_plus_dat"
        if dat_path is None and active_compare_context is not None:
            dat_path = Path(active_compare_context["dat_path"])
            data_context_source = "active_compare_context_reuse"
        if dat_path is None:
            return self.prepare_ygas_only_target_spectrum_dispatch_payload(
                selected_txt_paths,
                target_column,
                fs,
                requested_nsegment,
                overlap_ratio,
                start_dt=start_dt,
                end_dt=end_dt,
                reporter=reporter,
            )

        payload = self.prepare_target_spectrum_payload(
            ygas_paths=selected_txt_paths,
            dat_path=dat_path,
            fs_ui=fs,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            time_range_strategy=self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
            start_raw=self.time_start_var.get().strip(),
            end_raw=self.time_end_var.get().strip(),
            grouping_mode="每个 ygas 文件视为一个组",
            legacy_target_spectrum_mode=(
                self.legacy_target_spectrum_mode_var.get().strip()
                or core.LEGACY_TARGET_SPECTRUM_MODE_PSD
            ),
            use_requested_nsegment=bool(self.legacy_target_use_analysis_params_var.get()),
            forced_include_group_keys=self.get_selected_target_group_overrides(),
            reporter=reporter,
        )
        finalized = self.finalize_target_spectrum_dispatch_payload(
            payload,
            selection_file_count=int(selection_summary.get("selected_file_count", len(normalized_selected_files))),
            selected_txt_file_count=int(selection_summary.get("selected_txt_file_count", len(selected_txt_paths))),
            selected_dat_file_count=int(selection_summary.get("selected_dat_file_count", len(selected_dat_paths))),
            data_context_source=data_context_source,
            active_compare_context=active_compare_context,
            dat_context_file_name=str(dat_path.name),
        )
        finalized_target_metadata = dict(finalized.get("target_metadata", {}))
        finalized_target_metadata["target_spectrum_context_mode"] = "txt_plus_dat"
        finalized["target_metadata"] = finalized_target_metadata
        return finalized

    def prepare_selected_files_default_render_payload(
        self,
        selected_files: list[Path],
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        try:
            return self.prepare_default_target_spectrum_dispatch_payload(
                selected_files,
                target_column,
                fs,
                requested_nsegment,
                overlap_ratio,
                start_dt=start_dt,
                end_dt=end_dt,
                reporter=reporter,
            )
        except Exception as exc:
            return {
                "default_dispatch_family": "plain_spectral_fallback",
                "target_column": str(target_column),
                "target_spectrum_default_fallback_reason": str(exc),
            }

    def resolve_device_count_render_semantics(
        self,
        effective_device_count: int,
        *,
        single_device_compare_ready: bool = False,
        dual_compare_ready: bool = False,
    ) -> str:
        normalized_count = int(max(effective_device_count, 0))
        if normalized_count <= 0:
            return "empty_device_selection"
        if normalized_count == 1:
            if single_device_compare_ready:
                return "single_device_compare_psd"
            return "single_device"
        if normalized_count == 2:
            if dual_compare_ready:
                return "dual_psd_compare"
            return "multi_device_compare"
        return "multi_device_overlay"

    def resolve_device_count_plot_execution_path(
        self,
        effective_device_count: int,
        *,
        single_device_compare_ready: bool = False,
        dual_compare_ready: bool = False,
    ) -> str:
        normalized_count = int(max(effective_device_count, 0))
        if normalized_count <= 0:
            return "empty_device_selection"
        if normalized_count == 1:
            if single_device_compare_ready:
                return "single_device_compare_psd_render"
            return "single_device_spectrum"
        if normalized_count == 2:
            if dual_compare_ready:
                return "dual_psd_compare"
            return "multi_device_compare"
        return "multi_device_overlay"

    def prepare_multi_spectral_compare_payload(
        self,
        selected_files: list[Path],
        target_column: str,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        normalized_selected_files = [Path(path) for path in selected_files]
        series_results: list[dict[str, Any]] = []
        resolved_groups = self.resolve_effective_device_groups(
            normalized_selected_files,
            target_column=target_column,
            reporter=reporter,
        )
        selection_summary = dict(resolved_groups["selection_summary"])
        device_groups = list(resolved_groups["device_groups"])
        skipped_files = list(resolved_groups["skipped_files"])
        selected_txt_files = [Path(path) for path in selection_summary.get("selected_txt_paths", [])]

        active_compare_context: dict[str, Any] | None = None
        if selected_txt_files:
            try:
                active_compare_context = self.resolve_active_compare_context(selected_txt_files)
            except ValueError:
                active_compare_context = None

        compare_geometry_series_results: list[dict[str, Any]] = []
        if active_compare_context is not None:
            compare_geometry_series_results = list(
                self.build_compare_geometry_series_results(
                    context=active_compare_context,
                    target_column=target_column,
                    fs_ui=fs,
                    requested_nsegment=requested_nsegment,
                    overlap_ratio=overlap_ratio,
                )
            )

        for index, group in enumerate(device_groups, start=1):
            if reporter is not None:
                reporter(
                    f"正在计算设备图谱… {index}/{max(len(device_groups), 1)}："
                    f"{group['device_label']}（{group['file_count']} 个文件）"
                )
            try:
                merged_parsed = group["merged_parsed"]
                payload = core.compute_base_spectrum_payload(
                    merged_parsed,
                    target_column,
                    fs_ui=fs,
                    requested_nsegment=requested_nsegment,
                    overlap_ratio=overlap_ratio,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    require_timestamp=False,
                )
                details = dict(payload["details"])
                details["device_id"] = str(group["device_id"])
                details["device_label"] = str(group["device_label"])
                details["device_source"] = str(group["device_source"])
                details["group_device_source"] = str(group["device_source"])
                details["device_file_count"] = int(group["file_count"])
                details["device_group_merge_strategy"] = str(group["merge_metadata"].get("merge_strategy", ""))
                details["selection_merge_scope"] = str(group.get("selection_merge_scope") or "device_group")
                details["source_paths"] = [str(path) for path in group["source_paths"]]
                details["grouped_profile_name"] = str(merged_parsed.profile_name)
                details["group_resolved_target_columns"] = list(group.get("resolved_target_columns", []))
                details["group_target_resolution_sources"] = list(group.get("target_resolution_sources", []))
                details["raw_source_rows"] = int(
                    group["merge_metadata"].get("raw_rows", merged_parsed.source_row_count or len(merged_parsed.dataframe))
                )
                details["merged_rows"] = int(group["merge_metadata"].get("merged_points", len(merged_parsed.dataframe)))
                details["rendered_point_count"] = int(len(payload["freq"]))
                details.update(
                    core.build_single_time_range_metadata(
                        start_dt=start_dt,
                        end_dt=end_dt,
                        has_timestamp=bool(payload["series_meta"].get("has_timestamp")),
                        requested_start=details.get("base_requested_start"),
                        requested_end=details.get("base_requested_end"),
                        actual_start=details.get("base_actual_start"),
                        actual_end=details.get("base_actual_end"),
                    )
                )

                series_results.append(
                    {
                        "label": str(group["device_label"]),
                        "freq": np.asarray(payload["freq"], dtype=float),
                        "density": np.asarray(payload["density"], dtype=float),
                        "details": details,
                        "device_id": str(group["device_id"]),
                        "device_source": str(group["device_source"]),
                        "file_count": int(group["file_count"]),
                        "merge_strategy": str(group["merge_metadata"].get("merge_strategy", "")),
                        "selection_merge_scope": str(group.get("selection_merge_scope") or "device_group"),
                        "source_paths": [str(path) for path in group["source_paths"]],
                        "resolved_target_columns": list(group.get("resolved_target_columns", [])),
                        "target_resolution_sources": list(group.get("target_resolution_sources", [])),
                    }
                )
            except Exception as exc:
                skipped_files.append(f"{group['device_label']}(计算失败: {exc})")

        effective_device_count = int(len(series_results))
        effective_device_ids = [str(item.get("device_id") or "") for item in series_results if str(item.get("device_id") or "").strip()]
        selection_file_count = int(selection_summary.get("selected_file_count", len(normalized_selected_files)))
        selected_txt_file_count = int(selection_summary.get("selected_txt_file_count", 0))
        selected_dat_file_count = int(selection_summary.get("selected_dat_file_count", 0))
        exported_device_groups = [
            {
                "device_id": str(group["device_id"]),
                "device_label": str(group["device_label"]),
                "device_source": str(group["device_source"]),
                "group_device_source": str(group["device_source"]),
                "file_count": int(group["file_count"]),
                "merge_strategy": str(group["merge_metadata"].get("merge_strategy", "")),
                "selection_merge_scope": str(group.get("selection_merge_scope") or "device_group"),
                "raw_source_rows": int(
                    group["merge_metadata"].get(
                        "raw_rows",
                        group["merged_parsed"].source_row_count or len(group["merged_parsed"].dataframe),
                    )
                ),
                "merged_rows": int(group["merge_metadata"].get("merged_points", len(group["merged_parsed"].dataframe))),
                "resolved_target_columns": list(group.get("resolved_target_columns", [])),
                "target_resolution_sources": list(group.get("target_resolution_sources", [])),
            }
            for group in device_groups
        ]

        single_device_compare_payload: dict[str, Any] | None = None
        if effective_device_count == 1 and selected_txt_files:
            single_device_compare_payload, _fallback_reason = self.build_single_device_txt_compare_equivalent_payload(
                selected_txt_files,
                target_column=target_column,
                fs=fs,
                requested_nsegment=requested_nsegment,
                overlap_ratio=overlap_ratio,
                selection_summary=selection_summary,
            )
            if single_device_compare_payload is not None and not list(
                single_device_compare_payload.get("compare_geometry_series_results", [])
            ):
                single_device_compare_payload = None

        dual_compare_payload: dict[str, Any] | None = None
        if effective_device_count == 2:
            dual_compare_payload, _fallback_reason = self.build_dual_device_compare_dispatch_payload(
                selected_files=normalized_selected_files,
                target_column=target_column,
                fs=fs,
                requested_nsegment=requested_nsegment,
                overlap_ratio=overlap_ratio,
                start_dt=start_dt,
                end_dt=end_dt,
                active_compare_context=active_compare_context,
            )

        single_device_compare_ready = single_device_compare_payload is not None
        dual_compare_ready = dual_compare_payload is not None
        plot_execution_path = self.resolve_device_count_plot_execution_path(
            effective_device_count,
            single_device_compare_ready=single_device_compare_ready,
            dual_compare_ready=dual_compare_ready,
        )
        render_semantics = self.resolve_device_count_render_semantics(
            effective_device_count,
            single_device_compare_ready=single_device_compare_ready,
            dual_compare_ready=dual_compare_ready,
        )
        time_range_policy = ""
        if active_compare_context is not None:
            time_range_policy = str(active_compare_context.get("compare_context_time_range_policy") or "")
        elif series_results:
            time_range_policy = str(series_results[0]["details"].get("time_range_policy") or "")

        if effective_device_count == 1 and single_device_compare_payload is not None:
            final_payload = dict(single_device_compare_payload)
            final_data_context_source = "compare_context_reuse"
        elif effective_device_count == 2 and dual_compare_payload is not None:
            final_payload = dict(dual_compare_payload)
            final_data_context_source = (
                "compare_context_reuse"
                if active_compare_context is not None
                else str(resolved_groups.get("data_context_source") or "effective_device_grouping")
            )
        else:
            final_payload = {
                "target_column": target_column,
                "series_results": series_results,
                "skipped_files": skipped_files,
                "start_dt": start_dt,
                "end_dt": end_dt,
            }
            final_data_context_source = str(resolved_groups.get("data_context_source") or "effective_device_grouping")

        final_payload["target_column"] = target_column
        final_payload["skipped_files"] = list(final_payload.get("skipped_files", skipped_files))
        if "start_dt" not in final_payload:
            final_payload["start_dt"] = active_compare_context.get("start_dt") if active_compare_context is not None else start_dt
        if "end_dt" not in final_payload:
            final_payload["end_dt"] = active_compare_context.get("end_dt") if active_compare_context is not None else end_dt

        return self.finalize_device_dispatch_payload(
            final_payload,
            device_groups=exported_device_groups,
            effective_device_ids=effective_device_ids,
            effective_device_count=effective_device_count,
            plot_execution_path=plot_execution_path,
            render_semantics=render_semantics,
            selection_file_count=selection_file_count,
            selected_txt_file_count=selected_txt_file_count,
            selected_dat_file_count=selected_dat_file_count,
            data_context_source=final_data_context_source,
            time_range_policy=time_range_policy,
            active_compare_context=active_compare_context,
            compare_geometry_series_results=compare_geometry_series_results
            if compare_geometry_series_results
            else final_payload.get("compare_geometry_series_results"),
        )

    def dispatch_spectral_render_by_device_count(self, payload: dict[str, Any]) -> None:
        series_results = list(payload.get("series_results", []))
        effective_device_count = int(
            payload.get("effective_device_count", payload.get("device_count", len(series_results)))
        )
        if effective_device_count <= 0 or not series_results:
            raise ValueError("按有效设备数分流时没有可用设备。")
        if effective_device_count == 1 and str(payload.get("plot_execution_path") or "") == "single_device_compare_psd_render":
            self.render_single_device_compare_psd_payload(payload)
            return
        if effective_device_count == 1:
            self.render_single_device_spectrum_payload(payload)
            return
        if effective_device_count == 2 and str(payload.get("plot_execution_path") or "") == "dual_psd_compare":
            self.on_prepared_dual_plot_ready(payload)
            return
        self.render_multi_device_compare_payload(payload)

    def render_single_device_spectrum_payload(self, payload: dict[str, Any]) -> None:
        self.plot_multi_spectral_results(
            str(payload["target_column"]),
            list(payload["series_results"]),
            list(payload["skipped_files"]),
            payload=payload,
        )

    def render_multi_device_compare_payload(self, payload: dict[str, Any]) -> None:
        self.plot_multi_spectral_results(
            str(payload["target_column"]),
            list(payload["series_results"]),
            list(payload["skipped_files"]),
            payload=payload,
        )

    def on_multi_spectral_compare_ready(self, payload: dict[str, Any]) -> None:
        target_column = str(payload["target_column"])
        series_results = list(payload["series_results"])
        skipped_files = list(payload["skipped_files"])
        device_count = int(
            payload.get("effective_device_count", payload.get("device_count", len(series_results)))
        )
        if not series_results:
            self.status_var.set("设备图谱生成失败：没有有效系列")
            self.update_diagnostic_info(layout="设备分组模式", analysis_label=target_column, batch_success=0, batch_skips=len(skipped_files))
            self.render_plot_message("按设备分组后没有找到可用于生成图谱的有效系列。", level="warning")
            messagebox.showerror("分析失败", "所选文件在设备分组后没有可用于生成图谱的有效数据。")
            return
        if device_count == 1:
            mode_title = "单设备图谱"
        elif device_count == 2:
            mode_title = "设备对比图"
        else:
            mode_title = "多设备图谱"
        self.status_var.set(f"正在生成图：{mode_title} / {target_column}")
        self.root.update_idletasks()
        self.dispatch_spectral_render_by_device_count(payload)

    def on_selected_files_default_plot_ready(self, payload: dict[str, Any]) -> None:
        if str(payload.get("default_dispatch_family") or "") == "target_spectrum":
            target_element = str(payload.get("target_element") or payload.get("target_column") or "目标谱图")
            self.status_var.set(f"正在生成图：目标谱图 / {target_element}")
            self.root.update_idletasks()
            self.on_target_spectrum_ready(payload)
            return
        if str(payload.get("default_dispatch_family") or "") == "plain_spectral_fallback":
            fallback_reason = str(payload.get("target_spectrum_default_fallback_reason") or "").strip()
            target_column = str(payload.get("target_column") or (self.selected_columns[0] if self.selected_columns else ""))
            fallback_quiet = bool(self.pending_selected_files_default_render_quiet)
            if fallback_reason:
                self.render_plot_message(
                    f"目标谱图主链不可用，已回退普通功率谱密度：{fallback_reason}",
                    level="warning",
                )
            if target_column:
                self.status_var.set(f"正在生成图：功率谱密度 / {target_column}")
                self.root.update_idletasks()
            previous_reason = self.pending_direct_generate_target_spectrum_fallback_reason
            self.pending_direct_generate_target_spectrum_fallback_reason = (
                fallback_reason or "target_spectrum_payload_unavailable"
            )
            try:
                self.perform_analysis("spectral", quiet=fallback_quiet)
            finally:
                self.pending_direct_generate_target_spectrum_fallback_reason = previous_reason
            return
        self.on_multi_spectral_compare_ready(payload)

    def prepare_dual_compare_payload(
        self,
        *,
        ygas_paths: list[Path],
        dat_path: Path | None,
        selected_paths: list[Path],
        compare_mode: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        if dat_path is not None and ygas_paths:
            target_context: dict[str, Any] | None = None
            required_ygas_columns: list[str] | None = None
            required_dat_columns: list[str] | None = None
            if self.is_target_cross_compare_mode(compare_mode):
                target_element = self.resolve_target_element_name()
                required_ygas_columns = self.build_target_required_columns(
                    target_element=target_element,
                    device_kind="ygas",
                )
                required_dat_columns = self.build_target_required_columns(
                    target_element=target_element,
                    device_kind="dat",
                    include_reference=True,
                )
            parsed_a, label_a, parsed_b, label_b, selection_meta = self.build_compare_selection_from_paths(
                ygas_paths,
                dat_path,
                required_ygas_columns=required_ygas_columns,
                required_dat_columns=required_dat_columns,
                target_context=target_context,
                reporter=reporter,
            )
            if self.is_target_cross_compare_mode(compare_mode):
                selection_meta["target_spectral_context"] = self.resolve_target_spectral_context_for_parsed(
                    parsed_a,
                    parsed_b,
                    spectrum_mode=compare_mode,
                    comparison_is_target_spectral=True,
                )
                if selection_meta["target_spectral_context"] is None:
                    raise ValueError("当前 ygas+dat 目标频谱场景未找到完整的 Uz -> target canonical pair。")
            return {
                "parsed_a": parsed_a,
                "label_a": label_a,
                "parsed_b": parsed_b,
                "label_b": label_b,
                "selection_meta": selection_meta,
                "compare_mode": compare_mode,
                "start_dt": start_dt,
                "end_dt": end_dt,
            }

        if len(selected_paths) < 2:
            raise ValueError("请先选择两个文件，或选择多个 txt/log 加 1 个 dat 文件。")

        if reporter is not None:
            reporter("正在解析设备A文件…")
        parsed_a = self.parse_profiled_file(selected_paths[0])
        if reporter is not None:
            reporter("正在解析设备B文件…")
        parsed_b = self.parse_profiled_file(selected_paths[1])
        return {
            "parsed_a": parsed_a,
            "label_a": self.format_source_label(selected_paths[0]),
            "parsed_b": parsed_b,
            "label_b": self.format_source_label(selected_paths[1]),
            "selection_meta": {
                "txt_summary": None,
                "dat_summary": None,
            },
            "compare_mode": compare_mode,
            "start_dt": start_dt,
            "end_dt": end_dt,
        }

    def on_dual_compare_payload_ready(self, payload: dict[str, Any]) -> None:
        parsed_a = payload["parsed_a"]
        label_a = str(payload["label_a"])
        parsed_b = payload["parsed_b"]
        label_b = str(payload["label_b"])
        selection_meta = dict(payload["selection_meta"])
        target_context = core.get_target_spectral_context(selection_meta)
        compare_mode = str(payload["compare_mode"])
        start_dt = payload["start_dt"]
        end_dt = payload["end_dt"]

        if selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None:
            start_dt, end_dt = self.resolve_compare_time_range(selection_meta["txt_summary"], selection_meta["dat_summary"])

        self.update_compare_column_values(self.device_a_combo, self.device_a_column_var, parsed_a.suggested_columns)
        self.update_compare_column_values(self.device_b_combo, self.device_b_column_var, parsed_b.suggested_columns)
        hit_a, hit_b, mapping_name = self.apply_element_mapping(parsed_a, parsed_b)
        self.compare_file_info_var.set(
            f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 映射={mapping_name}"
        )
        if self.mapping_mode_var.get() == "预设映射" and (hit_a is None or hit_b is None):
            self.compare_file_info_var.set(
                f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 预设“{mapping_name}”未完全命中"
            )
        self.update_compare_summaries(selection_meta.get("txt_summary"), selection_meta.get("dat_summary"))

        if (
            self.mapping_mode_var.get() == "预设映射"
            and (hit_a is None or hit_b is None)
            and target_context is None
        ):
            raise ValueError(f"预设映射“{mapping_name}”未命中，请切换为手动映射或调整列选择。")

        if target_context is not None and self.is_target_cross_compare_mode(compare_mode):
            pairs = [
                {
                    "a_col": str(target_context.get("ygas_target_column") or target_context.get("target_column")),
                    "b_col": str(target_context["reference_column"]),
                    "label": f"{target_context['display_target_label']}(ygas) vs {target_context['reference_column']}",
                },
                {
                    "a_col": str(target_context.get("dat_target_column") or ""),
                    "b_col": str(target_context["reference_column"]),
                    "label": f"{target_context['display_target_label']}(dat) vs {target_context['reference_column']}",
                },
            ]
        else:
            pairs = self.build_compare_pairs()
            if not pairs:
                raise ValueError("当前没有可用于比对的列映射，请先选择设备A和设备B的目标列。")

        compare_scope = self.compare_scope_var.get().strip() or "单对单"
        alignment_strategy = self.get_alignment_strategy(compare_mode)
        plot_style = self.resolve_plot_style(compare_mode)
        plot_layout = self.resolve_plot_layout(compare_mode)
        fs_ui, requested_nsegment, overlap_ratio = self.get_analysis_params()
        match_tolerance = self.get_match_tolerance_seconds(default_value=0.2)
        spectrum_type = self.get_selected_cross_spectrum_type(compare_mode)
        scheme_name = self.scheme_name_var.get().strip() or "默认方案"
        self.start_background_task(
            status_text=f"正在准备双设备比对数据：{compare_mode}",
            worker=lambda reporter: self.prepare_dual_plot_payload(
                parsed_a=parsed_a,
                label_a=label_a,
                parsed_b=parsed_b,
                label_b=label_b,
                pairs=pairs,
                selection_meta=selection_meta,
                compare_mode=compare_mode,
                compare_scope=compare_scope,
                start_dt=start_dt,
                end_dt=end_dt,
                mapping_name=mapping_name,
                scheme_name=scheme_name,
                alignment_strategy=alignment_strategy,
                plot_style=plot_style,
                plot_layout=plot_layout,
                fs_ui=fs_ui,
                requested_nsegment=requested_nsegment,
                overlap_ratio=overlap_ratio,
                match_tolerance=match_tolerance,
                spectrum_type=spectrum_type,
                time_range_context={
                    "strategy_label": self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围",
                    "has_txt_dat_context": bool(
                        selection_meta.get("txt_summary") is not None and selection_meta.get("dat_summary") is not None
                    ),
                },
                reporter=reporter,
            ),
            on_success=self.on_prepared_dual_plot_ready,
            error_title="双设备比对",
        )

    def prepare_dual_plot_payload(
        self,
        *,
        parsed_a: ParsedFileResult,
        label_a: str,
        parsed_b: ParsedFileResult,
        label_b: str,
        pairs: list[dict[str, str]],
        selection_meta: dict[str, Any] | None,
        compare_mode: str,
        compare_scope: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        mapping_name: str,
        scheme_name: str,
        alignment_strategy: str,
        plot_style: str,
        plot_layout: str,
        fs_ui: float,
        requested_nsegment: int,
        overlap_ratio: float,
        match_tolerance: float,
        spectrum_type: str,
        time_range_context: dict[str, Any] | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        base_payload = {
            "parsed_a": parsed_a,
            "label_a": label_a,
            "parsed_b": parsed_b,
            "label_b": label_b,
            "pairs": pairs,
            "selection_meta": dict(selection_meta or {}),
            "compare_mode": compare_mode,
            "compare_scope": compare_scope,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "mapping_name": mapping_name,
            "scheme_name": scheme_name,
            "alignment_strategy": alignment_strategy,
            "plot_style": plot_style,
            "plot_layout": plot_layout,
            "fs_ui": fs_ui,
            "requested_nsegment": requested_nsegment,
            "overlap_ratio": overlap_ratio,
            "match_tolerance": match_tolerance,
            "spectrum_type": spectrum_type,
            "time_range_context": dict(time_range_context or {}),
        }

        if compare_mode == "时间序列对比":
            if reporter is not None:
                reporter("正在准备时间序列数据…")
            export_frames: list[pd.DataFrame] = []
            exported_a: set[str] = set()
            pair_data: list[dict[str, Any]] = []
            plotted_count = 0
            total_a_points = 0
            total_b_points = 0
            for index, pair in enumerate(pairs):
                series_a, meta_a = self.prepare_compare_series(parsed_a, pair["a_col"], start_dt, end_dt, "设备A")
                series_b, meta_b = self.prepare_compare_series(parsed_b, pair["b_col"], start_dt, end_dt, "设备B")
                a_name = meta_a["value_name"]
                b_name = meta_b["value_name"]
                pair_data.append(
                    {
                        "pair": pair,
                        "series_a": series_a,
                        "series_b": series_b,
                        "a_name": a_name,
                        "b_name": b_name,
                        "color_a": SERIES_COLORS[index % len(SERIES_COLORS)],
                        "color_b": SERIES_COLORS[(index + 1) % len(SERIES_COLORS)] if len(pairs) == 1 else SERIES_COLORS[(index + 3) % len(SERIES_COLORS)],
                    }
                )
                if pair["a_col"] not in exported_a:
                    export_frames.append(series_a[["时间戳", a_name]].rename(columns={a_name: f"{label_a}_{pair['a_col']}"}))
                    exported_a.add(pair["a_col"])
                export_frames.append(series_b[["时间戳", b_name]].rename(columns={b_name: f"{label_b}_{pair['b_col']}"}))
                plotted_count += 1
                total_a_points = max(total_a_points, meta_a["valid_count"])
                total_b_points = max(total_b_points, meta_b["valid_count"])
            if plotted_count == 0:
                raise ValueError("没有可用于时间序列对比的有效系列。")
            return {
                **base_payload,
                "kind": "time_series",
                "pair_data": pair_data,
                "export_frames": export_frames,
                "plotted_count": plotted_count,
                "total_a_points": total_a_points,
                "total_b_points": total_b_points,
            }

        if compare_mode == "散点一致性对比":
            if reporter is not None:
                reporter("正在时间对齐…")
            if len(pairs) == 1 and compare_scope == "单对单":
                pair = pairs[0]
                aligned, meta = self.align_compare_frames(
                    parsed_a,
                    pair["a_col"],
                    parsed_b,
                    pair["b_col"],
                    start_dt,
                    end_dt,
                    tolerance_seconds=match_tolerance,
                    alignment_strategy=alignment_strategy,
                )
                value_a = meta["meta_a"]["value_name"]
                value_b = meta["meta_b"]["value_name"]
                diff = aligned[value_a] - aligned[value_b]
                return {
                    **base_payload,
                    "kind": "scatter_single",
                    "pair": pair,
                    "aligned": aligned,
                    "meta": meta,
                    "mean_diff": float(diff.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                    "corr": float(aligned[value_a].corr(aligned[value_b])),
                }

            series_items: list[dict[str, Any]] = []
            export_frames: list[pd.DataFrame] = []
            success_count = 0
            matched_points_total = 0
            for index, pair in enumerate(pairs):
                aligned, meta = self.align_compare_frames(
                    parsed_a,
                    pair["a_col"],
                    parsed_b,
                    pair["b_col"],
                    start_dt,
                    end_dt,
                    tolerance_seconds=match_tolerance,
                    alignment_strategy=alignment_strategy,
                )
                value_a = meta["meta_a"]["value_name"]
                value_b = meta["meta_b"]["value_name"]
                series_items.append(
                    {
                        "pair": pair,
                        "x": aligned[value_a].to_numpy(dtype=float),
                        "y": aligned[value_b].to_numpy(dtype=float),
                        "color": SERIES_COLORS[index % len(SERIES_COLORS)],
                    }
                )
                export_frames.append(
                    aligned[["时间戳", value_a, value_b]].rename(
                        columns={value_a: f"{label_a}_{pair['label']}_A", value_b: f"{label_b}_{pair['label']}_B"}
                    )
                )
                success_count += 1
                matched_points_total += meta["matched_count"]
            if success_count == 0:
                raise ValueError("散点一致性对比没有得到可用的有效结果。")
            return {
                **base_payload,
                "kind": "scatter_multi",
                "series_items": series_items,
                "export_frames": export_frames,
                "success_count": success_count,
                "matched_points_total": matched_points_total,
            }

        if compare_mode == "时间段内 PSD 对比":
            if reporter is not None:
                reporter("正在时间对齐…")
            plotted_keys: set[tuple[str, str]] = set()
            series_results: list[dict[str, Any]] = []
            txt_points = 0
            dat_points = 0
            fs_a_last = DEFAULT_FS
            fs_b_last = DEFAULT_FS
            if reporter is not None:
                reporter("正在计算 PSD/CSD…")
            for pair in pairs:
                for side, label, column, parsed in (
                    ("txt", label_a, pair["a_col"], parsed_a),
                    ("dat", label_b, pair["b_col"], parsed_b),
                ):
                    key = (side, column)
                    if key in plotted_keys:
                        continue
                    plotted_keys.add(key)
                    psd_payload = core.compute_base_spectrum_payload(
                        parsed,
                        column,
                        fs_ui=fs_ui,
                        requested_nsegment=requested_nsegment,
                        overlap_ratio=overlap_ratio,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        require_timestamp=True,
                    )
                    details = dict(psd_payload["details"])
                    details.update(
                        core.build_compare_time_range_metadata(
                            strategy_label=str(base_payload["time_range_context"].get("strategy_label") or "使用 txt+dat 共同时间范围"),
                            start_dt=start_dt,
                            end_dt=end_dt,
                            has_txt_dat_context=bool(base_payload["time_range_context"].get("has_txt_dat_context")),
                            requested_start=details.get("base_requested_start"),
                            requested_end=details.get("base_requested_end"),
                            actual_start=details.get("base_actual_start"),
                            actual_end=details.get("base_actual_end"),
                        )
                    )
                    if side == "txt":
                        txt_points = max(txt_points, int(psd_payload["series_meta"].get("valid_points", 0)))
                        fs_a_last = float(psd_payload["effective_fs"])
                    else:
                        dat_points = max(dat_points, int(psd_payload["series_meta"].get("valid_points", 0)))
                        fs_b_last = float(psd_payload["effective_fs"])
                    series_results.append(
                        {
                            "label": f"{label} - {column}",
                            "side": side,
                            "column": column,
                            "freq": np.asarray(psd_payload["freq"], dtype=float),
                            "density": np.asarray(psd_payload["density"], dtype=float),
                            "details": details,
                        }
                    )
            if not series_results:
                raise ValueError("时间段内 PSD 对比没有得到可用的有效谱值。")
            return {
                **base_payload,
                "kind": "psd_compare",
                "series_results": series_results,
                "txt_points": txt_points,
                "dat_points": dat_points,
                "fs_a_last": fs_a_last,
                "fs_b_last": fs_b_last,
            }

        target_context = core.get_target_spectral_context(base_payload["selection_meta"])
        if compare_mode in {"互谱幅值", "协谱图", "正交谱图"} and target_context is not None:
            if reporter is not None:
                reporter("正在按目标谱图 canonical 双系列链路对齐 Uz 与目标量…")
            reference_series, reference_meta = self.prepare_compare_series(
                parsed_b,
                str(target_context["reference_column"]),
                start_dt,
                end_dt,
                "设备B",
            )
            reference_frame = reference_series[["时间戳", reference_meta["value_name"]]].rename(
                columns={reference_meta["value_name"]: "value"}
            )
            reference_fs = self.resolve_compare_fs(parsed_b, fs_ui)
            canonical_specs = [
                {
                    "series_role": "ygas_target_vs_uz",
                    "device_kind": "cross_ygas",
                    "source_kind": "ygas",
                    "target_column": str(target_context.get("ygas_target_column") or target_context.get("target_column") or ""),
                    "label": f"{target_context['display_target_label']}(ygas) vs {target_context['reference_column']}",
                    "parsed": parsed_a,
                },
                {
                    "series_role": "dat_target_vs_uz",
                    "device_kind": "cross_dat",
                    "source_kind": "dat",
                    "target_column": str(target_context.get("dat_target_column") or ""),
                    "label": f"{target_context['display_target_label']}(dat) vs {target_context['reference_column']}",
                    "parsed": parsed_b,
                },
            ]
            plot_series: list[dict[str, Any]] = []
            export_frames: list[pd.DataFrame] = []
            aligned_frames: list[pd.DataFrame] = []
            fs_used_values: list[float] = []
            series_details: list[dict[str, Any]] = []
            matched_points_total = 0
            total_a_points = 0
            total_b_points = int(reference_meta["valid_count"])
            for index, spec in enumerate(canonical_specs):
                target_column = str(spec["target_column"])
                if not target_column:
                    continue
                target_series, target_meta = self.prepare_compare_series(
                    spec["parsed"],
                    target_column,
                    start_dt,
                    end_dt,
                    "设备A" if spec["source_kind"] == "ygas" else "设备B",
                )
                target_frame = target_series[["时间戳", target_meta["value_name"]]].rename(
                    columns={target_meta["value_name"]: "value"}
                )
                target_fs = self.resolve_compare_fs(spec["parsed"], fs_ui)
                cross_payload = core.compute_target_cross_spectrum_payload(
                    target_frame=target_frame,
                    reference_frame=reference_frame,
                    target_fs=float(target_fs),
                    reference_fs=float(reference_fs),
                    target_element=str(target_context["display_target_label"]),
                    reference_column=str(target_context["reference_column"]),
                    target_column=target_column,
                    requested_nsegment=requested_nsegment,
                    overlap_ratio=overlap_ratio,
                    spectrum_type=spectrum_type,
                    use_requested_nsegment=True,
                    insufficient_message="当前数据不足以生成有效互谱/协谱结果。",
                    display_target_label=str(target_context["display_target_label"]),
                    target_context=target_context,
                    series_role=str(spec["series_role"]),
                    device_kind=str(spec["device_kind"]),
                    display_label=str(spec["label"]),
                )
                details = dict(cross_payload["details"])
                details["spectrum_mode"] = compare_mode
                details["target_spectral_context"] = target_context
                plot_series.append(
                    {
                        "label": str(spec["label"]),
                        "color": self.resolve_target_cross_series_color(str(spec["device_kind"]), 0)
                        or SERIES_COLORS[index % len(SERIES_COLORS)],
                        "freq": np.asarray(cross_payload["freq"])[np.asarray(cross_payload["mask"], dtype=bool)],
                        "values": np.asarray(cross_payload["values"])[np.asarray(cross_payload["mask"], dtype=bool)],
                        "series_role": str(spec["series_role"]),
                        "device_kind": str(spec["device_kind"]),
                    }
                )
                export_frames.append(cross_payload["export_frame"])
                aligned_frames.append(cross_payload["aligned_frame"])
                fs_used_values.append(float(details["effective_fs"]))
                series_details.append(details)
                matched_points_total += int(cross_payload["matched_count"])
                if spec["source_kind"] == "ygas":
                    total_a_points = max(total_a_points, int(target_meta["valid_count"]))
                total_b_points = max(total_b_points, int(target_meta["valid_count"]))
            if not plot_series:
                raise ValueError("当前 ygas+dat 目标频谱场景没有生成有效的 canonical cross 系列。")
            details = dict(series_details[0])
            details["generated_cross_series_count"] = int(len(plot_series))
            details["generated_cross_series_roles"] = [str(item["series_role"]) for item in plot_series]
            details["canonical_cross_pairs"] = list(target_context.get("canonical_cross_pairs") or [])
            details["ygas_target_column"] = target_context.get("ygas_target_column")
            details["dat_target_column"] = target_context.get("dat_target_column")
            return {
                **base_payload,
                "kind": "cross_compare",
                "plot_series": plot_series,
                "export_frames": export_frames,
                "aligned_frames": aligned_frames,
                "success_count": int(len(plot_series)),
                "matched_points_total": int(matched_points_total),
                "total_a_points": int(total_a_points),
                "total_b_points": int(total_b_points),
                "fs_used_values": fs_used_values,
                "last_details": details,
                "series_details": series_details,
                "canonical_pairs": canonical_specs,
                "target_spectral_context": target_context,
                "uses_target_spectral_context": True,
            }

        if compare_mode in {"互谱幅值", "协谱图", "正交谱图"}:
            if reporter is not None:
                reporter("正在时间对齐…")
            export_frames: list[pd.DataFrame] = []
            aligned_frames: list[pd.DataFrame] = []
            plot_series: list[dict[str, Any]] = []
            success_count = 0
            matched_points_total = 0
            total_a_points = 0
            total_b_points = 0
            fs_used_values: list[float] = []
            last_details: dict[str, Any] | None = None
            if reporter is not None:
                reporter("正在计算 PSD/CSD…")
            for index, pair in enumerate(pairs):
                aligned, meta = self.align_compare_frames(
                    parsed_a,
                    pair["a_col"],
                    parsed_b,
                    pair["b_col"],
                    start_dt,
                    end_dt,
                    tolerance_seconds=match_tolerance,
                    alignment_strategy=alignment_strategy,
                )
                value_a = meta["meta_a"]["value_name"]
                value_b = meta["meta_b"]["value_name"]
                fs_aligned = estimate_fs_from_timestamp(aligned, "时间戳") or fs_ui
                freq, values, details = self.compute_cross_spectrum_from_arrays_with_params(
                    aligned[value_a].to_numpy(dtype=float),
                    aligned[value_b].to_numpy(dtype=float),
                    fs_aligned,
                    requested_nsegment,
                    overlap_ratio,
                    spectrum_type=spectrum_type,
                    insufficient_message="当前数据不足以生成有效协谱图。",
                )
                details = dict(details)
                details.update(core.describe_generic_default_cross_implementation())
                details["alignment_strategy"] = alignment_strategy
                details["cross_reference_column"] = pair["a_col"]
                details["reference_column"] = pair["a_col"]
                details["target_column"] = pair["b_col"]
                details["cross_order"] = f"{pair['a_col']} -> {pair['b_col']}"
                values, mask, details = core.resolve_cross_display_output(
                    freq,
                    details,
                    analysis_context=details.get("analysis_context"),
                    cross_execution_path=details.get("cross_execution_path"),
                    spectrum_type=spectrum_type,
                    insufficient_message="当前数据不足以生成有效协谱图。",
                )
                if not np.any(mask):
                    continue
                export_frames.append(
                    self.build_cross_export_frame(
                        freq,
                        details,
                        mask,
                        prefix=None if len(pairs) == 1 else pair["label"],
                    )
                )
                aligned_frames.append(
                    aligned[["时间戳", value_a, value_b]].rename(
                        columns={
                            value_a: "设备A值" if len(pairs) == 1 else f"{pair['label']}_设备A值",
                            value_b: "设备B值" if len(pairs) == 1 else f"{pair['label']}_设备B值",
                        }
                    )
                )
                plot_series.append(
                    {
                        "label": pair["label"],
                        "color": SERIES_COLORS[index % len(SERIES_COLORS)],
                        "freq": freq[mask],
                        "values": values[mask],
                    }
                )
                success_count += 1
                matched_points_total += meta["matched_count"]
                total_a_points = max(total_a_points, meta["meta_a"]["valid_count"])
                total_b_points = max(total_b_points, meta["meta_b"]["valid_count"])
                fs_used_values.append(fs_aligned)
                last_details = details
            if success_count == 0:
                raise ValueError("当前数据不足以生成有效协谱图。")
            return {
                **base_payload,
                "kind": "cross_compare",
                "plot_series": plot_series,
                "export_frames": export_frames,
                "aligned_frames": aligned_frames,
                "success_count": success_count,
                "matched_points_total": matched_points_total,
                "total_a_points": total_a_points,
                "total_b_points": total_b_points,
                "fs_used_values": fs_used_values,
                "last_details": last_details,
            }

        if reporter is not None:
            reporter("正在时间对齐…")
        export_frames: list[pd.DataFrame] = []
        plot_series: list[dict[str, Any]] = []
        success_count = 0
        matched_points_total = 0
        if reporter is not None:
            reporter("正在计算结果序列…")
        for index, pair in enumerate(pairs):
            aligned, meta = self.align_compare_frames(
                parsed_a,
                pair["a_col"],
                parsed_b,
                pair["b_col"],
                start_dt,
                end_dt,
                tolerance_seconds=match_tolerance,
                alignment_strategy=alignment_strategy,
            )
            value_a = meta["meta_a"]["value_name"]
            value_b = meta["meta_b"]["value_name"]
            if compare_mode == "差值时间序列":
                transformed = aligned[value_a] - aligned[value_b]
                prefix = "差值"
            else:
                denominator = aligned[value_b].where(np.abs(aligned[value_b]) > 1e-12)
                transformed = aligned[value_a] / denominator
                prefix = "比值"
            transformed = transformed.replace([np.inf, -np.inf], np.nan)
            valid = np.isfinite(transformed.to_numpy(dtype=float))
            transformed_frame = pd.DataFrame(
                {
                    "时间戳": aligned.loc[valid, "时间戳"],
                    f"{prefix}_{pair['label']}": transformed.loc[valid],
                }
            ).dropna()
            if len(transformed_frame) < 2:
                continue
            plot_series.append(
                {
                    "label": pair["label"],
                    "color": SERIES_COLORS[index % len(SERIES_COLORS)],
                    "frame": transformed_frame,
                }
            )
            export_frames.append(transformed_frame)
            success_count += 1
            matched_points_total += meta["matched_count"]
        if success_count == 0:
            raise ValueError(f"{compare_mode}没有得到可用的有效结果。")
        return {
            **base_payload,
            "kind": "transformed",
            "plot_series": plot_series,
            "export_frames": export_frames,
            "success_count": success_count,
            "matched_points_total": matched_points_total,
        }

    def on_prepared_dual_plot_ready(self, payload: dict[str, Any]) -> None:
        kind = str(payload["kind"])
        self.status_var.set(f"正在生成图：{payload['compare_mode']}")
        self.root.update_idletasks()
        if kind == "time_series":
            self.render_time_series_compare_payload(payload)
            return
        if kind == "scatter_single":
            pair = payload["pair"]
            self.plot_scatter_consistency_compare(
                [payload["label_a"], payload["label_b"]],
                payload["parsed_a"],
                pair["a_col"],
                payload["parsed_b"],
                pair["b_col"],
                payload["aligned"],
                payload["meta"],
            )
            self.current_aligned_metadata.update(
                {
                    "方案名": payload["scheme_name"],
                    "映射名称": payload["mapping_name"],
                    "匹配点数": payload["meta"]["matched_count"],
                    "成功系列数": 1,
                }
            )
            return
        if kind == "scatter_multi":
            self.render_scatter_multi_payload(payload)
            return
        if kind == "psd_compare":
            self.render_psd_compare_series(
                payload["label_a"],
                payload["label_b"],
                payload["parsed_a"],
                payload["parsed_b"],
                payload["series_results"],
                mapping_name=payload["mapping_name"],
                start_dt=payload["start_dt"],
                end_dt=payload["end_dt"],
                txt_points=payload["txt_points"],
                dat_points=payload["dat_points"],
                fs_a_last=payload["fs_a_last"],
                fs_b_last=payload["fs_b_last"],
            )
            return
        if kind == "cross_compare":
            self.render_cross_compare_payload(payload)
            return
        if kind == "transformed":
            self.render_transformed_compare_payload(payload)
            return
        raise ValueError(f"未识别的双设备绘图数据类型：{kind}")

    def render_time_series_compare_payload(self, payload: dict[str, Any]) -> None:
        label_a = payload["label_a"]
        label_b = payload["label_b"]
        pair_data = payload["pair_data"]
        style = payload["plot_style"]
        layout = payload["plot_layout"]
        mapping_name = payload["mapping_name"]
        parsed_a = payload["parsed_a"]
        parsed_b = payload["parsed_b"]
        export_frames = payload["export_frames"]
        plotted_count = int(payload["plotted_count"])
        total_a_points = int(payload["total_a_points"])
        total_b_points = int(payload["total_b_points"])
        start_dt = payload["start_dt"]
        end_dt = payload["end_dt"]

        self.reset_plot_output_state()
        if layout == "叠加同图":
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            plotted_a: set[str] = set()
            for item in pair_data:
                pair = item["pair"]
                if pair["a_col"] not in plotted_a:
                    self.apply_series_style(
                        ax,
                        item["series_a"]["时间戳"],
                        item["series_a"][item["a_name"]],
                        color=item["color_a"],
                        label=f"{label_a} - {pair['a_col']}",
                        style=style,
                    )
                    plotted_a.add(pair["a_col"])
                self.apply_series_style(
                    ax,
                    item["series_b"]["时间戳"],
                    item["series_b"][item["b_name"]],
                    color=item["color_b"],
                    label=f"{label_b} - {pair['b_col']}",
                    style=style,
                )
            ax.set_title("时间序列对比")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.3)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="upper right")
            self.figure.autofmt_xdate()
            self.figure.tight_layout()
            self.refresh_canvas()
        elif layout in {"上下分图", "左右分图"}:
            self.figure.clear()
            axes = self.figure.subplots(2, 1, sharex=True) if layout == "上下分图" else self.figure.subplots(1, 2)
            ax_a, ax_b = axes[0], axes[1]
            plotted_a: set[str] = set()
            for item in pair_data:
                pair = item["pair"]
                if pair["a_col"] not in plotted_a:
                    self.apply_series_style(
                        ax_a,
                        item["series_a"]["时间戳"],
                        item["series_a"][item["a_name"]],
                        color=item["color_a"],
                        label=f"{label_a} - {pair['a_col']}",
                        style=style,
                    )
                    plotted_a.add(pair["a_col"])
                self.apply_series_style(
                    ax_b,
                    item["series_b"]["时间戳"],
                    item["series_b"][item["b_name"]],
                    color=item["color_b"],
                    label=f"{label_b} - {pair['b_col']}",
                    style=style,
                )
            ax_a.set_title(f"设备A：{'、'.join(sorted({item['pair']['a_col'] for item in pair_data}))}")
            ax_b.set_title(f"设备B：{'、'.join(sorted({item['pair']['b_col'] for item in pair_data}))}")
            for axis in (ax_a, ax_b):
                axis.set_xlabel("Time")
                axis.set_ylabel("Value")
                axis.grid(True, linestyle="--", alpha=0.3)
                if axis.get_legend_handles_labels()[1]:
                    axis.legend(loc="upper right")
            self.figure.autofmt_xdate()
            self.figure.tight_layout()
            self.refresh_canvas()
        else:
            self.figure.clear()
            axes = self.figure.subplots(2, 1, sharex=True)
            ax_a, ax_b = axes[0], axes[1]
            plotted_a: set[str] = set()
            for item in pair_data:
                pair = item["pair"]
                if pair["a_col"] in plotted_a:
                    continue
                self.apply_series_style(
                    ax_a,
                    item["series_a"]["时间戳"],
                    item["series_a"][item["a_name"]],
                    color=item["color_a"],
                    label=f"{label_a} - {pair['a_col']}",
                    style=style,
                )
                plotted_a.add(pair["a_col"])
            for item in pair_data:
                pair = item["pair"]
                self.apply_series_style(
                    ax_b,
                    item["series_b"]["时间戳"],
                    item["series_b"][item["b_name"]],
                    color=item["color_b"],
                    label=f"{label_b} - {pair['b_col']}",
                    style=style,
                )
            ax_a.set_title(f"设备A：{'、'.join(sorted({item['pair']['a_col'] for item in pair_data}))}")
            ax_b.set_title(f"设备B：{'、'.join(sorted({item['pair']['b_col'] for item in pair_data}))}")
            for axis in (ax_a, ax_b):
                axis.set_xlabel("Time")
                axis.set_ylabel("Value")
                axis.grid(True, linestyle="--", alpha=0.3)
                if axis.get_legend_handles_labels()[1]:
                    axis.legend(loc="upper right")
            self.figure.autofmt_xdate()
            self.figure.tight_layout()
            self.refresh_canvas(title="图形结果：主页面总览图")

            if self.use_separate_zoom_windows_var.get():
                entries: list[dict[str, Any]] = []
                fig_a = Figure(figsize=(6, 4), dpi=100)
                ax_a = fig_a.add_subplot(111)
                plotted_a = set()
                for item in pair_data:
                    pair = item["pair"]
                    if pair["a_col"] in plotted_a:
                        continue
                    self.apply_series_style(
                        ax_a,
                        item["series_a"]["时间戳"],
                        item["series_a"][item["a_name"]],
                        color=item["color_a"],
                        label=f"{label_a} - {pair['a_col']}",
                        style=style,
                    )
                    plotted_a.add(pair["a_col"])
                ax_a.set_title(f"设备A：{'、'.join(sorted({item['pair']['a_col'] for item in pair_data}))}")
                ax_a.set_xlabel("Time")
                ax_a.set_ylabel("Value")
                ax_a.grid(True, linestyle="--", alpha=0.3)
                if ax_a.get_legend_handles_labels()[1]:
                    ax_a.legend(loc="upper right")
                fig_a.autofmt_xdate()
                fig_a.tight_layout()
                entries.append(
                    {
                        "figure": fig_a,
                        "title": "设备A - 时间序列对比",
                        "filename": f"deviceA_{sanitize_filename(pair_data[0]['pair']['a_col'])}_{self.get_plot_style_tag(style)}.png",
                    }
                )
                fig_b = Figure(figsize=(6, 4), dpi=100)
                ax_b = fig_b.add_subplot(111)
                for item in pair_data:
                    pair = item["pair"]
                    self.apply_series_style(
                        ax_b,
                        item["series_b"]["时间戳"],
                        item["series_b"][item["b_name"]],
                        color=item["color_b"],
                        label=f"{label_b} - {pair['b_col']}",
                        style=style,
                    )
                ax_b.set_title(f"设备B：{'、'.join(sorted({item['pair']['b_col'] for item in pair_data}))}")
                ax_b.set_xlabel("Time")
                ax_b.set_ylabel("Value")
                ax_b.grid(True, linestyle="--", alpha=0.3)
                if ax_b.get_legend_handles_labels()[1]:
                    ax_b.legend(loc="upper right")
                fig_b.autofmt_xdate()
                fig_b.tight_layout()
                entries.append(
                    {
                        "figure": fig_b,
                        "title": "设备B - 时间序列对比",
                        "filename": f"deviceB_{sanitize_filename(pair_data[0]['pair']['b_col'])}_{self.get_plot_style_tag(style)}.png",
                    }
                )
                self.show_separate_plot_windows(entries)

        self.current_plot_kind = "aligned_time_series"
        self.current_plot_columns = [pair["pair"]["label"] for pair in pair_data]
        self.current_result_frame = None
        self.current_result_freq = None
        self.current_result_values = None
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = list(export_frames)
        self.current_compare_files = [label_a, label_b]
        self.register_plot_style_layout(style, layout)
        self.current_aligned_metadata = {
            "方案名": payload["scheme_name"],
            "映射名称": mapping_name,
            "对齐策略": "原始时间轴叠加",
            "时间范围": f"{start_dt or '自动'} ~ {end_dt or '自动'}",
            "匹配点数": "",
            "成功系列数": plotted_count,
            "出图样式": style,
            "出图布局": layout,
        }
        self.update_diagnostic_info(
            layout=f"{parsed_a.profile_name} / {parsed_b.profile_name}",
            analysis_label="；".join(item["pair"]["label"] for item in pair_data),
            extra_items=[
                f"当前要素映射={mapping_name}",
                f"实际命中设备A列={'、'.join(sorted({item['pair']['a_col'] for item in pair_data}))}",
                f"实际命中设备B列={'、'.join(sorted({item['pair']['b_col'] for item in pair_data}))}",
                "对齐策略=原始时间轴叠加",
                f"出图样式={style}",
                f"出图布局={layout}",
                f"时间范围={start_dt or '自动'} ~ {end_dt or '自动'}",
                f"txt 合并后点数={total_a_points}",
                f"dat 裁切后点数={total_b_points}",
                f"成功系列数={plotted_count}",
            ] + self.build_timestamp_quality_items(parsed_a, parsed_b),
        )
        status = f"图已生成：时间序列对比 / {mapping_name} / 成功系列数={plotted_count}"
        if layout == "分别生成两张图":
            if self.separate_plot_windows:
                status += f" | 已生成独立图窗口：{len(self.separate_plot_windows)} 个"
            else:
                status += " | 主页面保留总览，未弹出独立窗口"
        self.status_var.set(status)

    def render_transformed_compare_payload(self, payload: dict[str, Any]) -> None:
        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for item in payload["plot_series"]:
            self.apply_series_style(
                ax,
                item["frame"]["时间戳"],
                item["frame"].iloc[:, 1],
                color=item["color"],
                label=item["label"],
                style=payload["plot_style"],
            )
        ax.set_title(payload["compare_mode"])
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = "aligned_diff" if payload["compare_mode"] == "差值时间序列" else "aligned_ratio"
        self.current_plot_columns = [pair["label"] for pair in payload["pairs"]]
        self.current_result_frame = None
        self.current_result_freq = None
        self.current_result_values = None
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = list(payload["export_frames"])
        self.current_compare_files = [payload["label_a"], payload["label_b"]]
        self.register_plot_style_layout(payload["plot_style"], payload["plot_layout"])
        quality_warnings: list[str] = []
        if payload["matched_points_total"] < 20:
            quality_warnings.append("匹配点数偏少，散点一致性结果可能不稳定。")
        self.current_aligned_metadata = {
            "方案名": payload["scheme_name"],
            "映射名称": payload["mapping_name"],
            "对齐策略": payload["alignment_strategy"],
            "时间范围": f"{payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
            "匹配点数": payload["matched_points_total"],
            "成功系列数": payload["success_count"],
            "出图样式": payload["plot_style"],
            "出图布局": payload["plot_layout"],
        }
        self.update_diagnostic_info(
            layout=f"{payload['parsed_a'].profile_name} / {payload['parsed_b'].profile_name}",
            analysis_label=payload["compare_mode"],
            extra_items=[
                f"当前要素映射={payload['mapping_name']}",
                f"实际命中设备A列={'、'.join(sorted({pair['a_col'] for pair in payload['pairs']}))}",
                f"实际命中设备B列={'、'.join(sorted({pair['b_col'] for pair in payload['pairs']}))}",
                f"对齐策略={payload['alignment_strategy']}",
                f"出图样式={payload['plot_style']}",
                f"出图布局={payload['plot_layout']}",
                f"时间范围={payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
                f"匹配点数={payload['matched_points_total']}",
                f"成功系列数={payload['success_count']}",
            ] + self.build_timestamp_quality_items(payload["parsed_a"], payload["parsed_b"]) + quality_warnings,
        )
        self.status_var.set(f"图已生成：{payload['compare_mode']} / {payload['mapping_name']} / 成功系列数={payload['success_count']}")

    def render_scatter_multi_payload(self, payload: dict[str, Any]) -> None:
        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for item in payload["series_items"]:
            self.apply_series_style(ax, item["x"], item["y"], color=item["color"], label=item["pair"]["label"], style=payload["plot_style"])
        ax.set_title("散点一致性对比")
        ax.set_xlabel(payload["label_a"])
        ax.set_ylabel(payload["label_b"])
        ax.grid(True, linestyle="--", alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = "aligned_scatter"
        self.current_plot_columns = [pair["label"] for pair in payload["pairs"]]
        self.current_result_frame = None
        self.current_result_freq = None
        self.current_result_values = None
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = list(payload["export_frames"])
        self.current_compare_files = [payload["label_a"], payload["label_b"]]
        self.register_plot_style_layout(payload["plot_style"], payload["plot_layout"])
        self.current_aligned_metadata = {
            "方案名": payload["scheme_name"],
            "映射名称": payload["mapping_name"],
            "对齐策略": payload["alignment_strategy"],
            "时间范围": f"{payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
            "匹配点数": payload["matched_points_total"],
            "成功系列数": payload["success_count"],
            "出图样式": payload["plot_style"],
            "出图布局": payload["plot_layout"],
        }
        self.update_diagnostic_info(
            layout=f"{payload['parsed_a'].profile_name} / {payload['parsed_b'].profile_name}",
            analysis_label="；".join(pair["label"] for pair in payload["pairs"]),
            extra_items=[
                f"当前要素映射={payload['mapping_name']}",
                f"实际命中设备A列={'、'.join(sorted({pair['a_col'] for pair in payload['pairs']}))}",
                f"实际命中设备B列={'、'.join(sorted({pair['b_col'] for pair in payload['pairs']}))}",
                f"对齐策略={payload['alignment_strategy']}",
                f"出图样式={payload['plot_style']}",
                f"出图布局={payload['plot_layout']}",
                f"时间范围={payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
                f"匹配点数={payload['matched_points_total']}",
                f"成功系列数={payload['success_count']}",
            ] + self.build_timestamp_quality_items(payload["parsed_a"], payload["parsed_b"]),
        )
        self.status_var.set(f"图已生成：散点一致性对比 / {payload['mapping_name']} / 成功系列数={payload['success_count']}")

    def render_cross_compare_payload(self, payload: dict[str, Any]) -> None:
        display_meta = self.get_cross_spectrum_display_meta(payload["spectrum_type"])
        target_context = core.get_target_spectral_context(payload.get("target_spectral_context") or payload.get("selection_meta"))
        last_details = payload["last_details"]
        display_semantics = str(last_details.get("display_semantics") or "")
        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for item in payload["plot_series"]:
            self.apply_series_style(
                ax,
                item["freq"],
                item["values"],
                color=item["color"],
                label=item["label"],
                style=payload["plot_style"],
            )
        ax.set_xscale("log")
        if display_semantics == core.CROSS_DISPLAY_SEMANTICS_ABS or payload["spectrum_type"] == CROSS_SPECTRUM_MAGNITUDE:
            ax.set_yscale("log")
            all_freq = np.concatenate([item["freq"] for item in payload["plot_series"]])
            all_values = np.concatenate([item["values"] for item in payload["plot_series"]])
        else:
            all_values = np.concatenate([item["values"] for item in payload["plot_series"]])
            max_abs = float(np.nanmax(np.abs(all_values))) if all_values.size else 0.0
            linthresh = max(max_abs * 1e-3, 1e-12)
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        all_freq = np.concatenate([item["freq"] for item in payload["plot_series"]])
        all_values = np.concatenate([item["values"] for item in payload["plot_series"]])
        if all_freq.size and all_values.size:
            self.add_selected_reference_slopes(
                ax,
                all_freq,
                all_values,
                is_psd=False,
                spectrum_type=payload["spectrum_type"],
            )
        if target_context is not None:
            title = f"{display_meta['title_prefix']}：{target_context['reference_column']} vs {target_context['display_target_label']}"
        else:
            title = (
                f"{display_meta['title_prefix']}：{payload['pairs'][0]['a_col']} vs {payload['pairs'][0]['b_col']}"
                if len(payload["pairs"]) == 1
                else f"{display_meta['title_prefix']}对比"
            )
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(display_meta["ylabel"])
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = display_meta["plot_kind"]
        if target_context is not None:
            self.current_plot_columns = [str(target_context["reference_column"]), str(target_context["display_target_label"])]
        else:
            self.current_plot_columns = [pair["label"] for pair in payload["pairs"]]
        self.current_result_freq = None
        self.current_result_values = None
        self.current_result_frame = self.merge_frequency_frames(payload["export_frames"])
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = list(payload["aligned_frames"])
        self.current_compare_files = [payload["label_a"], payload["label_b"]]
        self.register_plot_style_layout(payload["plot_style"], payload["plot_layout"])
        fs_text = ",".join(f"{value:g}" for value in sorted({round(value, 6) for value in payload["fs_used_values"]}))
        valid_freq_points = int(sum(len(item["freq"]) for item in payload["plot_series"]))
        quality_warnings: list[str] = []
        if valid_freq_points < 10:
            quality_warnings.append("有效频点偏少，请检查时间范围、对齐结果或频谱参数。")
        if payload["matched_points_total"] < 20:
            quality_warnings.append("匹配点数偏少，当前双设备频域结果可能不稳定。")
        if display_semantics != core.CROSS_DISPLAY_SEMANTICS_ABS:
            all_values = np.concatenate([item["values"] for item in payload["plot_series"]]) if payload["plot_series"] else np.array([])
            if all_values.size:
                near_zero_ratio = float(np.mean(np.abs(all_values) <= 1e-12))
                if near_zero_ratio >= 0.8:
                    quality_warnings.append("协谱/正交谱大部分点接近 0，请确认列映射和时间范围。")
        self.current_aligned_metadata = {
            "方案名": payload["scheme_name"],
            "映射名称": payload["mapping_name"],
            "谱类型": payload["spectrum_type"],
            "对齐策略": payload["alignment_strategy"],
            "时间范围": f"{payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
            "匹配点数": payload["matched_points_total"],
            "成功系列数": payload["success_count"],
            "出图样式": payload["plot_style"],
            "出图布局": payload["plot_layout"],
            "FS": fs_text,
            "display_semantics": display_semantics or None,
        }
        if target_context is not None:
            self.current_aligned_metadata.update(
                {
                    "reference_column": str(target_context["reference_column"]),
                    "ygas_target_column": str(target_context.get("ygas_target_column") or target_context.get("target_column") or ""),
                    "dat_target_column": str(target_context.get("dat_target_column") or ""),
                    "canonical_cross_pairs": " | ".join(str(item) for item in (target_context.get("canonical_cross_pairs") or [])),
                    "generated_cross_series_count": int(payload.get("success_count", len(payload["plot_series"]))),
                    "generated_cross_series_roles": " | ".join(
                        sorted(
                            {
                                str(item.get("series_role"))
                                for item in payload["plot_series"]
                                if str(item.get("series_role") or "").strip()
                            }
                        )
                    ),
                }
            )
        if target_context is not None:
            actual_hit_items = [
                f"reference_column={target_context['reference_column']}",
                f"ygas_target_column={target_context.get('ygas_target_column')}",
                f"dat_target_column={target_context.get('dat_target_column')}",
                f"canonical_cross_pairs={' | '.join(str(item) for item in (target_context.get('canonical_cross_pairs') or []))}",
            ]
        else:
            actual_hit_items = [
                f"实际命中设备A列={'、'.join(sorted({pair['a_col'] for pair in payload['pairs']}))}",
                f"实际命中设备B列={'、'.join(sorted({pair['b_col'] for pair in payload['pairs']}))}",
            ]
        extra_items = [
            f"当前要素映射={payload['mapping_name']}",
            f"对齐策略={payload['alignment_strategy']}",
            f"谱类型={payload['spectrum_type']}",
            f"出图样式={payload['plot_style']}",
            f"出图布局={payload['plot_layout']}",
            f"时间范围={payload['start_dt'] or '自动'} ~ {payload['end_dt'] or '自动'}",
            f"匹配点数={payload['matched_points_total']}",
            f"有效频点数={valid_freq_points}",
            f"txt 合并后点数={payload['total_a_points']}",
            f"dat 裁切后点数={payload['total_b_points']}",
            f"成功系列数={payload['success_count']}",
        ] + actual_hit_items + self.build_timestamp_quality_items(payload["parsed_a"], payload["parsed_b"]) + quality_warnings
        if target_context is not None:
            extra_items.extend(
                [
                    f"cross_execution_path={last_details.get('cross_execution_path') if payload.get('last_details') else 'target_spectral_canonical'}",
                    f"cross_implementation_id={last_details.get('cross_implementation_id') if payload.get('last_details') else core.TARGET_COSPECTRUM_IMPLEMENTATION_ID}",
                    f"cross_implementation_label={last_details.get('cross_implementation_label') if payload.get('last_details') else core.resolve_target_cospectrum_implementation().get('implementation_label')}",
                    f"cross_reference_column={last_details.get('cross_reference_column', target_context['reference_column']) if payload.get('last_details') else target_context['reference_column']}",
                    f"alignment_strategy={last_details.get('alignment_strategy', payload['alignment_strategy']) if payload.get('last_details') else payload['alignment_strategy']}",
                    f"reference_column={target_context['reference_column']}",
                    f"ygas_target_column={target_context.get('ygas_target_column')}",
                    f"dat_target_column={target_context.get('dat_target_column')}",
                    f"canonical_cross_pairs={' | '.join(str(item) for item in (target_context.get('canonical_cross_pairs') or []))}",
                    f"generated_cross_series_count={payload.get('success_count', len(payload['plot_series']))}",
                    "generated_cross_series_roles="
                    + " | ".join(
                        sorted(
                            {
                                str(item.get("series_role"))
                                for item in payload["plot_series"]
                                if str(item.get("series_role") or "").strip()
                            }
                        )
                    ),
                    f"display_semantics={last_details.get('display_semantics')}",
                    f"display_value_source={last_details.get('display_value_source')}",
                ]
            )
        if payload["spectrum_type"] in CROSS_SPECTRUM_OPTIONS and last_details is not None:
            for key in (
                "cross_execution_path",
                "cross_implementation_id",
                "cross_implementation_label",
                "alignment_strategy",
                "cross_reference_column",
                "target_column",
                "cross_order",
                "series_role",
                "device_kind",
                "display_label",
                "display_semantics",
                "display_value_source",
            ):
                if last_details.get(key) is not None:
                    extra_items.append(f"{key}={last_details[key]}")
            for key in ("cross_kernel", "window", "detrend", "scaling", "return_onesided", "average"):
                if last_details.get(key) is not None:
                    extra_items.append(f"{key}={last_details[key]}")
        if last_details is not None:
            self.update_diagnostic_info(
                layout=f"{payload['parsed_a'].profile_name} / {payload['parsed_b'].profile_name}",
                analysis_label=title.replace(f"{display_meta['title_prefix']}：", "", 1),
                fs=payload["fs_used_values"][-1] if payload["fs_used_values"] else None,
                nsegment=int(last_details["nsegment"]),
                overlap_ratio=float(last_details["overlap_ratio"]),
                nperseg=int(last_details["nperseg"]),
                noverlap=int(last_details["noverlap"]),
                extra_items=extra_items,
            )
        else:
            self.update_diagnostic_info(
                layout=f"{payload['parsed_a'].profile_name} / {payload['parsed_b'].profile_name}",
                analysis_label=title.replace(f"{display_meta['title_prefix']}：", "", 1),
                extra_items=extra_items,
            )
        self.status_var.set(f"图已生成：{display_meta['title_prefix']} / {payload['mapping_name']} / 成功系列数={payload['success_count']}")

    def use_current_file_time_range(self) -> None:
        if self.current_file is None:
            messagebox.showwarning("提示", "请先选择一个文件。")
            return

        try:
            parsed = self.parse_profiled_file(self.current_file)
        except Exception as exc:
            self.render_plot_message(f"当前文件解析失败：{exc}", level="warning")
            self.status_var.set(f"时间范围提取失败：{exc}")
            return

        if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
            self.render_plot_message("未识别到时间列，无法填写时间范围。", level="warning")
            self.status_var.set("当前文件未识别到时间列")
            return

        timestamps = parse_mixed_timestamp_series(parsed.dataframe[parsed.timestamp_col]).dropna()
        if timestamps.empty:
            self.render_plot_message("当前文件时间列为空，无法填写时间范围。", level="warning")
            self.status_var.set("当前文件时间列为空")
            return

        self.time_start_var.set(timestamps.min().strftime("%Y-%m-%d %H:%M:%S"))
        self.time_end_var.set(timestamps.max().strftime("%Y-%m-%d %H:%M:%S"))
        self.status_var.set("已填入当前文件的时间范围")

    def use_selected_dat_time_range(self) -> None:
        try:
            _parsed_a, _label_a, parsed_b, _label_b, meta = self.build_compare_selection()
        except Exception as exc:
            self.render_plot_message(f"无法提取 dat 时间范围：{exc}", level="warning")
            self.status_var.set(f"dat 时间范围提取失败：{exc}")
            return

        dat_summary = meta["dat_summary"]
        self.time_start_var.set(dat_summary["start"].strftime("%Y-%m-%d %H:%M:%S"))
        self.time_end_var.set(dat_summary["end"].strftime("%Y-%m-%d %H:%M:%S"))
        self.time_range_strategy_var.set("使用 dat 时间范围")
        self.status_var.set("已填入 dat 文件时间范围")

    def use_common_selected_time_range(self) -> None:
        try:
            _parsed_a, _label_a, _parsed_b, _label_b, meta = self.build_compare_selection()
            common_start = max(meta["txt_summary"]["start"], meta["dat_summary"]["start"])
            common_end = min(meta["txt_summary"]["end"], meta["dat_summary"]["end"])
        except Exception as exc:
            self.render_plot_message(f"共同时间范围计算失败：{exc}", level="warning")
            self.status_var.set(f"共同时间范围计算失败：{exc}")
            return

        if common_start > common_end:
            self.render_plot_message(
                self.build_no_common_time_range_message(meta["txt_summary"], meta["dat_summary"]),
                level="warning",
            )
            self.status_var.set("txt 与 dat 没有共同时间范围")
            return

        self.time_start_var.set(common_start.strftime("%Y-%m-%d %H:%M:%S"))
        self.time_end_var.set(common_end.strftime("%Y-%m-%d %H:%M:%S"))
        self.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        self.status_var.set("已填入 txt 与 dat 的共同时间范围")

    def resolve_compare_common_time_range_from_meta(
        self,
        selection_meta: dict[str, Any],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        txt_summary = selection_meta.get("txt_summary")
        dat_summary = selection_meta.get("dat_summary")
        if txt_summary is None or dat_summary is None:
            raise ValueError("当前 compare 选择中缺少 txt/dat 时间摘要，无法计算共同时间窗。")
        common_start = max(pd.Timestamp(txt_summary["start"]), pd.Timestamp(dat_summary["start"]))
        common_end = min(pd.Timestamp(txt_summary["end"]), pd.Timestamp(dat_summary["end"]))
        if common_start > common_end:
            raise ValueError(self.build_no_common_time_range_message(txt_summary, dat_summary))
        return common_start, common_end

    def sync_compare_common_time_range_to_single_analysis(
        self,
        selection_meta: dict[str, Any] | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        try:
            if selection_meta is None:
                _parsed_a, _label_a, _parsed_b, _label_b, selection_meta = self.build_compare_selection()
            common_start, common_end = self.resolve_compare_common_time_range_from_meta(dict(selection_meta))
        except Exception as exc:
            self.render_plot_message(f"compare 共同时间窗同步失败：{exc}", level="warning")
            self.status_var.set(f"compare 共同时间窗同步失败：{exc}")
            return None

        self.time_start_var.set(common_start.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."))
        self.time_end_var.set(common_end.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."))
        if self.left_workflow_notebook is not None and self.single_analysis_tab is not None:
            self.left_workflow_notebook.select(self.single_analysis_tab)
        self.status_var.set("已将当前 compare 共同时间窗同步到单图分析；单图分析将按用户时间窗生成。")
        return common_start, common_end

    def auto_fill_txt_covering_dat_range(self) -> None:
        selected_files = self.get_selected_file_paths()
        if not selected_files:
            messagebox.showwarning("提示", "请先选择一个 dat 文件。")
            return

        dat_candidates = [path for path in selected_files if path.suffix.lower() == ".dat"]
        dat_path = dat_candidates[0] if dat_candidates else None
        if dat_path is None and self.current_file is not None and self.current_file.suffix.lower() == ".dat":
            dat_path = self.current_file
        if dat_path is None:
            messagebox.showwarning("提示", "请先在文件列表中选择一个 dat 文件。")
            return

        try:
            parsed_dat = self.parse_dat_file(dat_path)
            dat_start, dat_end, _count = self.get_parsed_time_bounds(parsed_dat)
        except Exception as exc:
            self.render_plot_message(f"dat 文件解析失败：{exc}", level="warning")
            self.status_var.set(f"dat 文件解析失败：{exc}")
            return

        matched_paths: list[Path] = []
        parsed_ranges: list[tuple[Path, pd.Timestamp, pd.Timestamp]] = []
        for path in self.file_paths:
            if path.suffix.lower() not in {".txt", ".log"}:
                continue
            try:
                parsed = self.parse_ygas_mode1_file(path)
                txt_start, txt_end, _txt_count = self.get_parsed_time_bounds(parsed)
            except Exception:
                continue
            parsed_ranges.append((path, txt_start, txt_end))
            if txt_end >= dat_start and txt_start <= dat_end:
                matched_paths.append(path)

        overall_start = min((start for _path, start, _end in parsed_ranges), default=None)
        overall_end = max((end for _path, _start, end in parsed_ranges), default=None)
        coverage_start = min((start for path, start, _end in parsed_ranges if path in matched_paths), default=None)
        coverage_end = max((end for path, _start, end in parsed_ranges if path in matched_paths), default=None)
        missing_before = max((dat_start - coverage_start).total_seconds() / 60.0, 0.0) if coverage_start is not None else None
        missing_after = max((dat_end - coverage_end).total_seconds() / 60.0, 0.0) if coverage_end is not None else None

        if not matched_paths:
            coverage_text = " | ".join(
                [
                    f"已找到 txt/log：0 个",
                    f"可解析 txt/log：{len(parsed_ranges)} 个",
                    f"整体覆盖开始：{overall_start if overall_start is not None else '无'}",
                    f"整体覆盖结束：{overall_end if overall_end is not None else '无'}",
                ]
            )
            self.txt_merge_summary_var.set(f"txt 合并摘要：{coverage_text}")
            self.render_plot_message("没有找到可覆盖当前 dat 时间段的 txt/log 文件。", level="warning")
            self.status_var.set("自动补齐失败：未找到重叠 txt/log")
            return

        self.file_listbox.selection_clear(0, tk.END)
        selected_target_paths = [dat_path, *matched_paths]
        for index, path in enumerate(self.file_paths):
            if path in selected_target_paths:
                self.file_listbox.selection_set(index)
                if path == dat_path:
                    self.file_listbox.activate(index)
                    self.file_listbox.see(index)
        self.refresh_compare_column_options()
        self.use_common_selected_time_range()
        coverage_status = (
            f"txt 合并摘要：文件数={len(matched_paths)} | 覆盖开始={coverage_start} | 覆盖结束={coverage_end} | "
            f"距 dat 开始还差={missing_before:.1f} 分钟 | 距 dat 结束还差={missing_after:.1f} 分钟"
            if coverage_start is not None and coverage_end is not None and missing_before is not None and missing_after is not None
            else f"txt 合并摘要：文件数={len(matched_paths)}"
        )
        self.txt_merge_summary_var.set(coverage_status)
        status = f"已自动补齐 {len(matched_paths)} 个 txt/log 文件"
        if (missing_before or 0.0) > 0 or (missing_after or 0.0) > 0:
            status += f" | 仍未完整覆盖 dat：前差 {missing_before:.1f} 分钟，后差 {missing_after:.1f} 分钟"
        else:
            status += " | 已覆盖当前 dat 时间段"
        self.status_var.set(status)

    def describe_layout(self, mode1_layout: dict[str, Any] | None) -> str:
        if mode1_layout and mode1_layout.get("matched"):
            variant = str(mode1_layout.get("variant", ""))
            if variant == "mode1-15":
                return "MODE1-15"
            if variant == "mode1-16-with-index":
                return "MODE1-16"
        return "普通表格"

    def update_diagnostic_info(
        self,
        *,
        layout: str | None = None,
        analysis_label: str | None = None,
        valid_points: int | None = None,
        fs: float | None = None,
        nsegment: int | None = None,
        overlap_ratio: float | None = None,
        nperseg: int | None = None,
        noverlap: int | None = None,
        batch_success: int | None = None,
        batch_skips: int | None = None,
        extra_items: list[str] | None = None,
    ) -> None:
        items: list[str] = []
        if self.current_data_source_label:
            items.append(f"当前数据源={self.current_data_source_label}")
        comparison_frame = self.current_comparison_frame
        if self.current_data_source_kind == "comparison_analysis" and self.raw_data is not None:
            items.append("当前分析表=对比汇总表")
            items.append(f"行数={len(self.raw_data)}")
            items.append(f"列数={len(self.raw_data.columns)}")
            if self.current_comparison_metadata.get("alignment_strategy"):
                items.append(f"对齐策略={self.current_comparison_metadata['alignment_strategy']}")
            if self.current_comparison_metadata.get("time_range_label"):
                items.append(f"时间范围={self.current_comparison_metadata['time_range_label']}")
        elif self.current_data_source_kind == "comparison_preview" and comparison_frame is not None:
            items.append("当前预览表=对比汇总表")
            items.append(f"行数={len(comparison_frame)}")
            items.append(f"列数={len(comparison_frame.columns)}")
            if self.current_comparison_metadata.get("alignment_strategy"):
                items.append(f"对齐策略={self.current_comparison_metadata['alignment_strategy']}")
            if self.current_comparison_metadata.get("time_range_label"):
                items.append(f"时间范围={self.current_comparison_metadata['time_range_label']}")
        target_context = self.get_current_target_spectral_context()
        if target_context is not None:
            items.append(f"reference_column={target_context['reference_column']}")
            items.append(f"ygas_target_column={target_context.get('ygas_target_column', target_context.get('target_column'))}")
            items.append(f"dat_target_column={target_context.get('dat_target_column')}")
            if target_context.get("canonical_cross_pairs"):
                items.append(
                    "canonical_cross_pairs="
                    + " | ".join(str(item) for item in (target_context.get("canonical_cross_pairs") or []))
                )
        items.append(f"已识别布局：{layout or self.current_layout_label}")
        if analysis_label:
            items.append(f"当前分析列：{analysis_label}")
        if valid_points is not None:
            items.append(f"有效点数：{valid_points}")
        if fs is not None:
            items.append(f"FS={fs:g}")
        if nsegment is not None:
            items.append(f"NSEGMENT={nsegment}")
        if overlap_ratio is not None:
            items.append(f"OVERLAP_RATIO={overlap_ratio:g}")
        if nperseg is not None:
            items.append(f"nperseg={nperseg}")
        if noverlap is not None:
            items.append(f"noverlap={noverlap}")
        if batch_success is not None:
            items.append(f"成功系列数={batch_success}")
        if batch_skips is not None:
            items.append(f"跳过文件数={batch_skips}")
        if self.current_plot_layout_label:
            items.append(f"当前主图模式={self.current_plot_layout_label}")
        if self.current_plot_layout_label == "分别生成两张图":
            items.append(f"已弹出独立窗口={'是' if bool(self.separate_plot_windows) else '否'}")
        if self.separate_plot_windows:
            items.append(f"独立图窗口数={len(self.separate_plot_windows)}")
        if extra_items:
            items.extend(extra_items)
        self.diagnostic_var.set(" | ".join(items))

    def render_plot_message(self, message: str, level: str = "info") -> None:
        color_map = {
            "info": "#4f5d75",
            "warning": "#c77d00",
            "error": "#c1121f",
        }
        text_color = color_map.get(level, "#4f5d75")

        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            message,
            ha="center",
            va="center",
            fontsize=16,
            color=text_color,
            wrap=True,
            transform=ax.transAxes,
        )
        self.figure.tight_layout()
        self.refresh_canvas(title="图形结果")

        self.current_plot_kind = None
        self.current_plot_columns = []
        self.current_result_freq = None
        self.current_result_values = None
        self.current_result_frame = None
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = []
        self.current_aligned_metadata = {}
        self.current_target_plot_metadata = {}
        self.current_compare_files = []

    def resolve_delimiter(self, delimiter_choice: str, custom_delimiter: str) -> str:
        delim_map = {
            "逗号 (,)": ",",
            "制表符": "\t",
            "空格": r"\s+",
            "分号 (;)": ";",
            "自定义": custom_delimiter or ",",
        }
        return delim_map.get(delimiter_choice, ",")

    def preview_suggests_no_header(self, preview: pd.DataFrame) -> bool:
        if preview.empty:
            return False

        values = [str(value).strip() for value in preview.iloc[0].tolist() if pd.notna(value) and str(value).strip()]
        if not values:
            return False

        data_like = sum(self.looks_like_number(value) or self.looks_like_timestamp(value) for value in values)
        header_like = sum(self.looks_like_header_name(value) for value in values)
        return data_like >= max(1, math.ceil(len(values) * 0.6)) and data_like > header_like

    def infer_read_settings(self, path: Path) -> dict[str, str | bool]:
        inferred: dict[str, str | bool] = {
            "delimiter": "逗号 (,)",
            "custom_delimiter": "",
            "start_row": "0",
            "header_row": "0",
            "no_header": False,
        }

        try:
            detected_profile = detect_file_profile(path, read_preview_lines(path))
            if detected_profile == "TOA5_DAT":
                return inferred
            preview = pd.read_csv(
                path,
                sep=",",
                skiprows=0,
                header=None,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                nrows=8,
            )
        except Exception:
            if path.suffix.lower() == ".txt":
                inferred["no_header"] = True
            return inferred

        if path.suffix.lower() == ".txt":
            inferred["no_header"] = True
            return inferred

        if path.suffix.lower() in TEXT_LIKE_SUFFIXES:
            mode1_layout = detect_mode1_layout(preview)
            if mode1_layout.get("matched") or self.preview_suggests_no_header(preview):
                inferred["no_header"] = True

        return inferred

    def should_use_profile_parser_for_display(self, path: Path) -> str | None:
        detected_profile = detect_file_profile(path, read_preview_lines(path))
        if detected_profile in {"YGAS_MODE1_15", "YGAS_MODE1_16", "TOA5_DAT"}:
            return detected_profile
        return None

    def build_loader_result_from_parsed(self, parsed: ParsedFileResult) -> LoaderResult:
        excluded_columns = self.get_profile_excluded_columns(parsed.profile_name, parsed.dataframe, parsed.timestamp_col)
        mode1_layout: dict[str, Any] | None = None
        if parsed.profile_name.startswith("YGAS_MODE1"):
            variant = "mode1-16-with-index" if parsed.profile_name == "YGAS_MODE1_16" else "mode1-15"
            mode1_layout = {
                "matched": True,
                "variant": variant,
                "display_columns": list(parsed.dataframe.columns),
                "excluded_cols": set(excluded_columns),
            }
        return LoaderResult(
            dataframe=parsed.dataframe.copy(),
            columns=list(parsed.dataframe.columns),
            available_columns=list(parsed.available_columns),
            suggested_columns=list(parsed.suggested_columns),
            used_mode1_template=parsed.profile_name.startswith("YGAS_MODE1"),
            mode1_layout=mode1_layout,
            profile_name=parsed.profile_name,
            excluded_columns=excluded_columns,
        )

    def get_read_settings_for_file(self, path: Path) -> dict[str, str | bool]:
        saved = self.saved_read_settings.get(str(path))
        if saved is not None:
            return {
                "delimiter": str(saved.get("delimiter", "逗号 (,)")),
                "custom_delimiter": str(saved.get("custom_delimiter", "")),
                "start_row": str(saved.get("start_row", "0")),
                "header_row": str(saved.get("header_row", "0")),
                "no_header": bool(saved.get("no_header", False)),
            }
        return self.infer_read_settings(path)

    def get_supported_files(self, folder: Path) -> list[Path]:
        return sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES],
            key=lambda path: path.name.lower(),
        )

    def populate_file_list(self, files: list[Path]) -> None:
        self.file_paths = files
        self.file_listbox.delete(0, tk.END)
        for path in files:
            self.file_listbox.insert(tk.END, path.name)

    def select_file_in_list(self, path: Path) -> bool:
        try:
            index = self.file_paths.index(path)
        except ValueError:
            return False

        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.activate(index)
        self.file_listbox.see(index)
        return True

    def select_paths_in_list(self, paths: list[Path], *, active_path: Path | None = None) -> bool:
        selected = {path.resolve() for path in paths}
        self.file_listbox.selection_clear(0, tk.END)
        active_index: int | None = None
        first_index: int | None = None

        for index, path in enumerate(self.file_paths):
            if path.resolve() not in selected:
                continue
            self.file_listbox.selection_set(index)
            if first_index is None:
                first_index = index
            if active_path is not None and path == active_path:
                active_index = index

        target_index = active_index if active_index is not None else first_index
        if target_index is None:
            return False

        self.file_listbox.activate(target_index)
        self.file_listbox.see(target_index)
        return True

    def build_summary_from_range_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        file_name: str | None = None,
    ) -> dict[str, Any] | None:
        if not entries:
            return None
        return {
            "start": min(pd.Timestamp(entry["start"]) for entry in entries),
            "end": max(pd.Timestamp(entry["end"]) for entry in entries),
            "total_points": int(sum(int(entry.get("points", 0)) for entry in entries)),
            "raw_rows": int(sum(int(entry.get("raw_rows", entry.get("points", 0))) for entry in entries)),
            "valid_timestamp_points": int(sum(int(entry.get("valid_timestamp_points", entry.get("points", 0))) for entry in entries)),
            "file_count": len(entries),
            "file_name": file_name or (entries[0]["path"].name if len(entries) == 1 else f"{len(entries)} 个文件"),
        }

    def activate_single_device_selection(
        self,
        infos: list[dict[str, Any]],
        *,
        device_kind: str,
    ) -> None:
        if not infos:
            raise ValueError("没有可用于单设备模式的文件。")

        selected_paths = [Path(info["path"]) for info in infos]
        preview_path = selected_paths[0]
        summary = self.build_summary_from_range_entries(
            infos,
            file_name=preview_path.name if len(selected_paths) == 1 else f"{len(selected_paths)} 个文件",
        )
        self.selected_dat_var.set("")
        self.select_paths_in_list(selected_paths, active_path=preview_path)

        if device_kind == "ygas":
            self.compare_file_info_var.set(
                f"已识别单设备 ygas 文件 {len(selected_paths)} 个，可直接生成单台设备图谱；双设备主路径需再补 1 个 dat 文件。"
            )
            self.update_compare_summaries(summary, None)
        else:
            self.compare_file_info_var.set(
                f"已识别单设备 dat 文件 {len(selected_paths)} 个，可直接生成单台设备图谱；双设备主路径需再补 1 个 txt/log 文件。"
            )
            self.update_compare_summaries(None, summary)

        summary_items = [
            f"单设备模式={device_kind}",
            f"文件数={len(selected_paths)}",
        ]
        if summary is not None:
            summary_items.append(f"时间范围={self.format_time_range_text(summary['start'], summary['end'])}")
            summary_items.append(f"有效时间戳点数={summary.get('valid_timestamp_points', summary['total_points'])}")
        self.folder_prepare_summary_var.set("自动准备摘要：" + " | ".join(summary_items))

        self.apply_file_read_defaults(preview_path)
        self.start_async_file_loading(preview_path)
        if self.left_workflow_notebook is not None and self.single_analysis_tab is not None:
            self.left_workflow_notebook.select(self.single_analysis_tab)
        self.render_plot_message(
            "当前目录只识别到 1 台设备的数据。\n\n可直接选择 1 列生成单台设备图谱；若勾选多个同设备文件，可按设备分组后合并生成图谱。",
            level="info",
        )
        self.status_var.set(f"单设备模式已就绪：{device_kind} 文件 {len(selected_paths)} 个")

    def format_time_range_text(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> str:
        if start is None or end is None:
            return "无"
        return f"{start.strftime('%Y-%m-%d %H:%M:%S')} ~ {end.strftime('%Y-%m-%d %H:%M:%S')}"

    def build_dat_option_label(self, dat_info: dict[str, Any]) -> str:
        start = pd.Timestamp(dat_info["start"]).strftime("%m-%d %H:%M")
        end = pd.Timestamp(dat_info["end"]).strftime("%H:%M")
        return f"{dat_info['path'].name} | {start}-{end}"

    def choose_best_dat_info(
        self,
        ygas_infos: list[dict[str, Any]],
        dat_infos: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not dat_infos:
            return None
        if not ygas_infos:
            return dat_infos[0]

        ygas_start = min(pd.Timestamp(info["start"]) for info in ygas_infos)
        ygas_end = max(pd.Timestamp(info["end"]) for info in ygas_infos)
        ygas_mid = ygas_start + (ygas_end - ygas_start) / 2

        def overlap_seconds(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            return max((overlap_end - overlap_start).total_seconds(), 0.0)

        def rank_value(dat_info: dict[str, Any]) -> tuple[int, float, float]:
            dat_start = pd.Timestamp(dat_info["start"])
            dat_end = pd.Timestamp(dat_info["end"])
            overlapping_groups = sum(
                1
                for info in ygas_infos
                if overlap_seconds(pd.Timestamp(info["start"]), pd.Timestamp(info["end"]), dat_start, dat_end) > 0
            )
            overlap_span = overlap_seconds(ygas_start, ygas_end, dat_start, dat_end)
            dat_mid = dat_start + (dat_end - dat_start) / 2
            midpoint_penalty = abs((dat_mid - ygas_mid).total_seconds())
            return overlapping_groups, overlap_span, -midpoint_penalty

        return max(dat_infos, key=rank_value)

    def build_auto_selection_for_dat(
        self,
        payload: dict[str, Any],
        dat_path: Path,
    ) -> dict[str, Any]:
        ygas_infos = list(payload.get("ygas_infos", []))
        dat_infos = list(payload.get("dat_infos", []))
        dat_info = next((info for info in dat_infos if Path(info["path"]) == dat_path), None)
        if dat_info is None:
            raise ValueError("未找到可用的 dat 文件信息。")

        dat_start = pd.Timestamp(dat_info["start"])
        dat_end = pd.Timestamp(dat_info["end"])
        overlapping_infos: list[dict[str, Any]] = []
        skipped_infos: list[dict[str, Any]] = []
        for info in ygas_infos:
            start = pd.Timestamp(info["start"])
            end = pd.Timestamp(info["end"])
            if end >= dat_start and start <= dat_end:
                overlapping_infos.append(info)
            else:
                skipped_infos.append(info)

        selected_infos = overlapping_infos if overlapping_infos else list(ygas_infos)
        txt_summary = self.build_summary_from_range_entries(selected_infos)
        dat_summary = self.build_summary_from_range_entries([dat_info], file_name=dat_info["path"].name)
        common_start: pd.Timestamp | None = None
        common_end: pd.Timestamp | None = None
        if txt_summary is not None and dat_summary is not None:
            common_start = max(pd.Timestamp(txt_summary["start"]), pd.Timestamp(dat_summary["start"]))
            common_end = min(pd.Timestamp(txt_summary["end"]), pd.Timestamp(dat_summary["end"]))
            if common_start > common_end:
                common_start = None
                common_end = None

        coverage_start = min((pd.Timestamp(info["start"]) for info in overlapping_infos), default=None)
        coverage_end = max((pd.Timestamp(info["end"]) for info in overlapping_infos), default=None)
        missing_before = None
        missing_after = None
        if coverage_start is not None:
            missing_before = max((dat_start - coverage_start).total_seconds() / 60.0, 0.0)
        if coverage_end is not None:
            missing_after = max((dat_end - coverage_end).total_seconds() / 60.0, 0.0)

        return {
            "dat_info": dat_info,
            "selected_ygas_infos": selected_infos,
            "selected_ygas_paths": [Path(info["path"]) for info in selected_infos],
            "skipped_ygas_infos": skipped_infos,
            "txt_summary": txt_summary,
            "dat_summary": dat_summary,
            "common_start": common_start,
            "common_end": common_end,
            "group_count": len(overlapping_infos),
            "expected_series_count": len(overlapping_infos) * 2,
            "missing_before_minutes": missing_before,
            "missing_after_minutes": missing_after,
        }

    def prepare_folder_auto_selection_payload(
        self,
        files: list[Path],
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        ygas_infos: list[dict[str, Any]] = []
        dat_infos: list[dict[str, Any]] = []
        ignored_files: list[str] = []
        parse_errors: list[str] = []

        total_files = len(files)
        for index, path in enumerate(files, start=1):
            if reporter is not None:
                reporter(f"正在自动识别文件… {index}/{total_files}：{path.name}")
            suffix = path.suffix.lower()
            try:
                profile = detect_file_profile(path, read_preview_lines(path))
                if suffix in {".txt", ".log"} or profile in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
                    parsed = self.parse_profiled_file(path)
                    start, end, points = self.get_parsed_time_bounds(parsed)
                    ygas_infos.append(
                        {
                            "path": path,
                            "profile": parsed.profile_name,
                            "start": start,
                            "end": end,
                            "points": points,
                            "raw_rows": int(parsed.source_row_count or len(parsed.dataframe)),
                            "valid_timestamp_points": int(parsed.timestamp_valid_count),
                        }
                    )
                    continue
                if suffix == ".dat" or profile == "TOA5_DAT":
                    parsed = self.parse_dat_file(path)
                    start, end, points = self.get_parsed_time_bounds(parsed)
                    dat_infos.append(
                        {
                            "path": path,
                            "profile": parsed.profile_name,
                            "start": start,
                            "end": end,
                            "points": points,
                            "raw_rows": int(parsed.source_row_count or len(parsed.dataframe)),
                            "valid_timestamp_points": int(parsed.timestamp_valid_count),
                        }
                    )
                    continue
                ignored_files.append(path.name)
            except Exception as exc:
                parse_errors.append(f"{path.name}：{exc}")

        payload: dict[str, Any] = {
            "ygas_infos": sorted(ygas_infos, key=lambda item: (pd.Timestamp(item["start"]), item["path"].name.lower())),
            "dat_infos": sorted(dat_infos, key=lambda item: (pd.Timestamp(item["start"]), item["path"].name.lower())),
            "ignored_files": ignored_files,
            "parse_errors": parse_errors,
        }
        best_dat = self.choose_best_dat_info(payload["ygas_infos"], payload["dat_infos"])
        if best_dat is not None:
            payload["selected_dat_path"] = Path(best_dat["path"])
            payload["selection"] = self.build_auto_selection_for_dat(payload, Path(best_dat["path"]))
        else:
            payload["selected_dat_path"] = None
            payload["selection"] = None
        return payload

    def apply_auto_prepared_selection(
        self,
        payload: dict[str, Any],
        *,
        selected_dat_path: Path | None = None,
        source: str = "directory_auto_prepare",
    ) -> None:
        selection = self.cache_auto_compare_context_payload(
            payload,
            source=source,
            selected_dat_path=selected_dat_path,
        )
        ygas_infos = list(payload.get("ygas_infos", []))
        dat_infos = list(payload.get("dat_infos", []))

        if not ygas_infos:
            if dat_infos:
                self.activate_single_device_selection(dat_infos, device_kind="dat")
                return
            self.compare_file_info_var.set("自动准备失败：未找到可用 ygas txt/log 文件")
            self.update_compare_summaries()
            self.folder_prepare_summary_var.set("自动准备摘要：未找到可用 ygas txt/log 文件")
            self.selected_dat_var.set("")
            self.render_plot_message("未找到可用 ygas txt/log 文件。", level="warning")
            self.status_var.set("自动准备失败：未找到可用 ygas txt/log")
            return

        if not dat_infos:
            self.activate_single_device_selection(ygas_infos, device_kind="ygas")
            return

        chosen_path = selected_dat_path or payload.get("selected_dat_path")
        if chosen_path is None:
            chosen_info = self.choose_best_dat_info(ygas_infos, dat_infos)
            chosen_path = Path(chosen_info["path"]) if chosen_info is not None else None
        if chosen_path is None:
            self.render_plot_message("自动准备失败：未找到可用 dat 文件。", level="warning")
            self.status_var.set("自动准备失败：未找到可用 dat 文件")
            return

        if selection is None:
            selection = self.build_auto_selection_for_dat(payload, Path(chosen_path))
        dat_label = next((label for label, path in self.auto_dat_options.items() if path == chosen_path), chosen_path.name)
        self.selected_dat_var.set(dat_label)
        self.element_preset_var.set("CO2")

        selected_paths = [*selection["selected_ygas_paths"], Path(selection["dat_info"]["path"])]
        preview_path = selection["selected_ygas_paths"][0] if selection["selected_ygas_paths"] else Path(selection["dat_info"]["path"])
        self.select_paths_in_list(selected_paths, active_path=preview_path)
        self.compare_file_info_var.set(
            f"自动准备完成：ygas={len(ygas_infos)} 个 | dat={len(dat_infos)} 个 | 当前 dat={Path(selection['dat_info']['path']).name}"
        )
        self.update_compare_summaries(selection["txt_summary"], selection["dat_summary"])

        common_text = self.format_time_range_text(selection["common_start"], selection["common_end"])
        summary_items = [
            f"共同时间范围={common_text}",
            f"可生成谱图的分组数={selection['group_count']}",
        ]
        if selection["skipped_ygas_infos"]:
            summary_items.append(f"已跳过不重叠 txt/log={len(selection['skipped_ygas_infos'])} 个")
        if selection["missing_before_minutes"] is not None and selection["missing_before_minutes"] > 0:
            summary_items.append(f"dat 开始前仍缺 {selection['missing_before_minutes']:.1f} 分钟")
        if selection["missing_after_minutes"] is not None and selection["missing_after_minutes"] > 0:
            summary_items.append(f"dat 结束后仍缺 {selection['missing_after_minutes']:.1f} 分钟")
        if payload.get("parse_errors"):
            summary_items.append(f"解析失败={len(payload['parse_errors'])} 个")
        self.folder_prepare_summary_var.set("自动准备摘要：" + " | ".join(summary_items))

        if selection["common_start"] is not None and selection["common_end"] is not None:
            self.time_start_var.set(selection["common_start"].strftime("%Y-%m-%d %H:%M:%S"))
            self.time_end_var.set(selection["common_end"].strftime("%Y-%m-%d %H:%M:%S"))
            self.time_range_strategy_var.set("使用 txt+dat 共同时间范围")
        else:
            dat_summary = selection["dat_summary"]
            self.time_start_var.set(pd.Timestamp(dat_summary["start"]).strftime("%Y-%m-%d %H:%M:%S"))
            self.time_end_var.set(pd.Timestamp(dat_summary["end"]).strftime("%Y-%m-%d %H:%M:%S"))
            self.time_range_strategy_var.set("使用 dat 时间范围")

        self.refresh_compare_column_options()
        self.apply_file_read_defaults(preview_path)
        self.start_async_file_loading(preview_path)
        if self.left_workflow_notebook is not None and self.dual_compare_tab is not None:
            self.left_workflow_notebook.select(self.dual_compare_tab)

        if selection["group_count"] <= 0:
            txt_summary = selection["txt_summary"]
            dat_summary = selection["dat_summary"]
            if txt_summary is not None and dat_summary is not None:
                self.render_plot_message(self.build_no_common_time_range_message(txt_summary, dat_summary), level="warning")
            else:
                self.render_plot_message("dat 与 txt/log 时间不重叠，暂时无法生成目标谱图。", level="warning")
            self.status_var.set("自动准备完成，但当前没有可用共同时间范围")
            return

        self.status_var.set(f"自动准备完成：已选 ygas {len(selection['selected_ygas_paths'])} 个，dat 1 个，可直接生成目标谱图")

    def async_open_directory(self) -> None:
        folder = filedialog.askdirectory(title="选择目录")
        if not folder:
            return

        self.remember_current_selection()
        self.remember_current_read_settings()
        self.current_folder = Path(folder)
        self.reset_auto_compare_context()
        self.current_target_group_preview_frame = None
        self.update_target_group_qc_panel(None)
        files = self.get_supported_files(self.current_folder)
        self.populate_file_list(files)

        if files:
            self.status_var.set(f"已载入目录：{self.current_folder}，正在自动识别文件…")
            self.current_target_plot_metadata = {}
            self.current_target_group_preview_frame = None
            self.current_result_frame = None
            self.current_result_freq = None
            self.current_result_values = None
            self.current_compare_files = []
            self.current_plot_kind = None
            self.current_plot_columns = []
            self.reset_plot_output_state()

            self.start_background_task(
                status_text="正在自动识别文件…",
                worker=lambda reporter: self.prepare_folder_auto_selection_payload(files, reporter),
                on_success=lambda payload: self.apply_auto_prepared_selection(payload, source="directory_auto_prepare"),
                error_title="自动准备文件夹",
            )
        else:
            self.compare_file_info_var.set("自动准备失败：当前目录没有可用数据文件")
            self.update_compare_summaries()
            self.folder_prepare_summary_var.set("自动准备摘要：当前目录没有可用的 txt/log/dat 文件")
            self.status_var.set("当前目录没有可用的数据文件")
            self.render_plot_message("当前目录没有可用的 txt/log/dat 文件。", level="warning")

    def async_open_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[
                ("高频数据文件", "*.txt *.log *.csv *.dat"),
                ("TXT 文件", "*.txt"),
                ("LOG 文件", "*.log"),
                ("CSV 文件", "*.csv"),
                ("DAT 文件", "*.dat"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return

        chosen_path = Path(file_path)
        if chosen_path.suffix.lower() not in ALLOWED_SUFFIXES:
            messagebox.showerror("不支持的文件", "请选择 txt、log、csv 或 dat 文件。")
            return

        self.remember_current_selection()
        self.remember_current_read_settings()
        self.current_folder = chosen_path.parent
        self.reset_auto_compare_context()
        self.current_target_group_preview_frame = None
        self.update_target_group_qc_panel(None)
        files = self.get_supported_files(self.current_folder)
        self.populate_file_list(files)

        if not self.select_file_in_list(chosen_path):
            self.status_var.set("所选文件未出现在当前目录列表中")
            return

        self.apply_file_read_defaults(chosen_path)
        self.start_async_file_loading(chosen_path)
        self.status_var.set(f"已选择文件：{chosen_path.name}")

    def on_selected_dat_changed(self, _event: tk.Event[Any] | None = None) -> None:
        label = self.selected_dat_var.get().strip()
        if not label or self.auto_prepare_payload is None:
            return
        dat_path = self.auto_dat_options.get(label)
        if dat_path is None:
            return
        try:
            self.apply_auto_prepared_selection(
                self.auto_prepare_payload,
                selected_dat_path=dat_path,
                source=self.auto_compare_context_source or "directory_auto_prepare",
            )
            self.status_var.set(f"已切换 dat：{dat_path.name}")
        except Exception as exc:
            self.render_plot_message(f"切换 dat 失败：{exc}", level="warning")
            self.status_var.set(f"切换 dat 失败：{exc}")

    def on_file_select(self, _event: tk.Event[Any] | None = None) -> None:
        selection = self.file_listbox.curselection()
        if not selection:
            return
        active_index = self.file_listbox.index(tk.ACTIVE)
        target_index = active_index if active_index in selection else selection[-1]
        new_path = self.file_paths[target_index]
        self.refresh_compare_column_options()
        if self.current_file == new_path:
            return

        self.remember_current_selection()
        self.remember_current_read_settings()
        self.apply_file_read_defaults(new_path)
        self.start_async_file_loading(new_path)

    def update_settings(self) -> None:
        self._update_delimiter_state()
        if self.suspend_setting_reload:
            return
        if self.current_file is None:
            return

        self.remember_current_read_settings()
        if self.reload_after_id is not None:
            self.root.after_cancel(self.reload_after_id)
        self.reload_after_id = self.root.after(300, lambda: self.start_async_file_loading(self.current_file))

    def remember_current_read_settings(self) -> None:
        if self.current_file is None:
            return
        self.saved_read_settings[str(self.current_file)] = {
            "delimiter": self.delimiter_var.get(),
            "custom_delimiter": self.custom_delimiter_var.get(),
            "start_row": self.start_row_var.get(),
            "header_row": self.header_row_var.get(),
            "no_header": self.no_header_var.get(),
        }

    def apply_read_settings(
        self,
        *,
        delimiter: str,
        custom_delimiter: str,
        start_row: str,
        header_row: str,
        no_header: bool,
    ) -> None:
        self.suspend_setting_reload = True
        try:
            self.delimiter_var.set(delimiter)
            self.custom_delimiter_var.set(custom_delimiter)
            self.start_row_var.set(start_row)
            self.header_row_var.set(header_row)
            self.no_header_var.set(no_header)
            self._update_delimiter_state()
        finally:
            self.suspend_setting_reload = False

    def apply_file_read_defaults(self, path: Path) -> None:
        settings = self.get_read_settings_for_file(path)
        self.apply_read_settings(
            delimiter=str(settings["delimiter"]),
            custom_delimiter=str(settings["custom_delimiter"]),
            start_row=str(settings["start_row"]),
            header_row=str(settings["header_row"]),
            no_header=bool(settings["no_header"]),
        )

    def _update_delimiter_state(self) -> None:
        custom_enabled = self.delimiter_var.get() == "自定义"
        state = "normal" if custom_enabled else "disabled"
        self.custom_delim_entry.configure(state=state)

        header_state = "disabled" if self.no_header_var.get() else "normal"
        self.header_spin.configure(state=header_state)

    def get_delimiter(self) -> str:
        return self.resolve_delimiter(self.delimiter_var.get(), self.custom_delimiter_var.get())

    def get_header_value(self) -> int | None:
        if self.no_header_var.get():
            return None
        return max(parse_int(self.header_row_var.get(), 0), 0)

    def looks_like_timestamp(self, value: str) -> bool:
        return bool(TIMESTAMP_PATTERN.match(value.strip()))

    def looks_like_number(self, value: str) -> bool:
        return bool(NUMERIC_PATTERN.match(value.strip()))

    def looks_like_header_name(self, value: str) -> bool:
        return bool(HEADER_NAME_PATTERN.match(value.strip()))

    def should_suggest_no_header(self, path: Path) -> bool:
        if path.suffix.lower() not in TEXT_LIKE_SUFFIXES:
            return False
        if self.no_header_var.get():
            return False

        state = self.no_header_prompt_state.get(str(path))
        if state in {"accepted", "rejected"}:
            return False

        try:
            preview = pd.read_csv(
                path,
                sep=self.get_delimiter(),
                skiprows=max(parse_int(self.start_row_var.get(), 0), 0),
                header=None,
                engine="python",
                on_bad_lines="skip",
                dtype=str,
                nrows=1,
            )
        except Exception:
            return False

        return self.preview_suggests_no_header(preview)

    def get_current_loader_options(self) -> tuple[str, int, int | None]:
        delimiter = self.get_delimiter()
        start_row = max(parse_int(self.start_row_var.get(), 0), 0)
        header_row = self.get_header_value()
        return delimiter, start_row, header_row

    def load_file_with_current_settings(self, path: Path) -> LoaderResult:
        delimiter, start_row, header_row = self.get_current_loader_options()
        return self.load_file_with_options(path, delimiter, start_row, header_row)

    def load_file_with_options(
        self,
        path: Path,
        delimiter: str,
        start_row: int,
        header_row: int | None,
    ) -> LoaderResult:
        profile_parser = self.should_use_profile_parser_for_display(path)
        if profile_parser in {"YGAS_MODE1_15", "YGAS_MODE1_16"}:
            return self.build_loader_result_from_parsed(self.parse_ygas_mode1_file(path))
        if profile_parser == "TOA5_DAT":
            return self.build_loader_result_from_parsed(self.parse_toa5_file(path))
        loader = AsyncFileLoader(
            file_path=str(path),
            delimiter=delimiter,
            start_row=start_row,
            header_row=header_row,
        )
        return loader.load()

    def load_file_with_file_settings(self, path: Path) -> LoaderResult:
        settings = self.get_read_settings_for_file(path)
        return self.load_file_with_options(
            path,
            self.resolve_delimiter(str(settings["delimiter"]), str(settings["custom_delimiter"])),
            max(parse_int(str(settings["start_row"]), 0), 0),
            None if bool(settings["no_header"]) else max(parse_int(str(settings["header_row"]), 0), 0),
        )

    def classify_dataframe_columns(
        self,
        df: pd.DataFrame,
        excluded_columns: set[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], set[str], set[str]]:
        return core.classify_numeric_columns(df, excluded_columns=excluded_columns)

    def get_selected_file_paths(self) -> list[Path]:
        indices = list(self.file_listbox.curselection())
        if indices:
            return [self.file_paths[index] for index in indices]
        if self.current_file is not None:
            return [self.current_file]
        return []

    def start_async_file_loading(self, path: Path) -> None:
        if self.reload_after_id is not None:
            self.root.after_cancel(self.reload_after_id)
            self.reload_after_id = None

        self.current_file = path
        self.active_load_token += 1
        self.loading_in_progress = True
        token = self.active_load_token
        delimiter, start_row, header_row = self.get_current_loader_options()
        self.status_var.set(f"正在加载：{path.name}")

        def task() -> None:
            try:
                result = self.load_file_with_options(path, delimiter, start_row, header_row)
                self.loading_queue.put(
                    {
                        "token": token,
                        "success": True,
                        "path": path,
                        "result": result,
                    }
                )
            except Exception as exc:  # pragma: no cover - GUI error path
                self.loading_queue.put(
                    {
                        "token": token,
                        "success": False,
                        "path": path,
                        "error": str(exc),
                    }
                )

        threading.Thread(target=task, daemon=True).start()
        if self.check_queue_after_id is None:
            self.check_queue_after_id = self.root.after(100, self.check_loading_status)

    def check_loading_status(self) -> None:
        processed = False
        while True:
            try:
                item = self.loading_queue.get_nowait()
            except queue.Empty:
                break

            processed = True
            if item["token"] != self.active_load_token:
                continue

            self.loading_in_progress = False
            if item["success"]:
                self.on_file_loaded(item["path"], item["result"])
            else:
                self.status_var.set(f"加载失败：{item['path'].name}")
                messagebox.showerror("加载失败", item["error"])

        if processed or not self.loading_queue.empty() or self.loading_in_progress:
            self.check_queue_after_id = self.root.after(100, self.check_loading_status)
        else:
            self.check_queue_after_id = None

    def start_background_task(
        self,
        *,
        status_text: str,
        worker: Any,
        on_success: Any,
        error_title: str,
    ) -> None:
        self.active_task_token += 1
        token = self.active_task_token
        self.background_task_in_progress = True
        self.status_var.set(status_text)
        self.root.update_idletasks()

        def report_progress(text: str) -> None:
            self.task_queue.put(
                {
                    "token": token,
                    "kind": "progress",
                    "text": text,
                }
            )

        def task() -> None:
            try:
                payload = worker(report_progress)
                self.task_queue.put(
                    {
                        "token": token,
                        "success": True,
                        "payload": payload,
                        "callback": on_success,
                    }
                )
            except Exception as exc:  # pragma: no cover - GUI error path
                self.task_queue.put(
                    {
                        "token": token,
                        "success": False,
                        "error": str(exc),
                        "error_title": error_title,
                    }
                )

        threading.Thread(target=task, daemon=True).start()
        if self.check_task_after_id is None:
            self.check_task_after_id = self.root.after(100, self.check_background_task_status)

    def check_background_task_status(self) -> None:
        processed = False
        while True:
            try:
                item = self.task_queue.get_nowait()
            except queue.Empty:
                break

            processed = True
            if item["token"] != self.active_task_token:
                continue

            if item.get("kind") == "progress":
                self.status_var.set(str(item.get("text", "正在处理…")))
                self.root.update_idletasks()
                continue

            self.background_task_in_progress = False
            if item["success"]:
                callback = item["callback"]
                try:
                    callback(item["payload"])
                except Exception as exc:
                    message = str(exc)
                    self.status_var.set(f"后台任务失败：{message}")
                    self.update_diagnostic_info(layout=self.current_layout_label)
                    self.render_plot_message(message, level="warning")
                    messagebox.showerror("后台任务失败", message)
            else:
                message = item["error"]
                self.status_var.set(f"{item['error_title']}失败：{message}")
                self.update_diagnostic_info(layout=self.current_layout_label)
                self.render_plot_message(message, level="warning")
                messagebox.showerror(item["error_title"], message)

        if processed or not self.task_queue.empty() or self.background_task_in_progress:
            self.check_task_after_id = self.root.after(100, self.check_background_task_status)
        else:
            self.check_task_after_id = None

    def on_file_loaded(self, path: Path, result: LoaderResult) -> None:
        df = result.dataframe
        self.preview_data = None
        self.current_data_source_kind = "file"
        self.current_data_source_label = f"当前文件：{path.name}"

        if self.should_suggest_no_header(path):
            prompt = (
                "检测到首行更像数据而不是表头。\n\n"
                "是否按无表头(header=None)重新加载？"
            )
            if messagebox.askyesno("检测到可能无表头", prompt):
                self.no_header_prompt_state[str(path)] = "accepted"
                self.no_header_var.set(True)
                self.start_async_file_loading(path)
                return
            self.no_header_prompt_state[str(path)] = "rejected"

        self.raw_data = df
        self.page_index = 0
        if result.profile_name:
            self.current_layout_label = result.profile_name
        else:
            self.current_layout_label = self.describe_layout(result.mode1_layout)
        self.excluded_analysis_cols = set(result.excluded_columns or set())
        if result.mode1_layout is not None:
            self.excluded_analysis_cols.update(set(result.mode1_layout.get("excluded_cols", set())))
        self.column_data, self.non_numeric_cols, self.unsuitable_spectrum_cols = self.classify_dataframe_columns(
            df,
            excluded_columns=self.excluded_analysis_cols,
        )
        self.current_used_mode1_template = result.used_mode1_template
        parsed_profile = str(result.profile_name or detect_file_profile(path, read_preview_lines(path)))
        timestamp_col = "时间戳" if "时间戳" in df.columns else self.guess_timestamp_column(df)
        self.current_file_parsed = core.build_parsed_result_from_dataframe(
            df.copy(),
            parsed_profile,
            timestamp_col,
            source_row_count=len(df),
            analyze_numeric_columns=False,
            suggested_columns_override=list(self.column_data.keys()) if self.column_data else None,
        )

        self.create_column_selector()
        self.update_table_view()
        self.refresh_compare_column_options()
        if path.suffix.lower() in TEXT_LIKE_SUFFIXES:
            self.ensure_auto_compare_context_for_selection([path], source="single_file_auto_bootstrap")
            self.refresh_single_compare_style_preview_state()

        numeric_count = len(self.column_data)
        non_numeric_count = len(self.non_numeric_cols)
        status_suffix = ""
        if self.unsuitable_spectrum_cols:
            status_suffix += f" | 已排除不适合谱分析的列：{len(self.unsuitable_spectrum_cols)} 列"
        if result.used_mode1_template:
            status_suffix += " | 已识别 MODE1 布局"
        elif result.profile_name == "TOA5_DAT":
            status_suffix += " | 已识别 TOA5 布局"
        self.status_var.set(
            f"已加载 {path.name} | 总列数 {len(df.columns)} | 可分析列 {numeric_count} | 非数值列 {non_numeric_count}{status_suffix}"
        )
        timestamp_items: list[str] = []
        timestamp_col = "时间戳" if "时间戳" in df.columns else self.guess_timestamp_column(df)
        if timestamp_col and timestamp_col in df.columns:
            parsed_timestamps = parse_mixed_timestamp_series(df[timestamp_col])
            timestamp_stats = build_timestamp_parse_stats(df[timestamp_col], parsed_timestamps)
            timestamp_items.append(
                f"时间戳有效率={timestamp_stats['valid_ratio']:.1%} "
                f"({timestamp_stats['valid_count']}/{timestamp_stats['row_count']})"
            )
            estimated_fs = estimate_fs_from_timestamp(df, timestamp_col)
            if estimated_fs is not None:
                timestamp_items.append(f"估算FS={estimated_fs:g}")
            if timestamp_stats["warning"]:
                timestamp_items.append(str(timestamp_stats["warning"]))
        self.update_diagnostic_info(layout=self.current_layout_label, extra_items=timestamp_items or None)

        if not self.column_data:
            if path.suffix.lower() in TEXT_LIKE_SUFFIXES and self.current_layout_label == "普通表格":
                self.render_plot_message("当前文件布局识别失败，请检查分隔符、表头或起始行设置。", level="warning")
            else:
                self.render_plot_message("当前文件没有可用于谱分析的有效列。", level="warning")

    def create_column_selector(self) -> None:
        for child in self.column_inner.winfo_children():
            child.destroy()

        preserved = self.restore_selection_candidates()
        self.column_vars.clear()

        display_columns = [
            col for col in (self.raw_data.columns if self.raw_data is not None else []) if col in self.column_data
        ]
        guidance_lines: list[tuple[str, str]] = []
        if self.current_data_source_kind == "comparison_analysis":
            target_context = core.get_target_spectral_context(self.current_comparison_metadata)
            guidance_lines.append(("汇总表模式：可勾选 1 列做 PSD，或勾选 2 列做互谱/协谱/正交谱。", "#234e70"))
            if target_context is not None:
                guidance_lines.append(
                    (
                        "当前属于目标频谱上下文，"
                        + "canonical pairs = "
                        + "；".join(str(item) for item in (target_context.get("canonical_cross_pairs") or []))
                        + "。",
                        "#2c5282",
                    )
                )
            else:
                guidance_lines.append(("建议优先选择一列 A_... 和一列 B_...，例如 A_CO2浓度 与 B_CO2_Avg。", "#666666"))

        if not display_columns:
            ttk.Label(self.column_inner, text="未检测到可用于谱分析的数值列").grid(
                row=0, column=0, sticky="w", padx=6, pady=6
            )
        else:
            row_cursor = 0
            if guidance_lines:
                for text, color in guidance_lines:
                    ttk.Label(
                        self.column_inner,
                        text=text,
                        foreground=color,
                        wraplength=300,
                        justify="left",
                    ).grid(row=row_cursor, column=0, sticky="w", padx=6, pady=(6 if row_cursor == 0 else 0, 4))
                    row_cursor += 1
            ttk.Label(
                self.column_inner,
                text="列较多时可用右侧滚动条或鼠标滚轮查看更多",
                foreground="#666666",
            ).grid(row=row_cursor, column=0, sticky="w", padx=6, pady=(6 if row_cursor == 0 else 0, 4))
            row_cursor += 1
        row_offset = row_cursor if display_columns else 0
        for idx, col in enumerate(display_columns):
            var = tk.BooleanVar(value=col in preserved)
            self.column_vars[col] = var
            cb = ttk.Checkbutton(
                self.column_inner,
                text=col,
                variable=var,
                command=lambda name=col: self.on_column_changed(name),
            )
            cb.grid(row=row_offset + idx, column=0, sticky="w", padx=6, pady=2)

        self.refresh_single_compare_style_preview_state()
        self.schedule_auto_analysis()

    def restore_selection_candidates(self) -> set[str]:
        if self.current_data_source_kind == "comparison_analysis":
            target_context = core.get_target_spectral_context(self.current_comparison_metadata)
            if target_context is not None:
                return {
                    column
                    for column in [
                        str(target_context.get("summary_ygas_target_column") or target_context.get("summary_target_column") or ""),
                        str(target_context.get("summary_reference_column") or ""),
                    ]
                    if column in self.column_data
                }
            return set()

        if not self.preserve_selection_var.get():
            if self.current_used_mode1_template:
                if "CO2浓度" in self.column_data:
                    return {"CO2浓度"}
                if "H2O浓度" in self.column_data:
                    return {"H2O浓度"}
            return set()

        current_key = str(self.current_file) if self.current_file else ""
        current_saved = set(self.saved_selections.get(current_key, []))
        if current_saved:
            return current_saved

        previous = set(self.selected_columns)
        restored = {col for col in previous if col in self.column_data}
        if restored:
            return restored

        if self.current_used_mode1_template:
            if "CO2浓度" in self.column_data:
                return {"CO2浓度"}
            if "H2O浓度" in self.column_data:
                return {"H2O浓度"}

        return set()

    @property
    def selected_columns(self) -> list[str]:
        return [name for name, var in self.column_vars.items() if var.get()]

    def remember_current_selection(self) -> None:
        if (
            not self.preserve_selection_var.get()
            or self.current_file is None
            or self.current_data_source_kind == "comparison_analysis"
        ):
            return
        self.saved_selections[str(self.current_file)] = list(self.selected_columns)

    def on_column_changed(self, changed_name: str) -> None:
        chosen = self.selected_columns
        if len(chosen) > 2:
            self.column_vars[changed_name].set(False)
            messagebox.showwarning("警告", "最多只能选择两列进行分析。")
            self.update_diagnostic_info(layout=self.current_layout_label)
            self.render_plot_message("当前最多支持 1 列功率谱密度或 2 列互谱分析。", level="warning")
            return

        if (
            self.preserve_selection_var.get()
            and self.current_file is not None
            and self.current_data_source_kind != "comparison_analysis"
        ):
            self.saved_selections[str(self.current_file)] = list(chosen)

        self.schedule_auto_analysis()

    def schedule_auto_analysis(self) -> None:
        if self.auto_analysis_after_id is not None:
            try:
                self.root.after_cancel(self.auto_analysis_after_id)
            except tk.TclError:
                pass
            self.auto_analysis_after_id = None
        self.auto_analysis_after_id = self.root.after(80, self.auto_perform_analysis)

    def auto_perform_analysis(self) -> None:
        self.auto_analysis_after_id = None
        chosen = self.selected_columns
        if len(chosen) == 1:
            if self.current_data_source_kind == "file" and self.get_selected_file_paths():
                self.perform_default_single_column_render(quiet=True)
            else:
                self.perform_analysis("spectral", quiet=True)
        elif len(chosen) == 2:
            self.perform_analysis("cross", quiet=True)
        elif len(chosen) == 0:
            if self.current_data_source_kind == "comparison_analysis":
                self.update_diagnostic_info(layout=self.current_layout_label)
                self.render_plot_message(
                    "当前主表是对比汇总表。\n\n请勾选 1 列做功率谱密度，或勾选 2 列做互谱/协谱/正交谱分析。",
                    level="info",
                )
                return
            if self.current_file is None:
                self.update_diagnostic_info(layout=self.current_layout_label)
                self.render_plot_message("请选择文件并勾选要分析的列。")
            elif self.column_data:
                self.update_diagnostic_info(layout=self.current_layout_label)
                self.render_plot_message("请先在左侧勾选 1 列做功率谱密度，或勾选 2 列做互谱分析。")
        else:
            self.status_var.set("请选择 1 列做 PSD，或选择 2 列做互谱。")
            self.update_diagnostic_info(layout=self.current_layout_label)
            if self.current_data_source_kind == "comparison_analysis":
                self.render_plot_message("对比汇总表当前最多支持 1 列做 PSD，或 2 列做互谱/协谱/正交谱分析。", level="warning")
            else:
                self.render_plot_message("当前最多支持 1 列功率谱密度或 2 列互谱分析。", level="warning")

    def generate_plot(self) -> None:
        chosen = self.selected_columns
        selected_files = self.get_selected_file_paths()

        if len(chosen) == 1:
            if selected_files:
                self.perform_multi_spectral_compare()
            else:
                self.perform_analysis("spectral")
            return

        if len(chosen) == 2:
            self.perform_analysis("cross")
            return

        if len(chosen) == 0:
            self.status_var.set("生成图失败：未选择分析列")
            self.update_diagnostic_info(layout=self.current_layout_label)
            self.render_plot_message("请先在左侧勾选 1 列或 2 列，然后再点击“生成图”。", level="warning")
            return

        self.status_var.set("生成图失败：选择列数量不正确")
        self.update_diagnostic_info(layout=self.current_layout_label)
        self.render_plot_message("当前最多支持 1 列功率谱密度或 2 列互谱分析。", level="warning")

    def perform_default_single_column_render(self, quiet: bool = False) -> None:
        selected_files = self.get_selected_file_paths()
        if not selected_files or self.current_data_source_kind == "comparison_analysis":
            self.perform_analysis("spectral", quiet=quiet)
            return

        try:
            target_column = self.selected_columns[0]
            fs, requested_nsegment, overlap_ratio = self.get_analysis_params()
            start_dt, end_dt = self.resolve_optional_time_range_inputs()
            payload = self.prepare_selected_files_default_render_payload(
                selected_files,
                target_column,
                fs,
                requested_nsegment,
                overlap_ratio,
                start_dt=start_dt,
                end_dt=end_dt,
            )
            previous_quiet = bool(self.pending_selected_files_default_render_quiet)
            self.pending_selected_files_default_render_quiet = bool(quiet)
            try:
                self.on_selected_files_default_plot_ready(payload)
            finally:
                self.pending_selected_files_default_render_quiet = previous_quiet
            self.last_param_error_message = None
        except Exception as exc:
            message = str(exc)
            self.status_var.set(f"分析失败：{message}")
            self.update_diagnostic_info(layout=self.current_layout_label)
            level = "warning" if message.startswith("参数错误") else "error"
            self.render_plot_message(message, level=level)
            is_param_error = message.startswith("参数错误")
            if is_param_error:
                if self.last_param_error_message != message and not quiet:
                    messagebox.showerror("参数错误", message)
                self.last_param_error_message = message
            elif not quiet:
                messagebox.showerror("分析失败", message)

    def perform_analysis(self, analysis_type: str, quiet: bool = False) -> None:
        try:
            if analysis_type == "spectral":
                if len(self.selected_columns) != 1:
                    raise ValueError("频谱分析需要且只能选择 1 列。")
                freq, density, details = self.spectral_analysis()
                self.plot_results("spectral", freq, density, self.selected_columns, details)
            else:
                if len(self.selected_columns) != 2:
                    raise ValueError("互谱分析需要且只能选择 2 列。")
                freq, density, details = self.cross_spectral_analysis()
                warning_message = str(details.get("selection_warning") or "").strip()
                if warning_message and not quiet:
                    messagebox.showwarning("提示", warning_message)
                self.plot_results("cross", freq, density, self.selected_columns, details)
            self.last_param_error_message = None
        except Exception as exc:
            message = str(exc)
            self.status_var.set(f"分析失败：{message}")
            self.update_diagnostic_info(layout=self.current_layout_label)
            level = "warning" if message.startswith("参数错误") else "error"
            self.render_plot_message(message, level=level)
            is_param_error = message.startswith("参数错误")
            if is_param_error:
                if self.last_param_error_message != message:
                    messagebox.showerror("参数错误", message)
                    self.last_param_error_message = message
            elif not quiet:
                messagebox.showerror("分析失败", message)

    def compute_psd_from_array_with_params(
        self,
        data: np.ndarray,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return core.compute_psd_from_array_with_params(data, fs, requested_nsegment, overlap_ratio)

    def compute_psd_from_array(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        fs, requested_nsegment, overlap_ratio = self.get_analysis_params()
        return self.compute_psd_from_array_with_params(data, fs, requested_nsegment, overlap_ratio)

    def compute_base_psd_payload(
        self,
        parsed: ParsedFileResult,
        value_column: str,
        *,
        start_dt: pd.Timestamp | None = None,
        end_dt: pd.Timestamp | None = None,
        require_timestamp: bool = False,
    ) -> dict[str, Any]:
        fs, requested_nsegment, overlap_ratio = self.get_analysis_params()
        return core.compute_base_spectrum_payload(
            parsed,
            value_column,
            fs_ui=fs,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            start_dt=start_dt,
            end_dt=end_dt,
            require_timestamp=require_timestamp,
        )

    def get_analysis_params(self) -> tuple[float, int, float]:
        # Practical completion: FS defaulted to 10.0 and exposed in UI for usability.
        try:
            fs = float(self.fs_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError("参数错误：FS 必须是大于 0 的数字。") from exc
        if fs <= 0:
            raise ValueError("参数错误：FS 必须大于 0。")

        try:
            nsegment = int(self.nsegment_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError("参数错误：NSEGMENT 必须是大于等于 2 的整数。") from exc
        if nsegment < 2:
            raise ValueError("参数错误：NSEGMENT 必须大于等于 2。")

        try:
            overlap_ratio = float(self.overlap_ratio_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError("参数错误：OVERLAP_RATIO 必须在 [0, 1) 范围内。") from exc
        if not (0 <= overlap_ratio < 1):
            raise ValueError("参数错误：OVERLAP_RATIO 必须在 [0, 1) 范围内。")

        return fs, nsegment, overlap_ratio

    def get_selected_cross_spectrum_type(self, compare_mode: str | None = None) -> str:
        if compare_mode == "互谱幅值":
            return CROSS_SPECTRUM_MAGNITUDE
        if compare_mode == "协谱图":
            return CROSS_SPECTRUM_REAL
        if compare_mode == "正交谱图":
            return CROSS_SPECTRUM_IMAG
        spectrum_type = self.cross_spectrum_type_var.get().strip() or CROSS_SPECTRUM_MAGNITUDE
        if spectrum_type not in CROSS_SPECTRUM_OPTIONS:
            return CROSS_SPECTRUM_MAGNITUDE
        return spectrum_type

    def get_selected_reference_slope_mode(self) -> str:
        return normalize_reference_slope_mode(self.reference_slope_mode_var.get())

    def get_cross_spectrum_display_meta(self, spectrum_type: str) -> dict[str, str]:
        return core.get_cross_spectrum_display_meta(spectrum_type)

    def build_reference_slope_diagnostic_items(
        self,
        *,
        is_psd: bool,
        spectrum_type: str | None = None,
    ) -> list[str]:
        selected_mode = self.get_selected_reference_slope_mode()
        specs = resolve_reference_slope_specs(selected_mode, is_psd=is_psd, spectrum_type=spectrum_type)
        return [
            f"reference_slope_mode={selected_mode}",
            f"reference_slope_selection={format_reference_slope_selection(specs)}",
        ]

    def add_selected_reference_slopes(
        self,
        ax: Any,
        freq: np.ndarray,
        density: np.ndarray,
        *,
        is_psd: bool,
        spectrum_type: str | None = None,
        use_fixed_power_law: bool = False,
        amplitude_at_1hz: float = 1.0,
    ) -> list[dict[str, float | str]]:
        specs = resolve_reference_slope_specs(
            self.get_selected_reference_slope_mode(),
            is_psd=is_psd,
            spectrum_type=spectrum_type,
        )
        for spec in specs:
            slope = float(spec["slope"])
            color = str(spec["color"])
            short_label = str(spec["short_label"])
            if use_fixed_power_law:
                self.add_fixed_reference_power_law(
                    ax,
                    freq,
                    slope=slope,
                    amplitude_at_1hz=amplitude_at_1hz,
                    label=short_label,
                    color=color,
                )
            else:
                self.add_reference_slope(
                    ax,
                    freq,
                    density,
                    slope,
                    color=color,
                    label=str(spec["label"]),
                )
        return specs

    def compute_cross_spectrum_from_arrays_with_params(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        fs: float,
        requested_nsegment: int,
        overlap_ratio: float,
        *,
        spectrum_type: str,
        insufficient_message: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return core.compute_cross_spectrum_from_arrays_with_params(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
        )

    def spectral_analysis(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        col = self.selected_columns[0]
        direct_generate_fallback_reason = str(
            self.pending_direct_generate_target_spectrum_fallback_reason or ""
        ).strip()
        force_plain_psd_fallback = bool(direct_generate_fallback_reason)
        if self.current_data_source_kind == "file" and self.current_file_parsed is not None:
            if not force_plain_psd_fallback:
                compare_equivalent_payload = self.build_single_txt_compare_equivalent_payload(col)
                if compare_equivalent_payload is not None:
                    details = dict(compare_equivalent_payload["details"])
                    return (
                        np.asarray(compare_equivalent_payload["freq"], dtype=float),
                        np.asarray(compare_equivalent_payload["density"], dtype=float),
                        details,
                    )
            start_dt, end_dt = self.resolve_optional_time_range_inputs()
            payload = self.compute_base_psd_payload(
                self.current_file_parsed,
                col,
                start_dt=start_dt,
                end_dt=end_dt,
                require_timestamp=False,
            )
            details = dict(payload["details"])
            details["analysis_context"] = "single_file_base_spectrum"
            details.update(
                core.build_single_time_range_metadata(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    has_timestamp=bool(payload["series_meta"].get("has_timestamp")),
                    requested_start=details.get("base_requested_start"),
                    requested_end=details.get("base_requested_end"),
                    actual_start=details.get("base_actual_start"),
                    actual_end=details.get("base_actual_end"),
                )
            )
            selection_summary = self.extract_single_device_txt_selection([Path(self.current_file)])
            device_id = core.resolve_device_identifier(self.current_file_parsed, Path(self.current_file)).get("device_id")
            annotate_device_dispatch_details(
                details,
                effective_device_ids=[str(device_id or Path(self.current_file).stem)],
                plot_execution_path="single_device_spectrum",
                render_semantics="single_device",
                selection_file_count=1,
                selected_txt_file_count=int(selection_summary.get("selected_txt_file_count", 0)),
                selected_dat_file_count=int(selection_summary.get("selected_dat_file_count", 0)),
                data_context_source="current_file",
            )
            if force_plain_psd_fallback:
                details["single_txt_execution_path"] = "direct_generate_plain_spectral_fallback"
                details["single_txt_selection_scope"] = "current_file"
                details["direct_generate_target_spectrum_fallback_reason"] = direct_generate_fallback_reason
            return np.asarray(payload["freq"], dtype=float), np.asarray(payload["density"], dtype=float), details
        data = self.column_data[col]
        return self.compute_psd_from_array(data)

    def cross_spectral_analysis(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        selected_columns = list(self.selected_columns)
        pair_info = core.resolve_target_cross_pair(selected_columns, self.current_comparison_metadata)
        ordered_columns = list(pair_info["ordered_columns"]) if pair_info["ordered_columns"] else selected_columns
        col1, col2 = ordered_columns
        data1 = self.column_data[col1]
        data2 = self.column_data[col2]
        fs, requested_nsegment, overlap_ratio = self.get_analysis_params()
        spectrum_type = self.get_selected_cross_spectrum_type()
        if bool(pair_info["uses_canonical_pair"]):
            freq, details = core.compute_target_cross_complex_from_selected_implementation(
                data1,
                data2,
                fs,
                requested_nsegment,
                overlap_ratio,
                implementation_id=core.TARGET_COSPECTRUM_IMPLEMENTATION_ID,
                insufficient_message="当前数据不足以生成有效互谱/协谱结果。",
            )
        else:
            freq, _values, details = self.compute_cross_spectrum_from_arrays_with_params(
                data1,
                data2,
                fs,
                requested_nsegment,
                overlap_ratio,
                spectrum_type=spectrum_type,
                insufficient_message="当前数据不足以生成有效协谱图。",
            )
        details = dict(details)
        resolved_pair = pair_info.get("resolved_pair") if isinstance(pair_info.get("resolved_pair"), dict) else None
        if bool(pair_info["uses_canonical_pair"]):
            details["cross_execution_path"] = "target_spectral_canonical"
        else:
            details.update(core.describe_generic_default_cross_implementation())
            details["alignment_strategy"] = core.GENERIC_SAME_FRAME_ALIGNMENT_STRATEGY
            details["cross_reference_column"] = str(col1)
            details["reference_column"] = str(col1)
            details["target_column"] = str(col2)
            details["cross_order"] = f"{col1} -> {col2}"
        details["ordered_columns"] = ordered_columns
        details["selected_columns"] = selected_columns
        details["uses_canonical_pair"] = bool(pair_info["uses_canonical_pair"])
        details["auto_reordered"] = bool(pair_info["auto_reordered"])
        details["selection_warning"] = pair_info.get("warning")
        if pair_info.get("context") is not None:
            details["target_spectral_context"] = pair_info["context"]
            if resolved_pair is not None and bool(pair_info["uses_canonical_pair"]):
                details["reference_column"] = resolved_pair.get("summary_reference_column", ordered_columns[0])
                details["cross_reference_column"] = resolved_pair.get("summary_reference_column", ordered_columns[0])
                details["target_column"] = resolved_pair.get("summary_target_column", ordered_columns[1])
                details["cross_order"] = resolved_pair.get("summary_cross_order", f"{ordered_columns[0]} -> {ordered_columns[1]}")
                details["series_role"] = resolved_pair.get("series_role")
                details["device_kind"] = resolved_pair.get("device_kind")
                details["display_label"] = resolved_pair.get("display_label")
            else:
                details["reference_column"] = pair_info["context"].get("reference_column")
                details["cross_reference_column"] = pair_info["context"].get("reference_column")
                details["target_column"] = pair_info["context"].get("target_column")
                details["cross_order"] = pair_info["context"].get("cross_order")
        analysis_context = (
            pair_info["context"].get("analysis_context")
            if isinstance(pair_info.get("context"), dict)
            else details.get("analysis_context")
        )
        values, _mask, details = core.resolve_cross_display_output(
            freq,
            details,
            analysis_context=analysis_context,
            cross_execution_path=details.get("cross_execution_path"),
            spectrum_type=spectrum_type,
            insufficient_message="当前数据不足以生成有效互谱/协谱结果。",
        )
        return freq, values, details

    def resolve_compare_fs(self, parsed: ParsedFileResult, ui_fs: float) -> float:
        resolved, _source = core.resolve_profile_aware_fs(parsed, ui_fs)
        return float(resolved)

    def compute_alignment_tolerance_seconds(self, timestamps_a: pd.Series, timestamps_b: pd.Series) -> float:
        intervals: list[float] = []
        for timestamps in (timestamps_a, timestamps_b):
            sorted_values = parse_mixed_timestamp_series(pd.Series(timestamps)).dropna().sort_values()
            deltas = sorted_values.diff().dropna().dt.total_seconds()
            deltas = deltas[deltas > 0]
            if not deltas.empty:
                intervals.append(float(deltas.median()))
        if not intervals:
            return 1.0
        return max(min(intervals) * 0.6, 0.001)

    def estimate_resample_interval_seconds(self, timestamps_a: pd.Series, timestamps_b: pd.Series) -> float:
        intervals: list[float] = []
        for timestamps in (timestamps_a, timestamps_b):
            sorted_values = parse_mixed_timestamp_series(pd.Series(timestamps)).dropna().sort_values()
            deltas = sorted_values.diff().dropna().dt.total_seconds()
            deltas = deltas[deltas > 0]
            if not deltas.empty:
                intervals.append(float(deltas.median()))
        if not intervals:
            return 1.0
        return max(max(intervals), 0.001)

    def get_match_tolerance_seconds(self, default_value: float | None = None) -> float:
        try:
            tolerance = float(self.match_tolerance_var.get())
        except (TypeError, ValueError):
            if default_value is not None:
                return default_value
            raise ValueError("匹配容差必须是大于 0 的数字。")
        if tolerance <= 0:
            raise ValueError("匹配容差必须大于 0。")
        return tolerance

    def get_alignment_strategy(self, compare_mode: str) -> str:
        if compare_mode == "时间序列对比":
            return "原始时间轴叠加"
        return self.alignment_strategy_var.get().strip() or "最近邻 + 容差"

    def resolve_plot_style(self, compare_mode: str) -> str:
        style = self.plot_style_var.get().strip() or "自动"
        if style != "自动":
            return style
        if compare_mode == "时间序列对比":
            return "连线图"
        if compare_mode in {"散点一致性对比", "时间段内 PSD 对比", "协谱图", "正交谱图"}:
            return "散点图"
        return "连线图"

    def resolve_plot_layout(self, compare_mode: str) -> str:
        layout = self.plot_layout_var.get().strip() or "叠加同图"
        if compare_mode in {"散点一致性对比", "差值时间序列", "比值时间序列", "互谱幅值", "协谱图", "正交谱图"}:
            return "叠加同图"
        return layout

    def get_plot_style_tag(self, style: str) -> str:
        return {
            "连线图": "line",
            "散点图": "scatter",
            "连线+散点": "line_scatter",
        }.get(style, "auto")

    def get_plot_layout_tag(self, layout: str) -> str:
        return {
            "叠加同图": "overlay",
            "上下分图": "split_vertical",
            "左右分图": "split_horizontal",
            "分别生成两张图": "separate",
        }.get(layout, "overlay")

    def clear_separate_plot_windows(self) -> None:
        for window in self.separate_plot_windows:
            try:
                window.destroy()
            except tk.TclError:
                pass
        self.separate_plot_windows = []
        self.separate_plot_entries = []

    def reset_plot_output_state(self) -> None:
        self.clear_separate_plot_windows()
        self.current_plot_style_label = ""
        self.current_plot_layout_label = ""
        self.current_lazy_aligned_frames = []
        self.current_point_count_contract = {}

    def apply_series_style(
        self,
        ax: Any,
        x: Any,
        y: Any,
        *,
        color: str,
        label: str,
        style: str,
        linewidth: float = 1.2,
        marker_size: int = 22,
        alpha: float = 0.8,
        marker: str | None = None,
        line_style: str | None = None,
        zorder: float | None = None,
    ) -> None:
        if style in {"连线图", "连线+散点"}:
            ax.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                linestyle=line_style or "-",
                label=label,
                alpha=alpha,
                zorder=zorder,
            )
        if style in {"散点图", "连线+散点"}:
            ax.scatter(
                x,
                y,
                color=color,
                s=marker_size,
                alpha=alpha,
                edgecolors="none",
                marker=marker or "o",
                label=label if style == "散点图" else None,
                zorder=zorder,
            )

    def build_spectrum_plot_mask(
        self,
        freq: np.ndarray,
        values: np.ndarray,
        spectrum_type: str,
        *,
        display_semantics: str | None = None,
    ) -> np.ndarray:
        return core.build_spectrum_plot_mask(
            freq,
            values,
            spectrum_type,
            display_semantics=display_semantics,
        )

    def configure_cross_spectrum_axis(
        self,
        ax: Any,
        freq: np.ndarray,
        values: np.ndarray,
        spectrum_type: str,
        *,
        display_semantics: str | None = None,
        title: str,
        style: str,
        label: str,
        color: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved_display_semantics = str(display_semantics or "").strip() or None
        mask = self.build_spectrum_plot_mask(
            freq,
            values,
            spectrum_type,
            display_semantics=resolved_display_semantics,
        )
        freq_plot = freq[mask]
        values_plot = values[mask]
        if len(freq_plot) == 0:
            if resolved_display_semantics == core.CROSS_DISPLAY_SEMANTICS_ABS or spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
                raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
            raise ValueError("当前数据不足以生成有效协谱图。")

        ax.set_xscale("log")
        if resolved_display_semantics == core.CROSS_DISPLAY_SEMANTICS_ABS or spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
            ax.set_yscale("log")
        else:
            max_abs = float(np.nanmax(np.abs(values_plot))) if len(values_plot) else 0.0
            linthresh = max(max_abs * 1e-3, 1e-12)
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

        self.apply_series_style(ax, freq_plot, values_plot, color=color, label=label, style=style)
        self.add_selected_reference_slopes(
            ax,
            freq_plot,
            values_plot,
            is_psd=False,
            spectrum_type=spectrum_type,
        )
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(self.get_cross_spectrum_display_meta(spectrum_type)["ylabel"])
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper right")
        return freq_plot, values_plot

    def build_time_series_axis(
        self,
        ax: Any,
        x: Any,
        y: Any,
        *,
        color: str,
        label: str,
        style: str,
        title: str,
    ) -> None:
        self.apply_series_style(ax, x, y, color=color, label=label, style=style)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper right")

    def build_psd_axis(
        self,
        ax: Any,
        freq: np.ndarray,
        density: np.ndarray,
        *,
        color: str,
        label: str,
        style: str,
        title: str,
        add_reference: bool = False,
    ) -> None:
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.apply_series_style(ax, freq, density, color=color, label=label, style=style)
        if add_reference:
            self.add_selected_reference_slopes(ax, freq, density, is_psd=True)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Density")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper right")

    def build_cross_export_frame(
        self,
        freq: np.ndarray,
        details: dict[str, Any],
        mask: np.ndarray,
        *,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        frame = core.build_cross_spectrum_export_frame(
            freq,
            details,
            mask,
            prefix=prefix,
            target_context=details.get("target_spectral_context"),
        )
        return self.attach_metadata_columns(
            frame,
            build_export_metadata_from_details(details, include_additional=True),
        )

    def merge_frequency_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for frame in frames:
            if merged is None:
                merged = frame.copy()
            else:
                merged = merged.merge(frame, on="frequency_hz", how="outer")
        if merged is None:
            return pd.DataFrame(columns=["frequency_hz"])
        return merged.sort_values("frequency_hz").reset_index(drop=True)

    def register_plot_style_layout(self, style: str, layout: str) -> None:
        self.current_plot_style_label = style
        self.current_plot_layout_label = layout

    def show_separate_plot_windows(self, entries: list[dict[str, Any]]) -> None:
        self.clear_separate_plot_windows()
        self.separate_plot_entries = entries
        for entry in entries:
            window = tk.Toplevel(self.root)
            window.title(entry["title"])
            window.geometry("780x520")
            icon_image = apply_window_icon(window)
            if icon_image is not None:
                window._icon_image_ref = icon_image
            canvas = FigureCanvasTkAgg(entry["figure"], master=window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            try:
                window.lift()
            except tk.TclError:
                pass
            try:
                window.focus_force()
            except tk.TclError:
                pass
            self.separate_plot_windows.append(window)

    def render_separate_window_summary(
        self,
        *,
        mode_title: str,
        device_a_summary: str,
        device_b_summary: str,
        window_count: int,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.80, mode_title, ha="center", va="center", fontsize=16, color="#234e70", weight="bold", transform=ax.transAxes)
        ax.text(0.08, 0.60, f"设备A：{device_a_summary}", ha="left", va="center", fontsize=13, color="#1f2937", transform=ax.transAxes)
        ax.text(0.08, 0.46, f"设备B：{device_b_summary}", ha="left", va="center", fontsize=13, color="#1f2937", transform=ax.transAxes)
        ax.text(
            0.5,
            0.24,
            f"已弹出独立图窗口：{window_count} 个\n可在独立窗口中查看、缩放和保存图片",
            ha="center",
            va="center",
            fontsize=13,
            color="#4f5d75",
            transform=ax.transAxes,
        )
        self.figure.tight_layout()
        self.refresh_canvas(title=mode_title)

    def resolve_compare_time_range(
        self,
        txt_summary: dict[str, Any],
        dat_summary: dict[str, Any],
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        strategy = self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围"
        if strategy == "手动输入时间范围":
            return self.resolve_time_range_inputs()
        if strategy == "使用 dat 时间范围":
            if txt_summary["start"] > dat_summary["start"] or txt_summary["end"] < dat_summary["end"]:
                raise ValueError("txt 覆盖不了 dat 时间段，请先自动补齐 txt 覆盖范围或改用共同时间范围。")
            return dat_summary["start"], dat_summary["end"]
        if strategy == "最近10分钟":
            return dat_summary["end"] - pd.Timedelta(minutes=10), dat_summary["end"]
        if strategy == "最近30分钟":
            return dat_summary["end"] - pd.Timedelta(minutes=30), dat_summary["end"]
        if strategy == "最近1小时":
            return dat_summary["end"] - pd.Timedelta(hours=1), dat_summary["end"]

        common_start = max(txt_summary["start"], dat_summary["start"])
        common_end = min(txt_summary["end"], dat_summary["end"])
        if common_start > common_end:
            raise ValueError(self.build_no_common_time_range_message(txt_summary, dat_summary))
        return common_start, common_end

    def build_no_common_time_range_message(
        self,
        txt_summary: dict[str, Any],
        dat_summary: dict[str, Any],
    ) -> str:
        return (
            "当前没有共同时间范围。\n\n"
            f"设备A开始时间：{txt_summary['start']}\n"
            f"设备A结束时间：{txt_summary['end']}\n"
            f"设备B开始时间：{dat_summary['start']}\n"
            f"设备B结束时间：{dat_summary['end']}\n"
            "共同时间范围：空\n\n"
            "建议：请选择覆盖该 dat 时间段的更多半小时 txt/log 文件。"
        )

    def resolve_compare_time_range_for_strategy(
        self,
        strategy: str,
        start_raw: str,
        end_raw: str,
        txt_summary: dict[str, Any] | None,
        dat_summary: dict[str, Any] | None,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if strategy == "手动输入时间范围":
            start_dt = self.parse_time_input(start_raw)
            end_dt = self.parse_time_input(end_raw)
            if start_dt is not None and end_dt is not None and start_dt > end_dt:
                raise ValueError("开始时间不能晚于结束时间。")
            return start_dt, end_dt

        if dat_summary is None:
            return self.parse_time_input(start_raw), self.parse_time_input(end_raw)

        if strategy == "使用 dat 时间范围":
            if txt_summary is not None and (txt_summary["start"] > dat_summary["start"] or txt_summary["end"] < dat_summary["end"]):
                raise ValueError("txt 覆盖不了 dat 时间段，请先自动补齐 txt 覆盖范围或改用共同时间范围。")
            return dat_summary["start"], dat_summary["end"]

        if strategy == "最近10分钟":
            return dat_summary["end"] - pd.Timedelta(minutes=10), dat_summary["end"]
        if strategy == "最近30分钟":
            return dat_summary["end"] - pd.Timedelta(minutes=30), dat_summary["end"]
        if strategy == "最近1小时":
            return dat_summary["end"] - pd.Timedelta(hours=1), dat_summary["end"]

        if txt_summary is None:
            return dat_summary["start"], dat_summary["end"]
        common_start = max(txt_summary["start"], dat_summary["start"])
        common_end = min(txt_summary["end"], dat_summary["end"])
        if common_start > common_end:
            raise ValueError(self.build_no_common_time_range_message(txt_summary, dat_summary))
        return common_start, common_end

    def resolve_target_time_range_for_strategy(
        self,
        strategy: str,
        start_raw: str,
        end_raw: str,
        txt_summary: dict[str, Any] | None,
        dat_summary: dict[str, Any] | None,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
        if strategy == "手动输入时间范围":
            start_dt = self.parse_time_input(start_raw)
            end_dt = self.parse_time_input(end_raw)
            if start_dt is not None and end_dt is not None and start_dt > end_dt:
                raise ValueError("开始时间不能晚于结束时间。")
            return start_dt, end_dt, "手动时间范围"

        if dat_summary is None:
            return self.parse_time_input(start_raw), self.parse_time_input(end_raw), strategy

        if strategy == "使用 dat 时间范围":
            return pd.Timestamp(dat_summary["start"]), pd.Timestamp(dat_summary["end"]), "使用 dat 时间范围"

        if strategy == "最近10分钟":
            return pd.Timestamp(dat_summary["end"]) - pd.Timedelta(minutes=10), pd.Timestamp(dat_summary["end"]), "最近10分钟"
        if strategy == "最近30分钟":
            return pd.Timestamp(dat_summary["end"]) - pd.Timedelta(minutes=30), pd.Timestamp(dat_summary["end"]), "最近30分钟"
        if strategy == "最近1小时":
            return pd.Timestamp(dat_summary["end"]) - pd.Timedelta(hours=1), pd.Timestamp(dat_summary["end"]), "最近1小时"

        if txt_summary is None:
            return pd.Timestamp(dat_summary["start"]), pd.Timestamp(dat_summary["end"]), "使用 dat 时间范围"

        common_start = max(pd.Timestamp(txt_summary["start"]), pd.Timestamp(dat_summary["start"]))
        common_end = min(pd.Timestamp(txt_summary["end"]), pd.Timestamp(dat_summary["end"]))
        if common_start > common_end:
            raise ValueError(self.build_no_common_time_range_message(txt_summary, dat_summary))
        return common_start, common_end, "使用 txt+dat 共同时间范围"

    def build_comparison_column_candidates(
        self,
        parsed: ParsedFileResult,
        single_value: str,
        multi_values: list[str],
    ) -> list[str]:
        ordered: list[str] = []
        for value in [single_value, *multi_values, *parsed.suggested_columns]:
            text = str(value).strip()
            if not text or text not in parsed.dataframe.columns or text in ordered:
                continue
            ordered.append(text)
        return ordered

    def prepare_comparison_side_frame(
        self,
        parsed: ParsedFileResult,
        columns: list[str],
        prefix: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
            raise ValueError("未识别到时间列，无法生成对比汇总表。")

        filtered = filter_by_time_range(parsed.dataframe, parsed.timestamp_col, start_dt, end_dt)
        if filtered.empty:
            raise ValueError("当前时间范围内没有数据，无法生成对比汇总表。")

        frame = pd.DataFrame({"时间戳": parse_mixed_timestamp_series(filtered[parsed.timestamp_col])})
        actual_columns: list[str] = []
        for column in columns:
            if column not in filtered.columns:
                continue
            prefixed = f"{prefix}_{column}"
            frame[prefixed] = pd.to_numeric(filtered[column], errors="coerce")
            actual_columns.append(column)

        frame = frame.dropna(subset=["时间戳"]).sort_values("时间戳").drop_duplicates("时间戳", keep="first").reset_index(drop=True)
        if frame.empty:
            raise ValueError("当前时间范围内没有有效时间戳，无法生成对比汇总表。")
        return frame, actual_columns

    def build_comparison_dataframe(
        self,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
        a_columns: list[str],
        b_columns: list[str],
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        alignment_strategy: str,
        tolerance_seconds: float,
        target_context: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame_a, actual_a_columns = self.prepare_comparison_side_frame(parsed_a, a_columns, "A", start_dt, end_dt)
        frame_b, actual_b_columns = self.prepare_comparison_side_frame(parsed_b, b_columns, "B", start_dt, end_dt)

        if alignment_strategy == "重采样后对齐":
            interval_seconds = self.estimate_resample_interval_seconds(frame_a["时间戳"], frame_b["时间戳"])
            interval_ms = max(int(round(interval_seconds * 1000.0)), 1)
            rule = f"{interval_ms}ms"
            resampled_a = (
                frame_a.set_index("时间戳")
                .sort_index()
                .resample(rule)
                .mean()
                .interpolate(method="time")
                .dropna(how="all")
            )
            resampled_b = (
                frame_b.set_index("时间戳")
                .sort_index()
                .resample(rule)
                .mean()
                .interpolate(method="time")
                .dropna(how="all")
            )
            comparison = resampled_a.join(resampled_b, how="inner").reset_index()
            comparison["匹配时间差_s"] = 0.0
        else:
            frame_b_aligned = frame_b.rename(columns={"时间戳": "B_匹配时间戳"})
            merge_kwargs: dict[str, Any] = {
                "left_on": "时间戳",
                "right_on": "B_匹配时间戳",
                "direction": "nearest",
            }
            if alignment_strategy == "最近邻 + 容差":
                merge_kwargs["tolerance"] = pd.to_timedelta(tolerance_seconds, unit="s")
            comparison = pd.merge_asof(
                frame_a.sort_values("时间戳"),
                frame_b_aligned.sort_values("B_匹配时间戳"),
                **merge_kwargs,
            ).dropna(subset=["B_匹配时间戳"])
            comparison["匹配时间差_s"] = (
                (comparison["时间戳"] - comparison["B_匹配时间戳"]).abs().dt.total_seconds()
            )

        a_prefixed = [f"A_{column}" for column in actual_a_columns if f"A_{column}" in comparison.columns]
        b_prefixed = [f"B_{column}" for column in actual_b_columns if f"B_{column}" in comparison.columns]
        if a_prefixed:
            comparison = comparison.loc[~comparison[a_prefixed].isna().all(axis=1)]
        if b_prefixed:
            comparison = comparison.loc[~comparison[b_prefixed].isna().all(axis=1)]
        comparison = comparison.sort_values("时间戳").reset_index(drop=True)
        if comparison.empty:
            raise ValueError("时间对齐后没有可用数据，无法生成对比汇总表。")

        start_text = pd.Timestamp(comparison["时间戳"].min()).strftime("%Y-%m-%d %H:%M:%S")
        end_text = pd.Timestamp(comparison["时间戳"].max()).strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "alignment_strategy": alignment_strategy,
            "time_range_label": f"{start_text} ~ {end_text}",
            "start": pd.Timestamp(comparison["时间戳"].min()),
            "end": pd.Timestamp(comparison["时间戳"].max()),
            "rows": int(len(comparison)),
            "columns": int(len(comparison.columns)),
            "a_columns": actual_a_columns,
            "b_columns": actual_b_columns,
        }
        resolved_target_context = core.get_target_spectral_context(target_context)
        if resolved_target_context is not None:
            metadata.update(resolved_target_context)
            metadata["target_spectral_context"] = resolved_target_context
            ordered_columns: list[str] = ["时间戳"]
            for column in [
                str(resolved_target_context.get("summary_ygas_target_column") or resolved_target_context.get("summary_target_column") or ""),
                str(resolved_target_context.get("summary_dat_target_column") or ""),
                str(resolved_target_context.get("summary_reference_column") or ""),
            ]:
                if column and column in comparison.columns and column not in ordered_columns:
                    ordered_columns.append(column)
            for column in comparison.columns:
                if column in {"时间戳", "B_匹配时间戳", "匹配时间差_s"} or column in ordered_columns:
                    continue
                ordered_columns.append(column)
            for column in ["B_匹配时间戳", "匹配时间差_s"]:
                if column in comparison.columns:
                    ordered_columns.append(column)
            comparison = comparison.loc[:, ordered_columns]
        return comparison, metadata

    def classify_comparison_analysis_frame(
        self,
        comparison_frame: pd.DataFrame,
        *,
        metadata: dict[str, Any] | None,
        excluded_columns: set[str],
    ) -> tuple[dict[str, np.ndarray], set[str], set[str]]:
        target_context = core.get_target_spectral_context(metadata)
        ordered_columns: list[str] = []
        if target_context is not None:
            for column in [
                str(target_context.get("summary_ygas_target_column") or target_context.get("summary_target_column") or ""),
                str(target_context.get("summary_dat_target_column") or ""),
                str(target_context.get("summary_reference_column") or ""),
            ]:
                if column and column in comparison_frame.columns and column not in excluded_columns and column not in ordered_columns:
                    ordered_columns.append(column)
        for column in comparison_frame.columns:
            if column in excluded_columns or column in ordered_columns:
                continue
            ordered_columns.append(column)

        column_data: dict[str, np.ndarray] = {}
        non_numeric_cols: set[str] = set()
        unsuitable_spectrum_cols: set[str] = set()
        for column in ordered_columns:
            numeric = pd.to_numeric(comparison_frame[column], errors="coerce")
            if int(numeric.notna().sum()) == 0:
                non_numeric_cols.add(column)
                continue
            values = numeric.to_numpy(dtype=float)
            column_data[column] = values
            finite = np.isfinite(values)
            if int(np.count_nonzero(finite)) < 2 or float(np.nanstd(values[finite])) <= 1e-12:
                unsuitable_spectrum_cols.add(column)
        return column_data, non_numeric_cols, unsuitable_spectrum_cols

    def prepare_comparison_summary_payload(
        self,
        *,
        ygas_paths: list[Path],
        dat_path: Path | None,
        selected_paths: list[Path],
        strategy: str,
        start_raw: str,
        end_raw: str,
        alignment_strategy: str,
        tolerance_seconds: float,
        a_single: str,
        b_single: str,
        a_multi: list[str],
        b_multi: list[str],
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        compare_payload = self.prepare_dual_compare_payload(
            ygas_paths=ygas_paths,
            dat_path=dat_path,
            selected_paths=selected_paths,
            compare_mode="时间序列对比",
            start_dt=None,
            end_dt=None,
            reporter=reporter,
        )
        parsed_a = compare_payload["parsed_a"]
        parsed_b = compare_payload["parsed_b"]
        label_a = str(compare_payload["label_a"])
        label_b = str(compare_payload["label_b"])
        selection_meta = dict(compare_payload["selection_meta"])

        actual_start, actual_end = self.resolve_compare_time_range_for_strategy(
            strategy,
            start_raw,
            end_raw,
            selection_meta.get("txt_summary"),
            selection_meta.get("dat_summary"),
        )

        if reporter is not None:
            reporter("正在时间对齐并生成对比汇总表…")

        target_context = self.resolve_target_spectral_context_for_parsed(
            parsed_a,
            parsed_b,
            spectrum_mode=self.compare_mode_var.get().strip() or None,
            comparison_is_target_spectral=True,
        )
        a_columns = self.build_comparison_column_candidates(parsed_a, a_single, a_multi)
        b_columns = self.build_comparison_column_candidates(parsed_b, b_single, b_multi)
        if target_context is not None:
            canonical_target = str(target_context.get("ygas_target_column") or target_context.get("target_column") or "")
            canonical_reference = str(target_context["reference_column"])
            canonical_dat_target = str(target_context.get("dat_target_column") or "")
            a_columns = [canonical_target, *[column for column in a_columns if column != canonical_target]]
            b_ordered = []
            if canonical_dat_target:
                b_ordered.append(canonical_dat_target)
            b_ordered.append(canonical_reference)
            b_columns = b_ordered + [column for column in b_columns if column not in b_ordered]
        comparison_frame, comparison_metadata = self.build_comparison_dataframe(
            parsed_a,
            parsed_b,
            a_columns,
            b_columns,
            actual_start,
            actual_end,
            alignment_strategy,
            tolerance_seconds,
            target_context=target_context,
        )
        comparison_metadata["label_a"] = label_a
        comparison_metadata["label_b"] = label_b
        comparison_metadata["txt_summary"] = selection_meta.get("txt_summary")
        comparison_metadata["dat_summary"] = selection_meta.get("dat_summary")
        return {
            "comparison_frame": comparison_frame,
            "comparison_metadata": comparison_metadata,
            "parsed_a": parsed_a,
            "parsed_b": parsed_b,
            "label_a": label_a,
            "label_b": label_b,
        }

    def generate_comparison_summary(self) -> None:
        ygas_paths, dat_path, _other = self.classify_selected_compare_files()
        try:
            if dat_path is None:
                raise ValueError("请先选择 1 个 dat 文件，再生成对比汇总表。")
            if not ygas_paths:
                raise ValueError("请先选择 1 个或多个 txt/log 文件作为设备A。")

            selected_paths = self.get_selected_file_paths()
            strategy = self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围"
            alignment_strategy = self.get_alignment_strategy("散点一致性对比")
            tolerance_seconds = self.get_match_tolerance_seconds(default_value=0.2)
            a_single = self.device_a_column_var.get().strip()
            b_single = self.device_b_column_var.get().strip()
            a_multi = self.get_listbox_selected_values(self.device_a_multi_listbox)
            b_multi = self.get_listbox_selected_values(self.device_b_multi_listbox)

            self.start_background_task(
                status_text="正在生成对比汇总表…",
                worker=lambda reporter: self.prepare_comparison_summary_payload(
                    ygas_paths=ygas_paths,
                    dat_path=dat_path,
                    selected_paths=selected_paths,
                    strategy=strategy,
                    start_raw=self.time_start_var.get().strip(),
                    end_raw=self.time_end_var.get().strip(),
                    alignment_strategy=alignment_strategy,
                    tolerance_seconds=tolerance_seconds,
                    a_single=a_single,
                    b_single=b_single,
                    a_multi=a_multi,
                    b_multi=b_multi,
                    reporter=reporter,
                ),
                on_success=self.on_comparison_summary_ready,
                error_title="生成对比汇总表",
            )
        except Exception as exc:
            message = str(exc)
            self.status_var.set(f"生成对比汇总表失败：{message}")
            self.update_diagnostic_info(layout="对比汇总表")
            self.render_plot_message(message, level="warning")
            messagebox.showerror("生成对比汇总表失败", message)

    def on_comparison_summary_ready(self, payload: dict[str, Any]) -> None:
        comparison_frame = payload["comparison_frame"].copy()
        comparison_metadata = dict(payload["comparison_metadata"])
        parsed_a = payload["parsed_a"]
        parsed_b = payload["parsed_b"]
        label_a = str(payload["label_a"])
        label_b = str(payload["label_b"])

        self.update_compare_column_values(self.device_a_combo, self.device_a_column_var, parsed_a.suggested_columns)
        self.update_compare_column_values(self.device_b_combo, self.device_b_column_var, parsed_b.suggested_columns)
        hit_a, hit_b, mapping_name = self.apply_element_mapping(parsed_a, parsed_b)
        self.compare_file_info_var.set(
            f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 映射={mapping_name}"
        )
        if self.mapping_mode_var.get() == "预设映射" and (hit_a is None or hit_b is None):
            self.compare_file_info_var.set(
                f"设备A={label_a}（{parsed_a.profile_name}） | 设备B={label_b}（{parsed_b.profile_name}） | 预设“{mapping_name}”未完全命中"
            )
        self.update_compare_summaries(
            comparison_metadata.get("txt_summary"),
            comparison_metadata.get("dat_summary"),
        )

        comparison_metadata["mapping_name"] = mapping_name
        self.current_comparison_frame = comparison_frame
        self.current_comparison_metadata = comparison_metadata
        self.preview_data = comparison_frame
        self.current_data_source_kind = "comparison_preview"
        self.current_file_parsed = None
        self.current_data_source_label = "对比汇总表预览"
        self.page_index = 0
        self.current_layout_label = "对比汇总表"
        self.update_table_view()
        self.show_table_tab()

        extra_items = [
            f"设备A列数={len(comparison_metadata.get('a_columns', []))}",
            f"设备B列数={len(comparison_metadata.get('b_columns', []))}",
            f"当前要素映射={mapping_name}",
        ]
        target_context = core.get_target_spectral_context(comparison_metadata)
        if target_context is not None:
            extra_items.extend(
                [
                    f"reference_column={target_context.get('reference_column')}",
                    f"ygas_target_column={target_context.get('ygas_target_column', target_context.get('target_column'))}",
                    f"dat_target_column={target_context.get('dat_target_column')}",
                    "canonical_cross_pairs="
                    + " | ".join(str(item) for item in (target_context.get("canonical_cross_pairs") or [])),
                ]
            )
        self.update_diagnostic_info(
            layout="对比汇总表",
            extra_items=extra_items,
        )
        self.status_var.set(
            f"已生成对比汇总表：{len(comparison_frame)} 行，{len(comparison_frame.columns)} 列"
        )

    def use_comparison_summary_for_analysis(self) -> None:
        if self.current_comparison_frame is None or self.current_comparison_frame.empty:
            messagebox.showwarning("警告", "当前没有可用的对比汇总表，请先点击“生成对比汇总表”。")
            return

        comparison_frame = self.current_comparison_frame.copy()
        excluded_columns = {"时间戳", "B_匹配时间戳", "匹配时间差_s"}
        target_context = core.get_target_spectral_context(self.current_comparison_metadata)
        self.raw_data = comparison_frame
        self.preview_data = None
        self.current_data_source_kind = "comparison_analysis"
        self.current_file_parsed = None
        self.current_data_source_label = "对比汇总表"
        self.current_layout_label = "对比汇总表"
        self.current_used_mode1_template = False
        self.excluded_analysis_cols = excluded_columns
        self.column_data, self.non_numeric_cols, self.unsuitable_spectrum_cols = self.classify_comparison_analysis_frame(
            comparison_frame,
            metadata=self.current_comparison_metadata,
            excluded_columns=excluded_columns,
        )
        self.current_plot_kind = None
        self.current_plot_columns = []
        self.current_result_freq = None
        self.current_result_values = None
        self.current_result_frame = None
        self.page_index = 0
        self.create_column_selector()
        self.update_table_view()
        if self.left_workflow_notebook is not None and self.single_analysis_tab is not None:
            self.left_workflow_notebook.select(self.single_analysis_tab)
        self.show_table_tab()
        self.update_diagnostic_info(layout="对比汇总表")
        self.status_var.set(
            f"已将对比汇总表载入分析区：{len(comparison_frame)} 行，{len(comparison_frame.columns)} 列 | 下一步请勾选 1 列或 2 列进行分析"
        )
        if target_context is not None:
            self.status_var.set(
                "已载入目标频谱上下文汇总表：canonical pairs = "
                + "；".join(str(item) for item in (target_context.get("canonical_cross_pairs") or []))
            )

    def export_comparison_summary(self) -> None:
        if self.current_comparison_frame is None or self.current_comparison_frame.empty:
            messagebox.showwarning("警告", "当前没有可导出的对比汇总表。")
            return

        path = filedialog.asksaveasfilename(
            title="导出汇总表",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialfile=f"{self.build_comparison_base_name()}.csv",
        )
        if not path:
            return

        self.current_comparison_frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.status_var.set(f"对比汇总表已导出到：{path}")

    def resolve_target_element_name(self) -> str:
        element_name = self.element_preset_var.get().strip() or "CO2"
        if element_name not in ELEMENT_PRESETS:
            return "CO2"
        return element_name

    def select_target_element_column(
        self,
        parsed: ParsedFileResult,
        *,
        device_key: str,
        target_element: str,
    ) -> str | None:
        return core.select_target_element_column(
            parsed,
            device_key=device_key,
            target_element=target_element,
        )
        preset = ELEMENT_PRESETS.get(target_element, {})
        candidates = list(preset.get(device_key, []))
        for columns in (parsed.suggested_columns, parsed.available_columns):
            match = self.select_first_matching_column(columns, candidates)
            if match:
                return match
        return None

    def build_target_source_hint(self, path: Path, default: str) -> str:
        return core.build_target_source_hint(path, default)

    def build_legacy_group_labels(
        self,
        *,
        target_element: str,
        window_start: pd.Timestamp,
        ygas_path: Path,
    ) -> dict[str, str]:
        return core.build_legacy_group_labels(
            target_element=target_element,
            window_start=window_start,
            ygas_path=ygas_path,
        )

    def get_legacy_target_series_style(
        self,
        group_index: int,
        device_kind: str,
        *,
        spectrum_mode: str = core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
    ) -> dict[str, Any]:
        marker = LEGACY_TARGET_MARKERS[group_index % len(LEGACY_TARGET_MARKERS)]
        line_style = LEGACY_TARGET_LINESTYLES[group_index % len(LEGACY_TARGET_LINESTYLES)]
        color_mode = self.legacy_target_color_mode_var.get().strip() or LEGACY_TARGET_COLOR_MODE_BY_DEVICE
        custom_cross_color = None if spectrum_mode == core.LEGACY_TARGET_SPECTRUM_MODE_PSD else self.resolve_target_cross_series_color(device_kind, group_index)
        if spectrum_mode == core.LEGACY_TARGET_SPECTRUM_MODE_PSD and color_mode == LEGACY_TARGET_COLOR_MODE_BY_DEVICE:
            color = LEGACY_TARGET_DEVICE_COLORS["dat" if device_kind == "dat" else "ygas"]
            return {
                "color": color,
                "alpha": 0.84 if device_kind == "dat" else 0.9,
                "size": 20 if device_kind == "dat" else 24,
                "zorder": 2 if device_kind == "dat" else 3,
                "marker": marker,
                "linestyle": line_style,
            }

        if custom_cross_color is not None:
            return {
                "color": custom_cross_color,
                "alpha": 0.78 if device_kind == "cross_dat" else 0.86,
                "size": 20 if device_kind == "cross_dat" else 24,
                "zorder": 2 if device_kind == "cross_dat" else 3,
                "marker": marker,
                "linestyle": line_style,
            }

        pair = LEGACY_TARGET_COLOR_PAIRS[group_index % len(LEGACY_TARGET_COLOR_PAIRS)]
        if device_kind in {"dat", "cross_dat"}:
            return {
                "color": pair["dat"],
                "alpha": 0.78,
                "size": 20,
                "zorder": 2,
                "marker": marker,
                "linestyle": line_style,
            }
        if device_kind in {"cross", "cross_ygas"}:
            return {
                "color": pair["ygas"],
                "alpha": 0.86,
                "size": 24,
                "zorder": 3,
                "marker": marker,
                "linestyle": line_style,
            }
        return {
            "color": pair["ygas"],
            "alpha": 0.88,
            "size": 24,
            "zorder": 3,
            "marker": marker,
            "linestyle": line_style,
        }

    def build_target_window_label(
        self,
        *,
        device_kind: str,
        target_element: str,
        window_start: pd.Timestamp,
        source_path: Path,
    ) -> str:
        labels = self.build_legacy_group_labels(
            target_element=target_element,
            window_start=window_start,
            ygas_path=source_path,
        )
        return labels["ygas_label"] if device_kind == "ygas" else labels["dat_label"]

    def build_target_window_series(
        self,
        parsed: ParsedFileResult,
        value_column: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame, meta = core.build_target_window_series(parsed, value_column, start_dt, end_dt)
        return frame.rename(columns={core.TIMESTAMP_COL: "时间戳", "value": "数值"}), meta

    def build_target_group_key(self, source_path: Path, window_start: pd.Timestamp) -> str:
        return core.build_target_group_key(source_path, window_start)

    def normalize_hex_color(self, value: str) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if not text.startswith("#") and len(text) == 6:
            text = f"#{text}"
        if HEX_COLOR_PATTERN.fullmatch(text):
            return text.lower()
        return None

    def choose_target_cross_color(self, device_kind: str) -> None:
        if device_kind == "cross_dat":
            variable = self.target_cross_dat_color_var
            title = "选择 dat 协谱颜色"
            fallback = LEGACY_TARGET_COLOR_PAIRS[0]["dat"]
        else:
            variable = self.target_cross_ygas_color_var
            title = "选择 ygas 协谱颜色"
            fallback = LEGACY_TARGET_COLOR_PAIRS[0]["ygas"]
        current = self.normalize_hex_color(variable.get()) or fallback
        _rgb, hex_color = colorchooser.askcolor(color=current, title=title)
        if hex_color:
            variable.set(str(hex_color).lower())

    def resolve_target_cross_series_color(self, device_kind: str, group_index: int) -> str | None:
        if device_kind in {"cross", "cross_ygas"}:
            custom = self.normalize_hex_color(self.target_cross_ygas_color_var.get())
            if custom:
                return custom
            return LEGACY_TARGET_COLOR_PAIRS[group_index % len(LEGACY_TARGET_COLOR_PAIRS)]["ygas"]
        if device_kind == "cross_dat":
            custom = self.normalize_hex_color(self.target_cross_dat_color_var.get())
            if custom:
                return custom
            return LEGACY_TARGET_COLOR_PAIRS[group_index % len(LEGACY_TARGET_COLOR_PAIRS)]["dat"]
        return None

    def evaluate_legacy_target_group_quality(
        self,
        group_payload: dict[str, Any],
        *,
        device_reference: dict[str, dict[str, float | None]],
        requested_nsegment: int,
        forced_include_keys: set[str],
    ) -> dict[str, Any]:
        return core.evaluate_legacy_target_group_quality(
            group_payload,
            device_reference=device_reference,
            requested_nsegment=requested_nsegment,
            forced_include_keys=forced_include_keys,
        )

    def evaluate_target_group_quality(
        self,
        group_payload: dict[str, Any],
        *,
        device_reference: dict[str, dict[str, float | None]],
        requested_nsegment: int,
        forced_include_keys: set[str],
    ) -> dict[str, Any]:
        return self.evaluate_legacy_target_group_quality(
            group_payload,
            device_reference=device_reference,
            requested_nsegment=requested_nsegment,
            forced_include_keys=forced_include_keys,
        )

    def build_legacy_target_group_preview_frame(self, group_records: list[dict[str, Any]]) -> pd.DataFrame:
        return core.build_legacy_target_group_preview_frame(group_records)

    def build_target_group_preview_frame(self, group_records: list[dict[str, Any]]) -> pd.DataFrame:
        return self.build_legacy_target_group_preview_frame(group_records)

    def build_target_group_qc_export_frame(self, group_records: list[dict[str, Any]]) -> pd.DataFrame:
        return core.build_legacy_target_group_qc_export_frame(group_records)

    def prepare_legacy_target_payload_core(
        self,
        *,
        ygas_paths: list[Path],
        dat_path: Path | None,
        fs_ui: float,
        requested_nsegment: int,
        overlap_ratio: float,
        time_range_strategy: str,
        start_raw: str,
        end_raw: str,
        grouping_mode: str,
        legacy_target_spectrum_mode: str = core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
        use_requested_nsegment: bool = False,
        forced_include_group_keys: set[str] | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        target_element = self.resolve_target_element_name()
        ygas_required_columns = self.build_target_required_columns(
            target_element=target_element,
            device_kind="ygas",
        )
        dat_required_columns = self.build_target_required_columns(
            target_element=target_element,
            device_kind="dat",
            include_reference=legacy_target_spectrum_mode != core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
        )

        def _resolve_fs(
            parsed: ParsedFileResult,
            frame: pd.DataFrame,
            ui_fs: float,
            device_kind: str,
        ) -> float:
            manual_override = not math.isclose(ui_fs, DEFAULT_FS, rel_tol=0.0, abs_tol=1e-9)
            if manual_override:
                return float(ui_fs)
            estimated = core.estimate_fs_from_timestamp(frame, core.TIMESTAMP_COL)
            if estimated:
                return float(estimated)
            if device_kind == "dat":
                return float(self.resolve_compare_fs(parsed, ui_fs))
            return float(DEFAULT_FS)

        return core.prepare_legacy_target_payload(
            ygas_paths=ygas_paths,
            dat_path=dat_path,
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            target_element=target_element,
            time_range_strategy=time_range_strategy,
            start_raw=start_raw,
            end_raw=end_raw,
            grouping_mode=grouping_mode,
            legacy_target_spectrum_mode=legacy_target_spectrum_mode,
            parse_ygas=lambda path: self.parse_target_profiled_file(
                path,
                device_kind="ygas",
                required_columns=ygas_required_columns,
            ),
            parse_dat=lambda path: self.parse_target_profiled_file(
                path,
                device_kind="dat",
                required_columns=dat_required_columns,
            ),
            resolve_time_range=self.resolve_target_time_range_for_strategy,
            resolve_fs=_resolve_fs,
            use_requested_nsegment=use_requested_nsegment,
            forced_include_group_keys=forced_include_group_keys,
            reporter=reporter,
        )

    def prepare_legacy_target_payload(
        self,
        *,
        ygas_paths: list[Path],
        dat_path: Path | None,
        fs_ui: float,
        requested_nsegment: int,
        overlap_ratio: float,
        time_range_strategy: str,
        start_raw: str,
        end_raw: str,
        grouping_mode: str,
        legacy_target_spectrum_mode: str = core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
        use_requested_nsegment: bool = False,
        forced_include_group_keys: set[str] | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        return self.prepare_legacy_target_payload_core(
            ygas_paths=ygas_paths,
            dat_path=dat_path,
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            time_range_strategy=time_range_strategy,
            start_raw=start_raw,
            end_raw=end_raw,
            grouping_mode=grouping_mode,
            legacy_target_spectrum_mode=legacy_target_spectrum_mode,
            use_requested_nsegment=use_requested_nsegment,
            forced_include_group_keys=forced_include_group_keys,
            reporter=reporter,
        )

    def prepare_target_spectrum_payload(
        self,
        *,
        ygas_paths: list[Path],
        dat_path: Path | None,
        fs_ui: float,
        requested_nsegment: int,
        overlap_ratio: float,
        time_range_strategy: str,
        start_raw: str,
        end_raw: str,
        grouping_mode: str,
        legacy_target_spectrum_mode: str = core.LEGACY_TARGET_SPECTRUM_MODE_PSD,
        use_requested_nsegment: bool = False,
        forced_include_group_keys: set[str] | None = None,
        reporter: Any | None = None,
    ) -> dict[str, Any]:
        return self.prepare_legacy_target_payload_core(
            ygas_paths=ygas_paths,
            dat_path=dat_path,
            fs_ui=fs_ui,
            requested_nsegment=requested_nsegment,
            overlap_ratio=overlap_ratio,
            time_range_strategy=time_range_strategy,
            start_raw=start_raw,
            end_raw=end_raw,
            grouping_mode=grouping_mode,
            legacy_target_spectrum_mode=legacy_target_spectrum_mode,
            use_requested_nsegment=use_requested_nsegment,
            forced_include_group_keys=forced_include_group_keys,
            reporter=reporter,
        )

    def generate_legacy_compatible_target_plot(self) -> None:
        ygas_paths, dat_path, other_paths = self.classify_selected_compare_files()
        try:
            self.ensure_legacy_target_selection_valid(ygas_paths, dat_path, other_paths)

            fs_ui, requested_nsegment, overlap_ratio = self.get_analysis_params()
            use_requested_nsegment = bool(self.legacy_target_use_analysis_params_var.get())
            legacy_target_spectrum_mode = (
                self.legacy_target_spectrum_mode_var.get().strip() or core.LEGACY_TARGET_SPECTRUM_MODE_PSD
            )
            forced_include_group_keys = self.get_selected_target_group_overrides()
            time_range_strategy = self.time_range_strategy_var.get().strip() or "使用 txt+dat 共同时间范围"
            grouping_mode = "每个 ygas 文件视为一个组"

            self.current_target_plot_metadata = {}
            self.current_target_group_preview_frame = None
            self.current_result_frame = None
            self.current_result_freq = None
            self.current_result_values = None
            self.current_compare_files = []
            self.current_plot_kind = None
            self.current_plot_columns = []
            self.update_target_group_qc_panel(None)
            self.reset_plot_output_state()
            self.render_plot_message("正在生成目标谱图...", level="info")

            self.start_background_task(
                status_text="正在生成目标谱图…",
                worker=lambda reporter: self.prepare_legacy_target_payload_core(
                    ygas_paths=ygas_paths,
                    dat_path=dat_path,
                    fs_ui=fs_ui,
                    requested_nsegment=requested_nsegment,
                    overlap_ratio=overlap_ratio,
                    time_range_strategy=time_range_strategy,
                    start_raw=self.time_start_var.get().strip(),
                    end_raw=self.time_end_var.get().strip(),
                    grouping_mode=grouping_mode,
                    legacy_target_spectrum_mode=legacy_target_spectrum_mode,
                    use_requested_nsegment=use_requested_nsegment,
                    forced_include_group_keys=forced_include_group_keys,
                    reporter=reporter,
                ),
                on_success=self.on_legacy_target_plot_ready,
                error_title="目标谱图",
            )
        except Exception as exc:
            message = str(exc)
            self.status_var.set(f"目标谱图生成失败：{message}")
            self.update_diagnostic_info(layout="目标谱图")
            self.render_plot_message(message, level="warning")
            messagebox.showerror("目标谱图生成失败", message)

    def generate_target_spectrum_plot(self) -> None:
        self.generate_legacy_compatible_target_plot()

    def on_legacy_target_plot_ready(self, payload: dict[str, Any]) -> None:
        self.on_target_spectrum_ready(payload)

    def on_target_spectrum_ready(self, payload: dict[str, Any]) -> None:
        payload = self.ensure_target_spectrum_dispatch_payload(payload)
        target_metadata = dict(payload["target_metadata"])
        self.current_data_source_kind = "legacy_target_spectrum"
        self.current_file_parsed = None
        self.current_data_source_label = str(target_metadata.get("mode_label", "目标谱图"))
        self.current_target_plot_metadata = target_metadata
        preview_frame = target_metadata.get("group_preview_frame")
        self.current_target_group_preview_frame = preview_frame.copy() if isinstance(preview_frame, pd.DataFrame) else None
        self.update_target_group_qc_panel(self.current_target_plot_metadata)
        self.plot_legacy_compatible_target_results(
            str(payload["target_element"]),
            list(payload["series_results"]),
            list(target_metadata.get("skipped_windows", [])),
        )

    def plot_legacy_compatible_target_results(
        self,
        target_element: str,
        series_results: list[dict[str, Any]],
        skipped_windows: list[str],
    ) -> None:
        self.plot_target_spectrum_results(target_element, series_results, skipped_windows)

    def plot_target_spectrum_results(
        self,
        target_element: str,
        series_results: list[dict[str, Any]],
        skipped_windows: list[str],
    ) -> None:
        target_metadata = dict(self.current_target_plot_metadata)
        spectrum_mode = str(target_metadata.get("spectrum_mode", core.LEGACY_TARGET_SPECTRUM_MODE_PSD))
        spectrum_type = core.resolve_legacy_target_cross_spectrum_type(spectrum_mode)
        is_psd_mode = spectrum_type is None
        if not series_results:
            raise ValueError(f"没有可用于绘制“{target_element}”目标谱图的有效系列。")
        display_semantics = (
            None
            if is_psd_mode
            else str(target_metadata.get("display_semantics") or series_results[0].get("details", {}).get("display_semantics") or "")
        )
        visible_series_scope = self.resolve_target_spectrum_visible_series_scope(
            target_metadata,
            is_psd_mode=is_psd_mode,
        )

        if is_psd_mode:
            series_results = sorted(
                series_results,
                key=lambda item: (0 if str(item.get("device_kind")) == "ygas" else 1, int(item.get("group_index", 0))),
            )
            plot_style_context = "时间段内 PSD 对比"
        else:
            role_order = {"cross_ygas": 0, "cross_dat": 1, "cross": 2}
            series_results = sorted(
                series_results,
                key=lambda item: (
                    int(item.get("group_index", 0)),
                    role_order.get(str(item.get("device_kind")), 99),
                ),
            )
            if spectrum_type == CROSS_SPECTRUM_REAL:
                plot_style_context = "协谱图"
            elif spectrum_type == CROSS_SPECTRUM_IMAG:
                plot_style_context = "正交谱图"
            else:
                plot_style_context = "互谱幅值"
        plot_style = self.resolve_plot_style(plot_style_context)
        color_mode = self.legacy_target_color_mode_var.get().strip() or LEGACY_TARGET_COLOR_MODE_BY_DEVICE

        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xscale("log")

        visible_series_results: list[dict[str, Any]] = []
        combined_freq: list[np.ndarray] = []
        combined_values: list[np.ndarray] = []
        reference_freq: list[np.ndarray] = []
        reference_values: list[np.ndarray] = []
        point_summaries: list[str] = []
        aligned_frames: list[pd.DataFrame] = []

        for index, item in enumerate(series_results):
            group_index = int(item.get("group_index", index))
            device_kind = str(item.get("device_kind", "ygas"))
            style = self.get_legacy_target_series_style(
                group_index,
                device_kind,
                spectrum_mode=spectrum_mode,
            )
            y_values = np.asarray(item.get("density", item.get("values")), dtype=float)
            freq_values = np.asarray(item["freq"], dtype=float)
            mask_type = CROSS_SPECTRUM_MAGNITUDE if is_psd_mode else spectrum_type
            item_display_semantics = None if is_psd_mode else str(item.get("details", {}).get("display_semantics") or display_semantics or "")
            mask = self.build_spectrum_plot_mask(
                freq_values,
                y_values,
                mask_type,
                display_semantics=item_display_semantics or None,
            )
            freq_plot = freq_values[mask]
            values_plot = y_values[mask]
            if len(freq_plot) == 0:
                continue
            reference_freq.append(freq_plot)
            reference_values.append(values_plot)
            if not self.is_target_spectrum_series_visible(item, visible_series_scope=visible_series_scope):
                continue

            self.apply_series_style(
                ax,
                freq_plot,
                values_plot,
                color=str(style["color"]),
                label=str(item["label"]),
                style=plot_style,
                linewidth=1.5,
                marker_size=int(style["size"]),
                alpha=float(style["alpha"]),
                marker=str(style["marker"]),
                line_style=str(style["linestyle"]),
                zorder=float(style["zorder"]),
            )
            visible_item = dict(item)
            visible_details = dict(visible_item.get("details", {}))
            visible_details["target_spectrum_visible_series_scope"] = visible_series_scope
            visible_details["target_spectrum_visible_series_count"] = 0
            visible_item["details"] = visible_details
            visible_series_results.append(visible_item)
            combined_freq.append(freq_plot)
            combined_values.append(values_plot)
            if is_psd_mode:
                point_summaries.append(
                    f"{item['label']}={item['valid_points']}点/{item['valid_freq_points']}频点"
                )
            else:
                matched_count = item.get("details", {}).get("matched_count", item.get("valid_points"))
                point_summaries.append(
                    f"{item['label']}=matched {matched_count}/{item['valid_freq_points']}频点"
                )
                aligned_frame = item.get("aligned_frame")
                if isinstance(aligned_frame, pd.DataFrame) and not aligned_frame.empty:
                    aligned_frames.append(aligned_frame)

        if not combined_freq:
            raise ValueError(f"没有可用于绘制“{target_element}”目标谱图的有效频点。")

        all_freq = np.concatenate(combined_freq)
        all_values = np.concatenate(combined_values)
        reference_all_freq = np.concatenate(reference_freq) if reference_freq else all_freq
        reference_all_values = np.concatenate(reference_values) if reference_values else all_values
        if is_psd_mode or display_semantics == core.CROSS_DISPLAY_SEMANTICS_ABS or spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
            ax.set_yscale("log")
        else:
            max_abs = float(np.nanmax(np.abs(reference_all_values))) if len(reference_all_values) else 0.0
            linthresh = max(max_abs * 1e-3, 1e-12)
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

        if is_psd_mode:
            self.add_selected_reference_slopes(
                ax,
                reference_all_freq,
                reference_all_values,
                is_psd=True,
                use_fixed_power_law=True,
                amplitude_at_1hz=1.0,
            )
            title = build_legacy_target_plot_title(target_element)
            ylabel = "Density"
        else:
            display_meta = self.get_cross_spectrum_display_meta(spectrum_type)
            reference_label = str(target_metadata.get("cross_reference_column") or "Uz")
            title = f"{display_meta['title_prefix']}：{reference_label} vs {target_element}"
            ylabel = display_meta["ylabel"]
            self.add_selected_reference_slopes(
                ax,
                reference_all_freq,
                reference_all_values,
                is_psd=False,
                spectrum_type=spectrum_type,
            )

        if visible_series_scope != "all_series" and is_psd_mode:
            x_limits = self.compute_positive_log_axis_limits(reference_all_freq)
            y_limits = self.compute_positive_log_axis_limits(reference_all_values)
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper right")
        self.figure.tight_layout()
        self.register_plot_style_layout(plot_style, "目标谱图叠加")
        self.refresh_canvas(title=title)

        self.current_plot_kind = "target_spectrum"
        if is_psd_mode:
            self.current_plot_columns = [target_element]
        else:
            self.current_plot_columns = [
                str(target_metadata.get("reference_column") or target_metadata.get("cross_reference_column") or "Uz"),
                str(target_element),
            ]
        self.current_result_freq = None
        self.current_result_values = None
        for visible_item in visible_series_results:
            visible_details = dict(visible_item.get("details", {}))
            visible_details["target_spectrum_visible_series_count"] = int(len(visible_series_results))
            visible_item["details"] = visible_details
        self.current_result_frame = self.build_overlay_export_frame(visible_series_results)
        self.current_aligned_frame = None
        if aligned_frames:
            self.current_target_plot_metadata["aligned_frames"] = aligned_frames
        else:
            self.current_target_plot_metadata.pop("aligned_frames", None)
        self.current_target_plot_metadata["target_spectrum_visible_series_scope"] = visible_series_scope
        self.current_target_plot_metadata["target_spectrum_visible_series_count"] = int(len(visible_series_results))
        self.current_aligned_metadata = {
            "spectrum_mode": spectrum_mode,
            "alignment_strategy": target_metadata.get("alignment_strategy"),
        }
        if display_semantics:
            self.current_aligned_metadata["display_semantics"] = display_semantics
        if not is_psd_mode:
            self.current_aligned_metadata.update(
                {
                    "reference_column": str(target_metadata.get("reference_column") or target_metadata.get("cross_reference_column") or "Uz"),
                    "ygas_target_column": str(target_metadata.get("ygas_target_column") or ""),
                    "dat_target_column": str(target_metadata.get("dat_target_column") or ""),
                    "canonical_cross_pairs": " | ".join(str(item) for item in (target_metadata.get("canonical_cross_pairs") or [])),
                    "generated_cross_series_count": int(target_metadata.get("generated_cross_series_count", len(series_results))),
                    "generated_cross_series_roles": " | ".join(
                        str(item)
                        for item in (target_metadata.get("generated_cross_series_roles") or [])
                        if str(item).strip()
                    ),
                }
            )
        self.current_compare_files = [str(item["label"]) for item in visible_series_results]

        first_details = dict(visible_series_results[0]["details"])
        mode_label = str(target_metadata.get("mode_label", "目标谱图"))
        group_records = list(target_metadata.get("group_records", []))
        kept_group_count = int(
            target_metadata.get(
                "kept_group_count",
                sum(1 for record in group_records if record.get("status") in {"保留", "手动保留"}),
            )
        )
        skipped_group_count = int(
            target_metadata.get(
                "skipped_group_count",
                sum(1 for record in group_records if record.get("status") == "跳过"),
            )
        )
        total_group_count = int(target_metadata.get("total_group_count", len(group_records)))
        expected_series_count = int(target_metadata.get("expected_series_count", len(series_results)))
        actual_series_count = int(target_metadata.get("actual_series_count", len(series_results)))
        visible_series_count = int(len(visible_series_results))
        requested_nsegment_ui = int(
            target_metadata.get(
                "requested_nsegment",
                first_details.get("requested_nsegment", first_details.get("nsegment", 0)),
            )
        )
        target_context_mode = str(target_metadata.get("target_spectrum_context_mode") or "").strip()
        target_context_dat_file_name = str(target_metadata.get("target_spectrum_context_dat_file_name") or "").strip()
        if self.current_result_frame is not None and not self.current_result_frame.empty:
            self.current_result_frame = self.attach_metadata_columns(
                self.current_result_frame,
                {
                    "current_plot_kind": "target_spectrum",
                    "plot_execution_path": "target_spectrum_render",
                    "render_semantics": str(target_metadata.get("render_semantics") or first_details.get("render_semantics") or ""),
                    "target_spectrum_group_count": kept_group_count,
                    "target_spectrum_visible_series_scope": visible_series_scope,
                    "target_spectrum_visible_series_count": visible_series_count,
                    "target_spectrum_context_mode": target_context_mode or None,
                    "target_spectrum_context_dat_file_name": target_context_dat_file_name or None,
                },
            )
        effective_nsegment_values = [
            int(value)
            for value in target_metadata.get("effective_nsegment_values", [])
            if value is not None
        ]
        positive_freq_points_min = target_metadata.get("positive_freq_points_min")
        positive_freq_points_max = target_metadata.get("positive_freq_points_max")
        first_positive_freq_min = target_metadata.get("first_positive_freq_min")
        first_positive_freq_max = target_metadata.get("first_positive_freq_max")
        kept_group_summaries: list[str] = []
        skipped_group_summaries = [
            f"{record.get('group_label', record.get('group_key', record.get('window_label', '未知组')))}({record.get('reason', '未知原因')})"
            for record in group_records
            if record.get("status") == "跳过"
        ]

        for record in group_records:
            group_name = record.get("group_label", record.get("group_key", record.get("window_label", "未知组")))
            if record.get("status") not in {"保留", "手动保留"}:
                continue
            if is_psd_mode:
                kept_group_summaries.append(
                    f"{group_name}: "
                    f"ygas={record.get('ygas_effective_nsegment')}/{record.get('ygas_positive_freq_points')}频点, "
                    f"dat={record.get('dat_effective_nsegment')}/{record.get('dat_positive_freq_points')}频点"
                )
            else:
                kept_group_summaries.append(
                    f"{group_name}: "
                    f"matched={record.get('matched_count')} "
                    f"effective_nsegment={record.get('cross_effective_nsegment')} "
                    f"positive_freq_points={record.get('cross_positive_freq_points')}"
                )

        extra_items = [
            f"当前模式={mode_label}",
            f"spectrum_mode={spectrum_mode}",
            f"plot_style={plot_style}",
            f"color_mode={color_mode}",
            f"当前目标要素={target_element}",
            *build_device_dispatch_items(first_details, include_plot_execution_path=True),
            f"target_spectrum_group_count={kept_group_count}",
            f"target_spectrum_visible_series_scope={visible_series_scope}",
            f"target_spectrum_visible_series_count={visible_series_count}",
            "current_plot_kind=target_spectrum",
            f"总组数={total_group_count}",
            f"保留组数={kept_group_count}",
            f"跳过组数={skipped_group_count}",
            f"预期系列数={expected_series_count}",
            f"实际系列数={actual_series_count}",
            f"可见系列数={visible_series_count}",
            f"目标谱图沿用当前分析参数={'是' if target_metadata.get('legacy_target_uses_requested_nsegment') else '否'}",
            f"requested_nsegment={requested_nsegment_ui}",
            f"系列有效点数={'；'.join(point_summaries)}",
        ]
        if target_context_mode:
            extra_items.append(f"target_spectrum_context_mode={target_context_mode}")
        if target_context_dat_file_name:
            extra_items.append(f"target_spectrum_context_dat_file_name={target_context_dat_file_name}")
        if effective_nsegment_values:
            extra_items.append(f"effective_nsegment={'/'.join(str(value) for value in effective_nsegment_values)}")
        if positive_freq_points_min is not None and positive_freq_points_max is not None:
            extra_items.append(f"positive_freq_points范围={positive_freq_points_min} ~ {positive_freq_points_max}")
        if first_positive_freq_min is not None and first_positive_freq_max is not None:
            extra_items.append(
                f"first_positive_freq范围={float(first_positive_freq_min):.12g} ~ {float(first_positive_freq_max):.12g} Hz"
            )
        extra_items.extend(
            self.build_reference_slope_diagnostic_items(
                is_psd=is_psd_mode,
                spectrum_type=None if is_psd_mode else spectrum_type,
            )
        )
        if is_psd_mode:
            extra_items.extend(
                [
                    f"psd_kernel={target_metadata.get('legacy_psd_kernel', first_details.get('psd_kernel'))}",
                    f"reference_line_mode={target_metadata.get('reference_line_mode', 'fixed_f_pow_-2_3')}",
                    f"reference_line_at_1hz={target_metadata.get('reference_line_at_1hz', 1.0)}",
                ]
            )
        else:
            matched_count_min = target_metadata.get("matched_count_min")
            matched_count_max = target_metadata.get("matched_count_max")
            tolerance_seconds_min = target_metadata.get("tolerance_seconds_min")
            tolerance_seconds_max = target_metadata.get("tolerance_seconds_max")
            extra_items.extend(
                [
                    f"cross_execution_path={target_metadata.get('cross_execution_path', first_details.get('cross_execution_path', 'target_spectral_canonical'))}",
                    f"cross_implementation_id={target_metadata.get('cross_implementation_id', first_details.get('cross_implementation_id'))}",
                    f"cross_implementation_label={target_metadata.get('cross_implementation_label', first_details.get('cross_implementation_label'))}",
                    f"alignment_strategy={target_metadata.get('alignment_strategy', '最近邻 + 容差')}",
                    f"cross_kernel={target_metadata.get('cross_kernel', first_details.get('cross_kernel'))}",
                    f"cross_reference_column={target_metadata.get('cross_reference_column', first_details.get('cross_reference_column', 'Uz'))}",
                    f"reference_column={target_metadata.get('reference_column', target_metadata.get('cross_reference_column', first_details.get('reference_column', 'Uz')))}",
                    f"ygas_target_column={target_metadata.get('ygas_target_column', first_details.get('ygas_target_column'))}",
                    f"dat_target_column={target_metadata.get('dat_target_column', first_details.get('dat_target_column'))}",
                    "canonical_cross_pairs="
                    + " | ".join(str(item) for item in (target_metadata.get("canonical_cross_pairs") or [])),
                    f"generated_cross_series_count={target_metadata.get('generated_cross_series_count', len(series_results))}",
                    "generated_cross_series_roles="
                    + " | ".join(
                        str(item)
                        for item in (target_metadata.get("generated_cross_series_roles") or [])
                        if str(item).strip()
                    ),
                    f"display_semantics={target_metadata.get('display_semantics', first_details.get('display_semantics'))}",
                    f"display_value_source={target_metadata.get('display_value_source', first_details.get('display_value_source'))}",
                ]
            )
            if spectrum_type in CROSS_SPECTRUM_OPTIONS:
                for key in ("window", "detrend", "scaling", "return_onesided", "average"):
                    value = target_metadata.get(key, first_details.get(key))
                    if value is not None:
                        extra_items.append(f"{key}={value}")
            if matched_count_min is not None and matched_count_max is not None:
                extra_items.append(f"matched_count范围={matched_count_min} ~ {matched_count_max}")
            if tolerance_seconds_min is not None and tolerance_seconds_max is not None:
                extra_items.append(
                    f"tolerance_seconds范围={float(tolerance_seconds_min):.6g} ~ {float(tolerance_seconds_max):.6g}"
                )
        if kept_group_summaries:
            extra_items.append(
                f"保留组摘要={'；'.join(kept_group_summaries[:4])}{' 等' if len(kept_group_summaries) > 4 else ''}"
            )
        if actual_series_count != expected_series_count:
            extra_items.append("某些组已被自动跳过，请查看组级质控列表。")
        if skipped_group_summaries:
            extra_items.append(
                f"跳过原因列表={'；'.join(skipped_group_summaries[:4])}{' 等' if len(skipped_group_summaries) > 4 else ''}"
            )
        elif skipped_windows:
            extra_items.append(
                f"被跳过组={'；'.join(skipped_windows[:4])}{' 等' if len(skipped_windows) > 4 else ''}"
            )

        self.update_diagnostic_info(
            layout=mode_label,
            analysis_label=target_element,
            fs=float(first_details.get("effective_fs", first_details.get("fs", 0.0))),
            nsegment=int(first_details.get("effective_nsegment", first_details["nsegment"])),
            overlap_ratio=float(first_details["overlap_ratio"]),
            nperseg=int(first_details["nperseg"]),
            noverlap=int(first_details["noverlap"]),
            batch_success=kept_group_count,
            batch_skips=skipped_group_count,
            extra_items=extra_items,
        )
        status_text = (
            f"{mode_label}已生成：保留 {kept_group_count}/{total_group_count} 组"
            f" | 输出 {visible_series_count}/{actual_series_count} 条可见系列"
        )
        status_items = build_device_dispatch_items(first_details)
        status_items.append(f"target_spectrum_group_count={kept_group_count}")
        status_items.append(f"target_spectrum_visible_series_scope={visible_series_scope}")
        status_items.append(f"target_spectrum_visible_series_count={visible_series_count}")
        if target_context_mode:
            status_items.append(f"target_spectrum_context_mode={target_context_mode}")
        if target_context_dat_file_name:
            status_items.append(f"target_spectrum_context_dat_file_name={target_context_dat_file_name}")
        status_items.append("current_plot_kind=target_spectrum")
        if status_items:
            status_text += f" | {' | '.join(status_items)}"
        if skipped_group_count:
            status_text += f" | 已自动跳过 {skipped_group_count} 组"
        self.status_var.set(status_text)

    def prepare_compare_series(
        self,
        parsed: ParsedFileResult,
        value_column: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        prefix: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        prepared, meta = core.prepare_base_spectrum_series(
            parsed,
            value_column,
            start_dt=start_dt,
            end_dt=end_dt,
            require_timestamp=True,
        )
        value_name = f"{prefix}_{value_column}"
        prepared = prepared.rename(columns={"value": value_name})

        return prepared, {
            "value_name": value_name,
            "original_count": int(meta.get("original_count", len(parsed.dataframe))),
            "time_filtered_count": int(meta.get("time_filtered_count", len(parsed.dataframe))),
            "valid_count": int(meta.get("valid_points", len(prepared))),
            "source_row_count": int(meta.get("source_row_count", parsed.source_row_count or len(parsed.dataframe))),
            "timestamp_valid_count": int(meta.get("timestamp_valid_count", parsed.timestamp_valid_count)),
            "timestamp_valid_ratio": float(meta.get("timestamp_valid_ratio", parsed.timestamp_valid_ratio)),
            "timestamp_warning": meta.get("timestamp_warning", parsed.timestamp_warning),
            "actual_start": meta.get("actual_start"),
            "actual_end": meta.get("actual_end"),
            "requested_start": meta.get("requested_start"),
            "requested_end": meta.get("requested_end"),
            "coverage_ratio": meta.get("coverage_ratio"),
        }

    def align_compare_frames(
        self,
        parsed_a: ParsedFileResult,
        col_a: str,
        parsed_b: ParsedFileResult,
        col_b: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        tolerance_seconds: float | None = None,
        alignment_strategy: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        series_a, meta_a = self.prepare_compare_series(parsed_a, col_a, start_dt, end_dt, "设备A")
        series_b, meta_b = self.prepare_compare_series(parsed_b, col_b, start_dt, end_dt, "设备B")

        alignment_strategy = alignment_strategy or "最近邻 + 容差"
        if alignment_strategy == "重采样后对齐":
            interval_seconds = self.estimate_resample_interval_seconds(series_a["时间戳"], series_b["时间戳"])
            interval_ms = max(int(round(interval_seconds * 1000.0)), 1)
            rule = f"{interval_ms}ms"
            a_name = meta_a["value_name"]
            b_name = meta_b["value_name"]
            resampled_a = (
                series_a.set_index("时间戳")[[a_name]]
                .sort_index()
                .resample(rule)
                .mean()
                .interpolate(method="time")
                .dropna()
            )
            resampled_b = (
                series_b.set_index("时间戳")[[b_name]]
                .sort_index()
                .resample(rule)
                .mean()
                .interpolate(method="time")
                .dropna()
            )
            aligned = resampled_a.join(resampled_b, how="inner").reset_index()
            tolerance_seconds = interval_seconds
        else:
            if tolerance_seconds is None:
                tolerance_seconds = self.compute_alignment_tolerance_seconds(series_a["时间戳"], series_b["时间戳"])
            merge_kwargs: dict[str, Any] = {
                "on": "时间戳",
                "direction": "nearest",
            }
            if alignment_strategy == "最近邻 + 容差":
                merge_kwargs["tolerance"] = pd.to_timedelta(tolerance_seconds, unit="s")
            aligned = pd.merge_asof(
                series_a.sort_values("时间戳"),
                series_b.sort_values("时间戳"),
                **merge_kwargs,
            ).dropna()
        if len(aligned) < 2:
            raise ValueError("两设备共同时间点不足，无法生成有效对齐结果。")

        return aligned, {
            "series_a": series_a,
            "series_b": series_b,
            "meta_a": meta_a,
            "meta_b": meta_b,
            "matched_count": len(aligned),
            "actual_start": pd.Timestamp(aligned["时间戳"].min()),
            "actual_end": pd.Timestamp(aligned["时间戳"].max()),
            "tolerance_seconds": tolerance_seconds,
            "alignment_strategy": alignment_strategy,
        }

    def run_dual_device_compare(self) -> None:
        ygas_paths, dat_path, _other = self.classify_selected_compare_files()
        try:
            selected_paths = self.get_compare_file_paths()
            if dat_path is None and len(selected_paths) < 2:
                raise ValueError("请先选择两个文件，或选择多个 txt/log 加 1 个 dat 文件。")
            mode = self.compare_mode_var.get().strip() or "时间序列对比"
            start_dt, end_dt = self.resolve_time_range_inputs()
            self.start_background_task(
                status_text="正在解析文件并准备双设备比对…",
                worker=lambda reporter: self.prepare_dual_compare_payload(
                    ygas_paths=ygas_paths,
                    dat_path=dat_path,
                    selected_paths=selected_paths,
                    compare_mode=mode,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    reporter=reporter,
                ),
                on_success=self.on_dual_compare_payload_ready,
                error_title="双设备比对",
            )
        except Exception as exc:
            message = str(exc)
            self.status_var.set(f"双设备比对失败：{message}")
            self.update_diagnostic_info(layout="双设备比对")
            self.render_plot_message(message, level="warning")
            messagebox.showerror("比对失败", message)

    def format_source_label(self, source: str | Path) -> str:
        if isinstance(source, Path):
            return source.name
        return str(source)

    def merge_timestamp_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for frame in frames:
            if merged is None:
                merged = frame.copy()
            else:
                merged = merged.merge(frame, on="时间戳", how="outer")
        if merged is None:
            return pd.DataFrame(columns=["时间戳"])
        return merged.sort_values("时间戳").reset_index(drop=True)

    def render_psd_compare_series(
        self,
        label_a: str,
        label_b: str,
        parsed_a: ParsedFileResult,
        parsed_b: ParsedFileResult,
        series_results: list[dict[str, Any]],
        *,
        mapping_name: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        txt_points: int,
        dat_points: int,
        fs_a_last: float,
        fs_b_last: float,
    ) -> None:
        self.render_psd_series_with_compare_semantics(
            series_results=series_results,
            geometry_reference_series_results=series_results,
            mapping_name=mapping_name,
            start_dt=start_dt,
            end_dt=end_dt,
            txt_points=txt_points,
            dat_points=dat_points,
            fs_a_last=fs_a_last,
            fs_b_last=fs_b_last,
            render_semantics="compare_psd_dual",
            label_a=label_a,
            label_b=label_b,
            parsed_a=parsed_a,
            parsed_b=parsed_b,
        )

    def should_render_single_device_with_compare_psd_semantics(self, payload: dict[str, Any]) -> bool:
        series_results = list(payload.get("series_results", []))
        if len(series_results) != 1:
            return False
        first_details = dict(series_results[0].get("details", {}))
        return str(first_details.get("single_device_execution_path") or "") == "compare_txt_side_equivalent"

    def render_single_device_compare_psd_payload(self, payload: dict[str, Any]) -> None:
        series_results = list(payload.get("series_results", []))
        if not series_results:
            raise ValueError("单设备 compare-style PSD 渲染缺少有效系列。")
        first_details = dict(series_results[0].get("details", {}))
        mapping_name = (
            self.element_preset_var.get().strip()
            if self.mapping_mode_var.get() == "预设映射"
            else str(payload.get("target_column") or first_details.get("base_value_column") or "PSD")
        ) or str(payload.get("target_column") or first_details.get("base_value_column") or "PSD")
        self.render_psd_series_with_compare_semantics(
            series_results=series_results,
            geometry_reference_series_results=list(payload.get("compare_geometry_series_results", [])),
            mapping_name=mapping_name,
            start_dt=payload.get("start_dt"),
            end_dt=payload.get("end_dt"),
            txt_points=int(first_details.get("valid_points", 0) or 0),
            dat_points=int(first_details.get("compare_dat_valid_points", 0) or 0),
            fs_a_last=float(first_details.get("effective_fs") or first_details.get("fs") or DEFAULT_FS),
            fs_b_last=float(first_details.get("compare_dat_effective_fs") or 0.0),
            render_semantics="compare_psd_single_side",
        )

    def normalize_compare_psd_series_results(
        self,
        series_results: list[dict[str, Any]],
        *,
        single_side_compare: bool,
        annotate_single_side_details: bool,
        preserve_original_side: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_series_results: list[dict[str, Any]] = []
        for index, raw_item in enumerate(series_results, start=1):
            item = dict(raw_item)
            details = dict(raw_item.get("details", {}))
            raw_side = str(item.get("side") or details.get("device_source") or details.get("group_device_source") or "")
            side = raw_side or ("txt" if single_side_compare and not preserve_original_side else "")
            column = str(
                item.get("column")
                or details.get("base_value_column")
                or details.get("target_column")
                or f"series_{index}"
            )
            label = str(item.get("label") or column)
            if single_side_compare and annotate_single_side_details:
                source_label = str(details.get("compare_txt_source_label") or label or "txt")
                label = f"{source_label} - {column}" if column else source_label
                side = "txt"
                details["single_device_render_semantics"] = "compare_psd_single_side"
                details["plot_execution_path"] = "single_device_compare_psd_render"
            if not side:
                side = "txt"
            item["side"] = side
            item["column"] = column
            item["label"] = label
            item["details"] = details
            normalized_series_results.append(item)
        return normalized_series_results

    def build_compare_psd_geometry_state(
        self,
        series_results: list[dict[str, Any]],
        *,
        style: str,
        color_offset: int = 0,
    ) -> dict[str, Any] | None:
        if not series_results:
            return None

        geometry_figure = Figure(figsize=(6, 4), dpi=100)
        geometry_axis = geometry_figure.add_subplot(111)
        try:
            for index, item in enumerate(series_results):
                color = SERIES_COLORS[(index + color_offset) % len(SERIES_COLORS)]
                self.apply_series_style(
                    geometry_axis,
                    item["freq"],
                    item["density"],
                    color=color,
                    label=str(item["label"]),
                    style=style,
                )
            reference_lines_start = len(geometry_axis.lines)
            self.add_selected_reference_slopes(
                geometry_axis,
                np.concatenate([np.asarray(item["freq"], dtype=float) for item in series_results]),
                np.concatenate([np.asarray(item["density"], dtype=float) for item in series_results]),
                is_psd=True,
            )
            geometry_axis.set_xscale("log")
            geometry_axis.set_yscale("log")
            geometry_axis.grid(True, which="both", linestyle="--", alpha=0.3)
            geometry_axis.autoscale(enable=True, axis="both", tight=False)
            reference_lines = []
            for line in geometry_axis.lines[reference_lines_start:]:
                reference_lines.append(
                    {
                        "x": np.asarray(line.get_xdata(), dtype=float),
                        "y": np.asarray(line.get_ydata(), dtype=float),
                        "label": str(line.get_label()),
                        "color": str(line.get_color()),
                        "linestyle": str(line.get_linestyle()),
                        "linewidth": float(line.get_linewidth()),
                        "alpha": None if line.get_alpha() is None else float(line.get_alpha()),
                        "zorder": None if line.get_zorder() is None else float(line.get_zorder()),
                    }
                )
            return {
                "xlim": tuple(float(value) for value in geometry_axis.get_xlim()),
                "ylim": tuple(float(value) for value in geometry_axis.get_ylim()),
                "reference_lines": reference_lines,
            }
        finally:
            geometry_figure.clf()

    def apply_compare_psd_geometry_state(self, ax: Any, geometry_state: dict[str, Any] | None) -> None:
        if not geometry_state:
            return
        for line_info in geometry_state.get("reference_lines", []):
            plot_kwargs: dict[str, Any] = {
                "linestyle": str(line_info.get("linestyle") or "--"),
                "linewidth": float(line_info.get("linewidth") or 1.4),
                "color": str(line_info.get("color") or "#d65f5f"),
                "label": str(line_info.get("label") or ""),
            }
            alpha = line_info.get("alpha")
            zorder = line_info.get("zorder")
            if alpha is not None:
                plot_kwargs["alpha"] = float(alpha)
            if zorder is not None:
                plot_kwargs["zorder"] = float(zorder)
            ax.plot(
                np.asarray(line_info.get("x", []), dtype=float),
                np.asarray(line_info.get("y", []), dtype=float),
                **plot_kwargs,
            )
        xlim = geometry_state.get("xlim")
        ylim = geometry_state.get("ylim")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    def render_psd_series_with_compare_semantics(
        self,
        *,
        series_results: list[dict[str, Any]],
        geometry_reference_series_results: list[dict[str, Any]] | None = None,
        mapping_name: str,
        start_dt: pd.Timestamp | None,
        end_dt: pd.Timestamp | None,
        txt_points: int,
        dat_points: int,
        fs_a_last: float,
        fs_b_last: float,
        render_semantics: str,
        label_a: str | None = None,
        label_b: str | None = None,
        parsed_a: ParsedFileResult | None = None,
        parsed_b: ParsedFileResult | None = None,
    ) -> None:
        if not series_results:
            raise ValueError("时间段内 PSD 对比没有得到可用的有效谱值。")

        style = self.resolve_plot_style("时间段内 PSD 对比")
        layout = self.resolve_plot_layout("时间段内 PSD 对比")
        self.reset_plot_output_state()
        single_side_compare = render_semantics == "compare_psd_single_side"
        normalized_series_results = self.normalize_compare_psd_series_results(
            series_results,
            single_side_compare=single_side_compare,
            annotate_single_side_details=single_side_compare,
        )
        geometry_reference_raw = list(geometry_reference_series_results or series_results)
        normalized_geometry_reference_series_results = self.normalize_compare_psd_series_results(
            geometry_reference_raw,
            single_side_compare=single_side_compare,
            annotate_single_side_details=False,
            preserve_original_side=True,
        )

        txt_series = [item for item in normalized_series_results if item["side"] == "txt"]
        dat_series = [item for item in normalized_series_results if item["side"] == "dat"]
        geometry_txt_series = [
            item for item in normalized_geometry_reference_series_results if str(item.get("side")) == "txt"
        ]
        geometry_dat_series = [
            item for item in normalized_geometry_reference_series_results if str(item.get("side")) == "dat"
        ]
        first_details = dict(normalized_series_results[0].get("details", {}))
        time_range_meta = core.extract_time_range_metadata(first_details)
        time_range_display_suffix = core.build_time_range_display_suffix(first_details)
        time_range_filename_suffix = core.build_time_range_filename_suffix(first_details)
        point_count_contract = build_series_point_count_contract(normalized_series_results)
        txt_point_count_contract = build_series_point_count_contract(txt_series)
        dat_point_count_contract = build_series_point_count_contract(dat_series)
        geometry_point_count_contract = build_series_point_count_contract(
            normalized_geometry_reference_series_results or normalized_series_results
        )
        total_valid_freq_points_across_series = int(point_count_contract["total_valid_freq_points_across_series"])
        quality_warnings: list[str] = []
        if total_valid_freq_points_across_series < 10:
            quality_warnings.append("有效频点偏少，请检查时间范围、FS 或 NSEGMENT 设置。")
        if txt_points < 32 or (not single_side_compare and dat_points < 32):
            quality_warnings.append("有效点数偏少，当前 PSD 结果可能不稳定。")
        if start_dt is not None and end_dt is not None:
            span_seconds = (pd.Timestamp(end_dt) - pd.Timestamp(start_dt)).total_seconds()
            if span_seconds < 60:
                quality_warnings.append("当前时间范围较窄，谱图只能反映很短时间窗内的变化。")

        if layout == "叠加同图":
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            for index, item in enumerate(normalized_series_results):
                color = SERIES_COLORS[index % len(SERIES_COLORS)]
                self.apply_series_style(ax, item["freq"], item["density"], color=color, label=item["label"], style=style)
            geometry_state = self.build_compare_psd_geometry_state(
                normalized_geometry_reference_series_results or normalized_series_results,
                style=style,
            )
            if geometry_state is not None:
                self.apply_compare_psd_geometry_state(ax, geometry_state)
            else:
                self.add_selected_reference_slopes(
                    ax,
                    np.concatenate([item["freq"] for item in normalized_series_results]),
                    np.concatenate([item["density"] for item in normalized_series_results]),
                    is_psd=True,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            title = f"时间段内 PSD 对比{time_range_display_suffix}"
            if single_side_compare:
                title = f"时间段内 PSD 对比（txt侧）{time_range_display_suffix}"
            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Density")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="upper right")
            self.figure.tight_layout()
            self.refresh_canvas()
        else:
            def plot_group(
                ax: Any,
                group_series: list[dict[str, Any]],
                *,
                geometry_series: list[dict[str, Any]] | None,
                title: str,
                color_offset: int = 0,
            ) -> None:
                for index, item in enumerate(group_series):
                    color = SERIES_COLORS[(index + color_offset) % len(SERIES_COLORS)]
                    self.apply_series_style(ax, item["freq"], item["density"], color=color, label=item["label"], style=style)
                geometry_state = self.build_compare_psd_geometry_state(
                    list(geometry_series or group_series),
                    style=style,
                    color_offset=color_offset,
                )
                if geometry_state is not None:
                    self.apply_compare_psd_geometry_state(ax, geometry_state)
                elif group_series:
                    self.add_selected_reference_slopes(
                        ax,
                        np.concatenate([item["freq"] for item in group_series]),
                        np.concatenate([item["density"] for item in group_series]),
                        is_psd=True,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(title)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Density")
                ax.grid(True, which="both", linestyle="--", alpha=0.3)
                if ax.get_legend_handles_labels()[1]:
                    ax.legend(loc="upper right")

            visible_groups: list[dict[str, Any]] = [
                {
                    "title": f"设备A：{','.join(item['column'] for item in txt_series) or '无数据'}{time_range_display_suffix}",
                    "series": txt_series,
                    "geometry_series": geometry_txt_series or txt_series,
                    "window_title": f"设备A - PSD 对比{time_range_display_suffix}",
                    "window_filename": f"deviceA_{sanitize_filename(txt_series[0]['column'] if txt_series else 'psd')}_{self.get_plot_style_tag(style)}{time_range_filename_suffix}.png",
                    "color_offset": 0,
                }
            ]
            if not single_side_compare:
                visible_groups.append(
                    {
                        "title": f"设备B：{','.join(item['column'] for item in dat_series) or '无数据'}{time_range_display_suffix}",
                        "series": dat_series,
                        "geometry_series": geometry_dat_series or dat_series,
                        "window_title": f"设备B - PSD 对比{time_range_display_suffix}",
                        "window_filename": f"deviceB_{sanitize_filename(dat_series[0]['column'] if dat_series else 'psd')}_{self.get_plot_style_tag(style)}{time_range_filename_suffix}.png",
                        "color_offset": len(txt_series),
                    }
                )

            self.figure.clear()
            if len(visible_groups) == 1:
                ax = self.figure.add_subplot(111)
                plot_group(
                    ax,
                    visible_groups[0]["series"],
                    geometry_series=list(visible_groups[0].get("geometry_series") or []),
                    title=visible_groups[0]["title"],
                    color_offset=int(visible_groups[0]["color_offset"]),
                )
            else:
                vertical_main_preview = layout in {"上下分图", "分别生成两张图"}
                axes = self.figure.subplots(2, 1) if vertical_main_preview else self.figure.subplots(1, 2)
                for ax, group in zip(list(axes), visible_groups):
                    plot_group(
                        ax,
                        group["series"],
                        geometry_series=list(group.get("geometry_series") or []),
                        title=group["title"],
                        color_offset=int(group["color_offset"]),
                    )
            self.figure.tight_layout()
            overview_title = "图形结果：主页面总览图" if layout == "分别生成两张图" else None
            self.refresh_canvas(title=overview_title)

            if layout == "分别生成两张图" and self.use_separate_zoom_windows_var.get():
                entries: list[dict[str, Any]] = []
                for group in visible_groups:
                    fig_zoom = Figure(figsize=(6, 4), dpi=100)
                    ax_zoom = fig_zoom.add_subplot(111)
                    plot_group(
                        ax_zoom,
                        group["series"],
                        geometry_series=list(group.get("geometry_series") or []),
                        title=group["title"].replace(time_range_display_suffix, ""),
                        color_offset=int(group["color_offset"]),
                    )
                    fig_zoom.tight_layout()
                    entries.append(
                        {
                            "figure": fig_zoom,
                            "title": group["window_title"],
                            "filename": group["window_filename"],
                        }
                    )
                if entries:
                    self.show_separate_plot_windows(entries)

        self.current_plot_kind = "single_device_compare_psd" if single_side_compare else "dual_psd_compare"
        for index, item in enumerate(normalized_series_results):
            details = dict(item.get("details", {}))
            details["current_plot_kind"] = self.current_plot_kind
            if single_side_compare:
                details["plot_execution_path"] = "single_device_compare_psd_render"
            elif not str(details.get("plot_execution_path") or "").strip():
                details["plot_execution_path"] = "dual_psd_compare"
            item["details"] = details
            if index == 0:
                first_details = details
        self.current_plot_columns = [item["label"] for item in normalized_series_results]
        self.current_result_freq = None
        self.current_result_values = None
        self.current_aligned_frame = None
        self.current_aligned_metadata = {}
        self.current_compare_files = (
            [str(first_details.get("compare_txt_source_label") or normalized_series_results[0]["label"])]
            if single_side_compare
            else [label_a, label_b]
        )
        self.current_result_frame = self.build_overlay_export_frame(normalized_series_results)
        self.current_point_count_contract = {"global": dict(point_count_contract), "txt": dict(txt_point_count_contract)}
        if dat_series:
            self.current_point_count_contract["dat"] = dict(dat_point_count_contract)
        series_point_items = [
            f"{item['label']} | valid_freq_points={item.get('details', {}).get('valid_freq_points')} | "
            f"frequency_point_count={item.get('details', {}).get('frequency_point_count')}"
            for item in normalized_series_results
        ]
        self.register_plot_style_layout(style, layout)
        if single_side_compare:
            annotate_compare_geometry_point_count_fields(
                first_details,
                visible_contract=txt_point_count_contract or point_count_contract,
                geometry_contract=geometry_point_count_contract,
            )
            compare_geometry_point_count_items = build_compare_geometry_point_count_items(first_details)
            extra_items = [
                f"当前要素映射={mapping_name}",
                "当前谱类型=PSD",
                *compare_geometry_point_count_items,
                *build_series_point_count_items(point_count_contract),
                *build_series_point_count_items(txt_point_count_contract, prefix="txt"),
                f"base_spectrum_builder={first_details.get('base_spectrum_builder')}",
                f"设备A_base_fs_source={next((item['details'].get('base_fs_source') for item in txt_series if item.get('details')), None)}",
                *core.build_time_range_diagnostic_items(first_details),
                f"实际命中设备A列={','.join(sorted({item['column'] for item in txt_series}))}",
                "对齐策略=时间窗裁切后独立PSD",
                f"出图样式={style}",
                f"出图布局={layout}",
                f"compare_common_time_range={start_dt or '自动'} ~ {end_dt or '自动'}",
                f"txt 合并后点数={txt_points}",
                f"txt_FS={fs_a_last:g}",
                f"成功系列数={len(normalized_series_results)}",
                *build_single_device_execution_items(first_details, include_plot_execution_path=True),
                *series_point_items,
            ]
            if dat_points > 0:
                extra_items.append(f"compare_dat_reference_points={dat_points}")
            if fs_b_last > 0:
                extra_items.append(f"compare_dat_reference_fs={fs_b_last:g}")
            extra_items.extend(quality_warnings)
            self.update_diagnostic_info(
                layout="时间段内 PSD 对比（单侧）",
                analysis_label="；".join(item["label"] for item in normalized_series_results),
                extra_items=extra_items,
            )
            status_items = [
                *compare_geometry_point_count_items,
                *build_series_point_count_status_items(txt_point_count_contract, prefix="txt", include_totals=False),
                f"series_count={point_count_contract['series_count']}",
                *build_single_device_execution_items(first_details, include_plot_execution_path=True),
            ]
            status = (
                f"图已生成：时间段内 PSD 对比 / {mapping_name} / 成功系列数={len(normalized_series_results)}"
                f" | {' | '.join(status_items)}"
                f" | 时间窗={time_range_meta.get('time_range_policy_label', first_details.get('time_range_policy_label', '默认'))}"
                f" | {time_range_meta.get('time_range_difference_hint', core.TIME_RANGE_DIFFERENCE_HINT)}"
            )
        else:
            extra_items = [
                f"当前要素映射={mapping_name}",
                f"当前谱类型=PSD",
                *build_device_dispatch_items(first_details, include_plot_execution_path=True),
                *build_series_point_count_items(point_count_contract),
                *build_series_point_count_items(txt_point_count_contract, prefix="txt"),
                *build_series_point_count_items(dat_point_count_contract, prefix="dat"),
                f"base_spectrum_builder={first_details.get('base_spectrum_builder')}",
                f"设备A_base_fs_source={next((item['details'].get('base_fs_source') for item in txt_series if item.get('details')), None)}",
                f"设备B_base_fs_source={next((item['details'].get('base_fs_source') for item in dat_series if item.get('details')), None)}",
                *core.build_time_range_diagnostic_items(first_details),
                f"实际命中设备A列={','.join(sorted({item['column'] for item in txt_series}))}",
                f"实际命中设备B列={','.join(sorted({item['column'] for item in dat_series}))}",
                "对齐策略=时间窗裁切后独立PSD",
                f"出图样式={style}",
                f"出图布局={layout}",
                f"compare_common_time_range={start_dt or '自动'} ~ {end_dt or '自动'}",
                f"txt 合并后点数={txt_points}",
                f"dat 裁切后点数={dat_points}",
                f"txt_FS={fs_a_last:g}",
                f"dat_FS={fs_b_last:g}",
                f"成功系列数={len(normalized_series_results)}",
                *series_point_items,
            ] + self.build_timestamp_quality_items(parsed_a, parsed_b)
            extra_items.extend(quality_warnings)
            self.update_diagnostic_info(
                layout=f"{parsed_a.profile_name} / {parsed_b.profile_name}",
                analysis_label="；".join(item["label"] for item in normalized_series_results),
                extra_items=extra_items,
            )
            status = (
                f"图已生成：时间段内 PSD 对比 / {mapping_name} / 成功系列数={len(normalized_series_results)}"
                f" | {' | '.join(build_device_dispatch_items(first_details, include_plot_execution_path=True))}"
                f" | {' | '.join(build_series_point_count_status_items(txt_point_count_contract, prefix='txt', include_totals=False))}"
                f" | {' | '.join(build_series_point_count_status_items(dat_point_count_contract, prefix='dat', include_totals=False))}"
                f" | series_count={point_count_contract['series_count']}"
                f" | total_valid_freq_points_across_series={point_count_contract['total_valid_freq_points_across_series']}"
                f" | total_frequency_points_across_series={point_count_contract['total_frequency_points_across_series']}"
                f" | 时间窗={time_range_meta.get('time_range_policy_label', first_details.get('time_range_policy_label', '默认'))}"
                f" | {time_range_meta.get('time_range_difference_hint', core.TIME_RANGE_DIFFERENCE_HINT)}"
            )
        if layout == "分别生成两张图":
            if self.separate_plot_windows:
                status += f" | 已生成独立图窗口：{len(self.separate_plot_windows)} 个"
            else:
                status += " | 主页面保留总览，未弹出独立窗口"
        if single_side_compare and bool(first_details.get("single_device_selection_filtered_to_txt_side")):
            status += " | 当前 selection 包含 dat，已自动忽略 dat 并按 txt-side 复用"
        if single_side_compare and str(first_details.get("single_device_compare_context_source") or "") == "auto_bootstrap_fallback":
            status += " | 当前 compare UI 状态不完整，单设备暂用 auto bootstrap 上下文"
        if (
            single_side_compare
            and "single_device_compare_context_dat_matches_selected_dat" in first_details
            and not bool(first_details.get("single_device_compare_context_dat_matches_selected_dat"))
        ):
            status += " | 警告：单设备复用的 dat 与当前 compare 页 dat 不一致"
        self.status_var.set(status)

    def plot_scatter_consistency_compare(
        self,
        paths: list[Path | str],
        parsed_a: ParsedFileResult,
        col_a: str,
        parsed_b: ParsedFileResult,
        col_b: str,
        aligned: pd.DataFrame,
        meta: dict[str, Any],
    ) -> None:
        value_a = meta["meta_a"]["value_name"]
        value_b = meta["meta_b"]["value_name"]
        label_a = self.format_source_label(paths[0])
        label_b = self.format_source_label(paths[1])
        mapping_name = self.element_preset_var.get().strip() if self.mapping_mode_var.get() == "预设映射" else "手动映射"
        style = self.resolve_plot_style("散点一致性对比")
        layout = "叠加同图"
        diff = aligned[value_a] - aligned[value_b]
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        mean_diff = float(diff.mean())
        corr = float(aligned[value_a].corr(aligned[value_b]))
        quality_warnings: list[str] = []
        if meta["matched_count"] < 20:
            quality_warnings.append("匹配点数偏少，散点一致性结果可能不稳定。")
        coverage_ratio = meta["matched_count"] / max(1, min(meta["meta_a"]["valid_count"], meta["meta_b"]["valid_count"]))
        if coverage_ratio < 0.3:
            quality_warnings.append("A/B 时间覆盖较弱，请检查时间范围或补齐更多文件。")

        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_series_style(ax, aligned[value_a], aligned[value_b], color=SERIES_COLORS[0], label=f"{col_a} vs {col_b}", style=style)
        min_val = float(min(aligned[value_a].min(), aligned[value_b].min()))
        max_val = float(max(aligned[value_a].max(), aligned[value_b].max()))
        if math.isfinite(min_val) and math.isfinite(max_val) and min_val < max_val:
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#666666", linewidth=1.0, label="y = x")
            ax.legend(loc="upper right")
        ax.set_title(f"散点一致性对比 - {col_a} vs {col_b}")
        ax.set_xlabel(f"{label_a} - {col_a}")
        ax.set_ylabel(f"{label_b} - {col_b}")
        ax.grid(True, linestyle="--", alpha=0.3)
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = "aligned_scatter"
        self.current_plot_columns = [col_a, col_b]
        self.current_result_freq = None
        self.current_result_values = None
        self.current_result_frame = None
        export_frame = aligned[["时间戳", value_a, value_b]].copy()
        export_frame.columns = ["时间戳", "txt值" if "txt" in label_a.lower() else "设备A值", "dat值" if "dat" in label_b.lower() else "设备B值"]
        self.current_aligned_frame = export_frame
        self.current_compare_files = [label_a, label_b]
        self.register_plot_style_layout(style, layout)
        self.current_aligned_metadata = {
            "方案名": self.scheme_name_var.get().strip() or "默认方案",
            "映射名称": mapping_name,
            "对齐策略": meta.get("alignment_strategy", self.get_alignment_strategy("散点一致性对比")),
            "时间范围": f"{meta['actual_start']} ~ {meta['actual_end']}",
            "匹配点数": meta["matched_count"],
            "成功系列数": 1,
            "出图样式": style,
            "出图布局": layout,
        }

        self.update_diagnostic_info(
            layout=f"{parsed_a.profile_name} / {parsed_b.profile_name}",
            analysis_label=f"{col_a} vs {col_b}",
            extra_items=[
                f"当前要素映射={mapping_name}",
                f"实际命中设备A列={col_a}",
                f"实际命中设备B列={col_b}",
                f"对齐策略={meta.get('alignment_strategy', self.get_alignment_strategy('散点一致性对比'))}",
                f"出图样式={style}",
                f"出图布局={layout}",
                f"成功系列数=1",
                f"当前时间窗={meta['actual_start']} ~ {meta['actual_end']}",
                f"txt 合并后点数={meta['meta_a']['valid_count']}",
                f"dat 裁切后点数={meta['meta_b']['valid_count']}",
                f"匹配点数={meta['matched_count']}",
                f"均值差={mean_diff:.4g}",
                f"RMSE={rmse:.4g}",
                f"相关系数r={corr:.4g}",
                f"匹配容差={meta['tolerance_seconds']:.3f}s",
            ] + self.build_timestamp_quality_items(parsed_a, parsed_b) + quality_warnings,
        )
        self.status_var.set(f"图已生成：散点一致性对比 / {col_a} vs {col_b}")

    def plot_results(
        self,
        analysis_type: str,
        freq: np.ndarray,
        density: np.ndarray,
        columns: list[str],
        details: dict[str, Any],
    ) -> None:
        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ordered_columns = list(details.get("ordered_columns", columns))
        target_context = core.get_target_spectral_context(details.get("target_spectral_context"))
        series_label = ordered_columns[0] if analysis_type == "spectral" else f"{ordered_columns[0]} vs {ordered_columns[1]}"
        quality_items: list[str] = []
        display_meta: dict[str, str] | None = None
        point_count_contract: dict[str, Any] = {}

        if analysis_type == "spectral":
            time_range_display_suffix = core.build_time_range_display_suffix(details)
            positive = (freq > 0) & np.isfinite(freq) & np.isfinite(density) & (density > 0)
            freq = freq[positive]
            density = density[positive]
            if len(freq) == 0:
                raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
            point_count_contract = build_series_point_count_contract(
                [
                    {
                        "label": series_label,
                        "freq": freq,
                        "details": details,
                    }
                ]
            )
            quality_items.append("当前谱类型=PSD")
            compare_geometry_point_count_items = build_compare_geometry_point_count_items(details)
            if compare_geometry_point_count_items:
                quality_items.extend(compare_geometry_point_count_items)
            quality_items.extend(build_series_point_count_items(point_count_contract))
            if details.get("base_spectrum_builder") is not None:
                quality_items.append(f"base_spectrum_builder={details.get('base_spectrum_builder')}")
            if details.get("base_fs_source") is not None:
                quality_items.append(f"base_fs_source={details.get('base_fs_source')}")
            quality_items.extend(core.build_time_range_diagnostic_items(details))
            quality_items.extend(build_single_txt_execution_items(details, include_plot_execution_path=True))
            if len(freq) < 10:
                quality_items.append("有效频点偏少，请检查时间范围、FS 或 NSEGMENT 设置。")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.scatter(
                freq,
                density,
                color=SERIES_COLORS[0],
                label=series_label,
                s=22,
                alpha=0.8,
                edgecolors="none",
            )
            title = f"功率谱密度：{columns[0]}{time_range_display_suffix}"
            ylabel = "Power Spectral Density"
            self.add_selected_reference_slopes(ax, freq, density, is_psd=True)
        else:
            spectrum_type = str(details.get("spectrum_type", CROSS_SPECTRUM_MAGNITUDE))
            display_meta = self.get_cross_spectrum_display_meta(spectrum_type)
            display_semantics = str(details.get("display_semantics") or "")
            title_prefix = display_meta["title_prefix"]
            if target_context is not None and details.get("uses_canonical_pair"):
                title = f"{title_prefix}：{target_context['reference_column']} vs {target_context['display_target_label']}"
                series_label = f"{target_context['reference_column']} vs {target_context['display_target_label']}"
            else:
                title = f"{title_prefix}：{ordered_columns[0]} vs {ordered_columns[1]}"
            raw_freq = freq
            raw_density = density
            freq, density = self.configure_cross_spectrum_axis(
                ax,
                raw_freq,
                raw_density,
                spectrum_type,
                display_semantics=display_semantics or None,
                title=title,
                style="散点图",
                label=series_label,
                color=SERIES_COLORS[0],
            )
            point_count_contract = build_series_point_count_contract(
                [
                    {
                        "label": series_label,
                        "freq": freq,
                        "details": details,
                    }
                ]
            )
            ylabel = self.get_cross_spectrum_display_meta(spectrum_type)["ylabel"]
            quality_items.append(f"当前谱类型={spectrum_type}")
            quality_items.extend(build_series_point_count_items(point_count_contract))
            if details.get("cross_execution_path") is not None:
                quality_items.append(f"cross_execution_path={details.get('cross_execution_path')}")
            if details.get("cross_implementation_id") is not None:
                quality_items.append(f"cross_implementation_id={details.get('cross_implementation_id')}")
            if details.get("cross_implementation_label") is not None:
                quality_items.append(f"cross_implementation_label={details.get('cross_implementation_label')}")
            if details.get("alignment_strategy") is not None:
                quality_items.append(f"alignment_strategy={details.get('alignment_strategy')}")
            if details.get("display_semantics") is not None:
                quality_items.append(f"display_semantics={details.get('display_semantics')}")
            if details.get("display_value_source") is not None:
                quality_items.append(f"display_value_source={details.get('display_value_source')}")
            if len(freq) < 10:
                quality_items.append("有效频点偏少，请检查时间范围、FS 或 NSEGMENT 设置。")
            if display_semantics != core.CROSS_DISPLAY_SEMANTICS_ABS and len(density) > 0:
                near_zero_ratio = float(np.mean(np.abs(density) <= 1e-12))
                if near_zero_ratio >= 0.8:
                    quality_items.append("协谱/正交谱大部分点接近 0，请确认列选择和时间范围。")
        quality_items.extend(
            self.build_reference_slope_diagnostic_items(
                is_psd=analysis_type == "spectral",
                spectrum_type=None if analysis_type == "spectral" else str(details.get("spectrum_type", CROSS_SPECTRUM_MAGNITUDE)),
            )
        )
        if target_context is not None:
            quality_items.extend(
                [
                    f"cross_execution_path={details.get('cross_execution_path')}",
                    f"cross_implementation_id={details.get('cross_implementation_id')}",
                    f"cross_implementation_label={details.get('cross_implementation_label')}",
                    f"cross_reference_column={details.get('cross_reference_column', target_context['reference_column'])}",
                    f"reference_column={details.get('reference_column', target_context['reference_column'])}",
                    f"ygas_target_column={target_context.get('ygas_target_column', target_context.get('target_column'))}",
                    f"dat_target_column={target_context.get('dat_target_column')}",
                    "canonical_cross_pairs="
                    + " | ".join(str(item) for item in (target_context.get("canonical_cross_pairs") or [])),
                    f"target_column={details.get('target_column')}",
                    f"cross_order={details.get('cross_order', target_context['cross_order'])}",
                    f"display_semantics={details.get('display_semantics')}",
                    f"display_value_source={details.get('display_value_source')}",
                ]
            )
            if spectrum_type in CROSS_SPECTRUM_OPTIONS:
                for key in ("cross_kernel", "window", "detrend", "scaling", "return_onesided", "average"):
                    if details.get(key) is not None:
                        quality_items.append(f"{key}={details[key]}")
            if details.get("auto_reordered"):
                quality_items.append("已自动修正列顺序为 Uz -> target")
            warning_message = str(details.get("selection_warning") or "").strip()
            if warning_message:
                quality_items.append(warning_message)

        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        if analysis_type == "spectral":
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = analysis_type
        if target_context is not None and details.get("uses_canonical_pair"):
            self.current_plot_columns = [str(target_context["reference_column"]), str(target_context["display_target_label"])]
        else:
            self.current_plot_columns = list(ordered_columns)
        self.current_result_freq = freq.copy()
        self.current_result_values = density.copy()
        self.current_aligned_frame = None
        self.current_lazy_aligned_frames = []
        self.current_aligned_metadata = {}
        self.current_compare_files = []
        if analysis_type == "spectral":
            self.current_result_frame = pd.DataFrame(
                {
                    "frequency_hz": self.current_result_freq,
                    "psd": self.current_result_values,
                }
            )
            self.current_result_frame = self.attach_metadata_columns(
                self.current_result_frame,
                build_export_metadata_from_details(details, include_additional=True),
            )
        else:
            export_mask = self.build_spectrum_plot_mask(
                raw_freq,
                raw_density,
                str(details.get("spectrum_type", CROSS_SPECTRUM_MAGNITUDE)),
                display_semantics=str(details.get("display_semantics") or "") or None,
            )
            self.current_result_frame = self.build_cross_export_frame(
                raw_freq,
                details,
                export_mask,
            )
        self.current_point_count_contract = {"global": dict(point_count_contract)}
        self.update_diagnostic_info(
            layout=self.current_layout_label,
            analysis_label=series_label,
            valid_points=int(details["valid_points"]),
            fs=float(details["fs"]),
            nsegment=int(details["nsegment"]),
            overlap_ratio=float(details["overlap_ratio"]),
            nperseg=int(details["nperseg"]),
            noverlap=int(details["noverlap"]),
            extra_items=quality_items,
        )
        if analysis_type == "spectral":
            policy_label = str(details.get("time_range_policy_label") or "默认")
            spectral_status_items: list[str] = []
            compare_geometry_point_count_items = build_compare_geometry_point_count_items(details)
            if compare_geometry_point_count_items:
                spectral_status_items.extend(compare_geometry_point_count_items)
                spectral_status_items.extend(
                    build_series_point_count_status_items(point_count_contract, include_totals=False)
                )
                spectral_status_items.append(f"series_count={int(point_count_contract.get('series_count', 0))}")
            else:
                spectral_status_items.extend(build_series_point_count_status_items(point_count_contract))
            spectral_status_items.extend(build_single_txt_execution_items(details))
            self.status_var.set(
                f"图已生成：功率谱密度 / {columns[0]} | 时间窗={policy_label} | "
                f"{' | '.join(spectral_status_items)} | "
                f"{details.get('time_range_difference_hint') or core.TIME_RANGE_DIFFERENCE_HINT}"
            )
        else:
            self.status_var.set(
                f"图已生成：{display_meta['title_prefix'] if display_meta else '互谱分析'} / {series_label}"
                f" | {' | '.join(build_series_point_count_status_items(point_count_contract))}"
            )

    def build_overlay_export_frame(self, series_results: list[dict[str, Any]]) -> pd.DataFrame:
        merged_frame: pd.DataFrame | None = None
        for item in series_results:
            y_values = item.get("density", item.get("values"))
            if y_values is None:
                continue
            series_frame = pd.DataFrame(
                {
                    "frequency_hz": item["freq"],
                    item["label"]: y_values,
                }
            )
            if merged_frame is None:
                merged_frame = series_frame
            else:
                merged_frame = merged_frame.merge(series_frame, on="frequency_hz", how="outer")

        if merged_frame is None:
            return pd.DataFrame(columns=["frequency_hz"])
        merged_frame = merged_frame.sort_values("frequency_hz").reset_index(drop=True)
        return self.attach_metadata_columns(
            merged_frame,
            build_aggregated_export_metadata(
                series_results,
                formatter=self.format_metadata_value,
                include_additional=True,
            ),
        )

    def plot_multi_spectral_results(
        self,
        target_column: str,
        series_results: list[dict[str, Any]],
        skipped_files: list[str],
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.reset_plot_output_state()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        payload = dict(payload or {})
        device_groups = list(payload.get("device_groups", []))
        effective_device_count = int(
            payload.get("effective_device_count", payload.get("device_count", len(series_results)))
        )
        file_count = int(
            payload.get(
                "selection_file_count",
                payload.get("file_count", sum(int(item.get("file_count", 1)) for item in series_results)),
            )
        )
        first_details = dict(series_results[0].get("details", {})) if series_results else {}
        render_semantics = str(
            payload.get("dispatch_render_semantics")
            or first_details.get("render_semantics")
            or self.resolve_device_count_render_semantics(effective_device_count)
        )
        time_range_display_suffix = core.build_time_range_display_suffix(first_details)

        combined_freq: list[np.ndarray] = []
        combined_density: list[np.ndarray] = []
        for index, item in enumerate(series_results):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            ax.scatter(
                item["freq"],
                item["density"],
                color=color,
                label=item["label"],
                s=22,
                alpha=0.8,
                edgecolors="none",
            )
            combined_freq.append(item["freq"])
            combined_density.append(item["density"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        all_freq = np.concatenate(combined_freq)
        all_density = np.concatenate(combined_density)
        self.add_selected_reference_slopes(ax, all_freq, all_density, is_psd=True)
        if effective_device_count == 1:
            title = f"单设备图谱 - {series_results[0]['label']} - {target_column}{time_range_display_suffix}"
        elif render_semantics in {"multi_device_compare", "dual_psd_compare"}:
            title = f"设备对比图 - {target_column}{time_range_display_suffix}"
        else:
            title = f"多设备图谱对比 - {target_column}{time_range_display_suffix}"
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Density")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.refresh_canvas()

        self.current_plot_kind = "multi_spectral"
        self.current_plot_columns = [target_column]
        self.current_result_freq = None
        self.current_result_values = None
        self.current_aligned_frame = None
        self.current_aligned_metadata = {}
        self.current_compare_files = [str(item["label"]) for item in series_results]
        self.current_result_frame = self.build_overlay_export_frame(series_results)
        point_count_contract = build_series_point_count_contract(series_results)
        self.current_point_count_contract = {"global": dict(point_count_contract)}
        if render_semantics == "single_device":
            self.current_plot_kind = "single_device_spectrum"
        elif render_semantics in {"multi_device_compare", "dual_psd_compare"}:
            self.current_plot_kind = "multi_device_compare"
        else:
            self.current_plot_kind = "multi_device_overlay"
        first_details = series_results[0]["details"]
        group_items = [
            f"{item['device_label']} | 文件数={item['file_count']} | 来源={item['device_source']} | "
            f"merge_scope={item.get('selection_merge_scope', 'device_group')} | 合并={item['merge_strategy']} | "
            f"raw_rows={item.get('raw_source_rows')} | merged_rows={item.get('merged_rows')} | "
            f"resolved_target_columns={','.join(str(value) for value in (item.get('resolved_target_columns') or []))} | "
            f"target_resolution_sources={','.join(str(value) for value in (item.get('target_resolution_sources') or []))}"
            for item in device_groups
        ]
        series_items = [
            f"{item['label']} | valid_points={item['details'].get('valid_points')} | "
            f"valid_freq_points={item['details'].get('valid_freq_points')} | "
            f"frequency_point_count={item['details'].get('frequency_point_count')} | "
            f"rendered_point_count={item['details'].get('rendered_point_count', len(item['freq']))}"
            for item in series_results
        ]
        self.update_diagnostic_info(
            layout=(
                "单设备图"
                if effective_device_count == 1
                else ("设备对比图" if render_semantics == "multi_device_compare" else "多设备图")
            ),
            analysis_label=target_column,
            fs=float(first_details["fs"]),
            nsegment=int(first_details["nsegment"]),
            overlap_ratio=float(first_details["overlap_ratio"]),
            nperseg=int(first_details["nperseg"]),
            noverlap=int(first_details["noverlap"]),
            batch_success=len(series_results),
            batch_skips=len(skipped_files),
            extra_items=[
                f"selection_file_count={file_count}",
                *build_device_dispatch_items(first_details, include_plot_execution_path=True),
                *build_series_point_count_items(point_count_contract),
                f"base_spectrum_builder={first_details.get('base_spectrum_builder')}",
                f"base_fs_source={first_details.get('base_fs_source')}",
                f"selection_merge_scope={first_details.get('selection_merge_scope')}",
                f"group_device_source={first_details.get('group_device_source')}",
                f"raw_source_rows={first_details.get('raw_source_rows')}",
                f"merged_rows={first_details.get('merged_rows')}",
                f"valid_points={first_details.get('valid_points')}",
                f"valid_freq_points={first_details.get('valid_freq_points')}",
                f"frequency_point_count={first_details.get('frequency_point_count')}",
                f"rendered_point_count={first_details.get('rendered_point_count', len(series_results[0]['freq']))}",
                *core.build_time_range_diagnostic_items(first_details),
                *group_items,
                *series_items,
                *self.build_reference_slope_diagnostic_items(is_psd=True),
            ],
        )

        status_prefix = (
            "单设备图谱"
            if effective_device_count == 1
            else ("设备对比图" if render_semantics in {"multi_device_compare", "dual_psd_compare"} else "多设备图谱对比")
        )
        status_items = build_device_dispatch_items(first_details)
        status_items.extend(build_series_point_count_status_items(point_count_contract))
        status = (
            f"图已生成：{status_prefix} / {target_column} / 成功系列数={len(series_results)}"
            f" | {' | '.join(status_items)}"
            f" | 时间窗={first_details.get('time_range_policy_label') or '默认'}"
            f" | {first_details.get('time_range_difference_hint') or core.TIME_RANGE_DIFFERENCE_HINT}"
        )
        if skipped_files:
            preview = "、".join(skipped_files[:3])
            if len(skipped_files) > 3:
                preview += " 等"
            status += f" | 已跳过 {len(skipped_files)} 个文件：{preview}"
        self.status_var.set(status)

    def perform_multi_spectral_compare(self) -> None:
        selected_files = self.get_selected_file_paths()
        if len(selected_files) < 1:
            self.status_var.set("设备图谱生成失败：文件数量不足")
            self.update_diagnostic_info(layout=self.current_layout_label, batch_success=0, batch_skips=0)
            self.render_plot_message("请先在文件列表中选择至少 1 个文件。", level="warning")
            messagebox.showwarning("提示", "请先在文件列表中选择至少 1 个文件。")
            return
        if len(self.selected_columns) != 1:
            self.status_var.set("设备图谱生成失败：未选中单一目标列")
            self.update_diagnostic_info(layout=self.current_layout_label, batch_success=0, batch_skips=0)
            self.render_plot_message("按设备分组生成图谱前，需要先在可分析列中选择 1 个目标列。", level="warning")
            messagebox.showwarning("提示", "按设备分组生成图谱前，需要先在可分析列中选择 1 个目标列。")
            return

        target_column = self.selected_columns[0]
        fs, requested_nsegment, overlap_ratio = self.get_analysis_params()
        start_dt, end_dt = self.resolve_optional_time_range_inputs()
        self.start_background_task(
            status_text=f"正在生成图：{target_column}",
            worker=lambda reporter: self.prepare_selected_files_default_render_payload(
                selected_files,
                target_column,
                fs,
                requested_nsegment,
                overlap_ratio,
                start_dt=start_dt,
                end_dt=end_dt,
                reporter=reporter,
            ),
            on_success=self.on_selected_files_default_plot_ready,
            error_title="生成图",
        )

    def add_reference_slope(
        self,
        ax: Any,
        freq: np.ndarray,
        density: np.ndarray,
        slope: float,
        *,
        color: str = "#d65f5f",
        label: str | None = None,
    ) -> None:
        freq_array = np.asarray(freq, dtype=float)
        density_array = np.asarray(density, dtype=float)
        base_mask = np.isfinite(freq_array) & (freq_array > 0) & np.isfinite(density_array)
        if np.count_nonzero(base_mask) < 2:
            return
        positive_mask = base_mask & (density_array > 0)
        if np.count_nonzero(positive_mask) >= 2:
            ref_x = np.asarray(freq_array[positive_mask], dtype=float)
            ref_anchor_values = np.asarray(density_array[positive_mask], dtype=float)
        else:
            fallback_values = np.abs(density_array[base_mask])
            fallback_mask = np.isfinite(fallback_values) & (fallback_values > 0)
            if np.count_nonzero(fallback_mask) < 2:
                return
            base_freq = np.asarray(freq_array[base_mask], dtype=float)
            ref_x = np.asarray(base_freq[fallback_mask], dtype=float)
            ref_anchor_values = np.asarray(fallback_values[fallback_mask], dtype=float)
        anchor_index = len(ref_x) // 2
        x_anchor = ref_x[anchor_index]
        y_anchor = ref_anchor_values[anchor_index]
        if not (np.isfinite(x_anchor) and np.isfinite(y_anchor) and x_anchor > 0 and y_anchor > 0):
            return
        ref = y_anchor * (ref_x / x_anchor) ** slope
        slope_text = float_to_fraction_str(slope)
        ref_label = label or f"Reference slope {slope_text}"
        ax.plot(ref_x, ref, linestyle="--", linewidth=1.4, color=color, label=ref_label)

    def add_fixed_reference_power_law(
        self,
        ax: Any,
        x_values: np.ndarray,
        *,
        slope: float = -2 / 3,
        amplitude_at_1hz: float = 1.0,
        label: str = "(-2/3)",
        color: str = "#d65f5f",
    ) -> None:
        x_array = np.asarray(x_values, dtype=float)
        mask = np.isfinite(x_array) & (x_array > 0)
        if np.count_nonzero(mask) < 2:
            return
        ref_x = np.unique(np.sort(x_array[mask]))
        ref_y = float(amplitude_at_1hz) * np.power(ref_x, slope)
        ax.plot(ref_x, ref_y, linestyle="--", linewidth=1.4, color=color, label=label, zorder=1)

    def clear_plot(self) -> None:
        self.update_diagnostic_info(layout=self.current_layout_label)
        self.render_plot_message("请先在左侧勾选要分析的列。")
        self.status_var.set("已清除图表。")

    def update_table_view(self) -> None:
        table_data = self.preview_data if self.preview_data is not None else self.raw_data
        if table_data is None:
            self.tree.delete(*self.tree.get_children())
            self.page_info_var.set("0 / 0")
            return

        self.tree.delete(*self.tree.get_children())
        columns = list(table_data.columns)
        self.tree["columns"] = columns
        for col in columns:
            self.tree.heading(col, text=col)

        self.adjust_column_widths()
        self.load_virtual_page()

    def adjust_column_widths(self) -> None:
        table_data = self.preview_data if self.preview_data is not None else self.raw_data
        if table_data is None:
            return

        sample_data = table_data.head(20)
        for col in table_data.columns:
            max_len = self.table_font.measure(col)
            for value in sample_data[col]:
                text = self.format_display_value(value)
            max_len = max(max_len, self.table_font.measure(text))
            self.tree.column(col, width=min(max_len + 24, 300), anchor="w")

    def load_virtual_page(self) -> None:
        table_data = self.preview_data if self.preview_data is not None else self.raw_data
        if table_data is None:
            return

        self.tree.delete(*self.tree.get_children())
        total_rows = len(table_data)
        if total_rows == 0:
            self.page_info_var.set("0 / 0")
            return

        max_page = max(math.ceil(total_rows / ROWS_PER_PAGE) - 1, 0)
        self.page_index = max(0, min(self.page_index, max_page))

        start_row = self.page_index * ROWS_PER_PAGE
        end_row = min(start_row + ROWS_PER_PAGE, total_rows)
        rows_to_insert = table_data.iloc[start_row:end_row]

        for _, row in rows_to_insert.iterrows():
            formatted_row = [self.format_display_value(value) for value in row.tolist()]
            self.tree.insert("", tk.END, values=formatted_row)

        self.page_info_var.set(f"{self.page_index + 1} / {max_page + 1}  ({start_row + 1}-{end_row} / {total_rows})")

    def format_display_value(self, value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, str):
            try:
                numeric = float(value)
            except ValueError:
                return value
            if math.isfinite(numeric):
                return f"{numeric:.4f}"
        return str(value)

    def prev_page(self) -> None:
        if self.page_index > 0:
            self.page_index -= 1
            self.load_virtual_page()

    def next_page(self) -> None:
        if self.raw_data is None:
            return
        max_page = max(math.ceil(len(self.raw_data) / ROWS_PER_PAGE) - 1, 0)
        if self.page_index < max_page:
            self.page_index += 1
            self.load_virtual_page()

    def save_figure(self) -> None:
        if self.current_plot_kind is None:
            messagebox.showwarning("警告", "没有可保存的图表。")
            return

        if self.separate_plot_entries:
            choice = messagebox.askyesnocancel(
                "保存图片",
                "当前主页面图仍可直接保存。\n\n"
                "选择“是”：保存主页面图\n"
                "选择“否”：批量保存独立窗口图\n"
                "选择“取消”：不执行保存",
            )
            if choice is None:
                return
            if not choice:
                target_folder = filedialog.askdirectory(title="选择保存目录")
                if not target_folder:
                    return
                saved_paths: list[str] = []
                for entry in self.separate_plot_entries:
                    path = Path(target_folder) / entry["filename"]
                    entry["figure"].savefig(path, dpi=300, bbox_inches="tight")
                    saved_paths.append(str(path))
                self.status_var.set(f"已分别保存 {len(saved_paths)} 张独立窗口图到：{target_folder}")
                return

        default_filename = self.build_default_filename()
        path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")],
            initialfile=default_filename,
        )
        if not path:
            return

        self.figure.savefig(path, dpi=300, bbox_inches="tight")
        self.status_var.set(f"主页面图已保存到：{path}")

    def build_compare_filename_core(self) -> str:
        if self.current_plot_columns:
            base = sanitize_filename(self.current_plot_columns[0])
        elif self.scheme_name_var.get().strip():
            base = sanitize_filename(self.scheme_name_var.get().strip())
        else:
            base = "compare"
        style_tag = self.get_plot_style_tag(self.current_plot_style_label) if self.current_plot_style_label else "auto"
        layout_tag = self.get_plot_layout_tag(self.current_plot_layout_label) if self.current_plot_layout_label else "overlay"
        return f"{base}_{layout_tag}_{style_tag}"

    def export_results(self) -> None:
        if self.current_plot_kind is None or self.current_result_frame is None:
            messagebox.showwarning("警告", "没有可导出的分析结果。")
            return

        default_filename = self.build_default_data_filename()
        path = filedialog.asksaveasfilename(
            title="导出结果",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialfile=default_filename,
        )
        if not path:
            return

        self.current_result_frame.to_csv(path, index=False, encoding="utf-8-sig")
        if self.current_plot_kind == "target_spectrum":
            export_path = Path(path)
            group_records = list(self.current_target_plot_metadata.get("group_records", []))
            qc_frame = self.current_target_plot_metadata.get("group_qc_export_frame")
            if not isinstance(qc_frame, pd.DataFrame):
                qc_frame = self.build_target_group_qc_export_frame(group_records)
            if isinstance(qc_frame, pd.DataFrame) and not qc_frame.empty:
                qc_path = export_path.with_name(f"{export_path.stem}_group_qc_summary.csv")
                qc_frame.to_csv(qc_path, index=False, encoding="utf-8-sig")
                self.status_var.set(f"结果已导出到：{path} | 组级QC：{qc_path}")
                return
        self.status_var.set(f"结果已导出到：{path}")

    def export_aligned_data(self) -> None:
        if (self.current_aligned_frame is None or self.current_aligned_frame.empty) and self.current_plot_kind == "target_spectrum":
            aligned_frames = self.current_target_plot_metadata.get("aligned_frames")
            if isinstance(aligned_frames, list) and aligned_frames:
                self.current_aligned_frame = self.merge_timestamp_frames(aligned_frames)
        if (self.current_aligned_frame is None or self.current_aligned_frame.empty) and self.current_lazy_aligned_frames:
            self.current_aligned_frame = self.merge_timestamp_frames(self.current_lazy_aligned_frames)
        if self.current_aligned_frame is None or self.current_aligned_frame.empty:
            messagebox.showwarning("警告", "当前没有可导出的对齐数据。")
            return

        path = filedialog.asksaveasfilename(
            title="导出对齐数据",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialfile=self.build_default_aligned_filename(),
        )
        if not path:
            return

        export_frame = self.current_aligned_frame.copy()
        if self.current_aligned_metadata:
            for key, value in self.current_aligned_metadata.items():
                export_frame[key] = str(value)
        export_frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.status_var.set(f"对齐数据已导出到：{path}")

    def build_comparison_base_name(self) -> str:
        start = self.current_comparison_metadata.get("start")
        end = self.current_comparison_metadata.get("end")
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            start_text = start.strftime("%Y%m%d_%H%M%S")
            end_text = end.strftime("%Y%m%d_%H%M%S")
            return f"compare_merged_{start_text}_to_{end_text}"
        return "compare_merged"

    def build_target_plot_base_name(self) -> str:
        start = self.current_target_plot_metadata.get("start")
        end = self.current_target_plot_metadata.get("end")
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            start_text = start.strftime("%Y%m%d_%H%M%S")
            end_text = end.strftime("%Y%m%d_%H%M%S")
            return f"target_windowed_{start_text}_to_{end_text}"
        return "target_windowed"

    def build_default_filename(self) -> str:
        time_range_filename_suffix = core.build_time_range_filename_suffix(self.current_result_frame)
        if self.current_data_source_kind == "comparison_analysis":
            base_name = self.build_comparison_base_name()
            if self.current_plot_kind == "spectral" and self.current_plot_columns:
                col = sanitize_filename(self.current_plot_columns[0])
                return f"{base_name}_{col}_spectral{time_range_filename_suffix}_plot.png"
            if self.current_plot_kind == "cross" and len(self.current_plot_columns) == 2:
                col1 = sanitize_filename(self.current_plot_columns[0])
                col2 = sanitize_filename(self.current_plot_columns[1])
                return f"{base_name}_{col1}_vs_{col2}_cross_plot.png"

        if self.current_plot_kind == "target_spectrum":
            element = sanitize_filename(self.current_plot_columns[0] if self.current_plot_columns else "target")
            return f"{self.build_target_plot_base_name()}_{element}_target_spectrum_plot.png"

        if self.current_file is None and self.current_plot_kind not in {
            "multi_spectral",
            "single_device_spectrum",
            "multi_device_compare",
            "multi_device_overlay",
            "aligned_time_series",
            "aligned_scatter",
            "dual_psd_compare",
            "single_device_compare_psd",
            "aligned_diff",
            "aligned_ratio",
            "aligned_cospectrum",
            "aligned_quadrature",
        }:
            return "spectral_plot.png"

        base_name = sanitize_filename(self.current_file.stem) if self.current_file is not None else "current_plot"
        if self.current_plot_kind == "spectral":
            return f"{base_name}_spectral{time_range_filename_suffix}_plot.png"

        if self.current_plot_kind == "cross" and len(self.current_plot_columns) == 2:
            col1 = sanitize_filename(self.current_plot_columns[0])
            col2 = sanitize_filename(self.current_plot_columns[1])
            return f"{base_name}_cross_{col1}_vs_{col2}_plot.png"

        if self.current_plot_kind in {"multi_spectral", "single_device_spectrum", "multi_device_compare", "multi_device_overlay"} and self.current_plot_columns:
            col = sanitize_filename(self.current_plot_columns[0])
            if self.current_plot_kind == "single_device_spectrum" or len(self.current_compare_files) == 1:
                device_name = sanitize_filename(self.current_compare_files[0]) or "single_device"
                return f"single_device_spectrum_{device_name}_{col}{time_range_filename_suffix}_plot.png"
            if self.current_plot_kind == "multi_device_compare":
                return f"multi_device_compare_{col}{time_range_filename_suffix}_plot.png"
            return f"multi_device_overlay_{col}{time_range_filename_suffix}_plot.png"

        if self.current_plot_kind in {
            "aligned_time_series",
            "aligned_scatter",
            "dual_psd_compare",
            "single_device_compare_psd",
            "aligned_diff",
            "aligned_ratio",
            "aligned_cospectrum",
            "aligned_quadrature",
        }:
            kind_name = {
                "aligned_time_series": "time_compare",
                "aligned_scatter": "scatter_compare",
                "dual_psd_compare": "psd_compare",
                "single_device_compare_psd": "psd_compare",
                "aligned_diff": "difference_compare",
                "aligned_ratio": "ratio_compare",
                "aligned_cospectrum": "cospectrum_compare",
                "aligned_quadrature": "quadrature_compare",
            }[self.current_plot_kind]
            return (
                f"{kind_name}_{self.build_compare_filename_core()}"
                f"{time_range_filename_suffix if self.current_plot_kind in {'dual_psd_compare', 'single_device_compare_psd'} else ''}_plot.png"
            )

        return "spectral_plot.png"

    def build_default_data_filename(self) -> str:
        time_range_filename_suffix = core.build_time_range_filename_suffix(self.current_result_frame)
        if self.current_data_source_kind == "comparison_analysis":
            base_name = self.build_comparison_base_name()
            if self.current_plot_kind == "spectral" and self.current_plot_columns:
                col = sanitize_filename(self.current_plot_columns[0])
                return f"{base_name}_{col}_spectral{time_range_filename_suffix}_data.csv"
            if self.current_plot_kind == "cross" and len(self.current_plot_columns) == 2:
                col1 = sanitize_filename(self.current_plot_columns[0])
                col2 = sanitize_filename(self.current_plot_columns[1])
                return f"{base_name}_{col1}_vs_{col2}_cross_data.csv"

        if self.current_plot_kind == "target_spectrum":
            element = sanitize_filename(self.current_plot_columns[0] if self.current_plot_columns else "target")
            return f"{self.build_target_plot_base_name()}_{element}_target_spectrum_data.csv"

        if self.current_file is None and self.current_plot_kind not in {
            "multi_spectral",
            "single_device_spectrum",
            "multi_device_compare",
            "multi_device_overlay",
            "dual_psd_compare",
            "single_device_compare_psd",
            "aligned_cospectrum",
            "aligned_quadrature",
        }:
            return "spectral_data.csv"

        base_name = sanitize_filename(self.current_file.stem) if self.current_file is not None else "current_plot"
        if self.current_plot_kind == "spectral":
            return f"{base_name}_spectral{time_range_filename_suffix}_data.csv"

        if self.current_plot_kind == "cross" and len(self.current_plot_columns) == 2:
            col1 = sanitize_filename(self.current_plot_columns[0])
            col2 = sanitize_filename(self.current_plot_columns[1])
            return f"{base_name}_cross_{col1}_vs_{col2}_data.csv"

        if self.current_plot_kind in {"multi_spectral", "single_device_spectrum", "multi_device_compare", "multi_device_overlay"} and self.current_plot_columns:
            col = sanitize_filename(self.current_plot_columns[0])
            if self.current_plot_kind == "single_device_spectrum" or len(self.current_compare_files) == 1:
                device_name = sanitize_filename(self.current_compare_files[0]) or "single_device"
                return f"single_device_spectrum_{device_name}_{col}{time_range_filename_suffix}_data.csv"
            if self.current_plot_kind == "multi_device_compare":
                return f"multi_device_compare_{col}{time_range_filename_suffix}_data.csv"
            return f"multi_device_overlay_{col}{time_range_filename_suffix}_data.csv"

        if self.current_plot_kind in {"dual_psd_compare", "single_device_compare_psd"}:
            return f"psd_compare_{self.build_compare_filename_core()}{time_range_filename_suffix}_data.csv"

        if self.current_plot_kind == "aligned_cospectrum":
            return f"cospectrum_compare_{self.build_compare_filename_core()}_data.csv"

        if self.current_plot_kind == "aligned_quadrature":
            return f"quadrature_compare_{self.build_compare_filename_core()}_data.csv"

        return "spectral_data.csv"

    def build_default_aligned_filename(self) -> str:
        return f"aligned_{self.build_compare_filename_core()}_data.csv"


def main() -> None:
    matplotlib.use("TkAgg")
    global FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _NavigationToolbar2Tk

    FigureCanvasTkAgg = _FigureCanvasTkAgg
    NavigationToolbar2Tk = _NavigationToolbar2Tk
    root = tk.Tk()
    app = FileViewerApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()

