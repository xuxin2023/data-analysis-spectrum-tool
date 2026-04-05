from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from matplotlib import mlab
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.signal as signal


DEFAULT_FS = 10.0
DEFAULT_NSEGMENT = 256
DEFAULT_OVERLAP_RATIO = 0.5
TARGET_SPECTRAL_ANALYSIS_CONTEXT = "target_spectral"
TARGET_SPECTRAL_SOURCE_MODE = "ygas+dat"
LEGACY_TARGET_NSEGMENT_CANDIDATES = (4096, 2048, 1024, 512)
LEGACY_TARGET_PSD_KERNEL_EXACT_WELCH = "exact_legacy_welch"
LEGACY_TARGET_PSD_KERNEL_DEFAULT = LEGACY_TARGET_PSD_KERNEL_EXACT_WELCH
LEGACY_TARGET_PSD_KERNEL_AUTO = "legacy_candidate_best"
LEGACY_TARGET_PSD_KERNEL_WELCH_UI = "welch_ui_requested"
LEGACY_TARGET_PSD_KERNEL_WELCH_4096 = "welch_4096_hann_50p"
LEGACY_TARGET_PSD_KERNEL_WELCH_FULL = "welch_full_hann_0p"
LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_BOXCAR = "periodogram_boxcar_false"
LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_HANN = "periodogram_hann_constant"
LEGACY_TARGET_PSD_KERNEL_MANUAL_RFFT = "manual_rfft_boxcar_false"
LEGACY_TARGET_PSD_KERNEL_CANDIDATES = (
    LEGACY_TARGET_PSD_KERNEL_WELCH_4096,
    LEGACY_TARGET_PSD_KERNEL_WELCH_FULL,
    LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_BOXCAR,
    LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_HANN,
    LEGACY_TARGET_PSD_KERNEL_MANUAL_RFFT,
)
LEGACY_TARGET_PSD_KERNEL_CHOICES = (
    LEGACY_TARGET_PSD_KERNEL_DEFAULT,
    LEGACY_TARGET_PSD_KERNEL_AUTO,
    *LEGACY_TARGET_PSD_KERNEL_CANDIDATES,
)
LEGACY_TARGET_REFERENCE_FIRST_FREQ_HZ = 0.00244140625
LEGACY_TARGET_REFERENCE_FIRST_FREQ_RANGE = (0.002, 0.005)

TIMESTAMP_COL = "时间戳"
INDEX_COL = "索引列"
STATUS_CHAR_COL = "状态字符"
STATUS_REGISTER_COL = "状态寄存器"
CHECKSUM_COL = "校验和"

TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$")

CROSS_SPECTRUM_MAGNITUDE = "互谱幅值 |Pxy|"
CROSS_SPECTRUM_REAL = "协谱 Re(Pxy)"
CROSS_SPECTRUM_IMAG = "正交谱 Im(Pxy)"
CROSS_SPECTRUM_OPTIONS = [
    CROSS_SPECTRUM_MAGNITUDE,
    CROSS_SPECTRUM_REAL,
    CROSS_SPECTRUM_IMAG,
]
CROSS_DISPLAY_SEMANTICS_ABS = "abs_pxy"
CROSS_DISPLAY_SEMANTICS_REAL = "real_pxy"
CROSS_DISPLAY_SEMANTICS_IMAG = "imag_pxy"
LEGACY_TARGET_SPECTRUM_MODE_PSD = "PSD 对比"
LEGACY_TARGET_SPECTRUM_MODE_CROSS_MAGNITUDE = "互谱幅值"
LEGACY_TARGET_SPECTRUM_MODE_COSPECTRUM = "协谱"
LEGACY_TARGET_SPECTRUM_MODE_QUADRATURE = "正交谱"
LEGACY_TARGET_SPECTRUM_MODE_CHOICES = (
    LEGACY_TARGET_SPECTRUM_MODE_PSD,
    LEGACY_TARGET_SPECTRUM_MODE_CROSS_MAGNITUDE,
    LEGACY_TARGET_SPECTRUM_MODE_COSPECTRUM,
    LEGACY_TARGET_SPECTRUM_MODE_QUADRATURE,
)
LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE = "最近邻 + 容差"
LEGACY_TARGET_ALIGNMENT_STRATEGY_ROUND_100MS_STRICT = "round(0.1s) + strict inner join"
LEGACY_TARGET_CROSS_KERNEL_DEFAULT = "csd_hann"
FRR_COMPATIBLE_CROSS_KERNEL_ID = "FRR_scipy_hann_csd_default"
SCIPY_DEFAULT_DIAGNOSTIC_VALUE = "scipy_default"
MLAB_DEFAULT_DIAGNOSTIC_VALUE = "mlab_default"
MANUAL_DEFAULT_DIAGNOSTIC_VALUE = "manual"
GENERIC_DEFAULT_CROSS_IMPLEMENTATION_ID = "B"
GENERIC_DEFAULT_CROSS_IMPLEMENTATION_LABEL = "Generic default scipy.csd wrapper"
TARGET_COSPECTRUM_IMPLEMENTATION_ID = "A"
GENERIC_SAME_FRAME_ALIGNMENT_STRATEGY = "same_dataframe_row_alignment"
TARGET_COSPECTRUM_DIAGNOSTIC_NOTES = (
    "当前工程里未发现 FRR_Sp_Alz_bete4 源码本体。",
    "当前只是对候选协谱 kernel 做对比排查。",
)
TARGET_COSPECTRUM_KERNEL_SELECTED_ID = "A"
TIME_RANGE_DIFFERENCE_HINT = "当前图形差异仅可能来自时间窗策略不同，请优先查看 time_range_policy 与 base_actual_time_range。"
TIME_RANGE_METADATA_EXPORT_KEYS = (
    "base_requested_start",
    "base_requested_end",
    "base_actual_start",
    "base_actual_end",
    "base_requested_time_range",
    "base_actual_time_range",
    "time_range_policy",
    "time_range_policy_label",
    "time_range_policy_note",
    "time_range_short_label",
    "time_range_file_tag",
    "time_range_difference_hint",
)
_TIME_RANGE_POLICY_SHORT_LABELS = {
    "full_series_no_timestamp": "无时间轴",
    "full_file": "全文件",
    "user_input_window": "用户时间窗",
    "manual_input_window": "用户时间窗",
    "explicit_window": "用户时间窗",
    "dat_window": "dat时间窗",
    "txt_dat_common_window": "txt+dat共同时间窗",
    "recent_10m": "最近10分钟",
    "recent_30m": "最近30分钟",
    "recent_1h": "最近1小时",
}
_TIME_RANGE_POLICY_FILE_TAGS = {
    "full_series_no_timestamp": "no_timestamp",
    "full_file": "full_file",
    "user_input_window": "user_window",
    "manual_input_window": "user_window",
    "explicit_window": "user_window",
    "dat_window": "dat_window",
    "txt_dat_common_window": "txt_dat_common_window",
    "recent_10m": "recent_10m",
    "recent_30m": "recent_30m",
    "recent_1h": "recent_1h",
}
_TIME_RANGE_LABEL_SHORT_LABELS = {
    "全数据行（无时间轴）": "无时间轴",
    "全文件时间范围": "全文件",
    "用户输入时间窗": "用户时间窗",
    "dat 时间范围": "dat时间窗",
    "txt+dat 共同时间范围": "txt+dat共同时间窗",
}
_CSD_DEFAULT_DETREND = "__default__"
TARGET_COSPECTRUM_KERNEL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "A",
        "candidate_name": "A_hann_default_density",
        "window_name": "hann",
        "detrend": _CSD_DEFAULT_DETREND,
        "scaling": "density",
        "description": "当前实现：hann + scipy 默认 detrend/scaling",
        "is_current_implementation": True,
    },
    {
        "candidate_id": "B",
        "candidate_name": "B_hann_constant_density",
        "window_name": "hann",
        "detrend": "constant",
        "scaling": "density",
        "description": "hann + detrend='constant' + scaling='density'",
        "is_current_implementation": False,
    },
    {
        "candidate_id": "C",
        "candidate_name": "C_hann_false_density",
        "window_name": "hann",
        "detrend": False,
        "scaling": "density",
        "description": "hann + detrend=False + scaling='density'",
        "is_current_implementation": False,
    },
    {
        "candidate_id": "D",
        "candidate_name": "D_hann_false_spectrum",
        "window_name": "hann",
        "detrend": False,
        "scaling": "spectrum",
        "description": "hann + detrend=False + scaling='spectrum'",
        "is_current_implementation": False,
    },
    {
        "candidate_id": "E",
        "candidate_name": "E_boxcar_false_density",
        "window_name": "boxcar",
        "detrend": False,
        "scaling": "density",
        "description": "boxcar + detrend=False + scaling='density'",
        "is_current_implementation": False,
    },
    {
        "candidate_id": "F",
        "candidate_name": "F_boxcar_false_spectrum",
        "window_name": "boxcar",
        "detrend": False,
        "scaling": "spectrum",
        "description": "boxcar + detrend=False + scaling='spectrum'",
        "is_current_implementation": False,
    },
)
TARGET_COSPECTRUM_ALIGNMENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "G",
        "alignment_strategy": LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
        "description": "当前实现：merge_asof(nearest, tolerance=auto)",
        "is_current_alignment": True,
    },
    {
        "candidate_id": "H",
        "alignment_strategy": LEGACY_TARGET_ALIGNMENT_STRATEGY_ROUND_100MS_STRICT,
        "description": "双方时间戳 round 到 0.1s 后 strict inner join",
        "is_current_alignment": False,
    },
)
TARGET_COSPECTRUM_IMPLEMENTATIONS: tuple[dict[str, Any], ...] = (
    {
        "implementation_id": "A",
        "implementation_label": "Current target helper scipy.csd hann defaults",
        "description": "signal.csd + hann + fs/window/nperseg/noverlap only",
        "cross_kernel": FRR_COMPATIBLE_CROSS_KERNEL_ID,
    },
    {
        "implementation_id": "B",
        "implementation_label": "Generic default compute_cross_spectrum wrapper",
        "description": "compute_cross_spectrum_from_arrays_with_params default behavior",
        "cross_kernel": GENERIC_DEFAULT_CROSS_IMPLEMENTATION_LABEL,
    },
    {
        "implementation_id": "C",
        "implementation_label": "SciPy two-sided positive half",
        "description": "signal.csd(return_onesided=False) and keep positive half only",
        "cross_kernel": "scipy_csd_two_sided_positive_half",
    },
    {
        "implementation_id": "D",
        "implementation_label": "SciPy two-sided manual one-sided x2",
        "description": "signal.csd(return_onesided=False) with manual one-sided mapping",
        "cross_kernel": "scipy_csd_two_sided_manual_one_sided",
    },
    {
        "implementation_id": "E",
        "implementation_label": "mlab.csd hanning nooverlap scale_by_freq",
        "description": "matplotlib.mlab.csd noverlap=0 scale_by_freq=True",
        "cross_kernel": "mlab_csd_hanning_nooverlap_scale_by_freq",
    },
    {
        "implementation_id": "F",
        "implementation_label": "mlab.csd hanning overlap scale_by_freq",
        "description": "matplotlib.mlab.csd noverlap=round(nperseg*overlap_ratio) scale_by_freq=True",
        "cross_kernel": "mlab_csd_hanning_overlap_scale_by_freq",
    },
    {
        "implementation_id": "G",
        "implementation_label": "mlab.csd hanning overlap no_freq_scale",
        "description": "matplotlib.mlab.csd noverlap=round(nperseg*overlap_ratio) scale_by_freq=False",
        "cross_kernel": "mlab_csd_hanning_overlap_no_freq_scale",
    },
    {
        "implementation_id": "H",
        "implementation_label": "manual FFT hann overlap raw",
        "description": "manual segmented FFT hann 50% overlap raw conj(X)*Y average",
        "cross_kernel": "manual_fft_hann_overlap_raw",
    },
    {
        "implementation_id": "I",
        "implementation_label": "manual FFT hann overlap density",
        "description": "manual segmented FFT hann 50% overlap density normalized",
        "cross_kernel": "manual_fft_hann_overlap_density",
    },
    {
        "implementation_id": "J",
        "implementation_label": "manual FFT boxcar overlap raw",
        "description": "manual segmented FFT boxcar 50% overlap raw conj(X)*Y average",
        "cross_kernel": "manual_fft_boxcar_overlap_raw",
    },
)

MODE1_COLUMN_TEMPLATE = [
    TIMESTAMP_COL,
    "通道1",
    "通道2",
    "通道3",
    "通道4",
    STATUS_CHAR_COL,
    "设备号",
    "CO2浓度",
    "H2O浓度",
    "CO2信号强度",
    "H2O信号强度",
    "温度",
    "压力",
    STATUS_REGISTER_COL,
    CHECKSUM_COL,
]

ELEMENT_PRESETS: dict[str, dict[str, list[str]]] = {
    "CO2": {
        "A": ["CO2浓度"],
        "B": ["CO2", "CO2_Avg", "co2", "co2_avg", "CO2浓度"],
    },
    "H2O": {
        "A": ["H2O浓度"],
        "B": ["H2O", "H2O_Avg", "h2o", "h2o_avg", "H2O浓度"],
    },
    "温度": {
        "A": ["温度"],
        "B": ["TA", "AirTC", "温度", "temperature", "Temperature"],
    },
    "压力": {
        "A": ["压力"],
        "B": ["PA", "PRESSURE", "压力", "pressure", "Pressure"],
    },
}


REFERENCE_COLUMN_PRESETS: dict[str, list[str]] = {
    "A": ["Uz", "uz", "W", "w", "通道3"],
    "B": ["Uz", "uz", "W", "w", "通道3"],
}

DEVICE_IDENTIFIER_COLUMN_CANDIDATES: tuple[str, ...] = (
    "device_id",
    "deviceid",
    "设备编号",
    "设备号",
    "sn",
    "serial_no",
    "serialno",
    "serial",
    "序列号",
)
DEVICE_FILENAME_TOKEN_BLACKLIST: set[str] = {
    "FILE",
    "FILES",
    "DATA",
    "SERIES",
    "TIME",
    "TIMESTAMP",
    "WINDOW",
    "RESULT",
    "OUTPUT",
    "EXPORT",
    "PLOT",
    "SPECTRUM",
    "PSD",
    "CROSS",
    "COMPARE",
    "ANALYSIS",
}


@dataclass
class ParsedDataResult:
    dataframe: pd.DataFrame
    profile_name: str
    timestamp_col: str | None
    suggested_columns: list[str]
    available_columns: list[str]
    source_row_count: int = 0
    timestamp_valid_count: int = 0
    timestamp_valid_ratio: float = 0.0
    timestamp_warning: str | None = None


def _unique_non_empty_strings(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    if not values:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def normalize_device_identifier(value: Any) -> str | None:
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None
    normalized = re.sub(r"\s+", "", text)
    if not normalized or normalized.lower() in {"nan", "none", "nat", "null"}:
        return None
    return normalized[:80]


def _normalize_identifier_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def extract_device_identifier_from_dataframe(df: pd.DataFrame) -> tuple[str | None, str | None]:
    normalized_candidates = {_normalize_identifier_column_name(name) for name in DEVICE_IDENTIFIER_COLUMN_CANDIDATES}
    for column in df.columns:
        if _normalize_identifier_column_name(column) not in normalized_candidates:
            continue
        series = df[column]
        values = [normalize_device_identifier(value) for value in series.tolist()]
        values = [value for value in values if value]
        if not values:
            continue
        counts = pd.Series(values, dtype="string").value_counts()
        if counts.empty:
            continue
        return str(counts.index[0]), str(column)
    return None, None


def infer_device_identifier_from_path(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return path.name

    prefixed_token = re.search(
        r"(?i)(?:^|[_\-.])((?:device|dev|sn|serial|ser)[A-Za-z0-9]{1,32})(?:[_\-.]|$)",
        stem,
    )
    if prefixed_token:
        explicit_token = normalize_device_identifier(prefixed_token.group(1))
        if explicit_token:
            return explicit_token

    explicit_match = re.search(
        r"(?i)(?:^|[_\-.])(?:device|dev|sn|serial|ser)[_\- ]*([A-Za-z0-9]{2,32})",
        stem,
    )
    if explicit_match:
        explicit = normalize_device_identifier(explicit_match.group(1))
        if explicit:
            return explicit

    leading_alpha = re.match(r"([A-Za-z]{2,16})", stem)
    if leading_alpha:
        prefix = leading_alpha.group(1)
        suffix = stem[len(prefix) :]
        if suffix and re.fullmatch(r"[\d_\-.]+", suffix):
            normalized_prefix = normalize_device_identifier(prefix)
            if normalized_prefix:
                return normalized_prefix

    for token in re.findall(r"[A-Za-z0-9]+", stem):
        normalized = normalize_device_identifier(token)
        if not normalized:
            continue
        upper = normalized.upper()
        if upper in DEVICE_FILENAME_TOKEN_BLACKLIST:
            continue
        if normalized.isdigit():
            if len(normalized) >= 8:
                continue
            if len(normalized) == 4 and 1900 <= int(normalized) <= 2100:
                continue
        return normalized

    return stem


def resolve_device_identifier(parsed: ParsedDataResult, path: Path) -> dict[str, str]:
    device_id, source_column = extract_device_identifier_from_dataframe(parsed.dataframe)
    if device_id:
        return {
            "device_id": device_id,
            "device_label": device_id,
            "device_source": f"column:{source_column}",
        }
    inferred = infer_device_identifier_from_path(path)
    return {
        "device_id": inferred,
        "device_label": inferred or path.stem,
        "device_source": "filename",
    }


def resolve_merged_profile_name(parsed_results: list[ParsedDataResult]) -> str:
    profiles = {str(parsed.profile_name) for parsed in parsed_results if str(parsed.profile_name).strip()}
    if profiles and all(profile.startswith("YGAS_MODE1") or profile == "YGAS_MERGED" for profile in profiles):
        return "YGAS_MERGED"
    if profiles and all(profile in {"TOA5_DAT", "TOA5_MERGED"} for profile in profiles):
        return "TOA5_MERGED"
    if len(profiles) == 1:
        return f"{next(iter(profiles))}_MERGED"
    return "GENERIC_MERGED"


def merge_parsed_results_for_device_group(
    parsed_results: list[ParsedDataResult],
    *,
    source_paths: list[Path] | None = None,
    device_id: str | None = None,
) -> tuple[ParsedDataResult, dict[str, Any]]:
    if not parsed_results:
        raise ValueError("没有可用于合并的设备文件。")

    raw_row_count = int(sum(int(parsed.source_row_count or len(parsed.dataframe)) for parsed in parsed_results))
    suggested_order: list[str] = []
    available_order: list[str] = []
    timestamp_merge_ready = True
    timestamp_frames: list[pd.DataFrame] = []
    merge_strategy = "row_concat"

    for parsed in parsed_results:
        for column in [*parsed.suggested_columns, *parsed.available_columns]:
            text = str(column).strip()
            if not text:
                continue
            if text not in suggested_order:
                suggested_order.append(text)
            if text not in available_order:
                available_order.append(text)

        timestamp_col = parsed.timestamp_col
        if timestamp_col is None or timestamp_col not in parsed.dataframe.columns:
            timestamp_merge_ready = False
            continue

        frame = parsed.dataframe.copy()
        frame[TIMESTAMP_COL] = parse_mixed_timestamp_series(frame[timestamp_col])
        frame = frame.dropna(subset=[TIMESTAMP_COL])
        if frame.empty:
            continue
        timestamp_frames.append(frame)

    if timestamp_merge_ready and timestamp_frames:
        merged = pd.concat(timestamp_frames, ignore_index=True, sort=False)
        merged = merged.sort_values(TIMESTAMP_COL).drop_duplicates(TIMESTAMP_COL, keep="first").reset_index(drop=True)
        timestamp_col = TIMESTAMP_COL
        merge_strategy = "timestamp_sorted_concat_dedup"
    else:
        merged = pd.concat([parsed.dataframe.copy() for parsed in parsed_results], ignore_index=True, sort=False)
        first_timestamp_col = parsed_results[0].timestamp_col
        timestamp_col = first_timestamp_col if first_timestamp_col in merged.columns else None

    merged_parsed = build_parsed_result_from_dataframe(
        merged,
        resolve_merged_profile_name(parsed_results),
        timestamp_col,
        source_row_count=raw_row_count,
    )
    ordered_suggested = [column for column in suggested_order if column in merged_parsed.available_columns]
    if ordered_suggested:
        merged_parsed.suggested_columns = prioritize_suggested_columns(
            ordered_suggested + [column for column in merged_parsed.suggested_columns if column not in ordered_suggested]
        )

    metadata = {
        "device_id": str(device_id or ""),
        "file_count": len(parsed_results),
        "source_paths": [str(path) for path in source_paths] if source_paths else [],
        "merge_strategy": merge_strategy,
        "profile_name": merged_parsed.profile_name,
        "timestamp_col": merged_parsed.timestamp_col,
        "raw_rows": raw_row_count,
        "merged_points": int(len(merged_parsed.dataframe)),
    }
    return merged_parsed, metadata


def resolve_profile_aware_fs(
    parsed: ParsedDataResult,
    ui_fs: float,
    *,
    timestamp_frame: pd.DataFrame | None = None,
) -> tuple[float, str]:
    manual_override = not math.isclose(float(ui_fs), DEFAULT_FS, rel_tol=0.0, abs_tol=1e-9)
    if manual_override:
        return float(ui_fs), "ui_override"

    profile_name = str(parsed.profile_name or "")
    if profile_name.startswith("YGAS_MODE1") or profile_name == "YGAS_MERGED":
        return float(DEFAULT_FS), "ygas_profile_default"

    estimated = None
    if timestamp_frame is not None and TIMESTAMP_COL in timestamp_frame.columns:
        estimated = estimate_fs_from_timestamp(timestamp_frame, TIMESTAMP_COL)
    if estimated is None:
        estimated = estimate_fs_from_timestamp(parsed.dataframe, parsed.timestamp_col)
    if estimated is not None:
        return float(estimated), "timestamp_estimate"
    return float(ui_fs), "ui_default_fallback"


def prepare_base_spectrum_series(
    parsed: ParsedDataResult,
    value_column: str,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
    *,
    require_timestamp: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if value_column not in parsed.dataframe.columns:
        raise ValueError(f"目标列不存在: {value_column}")

    original_count = int(len(parsed.dataframe))
    source_row_count = int(parsed.source_row_count or original_count)
    has_timestamp = bool(parsed.timestamp_col and parsed.timestamp_col in parsed.dataframe.columns)
    if require_timestamp and not has_timestamp:
        raise ValueError("未识别到时间列。")

    if has_timestamp:
        filtered = filter_by_time_range(parsed.dataframe, str(parsed.timestamp_col), start_dt, end_dt)
        time_filtered_count = int(len(filtered))
        source_frame = pd.DataFrame(
            {
                TIMESTAMP_COL: parse_mixed_timestamp_series(filtered[str(parsed.timestamp_col)]),
                "value": pd.to_numeric(filtered[value_column], errors="coerce"),
            }
        )
        source_frame = source_frame.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL)
        source_frame = source_frame.drop_duplicates(TIMESTAMP_COL, keep="first").reset_index(drop=True)
        if source_frame.empty:
            raise ValueError("时间窗内没有有效时间戳。")

        source_start = pd.Timestamp(source_frame[TIMESTAMP_COL].min())
        source_end = pd.Timestamp(source_frame[TIMESTAMP_COL].max())
        raw_points = int(len(source_frame))
        valid_mask = source_frame["value"].notna()
        valid_count = int(valid_mask.sum())
        non_null_ratio = float(valid_count / raw_points) if raw_points else 0.0
        first_valid_start = pd.Timestamp(source_frame.loc[valid_mask, TIMESTAMP_COL].iloc[0]) if valid_count else None
        last_valid_end = pd.Timestamp(source_frame.loc[valid_mask, TIMESTAMP_COL].iloc[-1]) if valid_count else None
        leading_invalid_gap_s = (
            float(max((first_valid_start - source_start).total_seconds(), 0.0))
            if first_valid_start is not None
            else None
        )
        trailing_invalid_gap_s = (
            float(max((source_end - last_valid_end).total_seconds(), 0.0))
            if last_valid_end is not None
            else None
        )

        frame = source_frame.dropna(subset=["value"]).reset_index(drop=True)
        if len(frame) < 2:
            raise ValueError("时间窗内有效数据点不足。")
        if float(frame["value"].std(ddof=0)) <= 1e-12:
            raise ValueError("时间窗内数据波动过小。")
        if int(frame["value"].nunique()) <= 1:
            raise ValueError("时间窗内数据近似恒定。")

        requested_start = pd.Timestamp(start_dt) if start_dt is not None else source_start
        requested_end = pd.Timestamp(end_dt) if end_dt is not None else source_end
        requested_duration = max((requested_end - requested_start).total_seconds(), 0.0)
        actual_start = pd.Timestamp(frame[TIMESTAMP_COL].min())
        actual_end = pd.Timestamp(frame[TIMESTAMP_COL].max())
        actual_duration = max((actual_end - actual_start).total_seconds(), 0.0)
        coverage_ratio = 1.0 if requested_duration <= 0 else min(actual_duration / requested_duration, 1.0)
        return frame, {
            "valid_points": int(len(frame)),
            "source_start": source_start,
            "source_end": source_end,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "first_valid_start": first_valid_start,
            "last_valid_end": last_valid_end,
            "leading_invalid_gap_s": leading_invalid_gap_s,
            "trailing_invalid_gap_s": trailing_invalid_gap_s,
            "non_null_ratio": non_null_ratio,
            "requested_start": requested_start,
            "requested_end": requested_end,
            "requested_duration_s": float(requested_duration),
            "actual_duration_s": float(actual_duration),
            "coverage_ratio": float(coverage_ratio),
            "original_count": original_count,
            "time_filtered_count": time_filtered_count,
            "source_row_count": source_row_count,
            "timestamp_valid_count": int(parsed.timestamp_valid_count),
            "timestamp_valid_ratio": float(parsed.timestamp_valid_ratio),
            "timestamp_warning": parsed.timestamp_warning,
            "has_timestamp": True,
        }

    numeric_series = pd.to_numeric(parsed.dataframe[value_column], errors="coerce")
    time_filtered_count = int(len(numeric_series))
    frame = pd.DataFrame({"value": numeric_series}).dropna().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("有效数据点不足。")
    if float(frame["value"].std(ddof=0)) <= 1e-12:
        raise ValueError("数据波动过小。")
    if int(frame["value"].nunique()) <= 1:
        raise ValueError("数据近似恒定。")
    return frame, {
        "valid_points": int(len(frame)),
        "source_start": None,
        "source_end": None,
        "actual_start": None,
        "actual_end": None,
        "first_valid_start": None,
        "last_valid_end": None,
        "leading_invalid_gap_s": None,
        "trailing_invalid_gap_s": None,
        "non_null_ratio": float(len(frame) / time_filtered_count) if time_filtered_count else 0.0,
        "requested_start": pd.Timestamp(start_dt) if start_dt is not None else None,
        "requested_end": pd.Timestamp(end_dt) if end_dt is not None else None,
        "requested_duration_s": None,
        "actual_duration_s": None,
        "coverage_ratio": None,
        "original_count": original_count,
        "time_filtered_count": time_filtered_count,
        "source_row_count": source_row_count,
        "timestamp_valid_count": int(parsed.timestamp_valid_count),
        "timestamp_valid_ratio": float(parsed.timestamp_valid_ratio),
        "timestamp_warning": parsed.timestamp_warning,
        "has_timestamp": False,
    }


def compute_base_spectrum_payload(
    parsed: ParsedDataResult,
    value_column: str,
    *,
    fs_ui: float,
    requested_nsegment: int,
    overlap_ratio: float,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
    require_timestamp: bool = False,
) -> dict[str, Any]:
    frame, series_meta = prepare_base_spectrum_series(
        parsed,
        value_column,
        start_dt=start_dt,
        end_dt=end_dt,
        require_timestamp=require_timestamp,
    )
    effective_fs, fs_source = resolve_profile_aware_fs(parsed, fs_ui, timestamp_frame=frame)
    freq, density, details = compute_psd_from_array_with_params(
        frame["value"].to_numpy(dtype=float),
        float(effective_fs),
        requested_nsegment,
        overlap_ratio,
    )
    positive = (freq > 0) & np.isfinite(freq) & np.isfinite(density) & (density > 0)
    freq_plot = np.asarray(freq[positive], dtype=float)
    density_plot = np.asarray(density[positive], dtype=float)
    if len(freq_plot) == 0:
        raise ValueError("当前数据没有可用于绘图的有效谱值。")

    details = dict(details)
    details.update(
        {
            "base_spectrum_builder": "shared_profile_based",
            "base_value_column": str(value_column),
            "base_profile_name": str(parsed.profile_name),
            "base_has_timestamp": bool(series_meta.get("has_timestamp")),
            "base_requested_start": series_meta.get("requested_start"),
            "base_requested_end": series_meta.get("requested_end"),
            "base_actual_start": series_meta.get("actual_start"),
            "base_actual_end": series_meta.get("actual_end"),
            "base_coverage_ratio": series_meta.get("coverage_ratio"),
            "base_valid_points": int(series_meta.get("valid_points", 0)),
            "base_original_count": int(series_meta.get("original_count", 0)),
            "base_time_filtered_count": int(series_meta.get("time_filtered_count", 0)),
            "base_fs_source": fs_source,
            "effective_fs": float(effective_fs),
        }
    )
    return {
        "freq": freq_plot,
        "density": density_plot,
        "details": details,
        "series_frame": frame,
        "series_meta": series_meta,
        "effective_fs": float(effective_fs),
    }


def format_metadata_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    return str(value)


def build_time_range_text(
    start: Any,
    end: Any,
    *,
    empty_start: str,
    empty_end: str,
) -> str:
    start_text = format_metadata_timestamp(start) or empty_start
    end_text = format_metadata_timestamp(end) or empty_end
    return f"{start_text} ~ {end_text}"


def resolve_time_range_policy_short_label(policy: Any, label: Any = None) -> str:
    policy_text = str(policy or "").strip()
    if policy_text in _TIME_RANGE_POLICY_SHORT_LABELS:
        return _TIME_RANGE_POLICY_SHORT_LABELS[policy_text]
    label_text = str(label or "").strip()
    return _TIME_RANGE_LABEL_SHORT_LABELS.get(label_text, label_text or "默认时间窗")


def resolve_time_range_policy_file_tag(policy: Any, label: Any = None) -> str:
    policy_text = str(policy or "").strip()
    if policy_text in _TIME_RANGE_POLICY_FILE_TAGS:
        return _TIME_RANGE_POLICY_FILE_TAGS[policy_text]
    short_label = resolve_time_range_policy_short_label(policy_text, label)
    return re.sub(r"[^a-z0-9_]+", "_", short_label.lower().replace("+", "_plus_")).strip("_")


def resolve_single_time_range_policy(
    *,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    has_timestamp: bool,
) -> dict[str, Any]:
    if not has_timestamp:
        return {
            "time_range_policy": "full_series_no_timestamp",
            "time_range_policy_label": "全数据行（无时间轴）",
            "time_range_policy_note": None,
        }
    if start_dt is None and end_dt is None:
        return {
            "time_range_policy": "full_file",
            "time_range_policy_label": "全文件时间范围",
            "time_range_policy_note": "注意：当前单图使用全文件；双设备 PSD 对比默认使用共同时间窗，因此图形可能不同。",
        }
    return {
        "time_range_policy": "user_input_window",
        "time_range_policy_label": "用户输入时间窗",
        "time_range_policy_note": None,
    }


def resolve_compare_time_range_policy(
    *,
    strategy_label: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    has_txt_dat_context: bool,
) -> dict[str, Any]:
    strategy = str(strategy_label or "").strip()
    mapping = {
        "手动输入时间范围": ("manual_input_window", "用户输入时间窗"),
        "使用 dat 时间范围": ("dat_window", "dat 时间范围"),
        "最近10分钟": ("recent_10m", "最近10分钟"),
        "最近30分钟": ("recent_30m", "最近30分钟"),
        "最近1小时": ("recent_1h", "最近1小时"),
        "使用 txt+dat 共同时间范围": ("txt_dat_common_window", "txt+dat 共同时间范围"),
    }
    if strategy in mapping:
        policy, label = mapping[strategy]
    elif has_txt_dat_context:
        policy, label = ("txt_dat_common_window", "txt+dat 共同时间范围")
    elif start_dt is None and end_dt is None:
        policy, label = ("full_file", "全文件时间范围")
    else:
        policy, label = ("explicit_window", strategy or "显式时间窗")
    note = None
    if policy != "full_file":
        note = f"注意：当前双设备 PSD 对比使用{label}；若单文件/单设备图谱使用全文件，图形可能不同。"
    return {
        "time_range_policy": policy,
        "time_range_policy_label": label,
        "time_range_policy_note": note,
    }


def build_time_range_metadata(
    *,
    time_range_policy: str,
    time_range_policy_label: str,
    time_range_policy_note: str | None,
    requested_start: Any,
    requested_end: Any,
    actual_start: Any,
    actual_end: Any,
) -> dict[str, Any]:
    metadata = {
        "time_range_policy": str(time_range_policy or "").strip(),
        "time_range_policy_label": str(time_range_policy_label or "").strip(),
        "time_range_policy_note": str(time_range_policy_note).strip() if time_range_policy_note else None,
        "base_requested_start": requested_start,
        "base_requested_end": requested_end,
        "base_actual_start": actual_start,
        "base_actual_end": actual_end,
    }
    metadata["base_requested_time_range"] = build_time_range_text(
        metadata["base_requested_start"],
        metadata["base_requested_end"],
        empty_start="自动",
        empty_end="自动",
    )
    metadata["base_actual_time_range"] = build_time_range_text(
        metadata["base_actual_start"],
        metadata["base_actual_end"],
        empty_start="无",
        empty_end="无",
    )
    metadata["time_range_short_label"] = resolve_time_range_policy_short_label(
        metadata["time_range_policy"],
        metadata["time_range_policy_label"],
    )
    metadata["time_range_file_tag"] = resolve_time_range_policy_file_tag(
        metadata["time_range_policy"],
        metadata["time_range_policy_label"],
    )
    metadata["time_range_difference_hint"] = TIME_RANGE_DIFFERENCE_HINT
    return metadata


def build_single_time_range_metadata(
    *,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    has_timestamp: bool,
    requested_start: Any,
    requested_end: Any,
    actual_start: Any,
    actual_end: Any,
) -> dict[str, Any]:
    policy_meta = resolve_single_time_range_policy(
        start_dt=start_dt,
        end_dt=end_dt,
        has_timestamp=has_timestamp,
    )
    return build_time_range_metadata(
        time_range_policy=str(policy_meta["time_range_policy"]),
        time_range_policy_label=str(policy_meta["time_range_policy_label"]),
        time_range_policy_note=policy_meta.get("time_range_policy_note"),
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
    )


def build_compare_time_range_metadata(
    *,
    strategy_label: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    has_txt_dat_context: bool,
    requested_start: Any,
    requested_end: Any,
    actual_start: Any,
    actual_end: Any,
) -> dict[str, Any]:
    policy_meta = resolve_compare_time_range_policy(
        strategy_label=strategy_label,
        start_dt=start_dt,
        end_dt=end_dt,
        has_txt_dat_context=has_txt_dat_context,
    )
    return build_time_range_metadata(
        time_range_policy=str(policy_meta["time_range_policy"]),
        time_range_policy_label=str(policy_meta["time_range_policy_label"]),
        time_range_policy_note=policy_meta.get("time_range_policy_note"),
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
    )


def extract_time_range_metadata(source: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(source, dict):
        for key in TIME_RANGE_METADATA_EXPORT_KEYS:
            if key in source and source.get(key) is not None:
                metadata[key] = source.get(key)
        return metadata
    if isinstance(source, pd.DataFrame) and not source.empty:
        first_row = source.iloc[0]
        for key in TIME_RANGE_METADATA_EXPORT_KEYS:
            if key in source.columns:
                value = first_row[key]
                if value is not None and not pd.isna(value):
                    metadata[key] = value
    return metadata


def build_time_range_display_suffix(source: Any) -> str:
    metadata = extract_time_range_metadata(source)
    short_label = str(metadata.get("time_range_short_label") or "").strip()
    if not short_label:
        short_label = resolve_time_range_policy_short_label(
            metadata.get("time_range_policy"),
            metadata.get("time_range_policy_label"),
        )
    return f" [{short_label}]" if short_label else ""


def build_time_range_filename_suffix(source: Any) -> str:
    metadata = extract_time_range_metadata(source)
    tag = str(metadata.get("time_range_file_tag") or "").strip()
    if not tag:
        tag = resolve_time_range_policy_file_tag(
            metadata.get("time_range_policy"),
            metadata.get("time_range_policy_label"),
        )
    return f"_{tag}" if tag else ""


def build_time_range_diagnostic_items(source: Any) -> list[str]:
    metadata = extract_time_range_metadata(source)
    items: list[str] = []
    if metadata.get("time_range_policy") is not None:
        items.append(f"time_range_policy={metadata.get('time_range_policy')}")
    if metadata.get("time_range_policy_label") is not None:
        items.append(f"time_range_policy_label={metadata.get('time_range_policy_label')}")
    requested_text = metadata.get("base_requested_time_range")
    if requested_text:
        items.append(f"base_requested_time_range={requested_text}")
    actual_text = metadata.get("base_actual_time_range")
    if actual_text:
        items.append(f"base_actual_time_range={actual_text}")
    note = metadata.get("time_range_policy_note")
    if note:
        items.append(str(note))
    hint = metadata.get("time_range_difference_hint")
    if hint:
        items.append(str(hint))
    return items


def build_target_summary_columns(
    *,
    reference_column: str,
    ygas_target_column: str,
    dat_target_column: str,
) -> dict[str, str]:
    reference_name = str(reference_column).strip()
    ygas_target_name = str(ygas_target_column).strip()
    dat_target_name = str(dat_target_column).strip()
    return {
        "summary_timestamp_column": TIMESTAMP_COL,
        "summary_target_column": f"A_{ygas_target_name}",
        "summary_ygas_target_column": f"A_{ygas_target_name}",
        "summary_dat_target_column": f"B_{dat_target_name}",
        "summary_reference_column": f"B_{reference_name}",
        "summary_matched_timestamp_column": "B_匹配时间戳",
        "summary_match_delta_column": "匹配时间差_s",
    }


def build_target_canonical_pair_specs(
    *,
    reference_column: str,
    ygas_target_column: str,
    dat_target_column: str,
    display_target_label: str,
) -> list[dict[str, Any]]:
    summary_columns = build_target_summary_columns(
        reference_column=reference_column,
        ygas_target_column=ygas_target_column,
        dat_target_column=dat_target_column,
    )
    reference_name = str(reference_column).strip()
    ygas_target_name = str(ygas_target_column).strip()
    dat_target_name = str(dat_target_column).strip()
    display_target = str(display_target_label or ygas_target_name).strip() or ygas_target_name
    summary_reference = str(summary_columns["summary_reference_column"])
    ygas_summary_target = str(summary_columns["summary_ygas_target_column"])
    dat_summary_target = str(summary_columns["summary_dat_target_column"])
    return [
        {
            "series_role": "ygas_target_vs_uz",
            "device_kind": "cross_ygas",
            "source_kind": "ygas",
            "reference_column": reference_name,
            "target_column": ygas_target_name,
            "summary_reference_column": summary_reference,
            "summary_target_column": ygas_summary_target,
            "cross_order": f"{reference_name} -> {ygas_target_name}",
            "summary_cross_order": f"{summary_reference} -> {ygas_summary_target}",
            "display_label": f"{display_target}(ygas) vs {reference_name}",
        },
        {
            "series_role": "dat_target_vs_uz",
            "device_kind": "cross_dat",
            "source_kind": "dat",
            "reference_column": reference_name,
            "target_column": dat_target_name,
            "summary_reference_column": summary_reference,
            "summary_target_column": dat_summary_target,
            "cross_order": f"{reference_name} -> {dat_target_name}",
            "summary_cross_order": f"{summary_reference} -> {dat_summary_target}",
            "display_label": f"{display_target}(dat) vs {reference_name}",
        },
    ]


def build_target_spectral_context(
    *,
    target_element: str,
    reference_column: str,
    ygas_target_column: str,
    dat_target_column: str,
    spectrum_mode: str | None,
    source_mode: str = TARGET_SPECTRAL_SOURCE_MODE,
    analysis_context: str = TARGET_SPECTRAL_ANALYSIS_CONTEXT,
    comparison_is_target_spectral: bool = False,
    display_target_label: str | None = None,
) -> dict[str, Any]:
    reference_name = str(reference_column).strip()
    ygas_target_name = str(ygas_target_column).strip()
    dat_target_name = str(dat_target_column).strip()
    display_target = str(display_target_label or target_element or ygas_target_name).strip() or ygas_target_name
    pair_specs = build_target_canonical_pair_specs(
        reference_column=reference_name,
        ygas_target_column=ygas_target_name,
        dat_target_column=dat_target_name,
        display_target_label=display_target,
    )
    context = {
        "target_element": str(target_element).strip() or display_target,
        "reference_column": reference_name,
        "target_column": ygas_target_name,
        "ygas_target_column": ygas_target_name,
        "dat_target_column": dat_target_name,
        "display_target_label": display_target,
        "cross_order": f"{reference_name} -> {display_target}",
        "canonical_cross_pairs": [f"{item['source_kind']}: {item['summary_cross_order']}" for item in pair_specs],
        "canonical_pair_specs": pair_specs,
        "analysis_context": analysis_context,
        "source_mode": source_mode,
        "spectrum_mode": spectrum_mode,
        "is_target_spectral_context": True,
        "comparison_is_target_spectral": bool(comparison_is_target_spectral),
    }
    context.update(
        build_target_summary_columns(
            reference_column=reference_name,
            ygas_target_column=ygas_target_name,
            dat_target_column=dat_target_name,
        )
    )
    return context


def get_target_spectral_context(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    nested = metadata.get("target_spectral_context")
    if isinstance(nested, dict) and nested.get("analysis_context") == TARGET_SPECTRAL_ANALYSIS_CONTEXT:
        return dict(nested)
    if metadata.get("analysis_context") != TARGET_SPECTRAL_ANALYSIS_CONTEXT and not metadata.get(
        "is_target_spectral_context"
    ):
        return None
    keys = {
        "target_element",
        "reference_column",
        "target_column",
        "ygas_target_column",
        "dat_target_column",
        "display_target_label",
        "cross_order",
        "canonical_cross_pairs",
        "canonical_pair_specs",
        "analysis_context",
        "source_mode",
        "spectrum_mode",
        "is_target_spectral_context",
        "comparison_is_target_spectral",
        "summary_timestamp_column",
        "summary_target_column",
        "summary_ygas_target_column",
        "summary_dat_target_column",
        "summary_reference_column",
        "summary_matched_timestamp_column",
        "summary_match_delta_column",
    }
    context = {key: metadata.get(key) for key in keys if key in metadata}
    return context or None


def resolve_target_cross_pair(
    selected_columns: list[str] | tuple[str, ...],
    target_context: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_columns = _unique_non_empty_strings(list(selected_columns))
    context = get_target_spectral_context(target_context)
    if context is None:
        return {
            "selected_columns": normalized_columns,
            "ordered_columns": normalized_columns,
            "uses_canonical_pair": False,
            "auto_reordered": False,
            "warning": None,
            "context": None,
        }

    ordered_columns = list(normalized_columns)
    warning: str | None = None
    uses_canonical_pair = False
    auto_reordered = False
    resolved_pair: dict[str, Any] | None = None
    canonical_pairs: list[dict[str, Any]] = [
        dict(item)
        for item in (context.get("canonical_pair_specs") or [])
        if isinstance(item, dict)
    ]
    if not canonical_pairs:
        canonical_reference = str(
            context.get("summary_reference_column") or context.get("reference_column") or ""
        ).strip()
        canonical_target = str(
            context.get("summary_target_column") or context.get("target_column") or ""
        ).strip()
        if canonical_reference and canonical_target:
            canonical_pairs = [
                {
                    "series_role": "ygas_target_vs_uz",
                    "device_kind": "cross_ygas",
                    "summary_reference_column": canonical_reference,
                    "summary_target_column": canonical_target,
                    "summary_cross_order": context.get("cross_order", f"{canonical_reference} -> {canonical_target}"),
                    "reference_column": context.get("reference_column"),
                    "target_column": context.get("target_column"),
                }
            ]

    if len(normalized_columns) == 2 and canonical_pairs:
        for pair in canonical_pairs:
            canonical_reference = str(pair.get("summary_reference_column") or "").strip()
            canonical_target = str(pair.get("summary_target_column") or "").strip()
            if not canonical_reference or not canonical_target:
                continue
            if normalized_columns == [canonical_reference, canonical_target]:
                ordered_columns = [canonical_reference, canonical_target]
                uses_canonical_pair = True
                resolved_pair = dict(pair)
                break
            if set(normalized_columns) == {canonical_reference, canonical_target}:
                ordered_columns = [canonical_reference, canonical_target]
                uses_canonical_pair = True
                auto_reordered = normalized_columns != ordered_columns
                resolved_pair = dict(pair)
                break
        if not uses_canonical_pair:
            pair_labels = [
                str(item.get("summary_cross_order") or "").strip()
                for item in canonical_pairs
                if str(item.get("summary_cross_order") or "").strip()
            ]
            warning = (
                "当前汇总表属于目标频谱上下文，"
                f"canonical pairs 包括 {'；'.join(pair_labels) if pair_labels else 'B_Uz -> A_target / B_target'}。"
                "本次将按普通双列互谱分析处理。"
            )

    return {
        "selected_columns": normalized_columns,
        "ordered_columns": ordered_columns,
        "uses_canonical_pair": uses_canonical_pair,
        "auto_reordered": auto_reordered,
        "warning": warning,
        "context": context,
        "resolved_pair": resolved_pair,
    }


def build_generated_column_names(count: int) -> list[str]:
    if count <= 0:
        return []
    names = [TIMESTAMP_COL]
    for index in range(1, count):
        names.append(f"数据列{index:02d}")
    return names[:count]


def parse_mixed_timestamp_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.Series(series, index=series.index, copy=True)
        valid_count = int(parsed.notna().sum())
        parsed.attrs["timestamp_parse_diagnostics"] = {
            "non_empty_input_count": valid_count,
            "mixed_valid_count": valid_count,
            "plain_valid_count": valid_count,
            "elementwise_valid_count": valid_count,
            "selected_strategy": "already-datetime",
            "plain_executed": False,
            "elementwise_executed": False,
            "elementwise_sampled": False,
            "fallback_reason": "input_already_datetime",
            "mixed_error": None,
            "plain_error": None,
            "elementwise_error": None,
            "elementwise_sample_valid_count": None,
            "elementwise_sample_error": None,
        }
        return parsed

    normalized = pd.Series(series, index=series.index, dtype="string").str.strip()
    invalid_mask = normalized.isna() | normalized.str.lower().isin({"", "nan", "none", "nat"})
    normalized = normalized.mask(invalid_mask, pd.NA)
    non_empty_input_count = int(normalized.notna().sum())
    quality_threshold = 0.995
    small_series_elementwise_limit = 20_000
    sample_chunk_size = 200

    def _safe_parse(
        values: pd.Series,
        *,
        strategy: str,
    ) -> tuple[pd.Series, str | None]:
        parse_error: str | None = None
        try:
            if strategy == "mixed":
                parsed_candidate = pd.to_datetime(values, errors="coerce", format="mixed")
            elif strategy == "plain":
                parsed_candidate = pd.to_datetime(values, errors="coerce")
            elif strategy == "elementwise":
                parsed_candidate = pd.Series(
                    [
                        pd.to_datetime(value, errors="coerce") if pd.notna(value) else pd.NaT
                        for value in values.tolist()
                    ],
                    index=values.index,
                )
            else:
                raise ValueError(f"Unsupported timestamp parse strategy: {strategy}")
        except (TypeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            parsed_candidate = pd.Series([pd.NaT] * len(values), index=values.index, dtype="datetime64[ns]")
        parsed_candidate = pd.Series(parsed_candidate, index=values.index)
        return pd.to_datetime(parsed_candidate, errors="coerce"), parse_error

    def _build_probe_sample(values: pd.Series) -> pd.Series:
        if len(values) <= sample_chunk_size * 3:
            return values.copy()
        middle_start = max(0, (len(values) // 2) - (sample_chunk_size // 2))
        sample = pd.concat(
            [
                values.iloc[:sample_chunk_size],
                values.iloc[middle_start : middle_start + sample_chunk_size],
                values.iloc[-sample_chunk_size:],
            ]
        )
        return sample[~sample.index.duplicated(keep="first")]

    parsed_mixed, mixed_error = _safe_parse(normalized, strategy="mixed")
    mixed_valid_count = int(parsed_mixed.notna().sum())
    plain_valid_count: int | None = None
    elementwise_valid_count: int | None = None
    plain_error: str | None = None
    elementwise_error: str | None = None
    elementwise_sample_valid_count: int | None = None
    elementwise_sample_error: str | None = None
    selected_strategy = "mixed"
    parsed = parsed_mixed
    plain_executed = False
    elementwise_executed = False
    elementwise_sampled = False
    fallback_reason = "mixed_selected_by_default"

    mixed_valid_ratio = mixed_valid_count / non_empty_input_count if non_empty_input_count else 0.0
    if non_empty_input_count == 0:
        fallback_reason = "no_non_empty_input"
    elif mixed_valid_count == non_empty_input_count:
        fallback_reason = "mixed_complete"
    elif mixed_valid_ratio >= quality_threshold:
        fallback_reason = "mixed_ratio_threshold"
    else:
        parsed_plain, plain_error = _safe_parse(normalized, strategy="plain")
        plain_valid_count = int(parsed_plain.notna().sum())
        plain_executed = True
        plain_valid_ratio = plain_valid_count / non_empty_input_count if non_empty_input_count else 0.0

        if plain_valid_count == non_empty_input_count:
            selected_strategy = "plain"
            parsed = parsed_plain
            fallback_reason = "plain_complete"
        elif plain_valid_count > mixed_valid_count:
            selected_strategy = "plain"
            parsed = parsed_plain
            fallback_reason = "plain_improves_over_mixed"
        elif plain_valid_ratio >= quality_threshold:
            selected_strategy = "plain"
            parsed = parsed_plain
            fallback_reason = "plain_ratio_threshold"
        else:
            best_baseline_strategy = "mixed"
            best_baseline_parsed = parsed_mixed
            best_baseline_valid_count = mixed_valid_count
            if plain_valid_count > mixed_valid_count:
                best_baseline_strategy = "plain"
                best_baseline_parsed = parsed_plain
                best_baseline_valid_count = plain_valid_count

            if non_empty_input_count <= small_series_elementwise_limit:
                parsed_elementwise, elementwise_error = _safe_parse(normalized, strategy="elementwise")
                elementwise_valid_count = int(parsed_elementwise.notna().sum())
                elementwise_executed = True
                if elementwise_valid_count > best_baseline_valid_count:
                    selected_strategy = "elementwise"
                    parsed = parsed_elementwise
                    fallback_reason = "elementwise_full_improves_over_best_baseline"
                else:
                    selected_strategy = best_baseline_strategy
                    parsed = best_baseline_parsed
                    fallback_reason = f"elementwise_full_no_gain_keep_{best_baseline_strategy}"
            else:
                probe_values = _build_probe_sample(normalized)
                probe_non_empty_count = int(probe_values.notna().sum())
                elementwise_sampled = True
                if probe_non_empty_count > 0:
                    parsed_elementwise_probe, elementwise_sample_error = _safe_parse(
                        probe_values,
                        strategy="elementwise",
                    )
                    elementwise_sample_valid_count = int(parsed_elementwise_probe.notna().sum())
                    mixed_probe_valid_count = int(parsed_mixed.loc[probe_values.index].notna().sum())
                    plain_probe_valid_count = (
                        int(parsed_plain.loc[probe_values.index].notna().sum()) if plain_executed else 0
                    )
                    best_probe_valid_count = max(mixed_probe_valid_count, plain_probe_valid_count)
                    sample_gain = elementwise_sample_valid_count - best_probe_valid_count
                    sample_valid_ratio = (
                        elementwise_sample_valid_count / probe_non_empty_count if probe_non_empty_count else 0.0
                    )
                    should_run_full_elementwise = sample_gain > 0 and (
                        sample_valid_ratio >= quality_threshold
                        or sample_gain >= max(3, int(math.ceil(probe_non_empty_count * 0.01)))
                    )
                    if should_run_full_elementwise:
                        parsed_elementwise, elementwise_error = _safe_parse(normalized, strategy="elementwise")
                        elementwise_valid_count = int(parsed_elementwise.notna().sum())
                        elementwise_executed = True
                        if elementwise_valid_count > best_baseline_valid_count:
                            selected_strategy = "elementwise"
                            parsed = parsed_elementwise
                            fallback_reason = "elementwise_full_after_sample_improves_over_best_baseline"
                        else:
                            selected_strategy = best_baseline_strategy
                            parsed = best_baseline_parsed
                            fallback_reason = f"elementwise_full_after_sample_no_gain_keep_{best_baseline_strategy}"
                    else:
                        selected_strategy = best_baseline_strategy
                        parsed = best_baseline_parsed
                        fallback_reason = f"large_series_sample_no_clear_elementwise_gain_keep_{best_baseline_strategy}"
                else:
                    selected_strategy = best_baseline_strategy
                    parsed = best_baseline_parsed
                    fallback_reason = f"large_series_probe_empty_keep_{best_baseline_strategy}"

    parsed = pd.Series(pd.to_datetime(parsed, errors="coerce"), index=series.index)
    parsed.attrs["timestamp_parse_diagnostics"] = {
        "non_empty_input_count": non_empty_input_count,
        "mixed_valid_count": mixed_valid_count,
        "plain_valid_count": plain_valid_count,
        "elementwise_valid_count": elementwise_valid_count,
        "selected_strategy": selected_strategy,
        "plain_executed": plain_executed,
        "elementwise_executed": elementwise_executed,
        "elementwise_sampled": elementwise_sampled,
        "fallback_reason": fallback_reason,
        "mixed_error": mixed_error,
        "plain_error": plain_error,
        "elementwise_error": elementwise_error,
        "elementwise_sample_valid_count": elementwise_sample_valid_count,
        "elementwise_sample_error": elementwise_sample_error,
    }
    return parsed


def build_timestamp_parse_stats(source: pd.Series, parsed: pd.Series) -> dict[str, Any]:
    row_count = int(len(source))
    valid_count = int(parsed.notna().sum())
    valid_ratio = float(valid_count / row_count) if row_count else 0.0
    warning = None
    if row_count > 0 and valid_ratio < 0.95:
        warning = (
            "时间戳解析有效率偏低，请检查文件格式或时间戳解析规则"
            f"（{valid_count}/{row_count}, {valid_ratio:.1%}）"
        )
    return {
        "row_count": row_count,
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "warning": warning,
    }


def read_preview_lines(path: Path, max_lines: int = 8) -> list[str]:
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            lines: list[str] = []
            with path.open("r", encoding=encoding, errors="ignore") as handle:
                for line in handle:
                    lines.append(line.rstrip("\r\n"))
                    if len(lines) >= max_lines:
                        break
            if lines:
                return lines
        except OSError:
            return []
    return []


def looks_like_timestamp_series(series: pd.Series) -> bool:
    values = [str(value).strip() for value in series.tolist() if pd.notna(value) and str(value).strip()]
    sample = values[:10]
    return len(sample) >= 3 and all(TIMESTAMP_PATTERN.match(value) for value in sample)


def looks_like_incremental_index(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    sample = numeric.iloc[:10]
    if len(sample) < 3:
        return False
    values = sample.to_numpy(dtype=float)
    if not np.allclose(values, np.round(values)):
        return False
    diffs = np.diff(values)
    return np.allclose(diffs, 1.0)


def detect_file_profile(path: Path, preview_lines: list[str]) -> str:
    cleaned = [line.strip() for line in preview_lines if line.strip()]
    upper_lines = [line.upper() for line in cleaned]
    if cleaned:
        first_fields = [field.strip().strip('"') for field in cleaned[0].split(",")]
        if len(first_fields) == 15 and first_fields and TIMESTAMP_PATTERN.match(first_fields[0]):
            return "YGAS_MODE1_15"

    if len(cleaned) >= 3:
        rows = [[field.strip().strip('"') for field in line.split(",")] for line in cleaned[:5]]
        if rows and len(rows[0]) == 16:
            first_col = pd.Series([row[0] if len(row) > 0 else None for row in rows])
            second_col = pd.Series([row[1] if len(row) > 1 else None for row in rows])
            if looks_like_incremental_index(first_col) and looks_like_timestamp_series(second_col):
                return "YGAS_MODE1_16"

    if upper_lines and "TOA5" in upper_lines[0] and any("TIMESTAMP" in line for line in upper_lines[:4]):
        return "TOA5_DAT"

    return "GENERIC_TABLE"


def filter_by_time_range(
    df: pd.DataFrame,
    timestamp_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        return df.iloc[0:0].copy()

    mask = pd.Series(True, index=df.index)
    if start_dt is not None:
        mask &= df[timestamp_col] >= start_dt
    if end_dt is not None:
        mask &= df[timestamp_col] <= end_dt
    return df.loc[mask].copy()


def estimate_fs_from_timestamp(df: pd.DataFrame, timestamp_col: str | None) -> float | None:
    if not timestamp_col or timestamp_col not in df.columns:
        return None
    timestamps = parse_mixed_timestamp_series(df[timestamp_col]).dropna().sort_values()
    if len(timestamps) < 3:
        return None
    deltas = timestamps.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None
    median_dt = float(deltas.median())
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def guess_timestamp_column(df: pd.DataFrame) -> str | None:
    preferred_names = [
        "时间戳",
        "TIMESTAMP",
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "datetime",
        "DateTime",
    ]
    for name in preferred_names:
        if name in df.columns:
            parsed = parse_mixed_timestamp_series(df[name])
            if parsed.notna().sum() >= 3:
                return str(name)

    for col in df.columns[: min(len(df.columns), 3)]:
        parsed = parse_mixed_timestamp_series(df[col])
        if parsed.notna().sum() >= max(3, int(len(df) * 0.6)):
            return str(col)
    return None


def get_profile_excluded_columns(
    profile_name: str,
    df: pd.DataFrame,
    timestamp_col: str | None,
) -> set[str]:
    excluded: set[str] = set()
    if timestamp_col and timestamp_col in df.columns:
        excluded.add(timestamp_col)
    if profile_name == "YGAS_MODE1_16" and INDEX_COL in df.columns:
        excluded.add(INDEX_COL)
    if profile_name.startswith("YGAS_MODE1") or profile_name == "YGAS_MERGED":
        for name in (STATUS_CHAR_COL, STATUS_REGISTER_COL, CHECKSUM_COL):
            if name in df.columns:
                excluded.add(name)
    return excluded


def classify_numeric_columns(
    df: pd.DataFrame,
    excluded_columns: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], set[str], set[str]]:
    column_data: dict[str, np.ndarray] = {}
    non_numeric_cols: set[str] = set()
    unsuitable_spectrum_cols: set[str] = set()
    excluded_columns = excluded_columns or set()

    for col in df.columns:
        if col in excluded_columns:
            unsuitable_spectrum_cols.add(str(col))
            continue
        numeric_data = pd.to_numeric(df[col], errors="coerce")
        total_count = len(numeric_data)
        valid_numeric = numeric_data.dropna()
        valid_count = len(valid_numeric)
        valid_ratio = 0.0 if total_count == 0 else valid_count / total_count
        if valid_ratio >= 0.8:
            std = float(valid_numeric.std(ddof=0)) if valid_count > 0 else 0.0
            nunique = int(valid_numeric.nunique())
            if valid_count < 16 or std <= 1e-12 or nunique <= 1:
                unsuitable_spectrum_cols.add(str(col))
            else:
                column_data[str(col)] = numeric_data.to_numpy(dtype=float)
        else:
            non_numeric_cols.add(str(col))

    return column_data, non_numeric_cols, unsuitable_spectrum_cols


def prioritize_suggested_columns(columns: list[str]) -> list[str]:
    priority = ["CO2浓度", "H2O浓度"]
    ordered = [name for name in priority if name in columns]
    ordered.extend(name for name in columns if name not in ordered)
    return ordered


def build_parsed_result_from_dataframe(
    df: pd.DataFrame,
    profile_name: str,
    timestamp_col: str | None,
    *,
    source_row_count: int | None = None,
    analyze_numeric_columns: bool = True,
    suggested_columns_override: list[str] | None = None,
) -> ParsedDataResult:
    parsed_df = df.copy()
    row_count = int(source_row_count if source_row_count is not None else len(parsed_df))
    timestamp_valid_count = 0
    timestamp_valid_ratio = 0.0
    timestamp_warning: str | None = None

    if timestamp_col and timestamp_col in parsed_df.columns:
        parsed_df[timestamp_col] = parse_mixed_timestamp_series(parsed_df[timestamp_col])
        timestamp_stats = build_timestamp_parse_stats(df[timestamp_col], parsed_df[timestamp_col])
        timestamp_valid_count = int(timestamp_stats["valid_count"])
        timestamp_valid_ratio = float(timestamp_stats["valid_ratio"])
        timestamp_warning = timestamp_stats["warning"]

    excluded = get_profile_excluded_columns(profile_name, parsed_df, timestamp_col)
    if analyze_numeric_columns:
        numeric_columns, _non_numeric, _unsuitable = classify_numeric_columns(parsed_df, excluded_columns=excluded)
        suggested = prioritize_suggested_columns(list(numeric_columns.keys()))
    else:
        suggested = prioritize_suggested_columns([str(col) for col in parsed_df.columns if str(col) not in excluded])
    if suggested_columns_override is not None:
        suggested = [str(col) for col in suggested_columns_override]
    return ParsedDataResult(
        dataframe=parsed_df,
        profile_name=profile_name,
        timestamp_col=timestamp_col,
        suggested_columns=suggested,
        available_columns=list(parsed_df.columns),
        source_row_count=row_count,
        timestamp_valid_count=timestamp_valid_count,
        timestamp_valid_ratio=timestamp_valid_ratio,
        timestamp_warning=timestamp_warning,
    )


def parse_ygas_mode1_file(path: Path) -> ParsedDataResult:
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=0,
        header=None,
        engine="c",
        on_bad_lines="skip",
        dtype=str,
    )
    profile_name = detect_file_profile(path, read_preview_lines(path))

    if profile_name == "YGAS_MODE1_16" and len(df.columns) >= len(MODE1_COLUMN_TEMPLATE) + 1:
        df = df.iloc[:, : len(MODE1_COLUMN_TEMPLATE) + 1].copy()
        df.columns = [INDEX_COL, *MODE1_COLUMN_TEMPLATE]
    else:
        df = df.iloc[:, : len(MODE1_COLUMN_TEMPLATE)].copy()
        df.columns = list(MODE1_COLUMN_TEMPLATE)
        profile_name = "YGAS_MODE1_15"

    return build_parsed_result_from_dataframe(df, profile_name, TIMESTAMP_COL)


def parse_toa5_file(path: Path) -> ParsedDataResult:
    df = pd.read_csv(
        path,
        sep=",",
        header=0,
        skiprows=[0, 2, 3],
        engine="c",
        on_bad_lines="skip",
        dtype=str,
    )
    df.columns = [str(col).strip().strip('"') for col in df.columns]
    timestamp_source = "TIMESTAMP" if "TIMESTAMP" in df.columns else guess_timestamp_column(df)
    if timestamp_source is None:
        raise ValueError("未识别到 TOA5 文件的时间列。")
    if timestamp_source != TIMESTAMP_COL:
        df[TIMESTAMP_COL] = parse_mixed_timestamp_series(df[timestamp_source])
    return build_parsed_result_from_dataframe(df, "TOA5_DAT", TIMESTAMP_COL)


def parse_ygas_mode1_file_fast(
    path: Path,
    *,
    required_columns: list[str] | None = None,
) -> ParsedDataResult:
    profile_name = detect_file_profile(path, read_preview_lines(path))
    if profile_name == "YGAS_MODE1_16":
        available_columns = [INDEX_COL, *MODE1_COLUMN_TEMPLATE]
    else:
        available_columns = list(MODE1_COLUMN_TEMPLATE)
        profile_name = "YGAS_MODE1_15"
    selected_columns = available_columns if required_columns is None else _unique_non_empty_strings(
        [TIMESTAMP_COL, *required_columns]
    )
    selected_columns = [column for column in selected_columns if column in available_columns]
    usecols = [available_columns.index(column) for column in selected_columns] if selected_columns else None
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=0,
        header=None,
        engine="c",
        on_bad_lines="skip",
        dtype=str,
        usecols=usecols,
    )
    if usecols is None:
        df.columns = available_columns[: len(df.columns)]
    else:
        df.columns = selected_columns

    return build_parsed_result_from_dataframe(
        df,
        profile_name,
        TIMESTAMP_COL,
        analyze_numeric_columns=False,
    )


def parse_toa5_file_fast(
    path: Path,
    *,
    required_columns: list[str] | None = None,
) -> ParsedDataResult:
    header_df = pd.read_csv(
        path,
        sep=",",
        header=0,
        skiprows=[0, 2, 3],
        engine="c",
        on_bad_lines="skip",
        dtype=str,
        nrows=0,
    )
    available_columns = [str(col).strip().strip('"') for col in header_df.columns]
    timestamp_source = "TIMESTAMP" if "TIMESTAMP" in available_columns else None
    if timestamp_source is None:
        for name in ("timestamp", "Timestamp", "time", "Time", "datetime", "DateTime"):
            if name in available_columns:
                timestamp_source = name
                break
    if timestamp_source is None:
        raise ValueError("未识别到 TOA5 文件的时间列。")
    selected_columns = list(available_columns)
    if required_columns is not None:
        selected_columns = [timestamp_source]
        for column in required_columns:
            text = str(column).strip()
            if not text:
                continue
            if text == TIMESTAMP_COL:
                text = timestamp_source
            if text in available_columns:
                selected_columns.append(text)
        selected_columns = _unique_non_empty_strings(selected_columns)
    df = pd.read_csv(
        path,
        sep=",",
        header=0,
        skiprows=[0, 2, 3],
        engine="c",
        on_bad_lines="skip",
        dtype=str,
        usecols=selected_columns,
    )
    df.columns = [str(col).strip().strip('"') for col in df.columns]
    if timestamp_source != TIMESTAMP_COL and TIMESTAMP_COL not in df.columns:
        df[TIMESTAMP_COL] = df[timestamp_source]
    return build_parsed_result_from_dataframe(
        df,
        "TOA5_DAT",
        TIMESTAMP_COL,
        analyze_numeric_columns=False,
    )


def load_and_merge_ygas_files_fast(
    paths: list[Path],
    *,
    required_columns: list[str] | None = None,
) -> tuple[ParsedDataResult, dict[str, Any]]:
    if not paths:
        raise ValueError("没有可用于拼接的 txt/log 文件。")

    frames: list[pd.DataFrame] = []
    suggested_columns: list[str] = []
    used_profiles: set[str] = set()
    raw_row_count = 0
    valid_timestamp_count = 0

    for path in paths:
        parsed = parse_ygas_mode1_file_fast(path, required_columns=required_columns)
        raw_row_count += int(parsed.source_row_count or len(parsed.dataframe))
        valid_timestamp_count += int(parsed.timestamp_valid_count)
        if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
            continue
        frame = parsed.dataframe.dropna(subset=[parsed.timestamp_col]).copy()
        if frame.empty:
            continue
        frames.append(frame)
        used_profiles.add(parsed.profile_name)
        if not suggested_columns:
            suggested_columns = list(parsed.suggested_columns)

    if not frames:
        raise ValueError("所选 txt/log 文件中没有可拼接的有效时间序列。")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(TIMESTAMP_COL).drop_duplicates(TIMESTAMP_COL, keep="first").reset_index(drop=True)
    merged_parsed = build_parsed_result_from_dataframe(
        merged,
        "YGAS_MERGED",
        TIMESTAMP_COL,
        source_row_count=raw_row_count,
        analyze_numeric_columns=False,
        suggested_columns_override=suggested_columns or None,
    )
    merged_parsed.timestamp_valid_count = int(valid_timestamp_count)
    merged_parsed.timestamp_valid_ratio = float(valid_timestamp_count / raw_row_count) if raw_row_count else 0.0
    if raw_row_count > 0 and merged_parsed.timestamp_valid_ratio < 0.95:
        merged_parsed.timestamp_warning = (
            "时间戳解析有效率偏低，请检查文件格式或时间戳解析规则"
            f"（{valid_timestamp_count}/{raw_row_count}, {merged_parsed.timestamp_valid_ratio:.1%}）"
        )

    summary = {
        "file_count": len(paths),
        "start": pd.Timestamp(merged[TIMESTAMP_COL].min()),
        "end": pd.Timestamp(merged[TIMESTAMP_COL].max()),
        "total_points": int(len(merged)),
        "raw_rows": int(raw_row_count),
        "valid_timestamp_points": int(valid_timestamp_count),
        "profiles": sorted(used_profiles),
    }
    return merged_parsed, summary


def load_and_merge_ygas_files(paths: list[Path]) -> tuple[ParsedDataResult, dict[str, Any]]:
    if not paths:
        raise ValueError("没有可用于拼接的 txt/log 文件。")

    frames: list[pd.DataFrame] = []
    suggested_columns: list[str] = []
    used_profiles: set[str] = set()
    raw_row_count = 0
    valid_timestamp_count = 0

    for path in paths:
        parsed = parse_ygas_mode1_file(path)
        raw_row_count += int(parsed.source_row_count or len(parsed.dataframe))
        valid_timestamp_count += int(parsed.timestamp_valid_count)
        if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
            continue
        frame = parsed.dataframe.dropna(subset=[parsed.timestamp_col]).copy()
        if frame.empty:
            continue
        frames.append(frame)
        used_profiles.add(parsed.profile_name)
        if not suggested_columns:
            suggested_columns = list(parsed.suggested_columns)

    if not frames:
        raise ValueError("所选 txt/log 文件中没有可拼接的有效时间序列。")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(TIMESTAMP_COL).drop_duplicates(TIMESTAMP_COL, keep="first").reset_index(drop=True)
    merged_parsed = build_parsed_result_from_dataframe(
        merged,
        "YGAS_MERGED",
        TIMESTAMP_COL,
        source_row_count=raw_row_count,
    )
    merged_parsed.timestamp_valid_count = int(valid_timestamp_count)
    merged_parsed.timestamp_valid_ratio = float(valid_timestamp_count / raw_row_count) if raw_row_count else 0.0
    if raw_row_count > 0 and merged_parsed.timestamp_valid_ratio < 0.95:
        merged_parsed.timestamp_warning = (
            "时间戳解析有效率偏低，请检查文件格式或时间戳解析规则"
            f"（{valid_timestamp_count}/{raw_row_count}, {merged_parsed.timestamp_valid_ratio:.1%}）"
        )
    if suggested_columns:
        merged_parsed.suggested_columns = prioritize_suggested_columns(
            [col for col in suggested_columns if col in merged_parsed.suggested_columns]
            + [col for col in merged_parsed.suggested_columns if col not in suggested_columns]
        )

    summary = {
        "file_count": len(paths),
        "start": pd.Timestamp(merged[TIMESTAMP_COL].min()),
        "end": pd.Timestamp(merged[TIMESTAMP_COL].max()),
        "total_points": int(len(merged)),
        "raw_rows": int(raw_row_count),
        "valid_timestamp_points": int(valid_timestamp_count),
        "profiles": sorted(used_profiles),
    }
    return merged_parsed, summary


def _compute_welch_psd_from_array(
    data: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    psd_kernel: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = _prepare_psd_input(data)

    nperseg = min(requested_nsegment, len(normalized))
    noverlap = int(nperseg * overlap_ratio)
    noverlap = min(noverlap, nperseg - 1)
    window = signal.windows.hann(nperseg)

    freq, density = signal.welch(
        normalized,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    details = _build_psd_details(
        freq,
        density,
        valid_points=len(normalized),
        fs=fs,
        requested_nsegment=requested_nsegment,
        effective_nsegment=nperseg,
        overlap_ratio=overlap_ratio,
        noverlap=noverlap,
        psd_kernel=psd_kernel,
        window_type="hann",
        detrend="constant",
        scaling_mode="density",
    )
    return freq, density, details


def compute_psd_from_array_with_params(
    data: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _compute_welch_psd_from_array(
        data,
        fs,
        requested_nsegment,
        overlap_ratio,
        psd_kernel="welch",
    )


def compute_exact_legacy_welch_psd_from_array(
    data: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _compute_welch_psd_from_array(
        data,
        fs,
        requested_nsegment,
        overlap_ratio,
        psd_kernel=LEGACY_TARGET_PSD_KERNEL_EXACT_WELCH,
    )


def _prepare_psd_input(data: np.ndarray) -> np.ndarray:
    normalized = np.asarray(data, dtype=float)
    normalized = normalized[~np.isnan(normalized)]
    if len(normalized) < 2:
        raise ValueError("数据长度不足，无法进行频谱分析。")
    if float(np.std(normalized)) <= 1e-12:
        raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    return normalized


def _build_psd_details(
    freq: np.ndarray,
    density: np.ndarray,
    *,
    valid_points: int,
    fs: float,
    requested_nsegment: int,
    effective_nsegment: int,
    overlap_ratio: float,
    noverlap: int,
    psd_kernel: str,
    window_type: str,
    detrend: str | bool,
    scaling_mode: str,
) -> dict[str, Any]:
    positive_mask = (freq > 0) & np.isfinite(freq) & np.isfinite(density)
    positive_freq_points = int(np.count_nonzero(positive_mask))
    valid_spectrum = positive_mask & (density > 0)
    if not np.any(valid_spectrum):
        raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    valid_freq_points = int(np.count_nonzero(valid_spectrum))
    first_positive_freq = float(freq[valid_spectrum][0]) if valid_freq_points else None
    return {
        "valid_points": int(valid_points),
        "fs": float(fs),
        "effective_fs": float(fs),
        "nsegment": int(requested_nsegment),
        "requested_nsegment": int(requested_nsegment),
        "overlap_ratio": float(overlap_ratio),
        "overlap": int(noverlap),
        "nperseg": int(effective_nsegment),
        "noverlap": int(noverlap),
        "effective_nsegment": int(effective_nsegment),
        "psd_kernel": str(psd_kernel),
        "window_type": str(window_type),
        "detrend": detrend,
        "scaling_mode": str(scaling_mode),
        "frequency_point_count": int(len(freq)),
        "positive_freq_points": positive_freq_points,
        "valid_freq_points": valid_freq_points,
        "first_positive_freq": first_positive_freq,
    }


def _log_spectrum_roughness(freq: np.ndarray, density: np.ndarray) -> float | None:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(density) & (density > 0)
    if int(np.count_nonzero(mask)) < 3:
        return None
    log_density = np.log10(density[mask])
    deltas = np.abs(np.diff(log_density))
    if deltas.size == 0:
        return None
    return float(np.nanmedian(deltas))


def _compute_periodogram_psd_from_array(
    data: np.ndarray,
    fs: float,
    *,
    requested_nsegment: int,
    window_type: str,
    detrend: str | bool,
    psd_kernel: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = _prepare_psd_input(data)
    effective_nsegment = int(len(normalized))
    window = signal.get_window(window_type, effective_nsegment, fftbins=True)
    freq, density = signal.periodogram(
        normalized,
        fs=fs,
        window=window,
        detrend=detrend,
        scaling="density",
        return_onesided=True,
    )
    details = _build_psd_details(
        freq,
        density,
        valid_points=len(normalized),
        fs=fs,
        requested_nsegment=requested_nsegment,
        effective_nsegment=effective_nsegment,
        overlap_ratio=0.0,
        noverlap=0,
        psd_kernel=psd_kernel,
        window_type=window_type,
        detrend=detrend,
        scaling_mode="density",
    )
    return freq, density, details


def _compute_manual_rfft_density_psd(
    data: np.ndarray,
    fs: float,
    *,
    requested_nsegment: int,
    window_type: str,
    detrend: str | bool,
    psd_kernel: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = _prepare_psd_input(data)
    effective_nsegment = int(len(normalized))
    working = normalized.copy()
    if detrend in {True, "constant"}:
        working = signal.detrend(working, type="constant")
    elif detrend not in {False, None}:
        raise ValueError(f"不支持的 detrend 配置: {detrend}")

    window = signal.get_window(window_type, effective_nsegment, fftbins=True)
    windowed = working * window
    fft_vals = np.fft.rfft(windowed, n=effective_nsegment)
    density = (np.abs(fft_vals) ** 2) / (float(fs) * float(np.sum(window ** 2)))
    if effective_nsegment % 2 == 0:
        if density.size > 2:
            density[1:-1] *= 2.0
    elif density.size > 1:
        density[1:] *= 2.0
    freq = np.fft.rfftfreq(effective_nsegment, d=1.0 / float(fs))
    details = _build_psd_details(
        freq,
        density,
        valid_points=len(normalized),
        fs=fs,
        requested_nsegment=requested_nsegment,
        effective_nsegment=effective_nsegment,
        overlap_ratio=0.0,
        noverlap=0,
        psd_kernel=psd_kernel,
        window_type=window_type,
        detrend=detrend,
        scaling_mode="density",
    )
    details["manual_density_formula"] = "Pxx = |rfft(x*w)|^2 / (fs * sum(w^2)); one-sided interior bins doubled"
    return freq, density, details


def compute_legacy_target_psd_from_array(
    data: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    psd_kernel: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_EXACT_WELCH:
        return compute_exact_legacy_welch_psd_from_array(
            data,
            fs,
            requested_nsegment,
            overlap_ratio,
        )
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_WELCH_4096:
        effective_nsegment = legacy_target_nsegment_resolver(len(_prepare_psd_input(data)))
        freq, density, details = compute_psd_from_array_with_params(
            data,
            fs,
            effective_nsegment,
            overlap_ratio,
        )
        details = dict(details)
        details["psd_kernel"] = psd_kernel
        return freq, density, details
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_WELCH_FULL:
        normalized = _prepare_psd_input(data)
        effective_nsegment = int(len(normalized))
        window = signal.windows.hann(effective_nsegment)
        freq, density = signal.welch(
            normalized,
            fs=fs,
            window=window,
            nperseg=effective_nsegment,
            noverlap=0,
            detrend="constant",
            scaling="density",
        )
        details = _build_psd_details(
            freq,
            density,
            valid_points=len(normalized),
            fs=fs,
            requested_nsegment=requested_nsegment,
            effective_nsegment=effective_nsegment,
            overlap_ratio=0.0,
            noverlap=0,
            psd_kernel=psd_kernel,
            window_type="hann",
            detrend="constant",
            scaling_mode="density",
        )
        return freq, density, details
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_BOXCAR:
        return _compute_periodogram_psd_from_array(
            data,
            fs,
            requested_nsegment=requested_nsegment,
            window_type="boxcar",
            detrend=False,
            psd_kernel=psd_kernel,
        )
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_PERIODOGRAM_HANN:
        return _compute_periodogram_psd_from_array(
            data,
            fs,
            requested_nsegment=requested_nsegment,
            window_type="hann",
            detrend="constant",
            psd_kernel=psd_kernel,
        )
    if psd_kernel == LEGACY_TARGET_PSD_KERNEL_MANUAL_RFFT:
        return _compute_manual_rfft_density_psd(
            data,
            fs,
            requested_nsegment=requested_nsegment,
            window_type="boxcar",
            detrend=False,
            psd_kernel=psd_kernel,
        )
    raise ValueError(f"未识别的 legacy-target 频谱核: {psd_kernel}")


def _score_legacy_target_candidate(
    *,
    median_first_positive_freq: float | None,
    median_positive_freq_points: float | None,
    median_roughness: float | None,
    failure_count: int,
) -> float:
    score = float(max(failure_count, 0) * 1000)
    if median_first_positive_freq is None or median_first_positive_freq <= 0:
        score += 500.0
    else:
        reference = LEGACY_TARGET_REFERENCE_FIRST_FREQ_HZ
        low, high = LEGACY_TARGET_REFERENCE_FIRST_FREQ_RANGE
        score += abs(math.log10(median_first_positive_freq / reference)) * 100.0
        if median_first_positive_freq < low:
            score += ((low - median_first_positive_freq) / low) * 35.0
        elif median_first_positive_freq > high:
            score += ((median_first_positive_freq - high) / high) * 35.0
    if median_positive_freq_points is not None and median_positive_freq_points > 0:
        score -= min(max(math.log2(median_positive_freq_points / 2048.0), 0.0), 3.0) * 5.0
    if median_roughness is not None:
        score += median_roughness * 12.0
    return float(score)


def summarize_legacy_target_psd_candidates(
    prepared_groups: list[dict[str, Any]],
    *,
    requested_nsegment: int,
    overlap_ratio: float,
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    low, high = LEGACY_TARGET_REFERENCE_FIRST_FREQ_RANGE
    for kernel_name in LEGACY_TARGET_PSD_KERNEL_CANDIDATES:
        first_positive_freqs: list[float] = []
        positive_freq_points: list[int] = []
        valid_freq_points: list[int] = []
        roughness_values: list[float] = []
        errors: list[str] = []
        for group in prepared_groups:
            for device_kind, values_key, fs_key in (
                ("ygas", "ygas_values", "ygas_fs"),
                ("dat", "dat_values", "dat_fs"),
            ):
                try:
                    freq, density, details = compute_legacy_target_psd_from_array(
                        group[values_key],
                        float(group[fs_key]),
                        requested_nsegment,
                        overlap_ratio,
                        psd_kernel=kernel_name,
                    )
                    if details.get("first_positive_freq") is not None:
                        first_positive_freqs.append(float(details["first_positive_freq"]))
                    positive_freq_points.append(int(details.get("positive_freq_points", 0)))
                    valid_freq_points.append(int(details.get("valid_freq_points", 0)))
                    roughness = _log_spectrum_roughness(freq, density)
                    if roughness is not None:
                        roughness_values.append(roughness)
                except Exception as exc:
                    errors.append(f"{group.get('group_label', group.get('group_key', device_kind))}/{device_kind}: {exc}")
        median_first_positive = (
            float(np.median(first_positive_freqs)) if first_positive_freqs else None
        )
        median_positive_points = (
            float(np.median(positive_freq_points)) if positive_freq_points else None
        )
        median_valid_points = float(np.median(valid_freq_points)) if valid_freq_points else None
        median_roughness = float(np.median(roughness_values)) if roughness_values else None
        score = _score_legacy_target_candidate(
            median_first_positive_freq=median_first_positive,
            median_positive_freq_points=median_positive_points,
            median_roughness=median_roughness,
            failure_count=len(errors),
        )
        point_summary = (
            f"仍在 2048 点附近（中位 {int(median_positive_points)}）"
            if median_positive_points is not None and median_positive_points <= 2300
            else f"已增至约 {int(median_positive_points or 0)} 点"
        )
        if median_first_positive is None:
            freq_summary = "没有稳定首个正频点"
        elif median_first_positive < low:
            freq_summary = f"低频起点比参考图已知区间更靠左（{median_first_positive:.12g} Hz）"
        elif median_first_positive > high:
            freq_summary = f"低频起点仍偏右（{median_first_positive:.12g} Hz）"
        else:
            freq_summary = f"低频起点落在参考图已知 0.002~0.005 Hz 区间内（{median_first_positive:.12g} Hz）"
        candidate_results.append(
            {
                "kernel_name": kernel_name,
                "score": score,
                "failure_count": len(errors),
                "error_samples": errors[:4],
                "median_first_positive_freq": median_first_positive,
                "min_first_positive_freq": min(first_positive_freqs) if first_positive_freqs else None,
                "max_first_positive_freq": max(first_positive_freqs) if first_positive_freqs else None,
                "median_positive_freq_points": median_positive_points,
                "median_valid_freq_points": median_valid_points,
                "min_positive_freq_points": min(positive_freq_points) if positive_freq_points else None,
                "max_positive_freq_points": max(positive_freq_points) if positive_freq_points else None,
                "median_log_roughness": median_roughness,
                "comparison_summary": f"{freq_summary}；{point_summary}",
            }
        )
    candidate_results.sort(key=lambda item: (float(item["score"]), LEGACY_TARGET_PSD_KERNEL_CANDIDATES.index(str(item["kernel_name"]))))
    selected = candidate_results[0]["kernel_name"] if candidate_results else LEGACY_TARGET_PSD_KERNEL_WELCH_4096
    return {
        "selected_kernel": selected,
        "selection_basis": (
            "未找到参考实现的分组版真实频谱函数，按已知参考图首频点区间、正频点密度和谱线粗糙度对候选核评分。"
        ),
        "candidate_results": candidate_results,
    }


def get_cross_spectrum_display_meta(spectrum_type: str) -> dict[str, str]:
    if spectrum_type == CROSS_SPECTRUM_REAL:
        return {
            "title_prefix": "协谱图",
            "ylabel": "Co-spectrum",
            "result_column": "cospectrum",
            "plot_kind": "aligned_cospectrum",
        }
    if spectrum_type == CROSS_SPECTRUM_IMAG:
        return {
            "title_prefix": "正交谱图",
            "ylabel": "Quadrature Spectrum",
            "result_column": "quadrature_spectrum",
            "plot_kind": "aligned_quadrature",
        }
    return {
        "title_prefix": "互谱幅值",
        "ylabel": "Cross Spectral Magnitude",
        "result_column": "cross_spectrum_abs",
        "plot_kind": "cross",
    }


def resolve_cross_display_semantics(
    analysis_context: str | None,
    cross_execution_path: str | None,
    spectrum_type: str,
) -> str:
    resolved_spectrum_type = str(spectrum_type or CROSS_SPECTRUM_MAGNITUDE)
    if resolved_spectrum_type == CROSS_SPECTRUM_IMAG:
        return CROSS_DISPLAY_SEMANTICS_IMAG
    if resolved_spectrum_type == CROSS_SPECTRUM_REAL:
        if (
            str(analysis_context or "").strip() == TARGET_SPECTRAL_ANALYSIS_CONTEXT
            and str(cross_execution_path or "").strip() == "target_spectral_canonical"
        ):
            return CROSS_DISPLAY_SEMANTICS_ABS
        return CROSS_DISPLAY_SEMANTICS_REAL
    return CROSS_DISPLAY_SEMANTICS_ABS


def select_cross_display_values(
    details: dict[str, Any],
    display_semantics: str,
) -> tuple[np.ndarray, str]:
    source_by_semantics = {
        CROSS_DISPLAY_SEMANTICS_ABS: "cross_spectrum_abs",
        CROSS_DISPLAY_SEMANTICS_REAL: "cross_spectrum_real",
        CROSS_DISPLAY_SEMANTICS_IMAG: "cross_spectrum_imag",
    }
    resolved_semantics = str(display_semantics or CROSS_DISPLAY_SEMANTICS_ABS)
    value_source = source_by_semantics.get(resolved_semantics)
    if value_source is None:
        raise ValueError(f"未识别的互谱显示语义: {display_semantics}")
    if value_source not in details:
        raise KeyError(f"互谱结果中缺少显示源字段: {value_source}")
    return np.asarray(details[value_source], dtype=float), value_source


def resolve_cross_display_output(
    freq: np.ndarray,
    details: dict[str, Any],
    *,
    analysis_context: str | None,
    cross_execution_path: str | None,
    spectrum_type: str,
    insufficient_message: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    freq_array = np.asarray(freq, dtype=float)
    display_semantics = resolve_cross_display_semantics(
        analysis_context,
        cross_execution_path,
        spectrum_type,
    )
    display_values, display_value_source = select_cross_display_values(details, display_semantics)
    mask = build_spectrum_plot_mask(
        freq_array,
        display_values,
        spectrum_type,
        display_semantics=display_semantics,
    )
    if not np.any(mask):
        if display_semantics == CROSS_DISPLAY_SEMANTICS_ABS:
            raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
        raise ValueError(insufficient_message)
    if display_semantics != CROSS_DISPLAY_SEMANTICS_ABS and not np.any(np.abs(display_values[mask]) > 1e-18):
        raise ValueError(insufficient_message)

    resolved_details = dict(details)
    resolved_details["spectrum_type"] = spectrum_type
    resolved_details["display_semantics"] = display_semantics
    resolved_details["display_value_source"] = display_value_source
    if analysis_context is not None:
        resolved_details["analysis_context"] = str(analysis_context)
    if cross_execution_path is not None:
        resolved_details["cross_execution_path"] = str(cross_execution_path)
    return display_values, mask, resolved_details


def get_target_cospectrum_kernel_candidates() -> list[dict[str, Any]]:
    return [dict(item) for item in TARGET_COSPECTRUM_KERNEL_CANDIDATES]


def get_target_cospectrum_alignment_candidates() -> list[dict[str, Any]]:
    return [dict(item) for item in TARGET_COSPECTRUM_ALIGNMENT_CANDIDATES]


def resolve_target_cospectrum_kernel_candidate(candidate_id: str | None = None) -> dict[str, Any]:
    resolved_id = str(candidate_id or TARGET_COSPECTRUM_KERNEL_SELECTED_ID).strip() or TARGET_COSPECTRUM_KERNEL_SELECTED_ID
    for item in TARGET_COSPECTRUM_KERNEL_CANDIDATES:
        if str(item["candidate_id"]) == resolved_id or str(item["candidate_name"]) == resolved_id:
            return dict(item)
    raise ValueError(f"未识别的目标协谱 kernel candidate: {candidate_id}")


def get_target_cospectrum_implementations() -> list[dict[str, Any]]:
    return [dict(item) for item in TARGET_COSPECTRUM_IMPLEMENTATIONS]


def resolve_target_cospectrum_implementation(implementation_id: str | None = None) -> dict[str, Any]:
    resolved_id = str(implementation_id or TARGET_COSPECTRUM_IMPLEMENTATION_ID).strip() or TARGET_COSPECTRUM_IMPLEMENTATION_ID
    for item in TARGET_COSPECTRUM_IMPLEMENTATIONS:
        if str(item["implementation_id"]) == resolved_id or str(item["implementation_label"]) == resolved_id:
            return dict(item)
    raise ValueError(f"未识别的目标协谱 implementation: {implementation_id}")


def describe_generic_default_cross_implementation() -> dict[str, str]:
    return {
        "cross_execution_path": "generic_dual_column",
        "cross_implementation_id": GENERIC_DEFAULT_CROSS_IMPLEMENTATION_ID,
        "cross_implementation_label": GENERIC_DEFAULT_CROSS_IMPLEMENTATION_LABEL,
    }


def _format_detrend_setting(detrend: Any) -> str:
    if detrend == _CSD_DEFAULT_DETREND:
        return "default"
    if detrend is False:
        return "False"
    return str(detrend)


def _build_csd_window(window_name: str, nperseg: int) -> np.ndarray:
    if window_name == "hann":
        return signal.windows.hann(nperseg)
    if window_name == "boxcar":
        return signal.windows.boxcar(nperseg)
    raise ValueError(f"未识别的协谱窗口类型: {window_name}")


def _compute_cross_spectrum_candidate(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
    window_name: str,
    detrend: Any,
    scaling: str,
    kernel_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    valid = ~(np.isnan(data1) | np.isnan(data2))
    data1 = data1[valid]
    data2 = data2[valid]
    min_len = min(len(data1), len(data2))
    if min_len < 2:
        raise ValueError("数据长度不足，无法进行互谱分析。")

    data1 = data1[:min_len]
    data2 = data2[:min_len]
    if float(np.std(data1)) <= 1e-12 or float(np.std(data2)) <= 1e-12:
        raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")

    nperseg = min(requested_nsegment, min_len)
    noverlap = int(nperseg * overlap_ratio)
    noverlap = min(noverlap, nperseg - 1)
    window = _build_csd_window(window_name, nperseg)

    csd_kwargs: dict[str, Any] = {
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "scaling": scaling,
    }
    if detrend != _CSD_DEFAULT_DETREND:
        csd_kwargs["detrend"] = detrend

    freq, pxy = signal.csd(
        data1,
        data2,
        **csd_kwargs,
    )
    cross_abs = np.abs(pxy)
    cross_real = np.real(pxy)
    cross_imag = np.imag(pxy)
    selected = {
        CROSS_SPECTRUM_MAGNITUDE: cross_abs,
        CROSS_SPECTRUM_REAL: cross_real,
        CROSS_SPECTRUM_IMAG: cross_imag,
    }[spectrum_type]

    finite_positive_freq = (freq > 0) & np.isfinite(freq) & np.isfinite(selected)
    if spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
        valid_spectrum = finite_positive_freq & (selected > 0)
        if not np.any(valid_spectrum):
            raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    else:
        valid_spectrum = finite_positive_freq
        if not np.any(valid_spectrum):
            raise ValueError(insufficient_message)
        if not np.any(np.abs(selected[valid_spectrum]) > 1e-18):
            raise ValueError(insufficient_message)

    return freq, selected, {
        "valid_points": min_len,
        "fs": float(fs),
        "nsegment": int(requested_nsegment),
        "overlap_ratio": float(overlap_ratio),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "spectrum_type": spectrum_type,
        "cross_spectrum_abs": cross_abs,
        "cross_spectrum_real": cross_real,
        "cross_spectrum_imag": cross_imag,
        "window": window_name,
        "detrend": _format_detrend_setting(detrend),
        "scaling": str(scaling),
        "return_onesided": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "average": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "cross_kernel": str(kernel_name),
    }


def compute_cross_spectrum_from_arrays_with_params(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
    window_name: str = "hann",
    detrend: Any = _CSD_DEFAULT_DETREND,
    scaling: str = "density",
    kernel_name: str = LEGACY_TARGET_CROSS_KERNEL_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _compute_cross_spectrum_candidate(
        data1,
        data2,
        fs,
        requested_nsegment,
        overlap_ratio,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        window_name=window_name,
        detrend=detrend,
        scaling=scaling,
        kernel_name=kernel_name,
    )


def compute_target_cross_spectrum_from_frr_kernel(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    valid = ~(np.isnan(data1) | np.isnan(data2))
    data1 = data1[valid]
    data2 = data2[valid]
    min_len = min(len(data1), len(data2))
    if min_len < 2:
        raise ValueError("数据长度不足，无法进行互谱分析。")

    data1 = data1[:min_len]
    data2 = data2[:min_len]
    if float(np.std(data1)) <= 1e-12 or float(np.std(data2)) <= 1e-12:
        raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")

    nperseg = min(requested_nsegment, min_len)
    noverlap = int(round(nperseg * overlap_ratio))
    noverlap = min(noverlap, nperseg - 1)
    window = signal.windows.hann(nperseg)
    freq, pxy = signal.csd(
        data1,
        data2,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    cross_abs = np.abs(pxy)
    cross_real = np.real(pxy)
    cross_imag = np.imag(pxy)
    selected = {
        CROSS_SPECTRUM_MAGNITUDE: cross_abs,
        CROSS_SPECTRUM_REAL: cross_real,
        CROSS_SPECTRUM_IMAG: cross_imag,
    }[spectrum_type]

    finite_positive_freq = (freq > 0) & np.isfinite(freq) & np.isfinite(selected)
    if spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
        valid_spectrum = finite_positive_freq & (selected > 0)
        if not np.any(valid_spectrum):
            raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    else:
        valid_spectrum = finite_positive_freq
        if not np.any(valid_spectrum):
            raise ValueError(insufficient_message)
        if not np.any(np.abs(selected[valid_spectrum]) > 1e-18):
            raise ValueError(insufficient_message)

    return freq, selected, {
        "valid_points": min_len,
        "fs": float(fs),
        "nsegment": int(requested_nsegment),
        "overlap_ratio": float(overlap_ratio),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "spectrum_type": spectrum_type,
        "cross_spectrum_abs": cross_abs,
        "cross_spectrum_real": cross_real,
        "cross_spectrum_imag": cross_imag,
        "window": "hann",
        "detrend": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "scaling": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "return_onesided": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "average": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        "cross_kernel": FRR_COMPATIBLE_CROSS_KERNEL_ID,
        "cross_implementation_id": "A",
        "cross_implementation_label": resolve_target_cospectrum_implementation("A")["implementation_label"],
    }


def _prepare_cross_input_arrays(data1: np.ndarray, data2: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    series1 = np.asarray(data1, dtype=float)
    series2 = np.asarray(data2, dtype=float)
    valid = ~(np.isnan(series1) | np.isnan(series2))
    series1 = series1[valid]
    series2 = series2[valid]
    min_len = min(len(series1), len(series2))
    if min_len < 2:
        raise ValueError("数据长度不足，无法进行互谱分析。")
    series1 = series1[:min_len]
    series2 = series2[:min_len]
    if float(np.std(series1)) <= 1e-12 or float(np.std(series2)) <= 1e-12:
        raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    return series1, series2, min_len


def _resolve_cross_segment_params(
    min_len: int,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    use_round: bool = False,
) -> tuple[int, int]:
    nperseg = min(int(requested_nsegment), int(min_len))
    overlap_value = float(overlap_ratio)
    noverlap = int(round(nperseg * overlap_value)) if use_round else int(nperseg * overlap_value)
    noverlap = max(0, min(noverlap, nperseg - 1))
    return int(nperseg), int(noverlap)


def _finalize_cross_spectrum_result(
    freq: np.ndarray,
    pxy: np.ndarray,
    *,
    valid_points: int,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    nperseg: int,
    noverlap: int,
    spectrum_type: str,
    insufficient_message: str,
    implementation: dict[str, Any],
    window: str,
    detrend: str,
    scaling: str,
    return_onesided: Any,
    average: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    freq = np.asarray(freq, dtype=float)
    pxy = np.asarray(pxy, dtype=complex)
    cross_abs = np.abs(pxy)
    cross_real = np.real(pxy)
    cross_imag = np.imag(pxy)
    selected = {
        CROSS_SPECTRUM_MAGNITUDE: cross_abs,
        CROSS_SPECTRUM_REAL: cross_real,
        CROSS_SPECTRUM_IMAG: cross_imag,
    }[spectrum_type]

    finite_positive_freq = (freq > 0) & np.isfinite(freq) & np.isfinite(selected)
    if spectrum_type == CROSS_SPECTRUM_MAGNITUDE:
        valid_spectrum = finite_positive_freq & (selected > 0)
        if not np.any(valid_spectrum):
            raise ValueError("当前所选列波动过小或为恒定列，无法生成有效谱图。")
    else:
        valid_spectrum = finite_positive_freq
        if not np.any(valid_spectrum):
            raise ValueError(insufficient_message)
        if not np.any(np.abs(selected[valid_spectrum]) > 1e-18):
            raise ValueError(insufficient_message)

    return freq, selected, {
        "valid_points": int(valid_points),
        "fs": float(fs),
        "nsegment": int(requested_nsegment),
        "overlap_ratio": float(overlap_ratio),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "spectrum_type": spectrum_type,
        "cross_spectrum_abs": cross_abs,
        "cross_spectrum_real": cross_real,
        "cross_spectrum_imag": cross_imag,
        "window": str(window),
        "detrend": str(detrend),
        "scaling": str(scaling),
        "return_onesided": return_onesided,
        "average": str(average),
        "cross_kernel": str(implementation["cross_kernel"]),
        "cross_implementation_id": str(implementation["implementation_id"]),
        "cross_implementation_label": str(implementation["implementation_label"]),
    }


def _compute_target_cross_spectrum_generic_default(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation("B")
    freq, values, details = compute_cross_spectrum_from_arrays_with_params(
        data1,
        data2,
        fs,
        requested_nsegment,
        overlap_ratio,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        kernel_name=str(implementation["cross_kernel"]),
    )
    details = dict(details)
    details.update(
        {
            "cross_implementation_id": str(implementation["implementation_id"]),
            "cross_implementation_label": str(implementation["implementation_label"]),
            "return_onesided": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
            "average": SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        }
    )
    return freq, values, details


def _compute_target_cross_spectrum_twosided_positive_half(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation("C")
    series1, series2, min_len = _prepare_cross_input_arrays(data1, data2)
    nperseg, noverlap = _resolve_cross_segment_params(min_len, requested_nsegment, overlap_ratio, use_round=True)
    window = signal.windows.hann(nperseg)
    freq, pxy = signal.csd(
        series1,
        series2,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )
    positive = np.asarray(freq, dtype=float) > 0
    return _finalize_cross_spectrum_result(
        np.asarray(freq, dtype=float)[positive],
        np.asarray(pxy, dtype=complex)[positive],
        valid_points=min_len,
        fs=fs,
        requested_nsegment=requested_nsegment,
        overlap_ratio=overlap_ratio,
        nperseg=nperseg,
        noverlap=noverlap,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        implementation=implementation,
        window="hann",
        detrend=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        scaling=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        return_onesided=False,
        average=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
    )


def _map_twosided_to_manual_onesided(pxy_twosided: np.ndarray, nperseg: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    pxy_twosided = np.asarray(pxy_twosided, dtype=complex)
    freq = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    pxy_one = np.zeros(len(freq), dtype=complex)
    pxy_one[0] = pxy_twosided[0]
    if nperseg % 2 == 0:
        if len(freq) > 2:
            pxy_one[1:-1] = 2.0 * pxy_twosided[1 : nperseg // 2]
        if len(freq) > 1:
            pxy_one[-1] = pxy_twosided[nperseg // 2]
    else:
        if len(freq) > 1:
            pxy_one[1:] = 2.0 * pxy_twosided[1 : (nperseg + 1) // 2]
    return freq, pxy_one


def _compute_target_cross_spectrum_twosided_manual_one_sided(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation("D")
    series1, series2, min_len = _prepare_cross_input_arrays(data1, data2)
    nperseg, noverlap = _resolve_cross_segment_params(min_len, requested_nsegment, overlap_ratio, use_round=True)
    window = signal.windows.hann(nperseg)
    _freq_twosided, pxy_twosided = signal.csd(
        series1,
        series2,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )
    freq, pxy = _map_twosided_to_manual_onesided(pxy_twosided, nperseg, fs)
    return _finalize_cross_spectrum_result(
        freq,
        pxy,
        valid_points=min_len,
        fs=fs,
        requested_nsegment=requested_nsegment,
        overlap_ratio=overlap_ratio,
        nperseg=nperseg,
        noverlap=noverlap,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        implementation=implementation,
        window="hann",
        detrend=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        scaling=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
        return_onesided=False,
        average=SCIPY_DEFAULT_DIAGNOSTIC_VALUE,
    )


def _compute_target_cross_spectrum_mlab(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
    implementation_id: str,
    noverlap: int,
    scale_by_freq: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation(implementation_id)
    series1, series2, min_len = _prepare_cross_input_arrays(data1, data2)
    nperseg = min(int(requested_nsegment), int(min_len))
    noverlap = max(0, min(int(noverlap), nperseg - 1))
    pxy, freq = mlab.csd(
        series1,
        series2,
        NFFT=nperseg,
        Fs=float(fs),
        detrend=mlab.detrend_none,
        window=mlab.window_hanning,
        noverlap=noverlap,
        scale_by_freq=bool(scale_by_freq),
    )
    return _finalize_cross_spectrum_result(
        np.asarray(freq, dtype=float),
        np.asarray(pxy, dtype=complex),
        valid_points=min_len,
        fs=fs,
        requested_nsegment=requested_nsegment,
        overlap_ratio=overlap_ratio,
        nperseg=nperseg,
        noverlap=noverlap,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        implementation=implementation,
        window="hanning",
        detrend="mlab.detrend_none",
        scaling=("scale_by_freq=True" if scale_by_freq else "scale_by_freq=False"),
        return_onesided=MLAB_DEFAULT_DIAGNOSTIC_VALUE,
        average="mlab_segment_mean",
    )


def _compute_manual_segmented_cross_spectrum(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
    implementation_id: str,
    window_name: str,
    density_normalized: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation(implementation_id)
    series1, series2, min_len = _prepare_cross_input_arrays(data1, data2)
    nperseg = min(int(requested_nsegment), int(min_len))
    noverlap = int(round(nperseg * 0.5))
    noverlap = max(0, min(noverlap, nperseg - 1))
    step = max(1, nperseg - noverlap)
    if window_name == "hann":
        window = signal.windows.hann(nperseg)
    elif window_name == "boxcar":
        window = signal.windows.boxcar(nperseg)
    else:
        raise ValueError(f"未识别的手写 FFT 窗类型: {window_name}")
    starts = list(range(0, len(series1) - nperseg + 1, step))
    if not starts:
        starts = [0]
    segments: list[np.ndarray] = []
    for start in starts:
        segment1 = series1[start : start + nperseg] * window
        segment2 = series2[start : start + nperseg] * window
        fft1 = np.fft.rfft(segment1, n=nperseg)
        fft2 = np.fft.rfft(segment2, n=nperseg)
        pxy_segment = np.conj(fft1) * fft2
        if density_normalized:
            scale = 1.0 / (float(fs) * float(np.sum(window * window)))
            pxy_segment = pxy_segment * scale
            if nperseg % 2 == 0:
                if len(pxy_segment) > 2:
                    pxy_segment[1:-1] *= 2.0
            elif len(pxy_segment) > 1:
                pxy_segment[1:] *= 2.0
        segments.append(pxy_segment)
    pxy = np.mean(np.vstack(segments), axis=0)
    freq = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    return _finalize_cross_spectrum_result(
        freq,
        pxy,
        valid_points=min_len,
        fs=fs,
        requested_nsegment=requested_nsegment,
        overlap_ratio=overlap_ratio,
        nperseg=nperseg,
        noverlap=noverlap,
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
        implementation=implementation,
        window=window_name,
        detrend=MANUAL_DEFAULT_DIAGNOSTIC_VALUE,
        scaling=("density_manual" if density_normalized else "raw_manual"),
        return_onesided="manual_rfft",
        average="manual_segment_mean",
    )


def compute_target_cross_spectrum_from_selected_implementation(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    spectrum_type: str,
    insufficient_message: str,
    implementation_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    implementation = resolve_target_cospectrum_implementation(implementation_id)
    resolved_id = str(implementation["implementation_id"])
    if resolved_id == "A":
        return compute_target_cross_spectrum_from_frr_kernel(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
        )
    if resolved_id == "B":
        return _compute_target_cross_spectrum_generic_default(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
        )
    if resolved_id == "C":
        return _compute_target_cross_spectrum_twosided_positive_half(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
        )
    if resolved_id == "D":
        return _compute_target_cross_spectrum_twosided_manual_one_sided(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
        )
    if resolved_id == "E":
        return _compute_target_cross_spectrum_mlab(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            noverlap=0,
            scale_by_freq=True,
        )
    if resolved_id == "F":
        nperseg, noverlap = _resolve_cross_segment_params(
            min(len(np.asarray(data1).ravel()), len(np.asarray(data2).ravel())),
            requested_nsegment,
            overlap_ratio,
            use_round=True,
        )
        return _compute_target_cross_spectrum_mlab(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            noverlap=noverlap,
            scale_by_freq=True,
        )
    if resolved_id == "G":
        nperseg, noverlap = _resolve_cross_segment_params(
            min(len(np.asarray(data1).ravel()), len(np.asarray(data2).ravel())),
            requested_nsegment,
            overlap_ratio,
            use_round=True,
        )
        return _compute_target_cross_spectrum_mlab(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            noverlap=noverlap,
            scale_by_freq=False,
        )
    if resolved_id == "H":
        return _compute_manual_segmented_cross_spectrum(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            window_name="hann",
            density_normalized=False,
        )
    if resolved_id == "I":
        return _compute_manual_segmented_cross_spectrum(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            window_name="hann",
            density_normalized=True,
        )
    if resolved_id == "J":
        return _compute_manual_segmented_cross_spectrum(
            data1,
            data2,
            fs,
            requested_nsegment,
            overlap_ratio,
            spectrum_type=spectrum_type,
            insufficient_message=insufficient_message,
            implementation_id=resolved_id,
            window_name="boxcar",
            density_normalized=False,
        )
    raise ValueError(f"未识别的目标协谱 implementation: {implementation_id}")


def build_spectrum_plot_mask(
    freq: np.ndarray,
    values: np.ndarray,
    spectrum_type: str,
    *,
    display_semantics: str | None = None,
) -> np.ndarray:
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    positive_only = (
        str(display_semantics or "").strip() == CROSS_DISPLAY_SEMANTICS_ABS
        or (
            display_semantics is None
            and spectrum_type == CROSS_SPECTRUM_MAGNITUDE
        )
    )
    if positive_only:
        mask &= values > 0
    return mask


def build_cross_spectrum_export_frame(
    freq: np.ndarray,
    details: dict[str, Any],
    mask: np.ndarray,
    *,
    prefix: str | None = None,
    target_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    point_count = int(np.count_nonzero(mask))
    columns: dict[str, Any] = {
        "frequency_hz": freq[mask],
        "cross_spectrum_abs": np.asarray(details["cross_spectrum_abs"])[mask],
        "cross_spectrum_real": np.asarray(details["cross_spectrum_real"])[mask],
        "cross_spectrum_imag": np.asarray(details["cross_spectrum_imag"])[mask],
    }
    display_value_source = str(details.get("display_value_source") or "").strip()
    if display_value_source and display_value_source in details:
        columns["display_values"] = np.asarray(details[display_value_source])[mask]
    context = get_target_spectral_context(target_context)
    repeated_keys: dict[str, Any] = {
        "cross_execution_path": details.get("cross_execution_path"),
        "cross_implementation_id": details.get("cross_implementation_id"),
        "cross_reference_column": details.get("cross_reference_column"),
        "reference_column": details.get("reference_column"),
        "target_column": details.get("target_column"),
        "cross_order": details.get("cross_order"),
        "series_role": details.get("series_role"),
        "device_kind": details.get("device_kind"),
        "display_label": details.get("display_label"),
        "display_semantics": details.get("display_semantics"),
        "display_value_source": details.get("display_value_source"),
    }
    if context is not None and point_count > 0:
        repeated_keys.setdefault("reference_column", context.get("reference_column"))
        repeated_keys["reference_column"] = context.get("reference_column")
        repeated_keys["target_column"] = details.get("target_column", context.get("target_column"))
        repeated_keys["cross_order"] = details.get("cross_order", context.get("cross_order"))
        repeated_keys["ygas_target_column"] = context.get("ygas_target_column")
        repeated_keys["dat_target_column"] = context.get("dat_target_column")
        repeated_keys["canonical_cross_pairs"] = context.get("canonical_cross_pairs")
    if point_count > 0:
        for key, value in repeated_keys.items():
            if value is not None:
                if isinstance(value, (list, tuple)):
                    text_value = " | ".join(str(item) for item in value if str(item).strip())
                else:
                    text_value = str(value)
                columns[key] = [text_value] * point_count
    frame = pd.DataFrame(columns)
    if not prefix:
        return frame
    rename_map = {
        "cross_spectrum_abs": f"{prefix}_cross_spectrum_abs",
        "cross_spectrum_real": f"{prefix}_cross_spectrum_real",
        "cross_spectrum_imag": f"{prefix}_cross_spectrum_imag",
        "display_values": f"{prefix}_display_values",
    }
    for key in (
        "cross_execution_path",
        "cross_implementation_id",
        "cross_reference_column",
        "reference_column",
        "target_column",
        "cross_order",
        "series_role",
        "device_kind",
        "display_label",
        "ygas_target_column",
        "dat_target_column",
        "canonical_cross_pairs",
        "display_semantics",
        "display_value_source",
    ):
        if key in frame.columns:
            rename_map[key] = f"{prefix}_{key}"
    return frame.rename(columns=rename_map)


def compute_target_cross_complex_from_selected_implementation(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    implementation_id: str | None = None,
    insufficient_message: str = "当前数据不足以生成有效互谱/协谱结果。",
) -> tuple[np.ndarray, dict[str, Any]]:
    freq, _values, details = compute_target_cross_spectrum_from_selected_implementation(
        data1,
        data2,
        fs,
        requested_nsegment,
        overlap_ratio,
        spectrum_type=CROSS_SPECTRUM_MAGNITUDE,
        insufficient_message=insufficient_message,
        implementation_id=implementation_id,
    )
    return np.asarray(freq, dtype=float), dict(details)


def compute_target_cross_complex_payload(
    *,
    target_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    target_fs: float,
    reference_fs: float,
    target_element: str,
    reference_column: str,
    target_column: str,
    requested_nsegment: int,
    overlap_ratio: float,
    spectrum_type: str,
    use_requested_nsegment: bool = False,
    alignment_strategy: str = LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
    insufficient_message: str = "当前数据不足以生成有效互谱/协谱结果。",
    display_target_label: str | None = None,
    target_context: dict[str, Any] | None = None,
    series_role: str | None = None,
    device_kind: str | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    aligned, alignment_meta = align_target_window_pair_by_time(
        target_frame,
        reference_frame,
        alignment_strategy=alignment_strategy,
    )
    matched_count = int(alignment_meta["matched_count"])
    cross_requested_nsegment = int(requested_nsegment)
    cross_effective_nsegment = (
        cross_requested_nsegment if use_requested_nsegment else legacy_target_nsegment_resolver(matched_count)
    )
    positive_fs = [
        float(value)
        for value in (reference_fs, target_fs)
        if math.isfinite(float(value)) and float(value) > 0
    ]
    effective_fs = estimate_fs_from_timestamp(aligned, TIMESTAMP_COL)
    if effective_fs is None:
        effective_fs = min(positive_fs) if positive_fs else float(DEFAULT_FS)

    selected_implementation = resolve_target_cospectrum_implementation()
    freq, details = compute_target_cross_complex_from_selected_implementation(
        aligned["dat_value"].to_numpy(dtype=float),
        aligned["ygas_value"].to_numpy(dtype=float),
        float(effective_fs),
        cross_effective_nsegment,
        overlap_ratio,
        implementation_id=str(selected_implementation["implementation_id"]),
        insufficient_message=insufficient_message,
    )
    resolved_target_context = get_target_spectral_context(target_context)
    if resolved_target_context is None:
        resolved_target_context = build_target_spectral_context(
            target_element=target_element,
            reference_column=reference_column,
            ygas_target_column=target_column,
            dat_target_column=target_column,
            display_target_label=display_target_label,
            spectrum_mode=spectrum_type,
            comparison_is_target_spectral=False,
        )
    pair_specs = [
        dict(item)
        for item in (resolved_target_context.get("canonical_pair_specs") or [])
        if isinstance(item, dict)
    ]
    selected_pair = next(
        (
            item
            for item in pair_specs
            if (
                (series_role and str(item.get("series_role")) == str(series_role))
                or (
                    str(item.get("reference_column") or "").strip() == str(reference_column).strip()
                    and str(item.get("target_column") or "").strip() == str(target_column).strip()
                )
            )
        ),
        None,
    )
    cross_order_label = (
        str(selected_pair.get("cross_order"))
        if isinstance(selected_pair, dict) and selected_pair.get("cross_order")
        else f"{reference_column} -> {target_column}"
    )
    resolved_display_label = str(
        display_label
        or (selected_pair.get("display_label") if isinstance(selected_pair, dict) else "")
        or display_target_label
        or target_column
    ).strip()
    details.update(
        {
            "analysis_context": TARGET_SPECTRAL_ANALYSIS_CONTEXT,
            "source_mode": TARGET_SPECTRAL_SOURCE_MODE,
            "effective_fs": float(effective_fs),
            "requested_nsegment": cross_requested_nsegment,
            "effective_nsegment": int(details.get("nperseg", cross_effective_nsegment)),
            "nsegment_source": "ui" if use_requested_nsegment else "target-spectral-preset",
            "cross_kernel": str(details.get("cross_kernel", FRR_COMPATIBLE_CROSS_KERNEL_ID)),
            "matched_count": matched_count,
            "tolerance_seconds": float(alignment_meta["tolerance_seconds"]),
            "alignment_strategy": str(alignment_meta["alignment_strategy"]),
            "cross_execution_path": "target_spectral_canonical",
            "cross_implementation_id": str(
                details.get("cross_implementation_id", selected_implementation["implementation_id"])
            ),
            "cross_implementation_label": str(
                details.get("cross_implementation_label", selected_implementation["implementation_label"])
            ),
            "reference_column": str(reference_column),
            "target_column": str(target_column),
            "cross_reference_column": str(reference_column),
            "cross_order": str(cross_order_label),
            "target_element": str(target_element),
            "series_role": str(
                series_role
                or (selected_pair.get("series_role") if isinstance(selected_pair, dict) else "")
            ),
            "device_kind": str(
                device_kind
                or (selected_pair.get("device_kind") if isinstance(selected_pair, dict) else "")
            ),
            "display_label": resolved_display_label,
            "ygas_target_column": resolved_target_context.get("ygas_target_column"),
            "dat_target_column": resolved_target_context.get("dat_target_column"),
            "canonical_cross_pairs": list(resolved_target_context.get("canonical_cross_pairs") or []),
            "target_spectral_context": resolved_target_context,
        }
    )
    aligned_frame = aligned[[TIMESTAMP_COL, "dat_value", "ygas_value"]].rename(
        columns={
            "dat_value": str(reference_column),
            "ygas_value": str(target_column),
        }
    )
    return {
        "freq": freq,
        "details": details,
        "aligned_frame": aligned_frame,
        "aligned": aligned,
        "alignment_meta": alignment_meta,
        "matched_count": matched_count,
        "target_spectral_context": resolved_target_context,
    }


def compute_target_cross_spectrum_payload(
    *,
    target_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    target_fs: float,
    reference_fs: float,
    target_element: str,
    reference_column: str,
    target_column: str,
    requested_nsegment: int,
    overlap_ratio: float,
    spectrum_type: str,
    use_requested_nsegment: bool = False,
    alignment_strategy: str = LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
    insufficient_message: str = "当前数据不足以生成有效互谱/协谱结果。",
    display_target_label: str | None = None,
    target_context: dict[str, Any] | None = None,
    series_role: str | None = None,
    device_kind: str | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    complex_payload = compute_target_cross_complex_payload(
        target_frame=target_frame,
        reference_frame=reference_frame,
        target_fs=target_fs,
        reference_fs=reference_fs,
        target_element=target_element,
        reference_column=reference_column,
        target_column=target_column,
        requested_nsegment=requested_nsegment,
        overlap_ratio=overlap_ratio,
        spectrum_type=spectrum_type,
        use_requested_nsegment=use_requested_nsegment,
        alignment_strategy=alignment_strategy,
        insufficient_message=insufficient_message,
        display_target_label=display_target_label,
        target_context=target_context,
        series_role=series_role,
        device_kind=device_kind,
        display_label=display_label,
    )
    freq = np.asarray(complex_payload["freq"], dtype=float)
    target_context = dict(complex_payload["target_spectral_context"])
    values, mask, details = resolve_cross_display_output(
        freq,
        complex_payload["details"],
        analysis_context=target_context.get("analysis_context"),
        cross_execution_path="target_spectral_canonical",
        spectrum_type=spectrum_type,
        insufficient_message=insufficient_message,
    )
    positive_freq_mask = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    details.update(
        {
            "positive_freq_points": int(np.count_nonzero(positive_freq_mask)),
            "valid_freq_points": int(np.count_nonzero(mask)),
            "first_positive_freq": float(freq[mask][0]),
            "target_spectral_context": target_context,
        }
    )
    export_frame = build_cross_spectrum_export_frame(
        freq,
        details,
        mask,
        target_context=target_context,
    )
    return {
        "freq": freq,
        "values": values,
        "details": details,
        "mask": mask,
        "aligned_frame": complex_payload["aligned_frame"],
        "aligned": complex_payload["aligned"],
        "alignment_meta": complex_payload["alignment_meta"],
        "matched_count": complex_payload["matched_count"],
        "export_frame": export_frame,
        "target_spectral_context": target_context,
    }


def resolve_legacy_target_cross_spectrum_type(legacy_target_spectrum_mode: str) -> str | None:
    if legacy_target_spectrum_mode == LEGACY_TARGET_SPECTRUM_MODE_CROSS_MAGNITUDE:
        return CROSS_SPECTRUM_MAGNITUDE
    if legacy_target_spectrum_mode == LEGACY_TARGET_SPECTRUM_MODE_COSPECTRUM:
        return CROSS_SPECTRUM_REAL
    if legacy_target_spectrum_mode == LEGACY_TARGET_SPECTRUM_MODE_QUADRATURE:
        return CROSS_SPECTRUM_IMAG
    return None


def compute_alignment_tolerance_seconds(timestamps_a: pd.Series, timestamps_b: pd.Series) -> float:
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


def align_target_window_pair_by_time(
    ygas_frame: pd.DataFrame,
    dat_frame: pd.DataFrame,
    *,
    tolerance_seconds: float | None = None,
    alignment_strategy: str = LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_columns = {TIMESTAMP_COL, "value"}
    if not required_columns.issubset(ygas_frame.columns):
        raise ValueError("ygas window frame is missing timestamp/value columns")
    if not required_columns.issubset(dat_frame.columns):
        raise ValueError("dat window frame is missing timestamp/value columns")

    ygas_prepared = (
        ygas_frame[[TIMESTAMP_COL, "value"]]
        .rename(columns={TIMESTAMP_COL: "ygas_timestamp", "value": "ygas_value"})
        .dropna()
        .sort_values("ygas_timestamp")
        .drop_duplicates("ygas_timestamp", keep="first")
        .reset_index(drop=True)
    )
    dat_prepared = (
        dat_frame[[TIMESTAMP_COL, "value"]]
        .rename(columns={TIMESTAMP_COL: "dat_timestamp", "value": "dat_value"})
        .dropna()
        .sort_values("dat_timestamp")
        .drop_duplicates("dat_timestamp", keep="first")
        .reset_index(drop=True)
    )
    if len(ygas_prepared) < 2 or len(dat_prepared) < 2:
        raise ValueError("两台设备共同时间点不足，无法生成有效对齐结果。")

    if alignment_strategy == LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE:
        if tolerance_seconds is None:
            tolerance_seconds = compute_alignment_tolerance_seconds(
                ygas_prepared["ygas_timestamp"],
                dat_prepared["dat_timestamp"],
            )

        aligned = pd.merge_asof(
            ygas_prepared,
            dat_prepared,
            left_on="ygas_timestamp",
            right_on="dat_timestamp",
            direction="nearest",
            tolerance=pd.to_timedelta(float(tolerance_seconds), unit="s"),
        ).dropna(subset=["dat_value"])
        aligned = aligned.reset_index(drop=True)
        if len(aligned) < 2:
            raise ValueError("两台设备共同时间点不足，无法生成有效对齐结果。")
        aligned[TIMESTAMP_COL] = aligned["ygas_timestamp"]
        return aligned, {
            "matched_count": int(len(aligned)),
            "actual_start": pd.Timestamp(aligned[TIMESTAMP_COL].min()),
            "actual_end": pd.Timestamp(aligned[TIMESTAMP_COL].max()),
            "tolerance_seconds": float(tolerance_seconds),
            "rounding_resolution_seconds": None,
            "alignment_strategy": alignment_strategy,
        }

    if alignment_strategy == LEGACY_TARGET_ALIGNMENT_STRATEGY_ROUND_100MS_STRICT:
        ygas_join = ygas_prepared.assign(aligned_timestamp=ygas_prepared["ygas_timestamp"].dt.round("100ms"))
        dat_join = dat_prepared.assign(aligned_timestamp=dat_prepared["dat_timestamp"].dt.round("100ms"))
        ygas_join = ygas_join.drop_duplicates("aligned_timestamp", keep="first").reset_index(drop=True)
        dat_join = dat_join.drop_duplicates("aligned_timestamp", keep="first").reset_index(drop=True)
        aligned = ygas_join.merge(
            dat_join,
            on="aligned_timestamp",
            how="inner",
            suffixes=("_ygas", "_dat"),
        )
        if len(aligned) < 2:
            raise ValueError("两台设备共同时间点不足，无法生成有效对齐结果。")
        aligned = aligned.rename(columns={"aligned_timestamp": TIMESTAMP_COL}).reset_index(drop=True)
        return aligned, {
            "matched_count": int(len(aligned)),
            "actual_start": pd.Timestamp(aligned[TIMESTAMP_COL].min()),
            "actual_end": pd.Timestamp(aligned[TIMESTAMP_COL].max()),
            "tolerance_seconds": None,
            "rounding_resolution_seconds": 0.1,
            "alignment_strategy": alignment_strategy,
        }

    raise ValueError(f"Unsupported legacy-target alignment strategy: {alignment_strategy}")


def compute_target_cospectrum_from_arrays_with_selected_kernel(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    *,
    insufficient_message: str = "当前数据不足以生成有效协谱结果。",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return compute_target_cross_spectrum_from_selected_implementation(
        data1,
        data2,
        fs,
        requested_nsegment,
        overlap_ratio,
        spectrum_type=CROSS_SPECTRUM_REAL,
        insufficient_message=insufficient_message,
        implementation_id=TARGET_COSPECTRUM_IMPLEMENTATION_ID,
    )


def compute_target_cospectrum_candidate_from_aligned_frame(
    aligned: pd.DataFrame,
    *,
    fs: float,
    requested_nsegment: int,
    overlap_ratio: float,
    kernel_candidate: dict[str, Any],
    alignment_strategy: str,
) -> dict[str, Any]:
    freq, values, details = compute_cross_spectrum_from_arrays_with_params(
        aligned["dat_value"].to_numpy(dtype=float),
        aligned["ygas_value"].to_numpy(dtype=float),
        fs,
        requested_nsegment,
        overlap_ratio,
        spectrum_type=CROSS_SPECTRUM_REAL,
        insufficient_message="当前数据不足以生成有效协谱结果。",
        window_name=str(kernel_candidate["window_name"]),
        detrend=kernel_candidate["detrend"],
        scaling=str(kernel_candidate["scaling"]),
        kernel_name=str(kernel_candidate["candidate_name"]),
    )
    mask = build_spectrum_plot_mask(freq, values, CROSS_SPECTRUM_REAL)
    if not np.any(mask):
        raise ValueError("无有效协谱频点。")

    freq_positive = freq[mask]
    real_positive = np.asarray(details["cross_spectrum_real"])[mask]
    imag_positive = np.asarray(details["cross_spectrum_imag"])[mask]
    abs_positive = np.asarray(details["cross_spectrum_abs"])[mask]
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    integral_real = float(trapz_fn(real_positive, freq_positive))
    first_positive_freq = float(freq_positive[0])
    first_positive_real = float(real_positive[0])
    matched_count = int(len(aligned))
    export_frame = pd.DataFrame(
        {
            "frequency_hz": freq_positive,
            "cross_spectrum_real": real_positive,
            "cross_spectrum_imag": imag_positive,
            "cross_spectrum_abs": abs_positive,
            "integral_real": np.full(len(freq_positive), integral_real, dtype=float),
            "first_positive_freq": np.full(len(freq_positive), first_positive_freq, dtype=float),
            "first_positive_real": np.full(len(freq_positive), first_positive_real, dtype=float),
            "matched_count": np.full(len(freq_positive), matched_count, dtype=int),
            "fs": np.full(len(freq_positive), float(fs), dtype=float),
            "window": [str(kernel_candidate["window_name"])] * len(freq_positive),
            "nperseg": np.full(len(freq_positive), int(details["nperseg"]), dtype=int),
            "noverlap": np.full(len(freq_positive), int(details["noverlap"]), dtype=int),
            "detrend": [str(details["detrend"])] * len(freq_positive),
            "scaling": [str(details["scaling"])] * len(freq_positive),
            "alignment_strategy": [str(alignment_strategy)] * len(freq_positive),
            "kernel_candidate_id": [str(kernel_candidate["candidate_id"])] * len(freq_positive),
            "kernel_candidate_name": [str(kernel_candidate["candidate_name"])] * len(freq_positive),
        }
    )
    details = dict(details)
    details.update(
        {
            "integral_real": integral_real,
            "first_positive_freq": first_positive_freq,
            "first_positive_real": first_positive_real,
            "matched_count": matched_count,
            "alignment_strategy": str(alignment_strategy),
            "kernel_candidate_id": str(kernel_candidate["candidate_id"]),
            "kernel_candidate_name": str(kernel_candidate["candidate_name"]),
        }
    )
    return {
        "freq": freq,
        "values": values,
        "details": details,
        "mask": mask,
        "spectrum_frame": export_frame,
    }


def build_target_cospectrum_candidate_figure(
    candidate_frames: dict[str, pd.DataFrame],
    *,
    low_frequency_max_hz: float | None = None,
) -> Figure:
    figure = Figure(figsize=(10, 6), dpi=120)
    ax = figure.add_subplot(111)
    if low_frequency_max_hz is None:
        ax.set_xscale("log")
    max_abs = 0.0
    for candidate_name, frame in candidate_frames.items():
        if frame.empty:
            continue
        subset = frame
        if low_frequency_max_hz is not None:
            subset = frame[(frame["frequency_hz"] >= 0.0) & (frame["frequency_hz"] <= float(low_frequency_max_hz))]
        if subset.empty:
            continue
        x = subset["frequency_hz"].to_numpy(dtype=float)
        y = subset["cross_spectrum_real"].to_numpy(dtype=float)
        max_abs = max(max_abs, float(np.nanmax(np.abs(y))) if len(y) else 0.0)
        ax.plot(x, y, linewidth=1.2, alpha=0.9, label=candidate_name)
    linthresh = max(max_abs * 1e-3, 1e-12)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    if low_frequency_max_hz is None:
        ax.set_title("Target Cospectrum Candidates (Full Range)")
    else:
        ax.set_title(f"Target Cospectrum Candidates (0 ~ {low_frequency_max_hz:.3f} Hz)")
        ax.set_xlim(0.0, float(low_frequency_max_hz))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Co-spectrum")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def _format_cross_fingerprint(values: np.ndarray, limit: int = 8) -> str:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return ""
    return "|".join(f"{float(value):.9g}" for value in array[:limit])


def _compare_cross_implementation_frames(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
) -> dict[str, Any]:
    merged = reference_frame[
        ["frequency_hz", "cross_spectrum_real", "cross_spectrum_imag", "cross_spectrum_abs"]
    ].merge(
        candidate_frame[["frequency_hz", "cross_spectrum_real", "cross_spectrum_imag", "cross_spectrum_abs"]],
        on="frequency_hz",
        how="inner",
        suffixes=("_ref", "_candidate"),
    )
    if merged.empty:
        return {
            "allclose": False,
            "first8_real_same": False,
            "max_abs_diff_real": None,
            "mean_abs_diff_real": None,
            "lowfreq_mean_abs_diff_real": None,
        }
    diff_real = merged["cross_spectrum_real_candidate"] - merged["cross_spectrum_real_ref"]
    lowfreq = merged[merged["frequency_hz"] <= 0.05]
    first8_ref = merged["cross_spectrum_real_ref"].to_numpy(dtype=float)[:8]
    first8_candidate = merged["cross_spectrum_real_candidate"].to_numpy(dtype=float)[:8]
    arrays_same = (
        np.allclose(
            merged["cross_spectrum_real_ref"].to_numpy(dtype=float),
            merged["cross_spectrum_real_candidate"].to_numpy(dtype=float),
        )
        and np.allclose(
            merged["cross_spectrum_imag_ref"].to_numpy(dtype=float),
            merged["cross_spectrum_imag_candidate"].to_numpy(dtype=float),
        )
        and np.allclose(
            merged["cross_spectrum_abs_ref"].to_numpy(dtype=float),
            merged["cross_spectrum_abs_candidate"].to_numpy(dtype=float),
        )
    )
    return {
        "allclose": bool(arrays_same),
        "first8_real_same": bool(np.allclose(first8_ref, first8_candidate)),
        "max_abs_diff_real": float(np.max(np.abs(diff_real))),
        "mean_abs_diff_real": float(np.mean(np.abs(diff_real))),
        "lowfreq_mean_abs_diff_real": (
            float(np.mean(np.abs(lowfreq["cross_spectrum_real_candidate"] - lowfreq["cross_spectrum_real_ref"])))
            if not lowfreq.empty
            else None
        ),
    }


def diagnose_target_cospectrum_implementations(
    *,
    ygas_paths: Path | list[Path] | tuple[Path, ...],
    dat_path: Path,
    target_element: str,
    requested_nsegment: int,
    overlap_ratio: float,
    time_range_strategy: str = "使用 txt+dat 共同时间范围",
) -> dict[str, Any]:
    normalized_ygas_paths = list(ygas_paths) if isinstance(ygas_paths, (list, tuple)) else [ygas_paths]
    if not normalized_ygas_paths:
        raise ValueError("至少需要 1 个 ygas 文件。")

    ygas_required_columns = list(ELEMENT_PRESETS.get(target_element, {}).get("A", []))
    dat_required_columns = [
        *ELEMENT_PRESETS.get(target_element, {}).get("B", []),
        *REFERENCE_COLUMN_PRESETS.get("B", []),
    ]
    parsed_ygas, txt_summary = load_and_merge_ygas_files_fast(normalized_ygas_paths, required_columns=ygas_required_columns)
    parsed_dat = parse_toa5_file_fast(dat_path, required_columns=dat_required_columns)

    target_column = select_target_element_column(parsed_ygas, device_key="A", target_element=target_element)
    dat_target_column = select_target_element_column(parsed_dat, device_key="B", target_element=target_element)
    reference_column = select_reference_uz_column(parsed_dat, device_key="B")
    if target_column is None:
        raise ValueError(f"ygas 文件中没有找到“{target_element}”对应列。")
    if dat_target_column is None:
        raise ValueError(f"dat 文件中没有找到“{target_element}”对应列。")
    if reference_column is None:
        raise ValueError("dat 文件中没有找到 Uz 参考列。")

    dat_start, dat_end, dat_points = get_parsed_time_bounds(parsed_dat)
    dat_summary = {
        "start": dat_start,
        "end": dat_end,
        "total_points": dat_points,
        "raw_rows": int(parsed_dat.source_row_count or len(parsed_dat.dataframe)),
        "valid_timestamp_points": int(parsed_dat.timestamp_valid_count),
        "file_name": dat_path.name,
    }
    start_dt, end_dt, resolved_strategy_label = default_legacy_target_time_range_resolver(
        time_range_strategy,
        "",
        "",
        txt_summary,
        dat_summary,
    )
    if start_dt is None or end_dt is None:
        raise ValueError("目标协谱实现诊断未能解析出有效时间范围。")

    ygas_frame, ygas_meta = build_target_window_series(parsed_ygas, target_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    dat_target_frame, dat_target_meta = build_target_window_series(parsed_dat, dat_target_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    reference_frame, reference_meta = build_target_window_series(parsed_dat, reference_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    aligned, alignment_meta = align_target_window_pair_by_time(
        ygas_frame,
        reference_frame,
        alignment_strategy=LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
    )
    effective_fs = estimate_fs_from_timestamp(aligned, TIMESTAMP_COL)
    if effective_fs is None:
        ygas_fs = estimate_fs_from_timestamp(ygas_frame, TIMESTAMP_COL) or DEFAULT_FS
        reference_fs = estimate_fs_from_timestamp(reference_frame, TIMESTAMP_COL) or DEFAULT_FS
        effective_fs = min(float(ygas_fs), float(reference_fs))

    implementation_frames: dict[str, pd.DataFrame] = {}
    implementation_summary_rows: list[dict[str, Any]] = []
    plot_frames: dict[str, pd.DataFrame] = {}
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    for implementation in get_target_cospectrum_implementations():
        implementation_id = str(implementation["implementation_id"])
        freq, values, details = compute_target_cross_spectrum_from_selected_implementation(
            aligned["dat_value"].to_numpy(dtype=float),
            aligned["ygas_value"].to_numpy(dtype=float),
            float(effective_fs),
            int(requested_nsegment),
            float(overlap_ratio),
            spectrum_type=CROSS_SPECTRUM_REAL,
            insufficient_message="当前数据不足以生成有效协谱结果。",
            implementation_id=implementation_id,
        )
        mask = build_spectrum_plot_mask(freq, values, CROSS_SPECTRUM_REAL)
        if not np.any(mask):
            raise ValueError(f"implementation {implementation_id} 没有有效协谱频点。")

        freq_positive = np.asarray(freq, dtype=float)[mask]
        real_positive = np.asarray(details["cross_spectrum_real"], dtype=float)[mask]
        imag_positive = np.asarray(details["cross_spectrum_imag"], dtype=float)[mask]
        abs_positive = np.asarray(details["cross_spectrum_abs"], dtype=float)[mask]
        integral_real = float(trapz_fn(real_positive, freq_positive))
        first_positive_freq = float(freq_positive[0])
        first_positive_real = float(real_positive[0])
        fingerprint_real = _format_cross_fingerprint(real_positive)
        fingerprint_imag = _format_cross_fingerprint(imag_positive)

        frame = pd.DataFrame(
            {
                "implementation_id": [implementation_id] * len(freq_positive),
                "implementation_label": [str(details["cross_implementation_label"])] * len(freq_positive),
                "frequency_hz": freq_positive,
                "cross_spectrum_real": real_positive,
                "cross_spectrum_imag": imag_positive,
                "cross_spectrum_abs": abs_positive,
                "matched_count": np.full(len(freq_positive), int(len(aligned)), dtype=int),
                "fs": np.full(len(freq_positive), float(details["fs"]), dtype=float),
                "nperseg": np.full(len(freq_positive), int(details["nperseg"]), dtype=int),
                "noverlap": np.full(len(freq_positive), int(details["noverlap"]), dtype=int),
                "window": [str(details["window"])] * len(freq_positive),
                "detrend": [str(details["detrend"])] * len(freq_positive),
                "scaling": [str(details["scaling"])] * len(freq_positive),
                "return_onesided": [str(details.get("return_onesided"))] * len(freq_positive),
                "average": [str(details.get("average"))] * len(freq_positive),
                "first_positive_freq": np.full(len(freq_positive), first_positive_freq, dtype=float),
                "first_positive_real": np.full(len(freq_positive), first_positive_real, dtype=float),
                "integral_real": np.full(len(freq_positive), integral_real, dtype=float),
                "fingerprint_first8_real": [fingerprint_real] * len(freq_positive),
                "fingerprint_first8_imag": [fingerprint_imag] * len(freq_positive),
            }
        )
        implementation_frames[implementation_id] = frame
        plot_frames[f"{implementation_id}: {details['cross_implementation_label']}"] = frame
        lowfreq = frame[frame["frequency_hz"] <= 0.05]
        implementation_summary_rows.append(
            {
                "implementation_id": implementation_id,
                "implementation_label": str(details["cross_implementation_label"]),
                "matched_count": int(len(aligned)),
                "fs": float(details["fs"]),
                "nperseg": int(details["nperseg"]),
                "noverlap": int(details["noverlap"]),
                "window": str(details["window"]),
                "detrend": str(details["detrend"]),
                "scaling": str(details["scaling"]),
                "return_onesided": str(details.get("return_onesided")),
                "average": str(details.get("average")),
                "first_positive_freq": first_positive_freq,
                "first_positive_real": first_positive_real,
                "integral_real": integral_real,
                "fingerprint_first8_real": fingerprint_real,
                "fingerprint_first8_imag": fingerprint_imag,
                "lowfreq_min_real": float(lowfreq["cross_spectrum_real"].min()) if not lowfreq.empty else None,
                "lowfreq_max_real": float(lowfreq["cross_spectrum_real"].max()) if not lowfreq.empty else None,
                "lowfreq_median_real": float(lowfreq["cross_spectrum_real"].median()) if not lowfreq.empty else None,
            }
        )

    summary = pd.DataFrame(implementation_summary_rows).sort_values("implementation_id").reset_index(drop=True)
    frame_a = implementation_frames["A"]
    frame_b = implementation_frames["B"]
    compare_a_vs_b = _compare_cross_implementation_frames(frame_a, frame_b)
    current_target_helper_equals_generic_default = bool(compare_a_vs_b["allclose"])
    max_abs_diff_vs_generic_default = compare_a_vs_b["max_abs_diff_real"]
    first8_real_same_as_generic_default = bool(compare_a_vs_b["first8_real_same"])

    for row in implementation_summary_rows:
        frame = implementation_frames[str(row["implementation_id"])]
        diff_vs_a = _compare_cross_implementation_frames(frame_a, frame)
        diff_vs_b = _compare_cross_implementation_frames(frame_b, frame)
        row["allclose_vs_current_helper"] = bool(diff_vs_a["allclose"])
        row["max_abs_diff_vs_current_helper"] = diff_vs_a["max_abs_diff_real"]
        row["mean_abs_diff_vs_current_helper"] = diff_vs_a["mean_abs_diff_real"]
        row["lowfreq_mean_abs_diff_vs_current_helper"] = diff_vs_a["lowfreq_mean_abs_diff_real"]
        row["allclose_vs_generic_default"] = bool(diff_vs_b["allclose"])
        row["max_abs_diff_vs_generic_default"] = diff_vs_b["max_abs_diff_real"]
        row["first8_real_same_as_generic_default"] = bool(diff_vs_b["first8_real_same"])

    summary = pd.DataFrame(implementation_summary_rows).sort_values("implementation_id").reset_index(drop=True)
    non_current = summary[summary["implementation_id"] != "A"].copy()
    largest_diff_row = (
        non_current.sort_values("max_abs_diff_vs_current_helper", ascending=False).iloc[0].to_dict()
        if not non_current.empty
        else None
    )
    most_lowfreq_diff_row = (
        non_current.sort_values("lowfreq_mean_abs_diff_vs_current_helper", ascending=False).iloc[0].to_dict()
        if not non_current.empty
        else None
    )

    summary_lines = [
        f"current_target_helper_equals_generic_default = {current_target_helper_equals_generic_default}",
        f"max_abs_diff_vs_generic_default = {max_abs_diff_vs_generic_default}",
        f"first8_real_same_as_generic_default = {first8_real_same_as_generic_default}",
    ]
    if current_target_helper_equals_generic_default:
        summary_lines.append("当前 target helper 只是把 generic default 包了一层，因此不会带来图形变化。")

    selected_implementation = resolve_target_cospectrum_implementation()
    runtime_paths = {
        "target_spectrum_main_path": {
            "cross_execution_path": "target_spectral_canonical",
            "cross_implementation_id": str(selected_implementation["implementation_id"]),
            "cross_implementation_label": str(selected_implementation["implementation_label"]),
        },
        "main_analysis_target_spectral": {
            "cross_execution_path": "target_spectral_canonical",
            "cross_implementation_id": str(selected_implementation["implementation_id"]),
            "cross_implementation_label": str(selected_implementation["implementation_label"]),
        },
        "advanced_target_spectral_cospectrum": {
            "cross_execution_path": "target_spectral_canonical",
            "cross_implementation_id": str(selected_implementation["implementation_id"]),
            "cross_implementation_label": str(selected_implementation["implementation_label"]),
        },
    }

    return {
        "implementation_summary": summary,
        "implementation_frames": implementation_frames,
        "overlay_figure": build_target_cospectrum_candidate_figure(plot_frames),
        "low_frequency_figure": build_target_cospectrum_candidate_figure(plot_frames, low_frequency_max_hz=0.05),
        "summary_lines": summary_lines,
        "summary_text": "\n".join(summary_lines),
        "current_target_helper_equals_generic_default": current_target_helper_equals_generic_default,
        "max_abs_diff_vs_generic_default": max_abs_diff_vs_generic_default,
        "first8_real_same_as_generic_default": first8_real_same_as_generic_default,
        "largest_diff_vs_current_helper": largest_diff_row,
        "most_lowfreq_diff_vs_current_helper": most_lowfreq_diff_row,
        "runtime_paths": runtime_paths,
        "alignment_strategy": str(alignment_meta["alignment_strategy"]),
        "time_range_label": (
            f"{resolved_strategy_label}: {pd.Timestamp(start_dt):%Y-%m-%d %H:%M:%S} ~ "
            f"{pd.Timestamp(end_dt):%Y-%m-%d %H:%M:%S}"
        ),
        "target_element": target_element,
        "target_column": target_column,
        "dat_target_column": dat_target_column,
        "reference_column": reference_column,
        "window_point_counts": {
            "ygas": int(ygas_meta["valid_points"]),
            "dat_target": int(dat_target_meta["valid_points"]),
            "reference": int(reference_meta["valid_points"]),
            "aligned": int(len(aligned)),
        },
        "notes": list(TARGET_COSPECTRUM_DIAGNOSTIC_NOTES),
    }


def diagnose_target_cospectrum_candidates(
    *,
    ygas_paths: Path | list[Path] | tuple[Path, ...],
    dat_path: Path,
    target_element: str,
    requested_nsegment: int,
    overlap_ratio: float,
    time_range_strategy: str = "使用 txt+dat 共同时间范围",
) -> dict[str, Any]:
    normalized_ygas_paths = list(ygas_paths) if isinstance(ygas_paths, (list, tuple)) else [ygas_paths]
    if not normalized_ygas_paths:
        raise ValueError("至少需要 1 个 ygas 文件。")

    ygas_required_columns = list(ELEMENT_PRESETS.get(target_element, {}).get("A", []))
    dat_required_columns = [
        *ELEMENT_PRESETS.get(target_element, {}).get("B", []),
        *REFERENCE_COLUMN_PRESETS.get("B", []),
    ]
    parsed_ygas, txt_summary = load_and_merge_ygas_files_fast(normalized_ygas_paths, required_columns=ygas_required_columns)
    parsed_dat = parse_toa5_file_fast(dat_path, required_columns=dat_required_columns)

    target_column = select_target_element_column(parsed_ygas, device_key="A", target_element=target_element)
    dat_target_column = select_target_element_column(parsed_dat, device_key="B", target_element=target_element)
    reference_column = select_reference_uz_column(parsed_dat, device_key="B")
    if target_column is None:
        raise ValueError(f"ygas 文件中没有找到“{target_element}”对应列。")
    if dat_target_column is None:
        raise ValueError(f"dat 文件中没有找到“{target_element}”对应列。")
    if reference_column is None:
        raise ValueError("dat 文件中没有找到 Uz 参考列。")

    dat_start, dat_end, dat_points = get_parsed_time_bounds(parsed_dat)
    dat_summary = {
        "start": dat_start,
        "end": dat_end,
        "total_points": dat_points,
        "raw_rows": int(parsed_dat.source_row_count or len(parsed_dat.dataframe)),
        "valid_timestamp_points": int(parsed_dat.timestamp_valid_count),
        "file_name": dat_path.name,
    }
    start_dt, end_dt, resolved_strategy_label = default_legacy_target_time_range_resolver(
        time_range_strategy,
        "",
        "",
        txt_summary,
        dat_summary,
    )
    if start_dt is None or end_dt is None:
        raise ValueError("目标协谱诊断未能解析出有效时间范围。")

    ygas_frame, ygas_meta = build_target_window_series(parsed_ygas, target_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    dat_target_frame, dat_target_meta = build_target_window_series(parsed_dat, dat_target_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    reference_frame, reference_meta = build_target_window_series(parsed_dat, reference_column, pd.Timestamp(start_dt), pd.Timestamp(end_dt))
    ygas_fs = estimate_fs_from_timestamp(ygas_frame, TIMESTAMP_COL) or DEFAULT_FS
    reference_fs = estimate_fs_from_timestamp(reference_frame, TIMESTAMP_COL) or estimate_fs_from_timestamp(dat_target_frame, TIMESTAMP_COL) or DEFAULT_FS

    kernel_candidates = get_target_cospectrum_kernel_candidates()
    alignment_candidates = get_target_cospectrum_alignment_candidates()
    candidate_frames: dict[str, pd.DataFrame] = {}
    candidate_summary_rows: list[dict[str, Any]] = []
    aligned_frames: dict[str, pd.DataFrame] = {}
    baseline_candidate_name = "G_A"

    for alignment_candidate in alignment_candidates:
        aligned, alignment_meta = align_target_window_pair_by_time(
            ygas_frame,
            reference_frame,
            alignment_strategy=str(alignment_candidate["alignment_strategy"]),
        )
        aligned_frames[str(alignment_candidate["candidate_id"])] = aligned.copy()
        effective_fs = estimate_fs_from_timestamp(aligned, TIMESTAMP_COL)
        if effective_fs is None:
            effective_fs = min(float(ygas_fs), float(reference_fs))
        covariance_centered = float(
            np.mean(
                (aligned["dat_value"] - aligned["dat_value"].mean())
                * (aligned["ygas_value"] - aligned["ygas_value"].mean())
            )
        )
        covariance_raw = float(np.mean(aligned["dat_value"] * aligned["ygas_value"]))
        for kernel_candidate in kernel_candidates:
            candidate_name = f"{alignment_candidate['candidate_id']}_{kernel_candidate['candidate_id']}"
            result = compute_target_cospectrum_candidate_from_aligned_frame(
                aligned,
                fs=float(effective_fs),
                requested_nsegment=int(requested_nsegment),
                overlap_ratio=float(overlap_ratio),
                kernel_candidate=kernel_candidate,
                alignment_strategy=str(alignment_candidate["alignment_strategy"]),
            )
            frame = result["spectrum_frame"].copy()
            frame["candidate_name"] = candidate_name
            candidate_frames[candidate_name] = frame

            lowfreq = frame[frame["frequency_hz"] <= 0.05]
            candidate_summary_rows.append(
                {
                    "candidate_name": candidate_name,
                    "alignment_candidate_id": str(alignment_candidate["candidate_id"]),
                    "kernel_candidate_id": str(kernel_candidate["candidate_id"]),
                    "alignment_strategy": str(alignment_candidate["alignment_strategy"]),
                    "matched_count": int(result["details"]["matched_count"]),
                    "fs": float(result["details"]["fs"]),
                    "window": str(result["details"]["window"]),
                    "nperseg": int(result["details"]["nperseg"]),
                    "noverlap": int(result["details"]["noverlap"]),
                    "detrend": str(result["details"]["detrend"]),
                    "scaling": str(result["details"]["scaling"]),
                    "integral_real": float(result["details"]["integral_real"]),
                    "first_positive_freq": float(result["details"]["first_positive_freq"]),
                    "first_positive_real": float(result["details"]["first_positive_real"]),
                    "lowfreq_min_real": float(lowfreq["cross_spectrum_real"].min()) if not lowfreq.empty else None,
                    "lowfreq_max_real": float(lowfreq["cross_spectrum_real"].max()) if not lowfreq.empty else None,
                    "lowfreq_median_real": float(lowfreq["cross_spectrum_real"].median()) if not lowfreq.empty else None,
                    "covariance_centered": covariance_centered,
                    "covariance_raw": covariance_raw,
                    "target_column": str(target_column),
                    "reference_column": str(reference_column),
                    "target_element": str(target_element),
                    "time_range_label": (
                        f"{resolved_strategy_label}: {pd.Timestamp(start_dt):%Y-%m-%d %H:%M:%S} ~ "
                        f"{pd.Timestamp(end_dt):%Y-%m-%d %H:%M:%S}"
                    ),
                }
            )

    baseline_frame = candidate_frames.get(baseline_candidate_name)
    if baseline_frame is None:
        raise ValueError("未找到当前实现对应的目标协谱 baseline candidate（G_A）。")
    baseline_real = baseline_frame[["frequency_hz", "cross_spectrum_real"]].rename(
        columns={"cross_spectrum_real": "baseline_cross_spectrum_real"}
    )
    for row in candidate_summary_rows:
        frame = candidate_frames[row["candidate_name"]]
        merged = baseline_real.merge(
            frame[["frequency_hz", "cross_spectrum_real"]],
            on="frequency_hz",
            how="inner",
        )
        if merged.empty:
            row["diff_vs_current_l1"] = None
            row["diff_vs_current_l2"] = None
            row["diff_vs_current_lowfreq_l1"] = None
            continue
        diff = merged["cross_spectrum_real"] - merged["baseline_cross_spectrum_real"]
        row["diff_vs_current_l1"] = float(np.mean(np.abs(diff)))
        row["diff_vs_current_l2"] = float(np.sqrt(np.mean(np.square(diff))))
        lowfreq_diff = merged[merged["frequency_hz"] <= 0.05]
        row["diff_vs_current_lowfreq_l1"] = (
            float(np.mean(np.abs(lowfreq_diff["cross_spectrum_real"] - lowfreq_diff["baseline_cross_spectrum_real"])))
            if not lowfreq_diff.empty
            else None
        )

    candidate_summary = pd.DataFrame(candidate_summary_rows).sort_values(
        ["alignment_candidate_id", "kernel_candidate_id"]
    ).reset_index(drop=True)
    overlay_figure = build_target_cospectrum_candidate_figure(candidate_frames)
    low_frequency_figure = build_target_cospectrum_candidate_figure(candidate_frames, low_frequency_max_hz=0.05)
    return {
        "candidate_summary": candidate_summary,
        "candidate_frames": candidate_frames,
        "overlay_figure": overlay_figure,
        "low_frequency_figure": low_frequency_figure,
        "aligned_frames": aligned_frames,
        "notes": list(TARGET_COSPECTRUM_DIAGNOSTIC_NOTES),
        "baseline_candidate_name": baseline_candidate_name,
        "selected_kernel_candidate_id": TARGET_COSPECTRUM_KERNEL_SELECTED_ID,
        "selected_alignment_strategy": LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
        "target_element": target_element,
        "target_column": target_column,
        "dat_target_column": dat_target_column,
        "reference_column": reference_column,
        "time_range_label": (
            f"{resolved_strategy_label}: {pd.Timestamp(start_dt):%Y-%m-%d %H:%M:%S} ~ "
            f"{pd.Timestamp(end_dt):%Y-%m-%d %H:%M:%S}"
        ),
        "window_point_counts": {
            "ygas": int(ygas_meta["valid_points"]),
            "dat_target": int(dat_target_meta["valid_points"]),
            "reference": int(reference_meta["valid_points"]),
        },
    }


def find_element_hit(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def build_range_summary_from_entries(
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
        "valid_timestamp_points": int(
            sum(int(entry.get("valid_timestamp_points", entry.get("points", 0))) for entry in entries)
        ),
        "file_count": len(entries),
        "file_name": file_name or (entries[0]["path"].name if len(entries) == 1 else f"{len(entries)} 个文件"),
    }


def parse_time_text(raw: str) -> pd.Timestamp | None:
    text = str(raw).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无法解析时间输入: {raw}")
    return pd.Timestamp(parsed)


def default_legacy_target_time_range_resolver(
    strategy: str,
    start_raw: str,
    end_raw: str,
    txt_summary: dict[str, Any] | None,
    dat_summary: dict[str, Any] | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    if strategy == "手动输入时间范围":
        start_dt = parse_time_text(start_raw)
        end_dt = parse_time_text(end_raw)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("开始时间不能晚于结束时间。")
        return start_dt, end_dt, "手动时间范围"

    if dat_summary is None:
        return parse_time_text(start_raw), parse_time_text(end_raw), strategy

    dat_start = pd.Timestamp(dat_summary["start"])
    dat_end = pd.Timestamp(dat_summary["end"])
    if strategy == "使用 dat 时间范围":
        return dat_start, dat_end, "使用 dat 时间范围"
    if strategy == "最近10分钟":
        return dat_end - pd.Timedelta(minutes=10), dat_end, "最近10分钟"
    if strategy == "最近30分钟":
        return dat_end - pd.Timedelta(minutes=30), dat_end, "最近30分钟"
    if strategy == "最近1小时":
        return dat_end - pd.Timedelta(hours=1), dat_end, "最近1小时"

    if txt_summary is None:
        return dat_start, dat_end, "使用 dat 时间范围"

    common_start = max(pd.Timestamp(txt_summary["start"]), dat_start)
    common_end = min(pd.Timestamp(txt_summary["end"]), dat_end)
    if common_start > common_end:
        raise ValueError("txt 与 dat 没有共同时间范围。")
    return common_start, common_end, "使用 txt+dat 共同时间范围"


def default_legacy_target_fs_resolver(
    parsed: ParsedDataResult,
    frame: pd.DataFrame,
    fs_ui: float,
    device_kind: str,
) -> float:
    estimated = estimate_fs_from_timestamp(frame, TIMESTAMP_COL)
    if estimated:
        return float(estimated)
    if device_kind == "ygas":
        return float(DEFAULT_FS)
    return float(fs_ui if math.isfinite(fs_ui) and fs_ui > 0 else DEFAULT_FS)


def legacy_target_nsegment_resolver(valid_points: int) -> int:
    valid_points = max(int(valid_points), 0)
    for candidate in LEGACY_TARGET_NSEGMENT_CANDIDATES:
        if valid_points >= candidate:
            return candidate
    return min(DEFAULT_NSEGMENT, valid_points)


def build_legacy_target_spectrum_fields(
    prefix: str,
    details: dict[str, Any] | None,
    *,
    valid_freq_points: int | None = None,
) -> dict[str, Any]:
    payload = dict(details or {})
    resolved_valid_freq_points = valid_freq_points
    if resolved_valid_freq_points is None:
        resolved_valid_freq_points = payload.get("valid_freq_points")
    return {
        f"{prefix}_effective_fs": payload.get("effective_fs", payload.get("fs")),
        f"{prefix}_requested_nsegment": payload.get("requested_nsegment", payload.get("nsegment")),
        f"{prefix}_effective_nsegment": payload.get("effective_nsegment", payload.get("nsegment")),
        f"{prefix}_noverlap": payload.get("noverlap"),
        f"{prefix}_nsegment_source": payload.get("nsegment_source"),
        f"{prefix}_psd_kernel": payload.get("psd_kernel"),
        f"{prefix}_positive_freq_points": payload.get("positive_freq_points"),
        f"{prefix}_first_positive_freq": payload.get("first_positive_freq"),
        f"{prefix}_window_type": payload.get("window_type"),
        f"{prefix}_detrend": payload.get("detrend"),
        f"{prefix}_overlap": payload.get("overlap"),
        f"{prefix}_scaling_mode": payload.get("scaling_mode"),
        f"{prefix}_valid_freq_points": resolved_valid_freq_points,
    }


def get_parsed_time_bounds(parsed: ParsedDataResult) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    if parsed.timestamp_col is None or parsed.timestamp_col not in parsed.dataframe.columns:
        raise ValueError("未识别到时间列。")
    timestamps = parse_mixed_timestamp_series(parsed.dataframe[parsed.timestamp_col]).dropna()
    if timestamps.empty:
        raise ValueError("时间列为空。")
    return pd.Timestamp(timestamps.min()), pd.Timestamp(timestamps.max()), int(len(timestamps))


def select_target_element_column(
    parsed: ParsedDataResult,
    *,
    device_key: str,
    target_element: str,
) -> str | None:
    preset = ELEMENT_PRESETS.get(target_element, {})
    candidates = list(preset.get(device_key, []))
    for columns in (parsed.suggested_columns, parsed.available_columns):
        match = find_element_hit(columns, candidates)
        if match:
            return match
    return None


def select_reference_uz_column(
    parsed: ParsedDataResult,
    *,
    device_key: str = "B",
) -> str | None:
    candidates = list(REFERENCE_COLUMN_PRESETS.get(device_key, []))
    for columns in (parsed.suggested_columns, parsed.available_columns):
        match = find_element_hit(columns, candidates)
        if match:
            return match
    return None


def build_target_source_hint(path: Path, default: str) -> str:
    token_candidates: list[str] = []
    for text in (path.stem, path.parent.name):
        token_candidates.extend(re.findall(r"[A-Za-z]+", text))
    generic_tokens = {
        "SAMPLE",
        "SERIES",
        "DATA",
        "LOG",
        "TXT",
        "DAT",
        "EXTRACTED",
        "WORK",
        "TMP",
        "TEMP",
        "TEST",
        "SMOKE",
        "OUTPUT",
        "LEGACY",
        "TARGET",
    }
    filtered_tokens = [token.upper() for token in token_candidates if token.upper() not in generic_tokens]
    preferred_tokens = [token for token in filtered_tokens if 2 <= len(token) <= 4]
    if preferred_tokens:
        return preferred_tokens[0][:10] or default
    if filtered_tokens:
        return filtered_tokens[0][:10] or default
    return default


def build_legacy_group_labels(
    *,
    target_element: str,
    window_start: pd.Timestamp,
    ygas_path: Path,
) -> dict[str, str]:
    source_hint = build_target_source_hint(ygas_path, "ZH")
    ygas_label = f"{target_element}-{source_hint}-{window_start:%H%M}"
    return {
        "group_label": ygas_label,
        "ygas_label": ygas_label,
        "dat_label": f"{window_start:%Y-%m-%d %H%M} csi",
    }


def build_target_group_key(source_path: Path, window_start: pd.Timestamp) -> str:
    return f"{source_path.name}@{window_start:%Y-%m-%d %H:%M}"


def build_target_window_series(
    parsed: ParsedDataResult,
    value_column: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, meta = prepare_base_spectrum_series(
        parsed,
        value_column,
        start_dt=pd.Timestamp(start_dt),
        end_dt=pd.Timestamp(end_dt),
        require_timestamp=True,
    )
    if len(frame) < 16:
        raise ValueError("时间窗内有效数据点不足。")
    return frame, meta


def evaluate_target_group_quality_psd(
    group_payload: dict[str, Any],
    *,
    device_reference: dict[str, dict[str, float | None]],
    requested_nsegment: int,
    forced_include_keys: set[str],
    legacy_compat_strict: bool = False,
) -> dict[str, Any]:
    min_points = max(32, min(requested_nsegment // 2, 128))
    min_freq_points = max(12, min(requested_nsegment // 8, 32))
    coverage_threshold = 0.98 if legacy_compat_strict else 0.80
    fs_ratio_limit = 2.0
    spectrum_log_threshold = 1.5

    reasons: list[str] = []
    ygas_meta = group_payload["ygas_meta"]
    dat_meta = group_payload["dat_meta"]
    ygas_details = group_payload["ygas_details"]
    dat_details = group_payload["dat_details"]
    ygas_fs = float(ygas_details.get("fs", 0.0))
    dat_fs = float(dat_details.get("fs", 0.0))

    positive_fs = [value for value in (ygas_fs, dat_fs) if math.isfinite(value) and value > 0]
    base_fs = min(positive_fs) if positive_fs else 0.0
    min_duration_s = max(60.0, min(300.0, (requested_nsegment / base_fs) * 8.0)) if base_fs > 0 else 60.0

    if float(ygas_meta.get("coverage_ratio", 0.0)) < coverage_threshold:
        reasons.append("ygas覆盖不足")
    if float(dat_meta.get("coverage_ratio", 0.0)) < coverage_threshold:
        reasons.append("dat覆盖不足")
    if int(ygas_meta.get("valid_points", 0)) < min_points:
        reasons.append("ygas有效点数不足")
    if int(dat_meta.get("valid_points", 0)) < min_points:
        reasons.append("dat有效点数不足")
    if float(ygas_meta.get("actual_duration_s", 0.0)) < float(ygas_meta.get("requested_duration_s", 0.0)) * coverage_threshold:
        reasons.append("ygas有效时长过短")
    if float(dat_meta.get("actual_duration_s", 0.0)) < float(dat_meta.get("requested_duration_s", 0.0)) * coverage_threshold:
        reasons.append("dat有效时长过短")
    if float(ygas_meta.get("actual_duration_s", 0.0)) < min_duration_s:
        reasons.append(f"ygas时间窗过短(<{min_duration_s:.0f}s)")
    if float(dat_meta.get("actual_duration_s", 0.0)) < min_duration_s:
        reasons.append(f"dat时间窗过短(<{min_duration_s:.0f}s)")
    if int(group_payload.get("ygas_freq_points", 0)) < min_freq_points:
        reasons.append("ygas有效频点过少")
    if int(group_payload.get("dat_freq_points", 0)) < min_freq_points:
        reasons.append("dat有效频点过少")

    if legacy_compat_strict:
        if float(ygas_meta.get("leading_invalid_gap_s") or 0.0) > 5.0:
            reasons.append("ygas起始有效数据滞后")
        if float(dat_meta.get("leading_invalid_gap_s") or 0.0) > 5.0:
            reasons.append("dat起始有效数据滞后")
        if float(ygas_meta.get("trailing_invalid_gap_s") or 0.0) > 5.0:
            reasons.append("ygas末尾有效数据提前结束")
        if float(dat_meta.get("trailing_invalid_gap_s") or 0.0) > 5.0:
            reasons.append("dat末尾有效数据提前结束")
        if float(ygas_meta.get("non_null_ratio") or 0.0) < 0.98:
            reasons.append("ygas有效值覆盖不足")
        if float(dat_meta.get("non_null_ratio") or 0.0) < 0.98:
            reasons.append("dat有效值覆盖不足")

    for device_kind, fs_value in (("ygas", ygas_fs), ("dat", dat_fs)):
        if not math.isfinite(fs_value) or fs_value <= 0:
            reasons.append(f"{device_kind}采样频率异常")
            continue
        median_fs = device_reference.get(device_kind, {}).get("median_fs")
        if median_fs and fs_value > 0:
            ratio = max(fs_value / float(median_fs), float(median_fs) / fs_value)
            if ratio > fs_ratio_limit:
                reasons.append(f"{device_kind}采样频率异常")

    for device_kind, log_median in (
        ("ygas", group_payload.get("ygas_log_median")),
        ("dat", group_payload.get("dat_log_median")),
    ):
        reference_log = device_reference.get(device_kind, {}).get("median_log")
        if log_median is None or reference_log is None:
            continue
        if abs(float(log_median) - float(reference_log)) > spectrum_log_threshold:
            reasons.append(f"{device_kind}谱值离群")

    forced_include = group_payload["group_key"] in forced_include_keys
    keep = not reasons or forced_include
    return {
        "group_key": group_payload["group_key"],
        "window_label": group_payload["window_label"],
        "ygas_start": ygas_meta.get("actual_start"),
        "ygas_end": ygas_meta.get("actual_end"),
        "dat_start": dat_meta.get("actual_start"),
        "dat_end": dat_meta.get("actual_end"),
        "ygas_points": int(ygas_meta.get("valid_points", 0)),
        "dat_points": int(dat_meta.get("valid_points", 0)),
        "ygas_fs": float(ygas_details.get("fs", 0.0)),
        "dat_fs": float(dat_details.get("fs", 0.0)),
        "ygas_freq_points": int(group_payload.get("ygas_freq_points", 0)),
        "dat_freq_points": int(group_payload.get("dat_freq_points", 0)),
        "ygas_coverage_ratio": float(ygas_meta.get("coverage_ratio", 0.0)),
        "dat_coverage_ratio": float(dat_meta.get("coverage_ratio", 0.0)),
        "ygas_leading_invalid_gap_s": ygas_meta.get("leading_invalid_gap_s"),
        "dat_leading_invalid_gap_s": dat_meta.get("leading_invalid_gap_s"),
        "ygas_trailing_invalid_gap_s": ygas_meta.get("trailing_invalid_gap_s"),
        "dat_trailing_invalid_gap_s": dat_meta.get("trailing_invalid_gap_s"),
        "ygas_non_null_ratio": ygas_meta.get("non_null_ratio"),
        "dat_non_null_ratio": dat_meta.get("non_null_ratio"),
        "keep": keep,
        "forced_include": forced_include,
        "forceable": bool(reasons),
        "reason": "；".join(reasons) if reasons else "通过组级质量控制",
    }


def evaluate_target_group_quality_cross(
    group_payload: dict[str, Any],
    *,
    requested_nsegment: int,
    forced_include_keys: set[str],
    cross_payloads: dict[str, dict[str, Any]] | None = None,
    cross_errors: dict[str, str] | None = None,
    legacy_compat_strict: bool = False,
) -> dict[str, Any]:
    min_points = max(32, min(requested_nsegment // 2, 128))
    min_freq_points = max(12, min(requested_nsegment // 8, 32))
    coverage_threshold = 0.98 if legacy_compat_strict else 0.80
    tolerance_limit = 5.0 if legacy_compat_strict else 10.0
    cross_payloads = dict(cross_payloads or {})
    cross_errors = {str(key): str(value) for key, value in (cross_errors or {}).items()}

    series_specs = {
        "ygas_target_vs_uz": {
            "target_meta": group_payload["ygas_meta"],
            "target_prefix": "ygas",
        },
        "dat_target_vs_uz": {
            "target_meta": group_payload["dat_meta"],
            "target_prefix": "dat",
        },
    }
    reference_meta = group_payload.get("reference_meta") or group_payload["dat_meta"]

    valid_series_roles: list[str] = []
    series_messages: list[str] = []
    role_metrics: dict[str, dict[str, Any]] = {}
    for role, spec in series_specs.items():
        payload = cross_payloads.get(role)
        target_meta = spec["target_meta"]
        prefix = str(spec["target_prefix"])
        role_reasons: list[str] = []
        if payload is None:
            if cross_errors.get(role):
                role_reasons.append(cross_errors[role])
            else:
                role_reasons.append("未生成有效 cross payload")
        if float(target_meta.get("coverage_ratio", 0.0)) < coverage_threshold:
            role_reasons.append(f"{prefix}覆盖不足")
        if float(reference_meta.get("coverage_ratio", 0.0)) < coverage_threshold:
            role_reasons.append("Uz覆盖不足")
        if int(target_meta.get("valid_points", 0)) < min_points:
            role_reasons.append(f"{prefix}有效点数不足")
        if int(reference_meta.get("valid_points", 0)) < min_points:
            role_reasons.append("Uz有效点数不足")
        if float(target_meta.get("actual_duration_s", 0.0)) < float(target_meta.get("requested_duration_s", 0.0)) * coverage_threshold:
            role_reasons.append(f"{prefix}有效时长过短")
        if float(reference_meta.get("actual_duration_s", 0.0)) < float(reference_meta.get("requested_duration_s", 0.0)) * coverage_threshold:
            role_reasons.append("Uz有效时长过短")
        if payload is not None:
            details = dict(payload.get("details", {}))
            matched_count = int(payload.get("matched_count", details.get("matched_count", 0)) or 0)
            valid_freq_points = int(details.get("valid_freq_points", 0) or 0)
            tolerance_seconds = details.get("tolerance_seconds")
            role_metrics[role] = {
                "matched_count": matched_count,
                "valid_freq_points": valid_freq_points,
                "tolerance_seconds": tolerance_seconds,
                "effective_nsegment": details.get("effective_nsegment"),
                "cross_order": details.get("cross_order"),
            }
            if matched_count < min_points:
                role_reasons.append("matched_count不足")
            if valid_freq_points < min_freq_points:
                role_reasons.append("有效频点过少")
            if tolerance_seconds is None or not math.isfinite(float(tolerance_seconds)):
                role_reasons.append("容差异常")
            elif float(tolerance_seconds) > tolerance_limit:
                role_reasons.append(f"容差过大(>{tolerance_limit:.0f}s)")
        if not role_reasons:
            valid_series_roles.append(role)
        else:
            series_messages.append(f"{role}={'；'.join(role_reasons)}")

    forced_include = group_payload["group_key"] in forced_include_keys
    keep = bool(valid_series_roles) or forced_include
    reason = (
        "通过组级质量控制"
        if keep and not series_messages
        else ("；".join(series_messages) if series_messages else "没有可用的 canonical cross 系列")
    )
    return {
        "group_key": group_payload["group_key"],
        "window_label": group_payload["window_label"],
        "ygas_start": group_payload["ygas_meta"].get("actual_start"),
        "ygas_end": group_payload["ygas_meta"].get("actual_end"),
        "dat_start": group_payload["dat_meta"].get("actual_start"),
        "dat_end": group_payload["dat_meta"].get("actual_end"),
        "ygas_points": int(group_payload["ygas_meta"].get("valid_points", 0)),
        "dat_points": int(group_payload["dat_meta"].get("valid_points", 0)),
        "ygas_fs": float(group_payload.get("ygas_fs", 0.0)),
        "dat_fs": float(group_payload.get("dat_fs", 0.0)),
        "ygas_freq_points": 0,
        "dat_freq_points": 0,
        "ygas_coverage_ratio": float(group_payload["ygas_meta"].get("coverage_ratio", 0.0)),
        "dat_coverage_ratio": float(group_payload["dat_meta"].get("coverage_ratio", 0.0)),
        "ygas_leading_invalid_gap_s": group_payload["ygas_meta"].get("leading_invalid_gap_s"),
        "dat_leading_invalid_gap_s": group_payload["dat_meta"].get("leading_invalid_gap_s"),
        "ygas_trailing_invalid_gap_s": group_payload["ygas_meta"].get("trailing_invalid_gap_s"),
        "dat_trailing_invalid_gap_s": group_payload["dat_meta"].get("trailing_invalid_gap_s"),
        "ygas_non_null_ratio": group_payload["ygas_meta"].get("non_null_ratio"),
        "dat_non_null_ratio": group_payload["dat_meta"].get("non_null_ratio"),
        "keep": keep,
        "forced_include": forced_include,
        "forceable": bool(series_messages),
        "reason": reason,
        "generated_cross_series_count": int(len(valid_series_roles)),
        "generated_cross_series_roles": list(valid_series_roles),
        "ygas_cross_matched_count": role_metrics.get("ygas_target_vs_uz", {}).get("matched_count"),
        "dat_cross_matched_count": role_metrics.get("dat_target_vs_uz", {}).get("matched_count"),
        "ygas_cross_valid_freq_points": role_metrics.get("ygas_target_vs_uz", {}).get("valid_freq_points"),
        "dat_cross_valid_freq_points": role_metrics.get("dat_target_vs_uz", {}).get("valid_freq_points"),
        "ygas_cross_tolerance_seconds": role_metrics.get("ygas_target_vs_uz", {}).get("tolerance_seconds"),
        "dat_cross_tolerance_seconds": role_metrics.get("dat_target_vs_uz", {}).get("tolerance_seconds"),
        "ygas_cross_order": role_metrics.get("ygas_target_vs_uz", {}).get("cross_order"),
        "dat_cross_order": role_metrics.get("dat_target_vs_uz", {}).get("cross_order"),
    }


def evaluate_legacy_target_group_quality(
    group_payload: dict[str, Any],
    *,
    device_reference: dict[str, dict[str, float | None]],
    requested_nsegment: int,
    forced_include_keys: set[str],
    legacy_compat_strict: bool = False,
) -> dict[str, Any]:
    return evaluate_target_group_quality_psd(
        group_payload,
        device_reference=device_reference,
        requested_nsegment=requested_nsegment,
        forced_include_keys=forced_include_keys,
        legacy_compat_strict=legacy_compat_strict,
    )


def build_legacy_target_group_preview_frame(group_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in group_records:
        rows.append(
            {
                "组标签": record.get("group_label", record.get("group_key", record.get("window_label", ""))),
                "时间窗": record.get("window_label", ""),
                "ygas图例": record.get("ygas_label", ""),
                "dat图例": record.get("dat_label", ""),
                "ygas开始": record.get("ygas_start"),
                "ygas结束": record.get("ygas_end"),
                "dat开始": record.get("dat_start"),
                "dat结束": record.get("dat_end"),
                "ygas点数": record.get("ygas_points"),
                "dat点数": record.get("dat_points"),
                "ygas_FS": record.get("ygas_fs"),
                "dat_FS": record.get("dat_fs"),
                "ygas_effective_fs": record.get("ygas_effective_fs", record.get("ygas_fs")),
                "dat_effective_fs": record.get("dat_effective_fs", record.get("dat_fs")),
                "ygas覆盖率": record.get("ygas_coverage_ratio"),
                "dat覆盖率": record.get("dat_coverage_ratio"),
                "ygas起始无效秒": record.get("ygas_leading_invalid_gap_s"),
                "dat起始无效秒": record.get("dat_leading_invalid_gap_s"),
                "ygas末尾无效秒": record.get("ygas_trailing_invalid_gap_s"),
                "dat末尾无效秒": record.get("dat_trailing_invalid_gap_s"),
                "ygas有效值占比": record.get("ygas_non_null_ratio"),
                "dat有效值占比": record.get("dat_non_null_ratio"),
                "ygas频点": record.get("ygas_freq_points"),
                "dat频点": record.get("dat_freq_points"),
                "ygas_requested_nsegment": record.get("ygas_requested_nsegment"),
                "dat_requested_nsegment": record.get("dat_requested_nsegment"),
                "ygas_effective_nsegment": record.get("ygas_effective_nsegment"),
                "dat_effective_nsegment": record.get("dat_effective_nsegment"),
                "ygas_noverlap": record.get("ygas_noverlap"),
                "dat_noverlap": record.get("dat_noverlap"),
                "ygas_nsegment_source": record.get("ygas_nsegment_source"),
                "dat_nsegment_source": record.get("dat_nsegment_source"),
                "ygas_psd_kernel": record.get("ygas_psd_kernel"),
                "dat_psd_kernel": record.get("dat_psd_kernel"),
                "ygas_positive_freq_points": record.get("ygas_positive_freq_points"),
                "dat_positive_freq_points": record.get("dat_positive_freq_points"),
                "ygas_first_positive_freq": record.get("ygas_first_positive_freq"),
                "dat_first_positive_freq": record.get("dat_first_positive_freq"),
                "ygas_window_type": record.get("ygas_window_type"),
                "dat_window_type": record.get("dat_window_type"),
                "ygas_detrend": record.get("ygas_detrend"),
                "dat_detrend": record.get("dat_detrend"),
                "ygas_overlap": record.get("ygas_overlap"),
                "dat_overlap": record.get("dat_overlap"),
                "ygas_scaling_mode": record.get("ygas_scaling_mode"),
                "dat_scaling_mode": record.get("dat_scaling_mode"),
                "ygas_valid_freq_points": record.get("ygas_valid_freq_points", record.get("ygas_freq_points")),
                "dat_valid_freq_points": record.get("dat_valid_freq_points", record.get("dat_freq_points")),
                "spectrum_mode": record.get("spectrum_mode"),
                "matched_count": record.get("matched_count"),
                "tolerance_seconds": record.get("tolerance_seconds"),
                "alignment_strategy": record.get("alignment_strategy"),
                "cross_kernel": record.get("cross_kernel"),
                "cross_effective_fs": record.get("cross_effective_fs"),
                "cross_requested_nsegment": record.get("cross_requested_nsegment"),
                "cross_effective_nsegment": record.get("cross_effective_nsegment"),
                "cross_noverlap": record.get("cross_noverlap"),
                "cross_nsegment_source": record.get("cross_nsegment_source"),
                "cross_positive_freq_points": record.get("cross_positive_freq_points"),
                "cross_first_positive_freq": record.get("cross_first_positive_freq"),
                "cross_valid_freq_points": record.get("cross_valid_freq_points"),
                "generated_cross_series_count": record.get("generated_cross_series_count"),
                "generated_cross_series_roles": record.get("generated_cross_series_roles"),
                "ygas_cross_matched_count": record.get("ygas_cross_matched_count"),
                "dat_cross_matched_count": record.get("dat_cross_matched_count"),
                "ygas_cross_valid_freq_points": record.get("ygas_cross_valid_freq_points"),
                "dat_cross_valid_freq_points": record.get("dat_cross_valid_freq_points"),
                "使用状态": record.get("status"),
                "原因": record.get("reason"),
            }
        )
    return pd.DataFrame(rows)


def build_legacy_target_group_qc_export_frame(group_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in group_records:
        status = str(record.get("status", ""))
        keep = bool(record.get("keep", status in {"保留", "手动保留"}))
        forced_include = bool(record.get("forced_include", status == "手动保留"))
        rows.append(
            {
                "group_key": record.get("group_key"),
                "group_label": record.get("group_label"),
                "ygas_label": record.get("ygas_label"),
                "dat_label": record.get("dat_label"),
                "ygas_start": record.get("ygas_start"),
                "ygas_end": record.get("ygas_end"),
                "dat_start": record.get("dat_start"),
                "dat_end": record.get("dat_end"),
                "ygas_points": record.get("ygas_points"),
                "dat_points": record.get("dat_points"),
                "ygas_fs": record.get("ygas_fs"),
                "dat_fs": record.get("dat_fs"),
                "ygas_freq_points": record.get("ygas_freq_points"),
                "dat_freq_points": record.get("dat_freq_points"),
                "ygas_effective_fs": record.get("ygas_effective_fs", record.get("ygas_fs")),
                "dat_effective_fs": record.get("dat_effective_fs", record.get("dat_fs")),
                "ygas_requested_nsegment": record.get("ygas_requested_nsegment"),
                "dat_requested_nsegment": record.get("dat_requested_nsegment"),
                "ygas_effective_nsegment": record.get("ygas_effective_nsegment"),
                "dat_effective_nsegment": record.get("dat_effective_nsegment"),
                "ygas_noverlap": record.get("ygas_noverlap"),
                "dat_noverlap": record.get("dat_noverlap"),
                "ygas_nsegment_source": record.get("ygas_nsegment_source"),
                "dat_nsegment_source": record.get("dat_nsegment_source"),
                "ygas_psd_kernel": record.get("ygas_psd_kernel"),
                "dat_psd_kernel": record.get("dat_psd_kernel"),
                "ygas_positive_freq_points": record.get("ygas_positive_freq_points"),
                "dat_positive_freq_points": record.get("dat_positive_freq_points"),
                "ygas_first_positive_freq": record.get("ygas_first_positive_freq"),
                "dat_first_positive_freq": record.get("dat_first_positive_freq"),
                "ygas_window_type": record.get("ygas_window_type"),
                "dat_window_type": record.get("dat_window_type"),
                "ygas_detrend": record.get("ygas_detrend"),
                "dat_detrend": record.get("dat_detrend"),
                "ygas_overlap": record.get("ygas_overlap"),
                "dat_overlap": record.get("dat_overlap"),
                "ygas_scaling_mode": record.get("ygas_scaling_mode"),
                "dat_scaling_mode": record.get("dat_scaling_mode"),
                "ygas_valid_freq_points": record.get("ygas_valid_freq_points", record.get("ygas_freq_points")),
                "dat_valid_freq_points": record.get("dat_valid_freq_points", record.get("dat_freq_points")),
                "spectrum_mode": record.get("spectrum_mode"),
                "matched_count": record.get("matched_count"),
                "tolerance_seconds": record.get("tolerance_seconds"),
                "alignment_strategy": record.get("alignment_strategy"),
                "cross_kernel": record.get("cross_kernel"),
                "cross_effective_fs": record.get("cross_effective_fs"),
                "cross_requested_nsegment": record.get("cross_requested_nsegment"),
                "cross_effective_nsegment": record.get("cross_effective_nsegment"),
                "cross_noverlap": record.get("cross_noverlap"),
                "cross_nsegment_source": record.get("cross_nsegment_source"),
                "cross_positive_freq_points": record.get("cross_positive_freq_points"),
                "cross_first_positive_freq": record.get("cross_first_positive_freq"),
                "cross_valid_freq_points": record.get("cross_valid_freq_points"),
                "generated_cross_series_count": record.get("generated_cross_series_count"),
                "generated_cross_series_roles": record.get("generated_cross_series_roles"),
                "ygas_cross_matched_count": record.get("ygas_cross_matched_count"),
                "dat_cross_matched_count": record.get("dat_cross_matched_count"),
                "ygas_cross_valid_freq_points": record.get("ygas_cross_valid_freq_points"),
                "dat_cross_valid_freq_points": record.get("dat_cross_valid_freq_points"),
                "ygas_coverage_ratio": record.get("ygas_coverage_ratio"),
                "dat_coverage_ratio": record.get("dat_coverage_ratio"),
                "ygas_leading_invalid_gap_s": record.get("ygas_leading_invalid_gap_s"),
                "dat_leading_invalid_gap_s": record.get("dat_leading_invalid_gap_s"),
                "ygas_trailing_invalid_gap_s": record.get("ygas_trailing_invalid_gap_s"),
                "dat_trailing_invalid_gap_s": record.get("dat_trailing_invalid_gap_s"),
                "ygas_non_null_ratio": record.get("ygas_non_null_ratio"),
                "dat_non_null_ratio": record.get("dat_non_null_ratio"),
                "status": status,
                "keep": keep,
                "forced_include": forced_include,
                "reason": record.get("reason"),
            }
        )
    return pd.DataFrame(rows)


def prepare_legacy_target_payload(
    *,
    ygas_paths: list[Path],
    dat_path: Path | None,
    fs_ui: float,
    requested_nsegment: int,
    overlap_ratio: float,
    target_element: str,
    time_range_strategy: str,
    start_raw: str,
    end_raw: str,
    grouping_mode: str,
    parse_ygas: Callable[[Path], ParsedDataResult] = parse_ygas_mode1_file,
    parse_dat: Callable[[Path], ParsedDataResult] = parse_toa5_file,
    resolve_time_range: Callable[
        [str, str, str, dict[str, Any] | None, dict[str, Any] | None],
        tuple[pd.Timestamp | None, pd.Timestamp | None, str],
    ] = default_legacy_target_time_range_resolver,
    resolve_fs: Callable[[ParsedDataResult, pd.DataFrame, float, str], float] = default_legacy_target_fs_resolver,
    legacy_target_spectrum_mode: str = LEGACY_TARGET_SPECTRUM_MODE_PSD,
    use_requested_nsegment: bool = False,
    legacy_psd_kernel: str = LEGACY_TARGET_PSD_KERNEL_DEFAULT,
    forced_include_group_keys: set[str] | None = None,
    reporter: Any | None = None,
) -> dict[str, Any]:
    if dat_path is None:
        raise ValueError("请先选择 1 个 dat 文件，再生成目标谱图。")

    forced_include_group_keys = forced_include_group_keys or set()
    legacy_target_spectrum_mode = str(legacy_target_spectrum_mode or LEGACY_TARGET_SPECTRUM_MODE_PSD)
    if legacy_target_spectrum_mode not in LEGACY_TARGET_SPECTRUM_MODE_CHOICES:
        raise ValueError(f"未识别的目标谱图谱类型: {legacy_target_spectrum_mode}")
    selected_cross_spectrum_type = resolve_legacy_target_cross_spectrum_type(legacy_target_spectrum_mode)
    is_psd_mode = selected_cross_spectrum_type is None
    mode_label = "目标谱图"
    if reporter is not None:
        reporter("正在解析 dat 文件...")
    parsed_dat = parse_dat(dat_path)
    dat_column = select_target_element_column(parsed_dat, device_key="B", target_element=target_element)
    if dat_column is None:
        raise ValueError(f"dat 文件中没有找到“{target_element}”对应列，请检查列映射。")
    reference_column = None
    cross_reference_label = None
    cross_order_label = None
    if not is_psd_mode:
        reference_column = select_reference_uz_column(parsed_dat, device_key="B")
        if reference_column is None:
            raise ValueError("dat 文件中没有找到 Uz 列，无法生成互谱/协谱/正交谱。")
        cross_reference_label = str(reference_column)
        cross_order_label = f"{cross_reference_label} -> {target_element}"

    ygas_windows: list[dict[str, Any]] = []
    skipped_windows: list[str] = []
    precheck_group_records: list[dict[str, Any]] = []
    total_paths = len(ygas_paths)
    for index, path in enumerate(ygas_paths, start=1):
        if reporter is not None:
            reporter(f"正在解析 ygas 窗口 {index}/{total_paths}: {path.name}")
        try:
            parsed_ygas = parse_ygas(path)
            window_start, window_end, _window_points = get_parsed_time_bounds(parsed_ygas)
            ygas_column = select_target_element_column(parsed_ygas, device_key="A", target_element=target_element)
            labels = build_legacy_group_labels(
                target_element=target_element,
                window_start=pd.Timestamp(window_start),
                ygas_path=path,
            )
            group_key = build_target_group_key(path, pd.Timestamp(window_start))
            window_label = f"{pd.Timestamp(window_start):%Y-%m-%d %H:%M} ~ {pd.Timestamp(window_end):%H:%M}"
            if ygas_column is None:
                skipped_windows.append(f"{group_key}(未找到{target_element}列)")
                precheck_group_records.append(
                    {
                        "group_key": group_key,
                        "group_label": labels["group_label"],
                        "window_label": window_label,
                        "ygas_start": pd.Timestamp(window_start),
                        "ygas_end": pd.Timestamp(window_end),
                        "dat_start": None,
                        "dat_end": None,
                        "ygas_points": int(parsed_ygas.timestamp_valid_count),
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
                        "ygas_column": None,
                        "dat_column": dat_column,
                        "ygas_label": labels["ygas_label"],
                        "dat_label": labels["dat_label"],
                        "keep": False,
                        "forced_include": False,
                        "reason": f"未找到{target_element}列",
                        "status": "跳过",
                        "forceable": False,
                    }
                )
                continue
            ygas_windows.append(
                {
                    "path": path,
                    "parsed": parsed_ygas,
                    "column": ygas_column,
                    "start": window_start,
                    "end": window_end,
                }
            )
        except Exception as exc:
            skipped_windows.append(f"{path.name}({exc})")
            precheck_group_records.append(
                {
                    "group_key": path.name,
                    "group_label": path.name,
                    "window_label": path.name,
                    "ygas_start": None,
                    "ygas_end": None,
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
                    "ygas_column": None,
                    "dat_column": dat_column,
                    "ygas_label": path.name,
                    "dat_label": path.name,
                    "keep": False,
                    "forced_include": False,
                    "reason": str(exc),
                    "status": "跳过",
                    "forceable": False,
                }
            )

    if not ygas_windows:
        raise ValueError(f"没有找到可用于生成“{target_element}”目标谱图的 ygas 时间窗口。")

    txt_summary = build_range_summary_from_entries(
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
    dat_start, dat_end, dat_points = get_parsed_time_bounds(parsed_dat)
    dat_summary = {
        "start": dat_start,
        "end": dat_end,
        "total_points": dat_points,
        "raw_rows": int(parsed_dat.source_row_count or len(parsed_dat.dataframe)),
        "valid_timestamp_points": int(parsed_dat.timestamp_valid_count),
        "file_name": dat_path.name,
    }
    target_start, target_end, resolved_strategy_label = resolve_time_range(
        time_range_strategy,
        start_raw,
        end_raw,
        txt_summary,
        dat_summary,
    )

    def _median_log_density(freq: np.ndarray, density: np.ndarray) -> float | None:
        mask = (freq > 0) & np.isfinite(freq) & np.isfinite(density) & (density > 0)
        if not np.any(mask):
            return None
        return float(np.median(np.log10(density[mask])))

    prepared_groups: list[dict[str, Any]] = []
    selected_psd_kernel = str(legacy_psd_kernel or LEGACY_TARGET_PSD_KERNEL_DEFAULT)

    for item in sorted(ygas_windows, key=lambda value: pd.Timestamp(value["start"])):
        parsed_ygas = item["parsed"]
        ygas_column = str(item["column"])
        window_start = pd.Timestamp(item["start"])
        window_end = pd.Timestamp(item["end"])
        labels = build_legacy_group_labels(
            target_element=target_element,
            window_start=window_start,
            ygas_path=Path(item["path"]),
        )
        group_key = build_target_group_key(Path(item["path"]), window_start)
        effective_start = max(
            [window_start, pd.Timestamp(dat_start)] + ([pd.Timestamp(target_start)] if target_start is not None else [])
        )
        effective_end = min(
            [window_end, pd.Timestamp(dat_end)] + ([pd.Timestamp(target_end)] if target_end is not None else [])
        )
        window_label = f"{effective_start:%Y-%m-%d %H:%M} ~ {effective_end:%H:%M}"
        if effective_start >= effective_end:
            reason = "不在当前目标时间范围内"
            skipped_windows.append(f"{group_key}({reason})")
            precheck_group_records.append(
                {
                    "group_key": group_key,
                    "group_label": labels["group_label"],
                    "window_label": window_label,
                    "ygas_start": effective_start,
                    "ygas_end": effective_end,
                    "dat_start": effective_start,
                    "dat_end": effective_end,
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
                    "ygas_column": ygas_column,
                    "dat_column": dat_column,
                    "ygas_label": labels["ygas_label"],
                    "dat_label": labels["dat_label"],
                    "keep": False,
                    "forced_include": False,
                    "reason": reason,
                    "status": "跳过",
                    "forceable": False,
                    "spectrum_mode": legacy_target_spectrum_mode,
                }
            )
            continue

        try:
            ygas_frame, ygas_meta = build_target_window_series(parsed_ygas, ygas_column, effective_start, effective_end)
            dat_frame, dat_meta = build_target_window_series(parsed_dat, dat_column, effective_start, effective_end)
            reference_frame = None
            reference_meta = None
            if not is_psd_mode and reference_column is not None:
                reference_frame, reference_meta = build_target_window_series(
                    parsed_dat,
                    reference_column,
                    effective_start,
                    effective_end,
                )
            candidate_points = [int(len(ygas_frame)), int(len(dat_frame))]
            if reference_frame is not None:
                candidate_points.append(int(len(reference_frame)))
            group_requested_nsegment = int(
                requested_nsegment if use_requested_nsegment else legacy_target_nsegment_resolver(min(candidate_points))
            )
            ygas_fs = float(resolve_fs(parsed_ygas, ygas_frame, fs_ui, "ygas"))
            dat_fs = float(resolve_fs(parsed_dat, dat_frame, fs_ui, "dat"))
            prepared_groups.append(
                {
                    "group_key": group_key,
                    "group_label": labels["group_label"],
                    "window_label": window_label,
                    "ygas_label": labels["ygas_label"],
                    "dat_label": labels["dat_label"],
                    "ygas_path": Path(item["path"]),
                    "ygas_column": ygas_column,
                    "dat_column": dat_column,
                    "reference_column": reference_column,
                    "window_start": effective_start,
                    "window_end": effective_end,
                    "ygas_frame": ygas_frame,
                    "dat_frame": dat_frame,
                    "reference_frame": reference_frame,
                    "reference_meta": reference_meta,
                    "ygas_meta": ygas_meta,
                    "dat_meta": dat_meta,
                    "ygas_fs": ygas_fs,
                    "dat_fs": dat_fs,
                    "group_requested_nsegment": group_requested_nsegment,
                }
            )
        except Exception as exc:
            skipped_windows.append(f"{group_key}({exc})")
            precheck_group_records.append(
                {
                    "group_key": group_key,
                    "group_label": labels["group_label"],
                    "window_label": window_label,
                    "ygas_start": effective_start,
                    "ygas_end": effective_end,
                    "dat_start": effective_start,
                    "dat_end": effective_end,
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
                    "ygas_column": ygas_column,
                    "dat_column": dat_column,
                    "ygas_label": labels["ygas_label"],
                    "dat_label": labels["dat_label"],
                    "keep": False,
                    "forced_include": False,
                    "reason": str(exc),
                    "status": "跳过",
                    "forceable": False,
                    "spectrum_mode": legacy_target_spectrum_mode,
                }
            )

    if not prepared_groups:
        raise ValueError("选定时间范围内没有可用的目标谱图时间窗口。")

    base_target_context = None
    if not is_psd_mode and reference_column is not None:
        base_target_context = build_target_spectral_context(
            target_element=target_element,
            reference_column=str(reference_column),
            ygas_target_column=str(prepared_groups[0]["ygas_column"]),
            dat_target_column=str(dat_column),
            display_target_label=target_element,
            spectrum_mode=legacy_target_spectrum_mode,
            comparison_is_target_spectral=False,
        )

    if is_psd_mode and selected_psd_kernel == LEGACY_TARGET_PSD_KERNEL_AUTO:
        candidate_summary = summarize_legacy_target_psd_candidates(
            prepared_groups,
            requested_nsegment=max(int(group["group_requested_nsegment"]) for group in prepared_groups),
            overlap_ratio=overlap_ratio,
        )
        selected_psd_kernel = str(candidate_summary["selected_kernel"])
        legacy_psd_kernel_selection_basis = str(candidate_summary["selection_basis"])
        legacy_psd_candidate_results = list(candidate_summary["candidate_results"])
    elif is_psd_mode:
        legacy_psd_kernel_selection_basis = "显式指定目标谱图频谱核。"
        legacy_psd_candidate_results = []
    else:
        legacy_psd_kernel_selection_basis = "非 PSD 模式跳过 PSD 候选计算。"
        legacy_psd_candidate_results = []

    device_reference: dict[str, dict[str, float | None]] = {"ygas": {"median_fs": None, "median_log": None}, "dat": {"median_fs": None, "median_log": None}}
    if is_psd_mode:
        device_fs_samples: dict[str, list[float]] = {"ygas": [], "dat": []}
        device_log_samples: dict[str, list[float]] = {"ygas": [], "dat": []}
        for group in prepared_groups:
            ygas_values = group["ygas_frame"]["value"].to_numpy(dtype=float)
            dat_values = group["dat_frame"]["value"].to_numpy(dtype=float)
            ygas_freq, ygas_density, ygas_details = compute_legacy_target_psd_from_array(
                ygas_values,
                float(group["ygas_fs"]),
                int(group["group_requested_nsegment"]),
                overlap_ratio,
                psd_kernel=selected_psd_kernel,
            )
            dat_freq, dat_density, dat_details = compute_legacy_target_psd_from_array(
                dat_values,
                float(group["dat_fs"]),
                int(group["group_requested_nsegment"]),
                overlap_ratio,
                psd_kernel=selected_psd_kernel,
            )
            ygas_mask = build_spectrum_plot_mask(ygas_freq, ygas_density, CROSS_SPECTRUM_MAGNITUDE)
            dat_mask = build_spectrum_plot_mask(dat_freq, dat_density, CROSS_SPECTRUM_MAGNITUDE)
            group["ygas_freq"] = ygas_freq
            group["ygas_density"] = ygas_density
            group["ygas_details"] = ygas_details
            group["ygas_freq_points"] = int(np.count_nonzero(ygas_mask))
            group["ygas_log_median"] = _median_log_density(ygas_freq, ygas_density)
            group["dat_freq"] = dat_freq
            group["dat_density"] = dat_density
            group["dat_details"] = dat_details
            group["dat_freq_points"] = int(np.count_nonzero(dat_mask))
            group["dat_log_median"] = _median_log_density(dat_freq, dat_density)
            device_fs_samples["ygas"].append(float(ygas_details.get("effective_fs", group["ygas_fs"])))
            device_fs_samples["dat"].append(float(dat_details.get("effective_fs", group["dat_fs"])))
            if group["ygas_log_median"] is not None:
                device_log_samples["ygas"].append(float(group["ygas_log_median"]))
            if group["dat_log_median"] is not None:
                device_log_samples["dat"].append(float(group["dat_log_median"]))

        device_reference = {
            device_kind: {
                "median_fs": (float(np.median(values)) if values else None),
                "median_log": (float(np.median(device_log_samples[device_kind])) if device_log_samples[device_kind] else None),
            }
            for device_kind, values in device_fs_samples.items()
        }

    series_results: list[dict[str, Any]] = []
    group_records = list(precheck_group_records)
    for group_index, group in enumerate(prepared_groups):
        cross_payloads: dict[str, dict[str, Any]] = {}
        cross_errors: dict[str, str] = {}
        if is_psd_mode:
            record = evaluate_target_group_quality_psd(
                group,
                device_reference=device_reference,
                requested_nsegment=int(group["group_requested_nsegment"]),
                forced_include_keys=forced_include_group_keys,
            )
        else:
            pair_specs = [
                dict(item)
                for item in (base_target_context.get("canonical_pair_specs") if base_target_context else [])
                if isinstance(item, dict)
            ]
            for pair_spec in pair_specs:
                role = str(pair_spec.get("series_role") or "")
                try:
                    target_frame = group["ygas_frame"] if role == "ygas_target_vs_uz" else group["dat_frame"]
                    target_fs = float(group["ygas_fs"] if role == "ygas_target_vs_uz" else group["dat_fs"])
                    payload = compute_target_cross_spectrum_payload(
                        target_frame=target_frame,
                        reference_frame=group["reference_frame"],
                        target_fs=target_fs,
                        reference_fs=float(group["dat_fs"]),
                        target_element=target_element,
                        reference_column=str(reference_column),
                        target_column=str(pair_spec.get("target_column") or ""),
                        requested_nsegment=int(group["group_requested_nsegment"]),
                        overlap_ratio=overlap_ratio,
                        spectrum_type=str(selected_cross_spectrum_type),
                        use_requested_nsegment=use_requested_nsegment,
                        display_target_label=target_element,
                        target_context=base_target_context,
                        series_role=role,
                        device_kind=str(pair_spec.get("device_kind") or ""),
                        display_label=str(pair_spec.get("display_label") or ""),
                    )
                    cross_payloads[role] = payload
                except Exception as exc:
                    cross_errors[role] = str(exc)
            record = evaluate_target_group_quality_cross(
                group,
                requested_nsegment=int(group["group_requested_nsegment"]),
                forced_include_keys=forced_include_group_keys,
                cross_payloads=cross_payloads,
                cross_errors=cross_errors,
            )
        record.update(
            {
                "group_label": group["group_label"],
                "ygas_label": group["ygas_label"],
                "dat_label": group["dat_label"],
                "ygas_column": group["ygas_column"],
                "dat_column": group["dat_column"],
                "spectrum_mode": legacy_target_spectrum_mode,
                "reference_column": reference_column,
                "ygas_target_column": group["ygas_column"],
                "dat_target_column": group["dat_column"],
                "canonical_cross_pairs": list(base_target_context.get("canonical_cross_pairs", [])) if base_target_context else [],
                **build_legacy_target_spectrum_fields("ygas", group.get("ygas_details"), valid_freq_points=group.get("ygas_freq_points")),
                **build_legacy_target_spectrum_fields("dat", group.get("dat_details"), valid_freq_points=group.get("dat_freq_points")),
            }
        )

        if record.get("keep"):
            record["status"] = "手动保留" if record.get("forced_include") else "保留"
        else:
            record["status"] = "跳过"
            skipped_windows.append(f"{group['group_key']}({record.get('reason')})")

        if not is_psd_mode and cross_payloads:
            valid_payloads = list(cross_payloads.values())
            first_cross_details = dict(valid_payloads[0]["details"])
            matched_values = [int(item.get("matched_count", 0)) for item in valid_payloads]
            tolerance_values = [
                float(item["details"]["tolerance_seconds"])
                for item in valid_payloads
                if item.get("details", {}).get("tolerance_seconds") is not None
            ]
            valid_freq_values = [
                int(item["details"].get("valid_freq_points", 0))
                for item in valid_payloads
            ]
            record.update(
                {
                    "matched_count": max(matched_values) if matched_values else None,
                    "tolerance_seconds": max(tolerance_values) if tolerance_values else None,
                    "alignment_strategy": first_cross_details.get("alignment_strategy"),
                    "cross_kernel": first_cross_details.get("cross_kernel"),
                    "cross_effective_fs": first_cross_details.get("effective_fs"),
                    "cross_requested_nsegment": first_cross_details.get("requested_nsegment"),
                    "cross_effective_nsegment": first_cross_details.get("effective_nsegment"),
                    "cross_noverlap": first_cross_details.get("noverlap"),
                    "cross_nsegment_source": first_cross_details.get("nsegment_source"),
                    "cross_positive_freq_points": max(valid_freq_values) if valid_freq_values else None,
                    "cross_first_positive_freq": first_cross_details.get("first_positive_freq"),
                    "cross_valid_freq_points": max(valid_freq_values) if valid_freq_values else None,
                    "cross_reference_column": first_cross_details.get("reference_column"),
                    "cross_order": "；".join(
                        str(item["details"].get("cross_order"))
                        for item in valid_payloads
                        if item.get("details", {}).get("cross_order") is not None
                    ),
                    "target_column": first_cross_details.get("target_column"),
                    "display_semantics": first_cross_details.get("display_semantics"),
                    "display_value_source": first_cross_details.get("display_value_source"),
                    "generated_cross_series_count": int(
                        record.get("generated_cross_series_count", len(cross_payloads))
                    ),
                    "generated_cross_series_roles": list(
                        record.get("generated_cross_series_roles", list(cross_payloads.keys()))
                    ),
                }
            )

        group_records.append(record)
        if not record.get("keep"):
            continue

        if is_psd_mode:
            series_results.extend(
                [
                    {
                        "label": group["ygas_label"],
                        "device_kind": "ygas",
                        "group_index": group_index,
                        "freq": group["ygas_freq"],
                        "density": group["ygas_density"],
                        "valid_points": int(group["ygas_meta"]["valid_points"]),
                        "valid_freq_points": int(group["ygas_freq_points"]),
                        "details": dict(group["ygas_details"]),
                    },
                    {
                        "label": group["dat_label"],
                        "device_kind": "dat",
                        "group_index": group_index,
                        "freq": group["dat_freq"],
                        "density": group["dat_density"],
                        "valid_points": int(group["dat_meta"]["valid_points"]),
                        "valid_freq_points": int(group["dat_freq_points"]),
                        "details": dict(group["dat_details"]),
                    },
                ]
            )
        else:
            append_roles = list(record.get("generated_cross_series_roles") or [])
            if record.get("forced_include") and not append_roles:
                append_roles = list(cross_payloads.keys())
            pair_specs = {
                str(item.get("series_role") or ""): dict(item)
                for item in (base_target_context.get("canonical_pair_specs") if base_target_context else [])
                if isinstance(item, dict)
            }
            for role in append_roles:
                cross_payload = cross_payloads.get(str(role))
                pair_spec = pair_specs.get(str(role), {})
                if cross_payload is None:
                    continue
                series_details = dict(cross_payload["details"])
                series_results.append(
                    {
                        "label": f"{group['group_label']} - {pair_spec.get('display_label', series_details.get('display_label', role))}",
                        "device_kind": str(pair_spec.get("device_kind") or series_details.get("device_kind") or "cross"),
                        "group_index": group_index,
                        "freq": cross_payload["freq"],
                        "values": cross_payload["values"],
                        "valid_points": int(cross_payload["matched_count"]),
                        "valid_freq_points": int(cross_payload["details"]["valid_freq_points"]),
                        "series_role": str(role),
                        "reference_column": series_details.get("reference_column"),
                        "target_column": series_details.get("target_column"),
                        "cross_order": series_details.get("cross_order"),
                        "display_label": series_details.get("display_label"),
                        "details": series_details,
                        "aligned_frame": cross_payload["aligned_frame"],
                    }
                )

    kept_group_records = [record for record in group_records if record.get("status") in {"保留", "手动保留"}]
    if not series_results:
        raise ValueError("所有目标谱图时间组都被跳过，未生成有效输出。")

    if is_psd_mode:
        mode_freq_points = [int(item.get("valid_freq_points", 0)) for item in series_results]
        mode_first_positive_freqs = [
            float(item["details"]["first_positive_freq"])
            for item in series_results
            if item.get("details", {}).get("first_positive_freq") is not None
        ]
    else:
        mode_freq_points = [
            int(item.get("valid_freq_points", 0))
            for item in series_results
            if item.get("valid_freq_points") is not None
        ]
        mode_first_positive_freqs = [
            float(item["details"]["first_positive_freq"])
            for item in series_results
            if item.get("details", {}).get("first_positive_freq") is not None
        ]

    matched_counts = [
        int(item.get("details", {}).get("matched_count"))
        for item in series_results
        if item.get("details", {}).get("matched_count") is not None
    ]
    tolerance_values = [
        float(item.get("details", {}).get("tolerance_seconds"))
        for item in series_results
        if item.get("details", {}).get("tolerance_seconds") is not None
    ]
    effective_nsegment_values = sorted(
        {
            int(item.get("details", {}).get("effective_nsegment"))
            for item in series_results
            if item.get("details", {}).get("effective_nsegment") is not None
        }
    )
    requested_nsegment_summary = int(
        requested_nsegment if use_requested_nsegment else max(group["group_requested_nsegment"] for group in prepared_groups)
    )
    selected_cross_implementation = resolve_target_cospectrum_implementation() if not is_psd_mode else None
    resolved_cross_kernel = (
        str(selected_cross_implementation["cross_kernel"]) if selected_cross_implementation is not None else None
    )

    target_metadata = {
        "mode_label": mode_label,
        "spectrum_mode": legacy_target_spectrum_mode,
        "ygas_column": kept_group_records[0].get("ygas_column") if kept_group_records else None,
        "dat_column": dat_column,
        "ygas_target_column": kept_group_records[0].get("ygas_target_column") if kept_group_records else None,
        "dat_target_column": kept_group_records[0].get("dat_target_column") if kept_group_records else None,
        "time_range_label": (
            f"{resolved_strategy_label}: "
            f"{(pd.Timestamp(target_start) if target_start is not None else pd.Timestamp(txt_summary['start'])):%Y-%m-%d %H:%M:%S}"
            " ~ "
            f"{(pd.Timestamp(target_end) if target_end is not None else pd.Timestamp(dat_summary['end'])):%Y-%m-%d %H:%M:%S}"
        ),
        "legacy_target_uses_requested_nsegment": bool(use_requested_nsegment),
        "requested_nsegment": requested_nsegment_summary,
        "legacy_psd_kernel_requested": legacy_psd_kernel,
        "legacy_psd_kernel": selected_psd_kernel,
        "legacy_psd_kernel_selection_basis": legacy_psd_kernel_selection_basis,
        "legacy_psd_candidate_results": legacy_psd_candidate_results,
        "reference_line_mode": "fixed_f_pow_-2_3",
        "reference_line_at_1hz": 1.0,
        "alignment_strategy": LEGACY_TARGET_ALIGNMENT_STRATEGY_NEAREST_TOLERANCE,
        "cross_kernel": resolved_cross_kernel,
        "cross_execution_path": ("target_spectral_canonical" if not is_psd_mode else None),
        "cross_implementation_id": (
            str(selected_cross_implementation["implementation_id"]) if selected_cross_implementation is not None else None
        ),
        "cross_implementation_label": (
            str(selected_cross_implementation["implementation_label"])
            if selected_cross_implementation is not None
            else None
        ),
        "window": ("hann" if not is_psd_mode else None),
        "detrend": (SCIPY_DEFAULT_DIAGNOSTIC_VALUE if not is_psd_mode else None),
        "scaling": (SCIPY_DEFAULT_DIAGNOSTIC_VALUE if not is_psd_mode else None),
        "return_onesided": (SCIPY_DEFAULT_DIAGNOSTIC_VALUE if not is_psd_mode else None),
        "average": (SCIPY_DEFAULT_DIAGNOSTIC_VALUE if not is_psd_mode else None),
        "cross_reference_column": cross_reference_label,
        "cross_order": cross_order_label,
        "reference_column": cross_reference_label,
        "canonical_cross_pairs": list(base_target_context.get("canonical_cross_pairs", [])) if base_target_context else [],
        "expected_series_count": int(
            len(kept_group_records)
            * (2 if is_psd_mode else len(base_target_context.get("canonical_pair_specs", [])) if base_target_context else 0)
        ),
        "actual_series_count": int(len(series_results)),
        "total_group_count": int(len(group_records)),
        "kept_group_count": int(len(kept_group_records)),
        "skipped_group_count": int(sum(1 for record in group_records if record.get("status") == "跳过")),
        "manual_override_groups": sorted(str(key) for key in forced_include_group_keys),
        "effective_nsegment_values": effective_nsegment_values,
        "positive_freq_points_min": min(mode_freq_points) if mode_freq_points else None,
        "positive_freq_points_max": max(mode_freq_points) if mode_freq_points else None,
        "first_positive_freq_min": min(mode_first_positive_freqs) if mode_first_positive_freqs else None,
        "first_positive_freq_max": max(mode_first_positive_freqs) if mode_first_positive_freqs else None,
        "matched_count_min": min(matched_counts) if matched_counts else None,
        "matched_count_max": max(matched_counts) if matched_counts else None,
        "tolerance_seconds_min": min(tolerance_values) if tolerance_values else None,
        "tolerance_seconds_max": max(tolerance_values) if tolerance_values else None,
        "generated_cross_series_count": int(len(series_results)) if not is_psd_mode else None,
        "generated_cross_series_roles": (
            sorted(
                {
                    str(item.get("series_role"))
                    for item in series_results
                    if str(item.get("series_role") or "").strip()
                }
            )
            if not is_psd_mode
            else None
        ),
        "group_records": group_records,
        "group_preview_frame": build_legacy_target_group_preview_frame(group_records),
        "skipped_windows": skipped_windows,
    }
    if not is_psd_mode:
        first_cross_details = next(
            (
                dict(item.get("details", {}))
                for item in series_results
                if str(item.get("device_kind", "")).startswith("cross") and item.get("details") is not None
            ),
            {},
        )
        for key in (
            "cross_kernel",
            "cross_execution_path",
            "cross_implementation_id",
            "cross_implementation_label",
            "display_semantics",
            "display_value_source",
            "window",
            "detrend",
            "scaling",
            "return_onesided",
            "average",
        ):
            if first_cross_details.get(key) is not None:
                target_metadata[key] = first_cross_details.get(key)
        for item in series_results:
            details = item.get("details")
            if isinstance(details, dict):
                details["generated_cross_series_count"] = target_metadata.get("generated_cross_series_count")
                details["generated_cross_series_roles"] = list(target_metadata.get("generated_cross_series_roles") or [])
                details["canonical_cross_pairs"] = list(target_metadata.get("canonical_cross_pairs") or [])
                details["ygas_target_column"] = target_metadata.get("ygas_target_column")
                details["dat_target_column"] = target_metadata.get("dat_target_column")
    if not is_psd_mode and cross_reference_label is not None:
        target_metadata.update(
            {
                "reference_column": cross_reference_label,
                "target_column": kept_group_records[0].get("ygas_column") if kept_group_records else None,
                "analysis_context": TARGET_SPECTRAL_ANALYSIS_CONTEXT,
                "source_mode": TARGET_SPECTRAL_SOURCE_MODE,
                "target_spectral_context": base_target_context,
            }
        )

    return {
        "target_element": target_element,
        "series_results": series_results,
        "target_metadata": target_metadata,
    }

