# 数据分析软件说明

## 交付文件
- `fr_r_spectrum_tool_rebuild.py`
  GUI 主程序入口
- `spectrum_core.py`
  不依赖 Tkinter 的纯逻辑 core 模块
- `smoke_check_data_pipeline.py`
  无 GUI 自检脚本
- `rebuild_notes.md`
  版本补丁说明
- `requirements.txt`
  依赖列表

## 依赖安装
```bash
python -m pip install -r requirements.txt
```

## GUI 启动
```bash
python D:\数据分析\fr_r_spectrum_tool_rebuild.py
```

## 无 GUI 自检
```bash
python D:\数据分析\smoke_check_data_pipeline.py --mode single --ygas "C:\Users\A\Desktop\SAMPLE202603270700-202603270730.log"
python D:\数据分析\smoke_check_data_pipeline.py --mode single --dat "你的_TOA5_文件.dat"
python D:\数据分析\smoke_check_data_pipeline.py --mode dual --ygas "你的_ygas.log" --dat "你的_TOA5_文件.dat" --element CO2
```

## 推荐测试文件
- YGAS 高频 `txt / log`
- TOA5 `dat`
- 同时覆盖同一时间段的 `ygas + dat` 组合样本
