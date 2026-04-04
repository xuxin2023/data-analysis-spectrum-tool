# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\数据分析\\fr_r_spectrum_tool_rebuild.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\数据分析\\assets', 'assets')],
    hiddenimports=['matplotlib.backends.backend_tkagg', 'matplotlib.backends._backend_tk'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='data_analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\数据分析\\assets\\app_icon.ico'],
)
