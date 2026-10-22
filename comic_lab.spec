# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

BASE_DIR = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(BASE_DIR, 'web'), 'web'),
    (os.path.join(BASE_DIR, 'fonts'), 'fonts'),
    (os.path.join(BASE_DIR, 'assets'), 'assets'),
    (os.path.join(BASE_DIR, 'glossary.json'), '.'),
    (os.path.join(BASE_DIR, 'naturalization_rules.py'), '.'),
    (os.path.join(BASE_DIR, 'character_memory.json'), '.'),
    (os.path.join(BASE_DIR, 'version_manifest.json'), '.'),
    (os.path.join(BASE_DIR, 'version.py'), '.'),
]

hiddenimports = [
    'version',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'starlette',
    'starlette.applications',
    'starlette.routing',
    'starlette.staticfiles',
    'starlette.responses',
    'starlette.middleware',
    'starlette.middleware.cors',
    'pydantic',
    'pydantic_core',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'webview',
    'webview.platforms',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'clr_loader',
    'pythonnet',
    'engine',
    'gemini_enhancements',
    'glossary_manager',
    'naturalization_rules',
    'bubble_lettering',
    'sfx_engine',
    'reading_order',
    'speaker_detector',
    'psutil',
    'requests',
    'dotenv',
]

excludes = [
    'torch',
    'torchvision',
    'easyocr',
    'streamlit',
    'altair',
    'pandas',
    'pyarrow',
    'sympy',
    'matplotlib',
    'scipy',
    'IPython',
    'jupyter',
    'pytest',
]

a = Analysis(
    ['desktop_app.py'],
    pathex=[BASE_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash(
    os.path.join(BASE_DIR, 'assets', 'splash.png'),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(66, 398),
    text_size=11,
    text_font='Segoe UI',
    text_color='#00e676',
    text_default='Studio ishga tushirilmoqda...',
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    [],
    exclude_binaries=True,
    name='ComicLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(BASE_DIR, 'assets', 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    splash.binaries,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ComicLab',
)
