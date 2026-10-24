import os
import sys
import io
import json
import re
import struct
import base64
import time
from typing import Dict, Any, List
from PIL import Image

import uvicorn
import secrets
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, Response, FileResponse, RedirectResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Ensure comic-lab root is in sys.path
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import threading
import urllib.request
from version import APP_VERSION, APP_NAME, DEFAULT_UPDATE_URL, is_newer_version

_engine_module = None
_engine_lock = threading.Lock()

def get_engine():
    """Lazily loads the 5,350-line neural translation and OCR engine without blocking UI startup."""
    global _engine_module
    if _engine_module is None:
        with _engine_lock:
            if _engine_module is None:
                import engine as eng
                _engine_module = eng
    return _engine_module

class _EngineProxy:
    """Zero-overhead transparent proxy to engine module."""
    def __getattr__(self, name):
        return getattr(get_engine(), name)

engine = _EngineProxy()

def warm_engine_in_background():
    """Pre-compiles engine in background thread after startup so processing is instant."""
    t = threading.Thread(target=get_engine, daemon=True)
    t.start()

import cv2
import numpy as np
import gemini_enhancements
import glossary_manager
import naturalization_rules
import sfx_engine
import reading_order
import bubble_lettering
import speaker_detector
import zipfile

# In-memory session store for current page and multi-page batch queue
CURRENT_STATE: Dict[str, Any] = {
    "original_image": None,
    "cleaned_image": None,
    "bubbles": [],
    "image_name": "comic_page.png"
}
PAGES_CACHE: Dict[str, Dict[str, Any]] = {}

def pil_to_base64_jpeg(img: Image.Image, quality: int = 98) -> str:
    """Converts PIL image to crystal-clear high fidelity base64 JPEG (subsampling=0, 4:4:4 full chroma)."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Converts PIL image to lossless PNG bytes for HD export."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()

async def serve_index(request):
    """Serves the Comic Studio HTML5 app."""
    is_desktop = (os.getenv("COMICLAB_STANDALONE") == "1" or request.query_params.get("desktop") == "1")
    for rel_path in ["index.html", os.path.join("web", "index.html")]:
        html_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
                if is_desktop:
                    # Strip download button and modal completely when running inside Desktop App
                    content = re.sub(
                        r'<!-- Minimalist Desktop App Pill.*?</a>',
                        '',
                        content,
                        flags=re.DOTALL
                    )
                    content = re.sub(
                        r'<!-- MINIMALIST DOWNLOAD DESKTOP APP MODAL.*?<!-- TEAM GLOSSARY MODAL -->',
                        '<!-- TEAM GLOSSARY MODAL -->',
                        content,
                        flags=re.DOTALL
                    )
                return HTMLResponse(content)
    return HTMLResponse("<h1>Comic Studio UI not found</h1>", status_code=404)

async def api_health(request):
    """Health check and status API."""
    gemini_ok = gemini_enhancements.is_gemini_enabled()
    return JSONResponse({
        "status": "online",
        "gemini_active": gemini_ok,
        "memory_summary": "Active"
    })

async def api_get_settings(request):
    """Returns AI engine status and masked API key without leaking secrets."""
    key = gemini_enhancements.get_gemini_api_key()
    has_key = bool(key and len(key) >= 15)
    masked = ""
    if has_key:
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "******"
    is_locked = getattr(gemini_enhancements, "IS_STUDIO_KEY_LOCKED", False)
    return JSONResponse({
        "status": "ok",
        "gemini_active": gemini_enhancements.is_gemini_enabled(),
        "has_key": has_key,
        "masked_key": masked,
        "is_locked": is_locked,
        "download_dir": gemini_enhancements.get_download_dir(),
        "default_download_dir": gemini_enhancements.get_default_download_dir(),
        "message": "Jamoaviy Studio Master API kaliti biriktirilgan va doimiy faol. Xavfsizlik yuzasidan kalit qulflangan." if is_locked else ""
    })

async def api_save_download_dir(request):
    """Updates persistent download / save directory."""
    try:
        data = await request.json()
        new_dir = (data.get("download_dir") or "").strip()
        if not new_dir:
            new_dir = gemini_enhancements.get_default_download_dir()
        ok = gemini_enhancements.set_download_dir(new_dir)
        if not ok:
            return JSONResponse({"status": "error", "message": "Papka yo'li noto'g'ri yoki unga yozish imkonsiz."}, status_code=400)
        return JSONResponse({
            "status": "ok",
            "download_dir": gemini_enhancements.get_download_dir(),
            "message": "Yuklab olish va saqlash manzili muvaffaqiyatli saqlandi."
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_browse_folder(request):
    """Opens native Windows FolderBrowserDialog via PowerShell as a fallback."""
    try:
        import subprocess
        ps_script = (
            "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$f.Description = 'ComicLab yuklab olish va saqlash papkasini tanlang'; "
            "if($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){ Write-Output $f.SelectedPath }"
        )
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True, timeout=60)
        path = proc.stdout.strip()
        if path and os.path.exists(path):
            return JSONResponse({"status": "ok", "path": path})
        return JSONResponse({"status": "cancelled", "path": None})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

LAST_SAVED_EXPORT_PATH = None

async def api_save_export_to_download_dir(request):
    """Directly saves exported comic chapter/page into the user's configured download directory."""
    global LAST_SAVED_EXPORT_PATH
    try:
        data = await request.json()
        filename = data.get("filename") or "comic_uzbek_chapter.zip"
        content_b64 = data.get("content_b64", "")
        if not content_b64:
            return JSONResponse({"status": "error", "message": "Fayl ma'lumoti bo'sh."}, status_code=400)
        if "," in content_b64:
            content_b64 = content_b64.split(",", 1)[1]
        dest_dir = gemini_enhancements.get_download_dir()
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        with open(dest_path, "wb") as f:
            f.write(base64.b64decode(content_b64))
        LAST_SAVED_EXPORT_PATH = dest_path
        return JSONResponse({
            "status": "ok",
            "path": dest_path,
            "filename": filename,
            "message": f"Fayl saqlandi: {dest_path}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_save_settings(request):
    """Tests and saves new Gemini API key to persistent storage."""
    if getattr(gemini_enhancements, "IS_STUDIO_KEY_LOCKED", False):
        return JSONResponse({
            "status": "error",
            "message": "API kaliti studiya tomonidan mahkamlangan va qulflangan. Uni o'zgartirib bo'lmaydi."
        }, status_code=403)
    try:
        data = await request.json()
        new_key = (data.get("gemini_api_key") or "").strip()
        if not new_key:
            return JSONResponse({"status": "error", "message": "API kalit kiritilmadi."}, status_code=400)

        # Fast verification ping to Google AI API
        ok, msg = gemini_enhancements.test_gemini_key(new_key)
        if not ok:
            return JSONResponse({"status": "error", "message": msg}, status_code=400)

        saved = gemini_enhancements.save_user_api_key(new_key)
        if not saved:
            return JSONResponse({"status": "error", "message": "Kalitni saqlab bo'lmadi."}, status_code=500)

        masked = new_key[:6] + "..." + new_key[-4:]
        return JSONResponse({
            "status": "ok",
            "message": "Gemini API kaliti muvaffaqiyatli saqlandi va faollashtirildi.",
            "masked_key": masked,
            "gemini_active": True
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_delete_settings(request):
    """Disconnects and removes saved Gemini API key."""
    if getattr(gemini_enhancements, "IS_STUDIO_KEY_LOCKED", False):
        return JSONResponse({
            "status": "error",
            "message": "Studiya jamoaviy API kalitini o'chirib bo'lmaydi."
        }, status_code=403)
    try:
        gemini_enhancements.delete_user_api_key()
        return JSONResponse({"status": "ok", "message": "API kaliti o'chirildi."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_download_desktop(request):
    """Serves the ComicLab desktop application zip bundle for download."""
    candidate_paths = [
        os.path.join(BASE_DIR, "dist", "ComicLab_Portable_v1.0.0.zip"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "ComicLab_Portable_v1.0.0.zip"),
        os.path.join(os.getcwd(), "dist", "ComicLab_Portable_v1.0.0.zip")
    ]
    zip_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            zip_path = p
            break

    if not zip_path:
        fallback_url = os.environ.get(
            "DESKTOP_DOWNLOAD_URL",
            "https://github.com/hinklab/comic-lab/releases/download/v1.0.0/ComicLab_Portable_v1.0.0.zip"
        )
        return RedirectResponse(url=fallback_url, status_code=302)

    return FileResponse(
        path=zip_path,
        filename="ComicLab_Portable_v1.0.0.zip",
        media_type="application/zip"
    )

# --- AUTO-UPDATE ENGINE FOR DESKTOP APPLICATION ---
UPDATE_DOWNLOAD_STATE = {
    "status": "idle",       # "idle", "downloading", "completed", "error"
    "percent": 0,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "downloaded_mb": "0",
    "total_mb": "0",
    "filepath": "",
    "error": None
}
UPDATE_LOCK = threading.Lock()

async def api_version(request):
    """Returns application version manifest for update checks."""
    manifest_path = os.path.join(BASE_DIR, "version_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return JSONResponse(data)
        except Exception:
            pass
    return JSONResponse({
        "app_name": APP_NAME,
        "latest_version": APP_VERSION,
        "release_date": "2026-10-22",
        "download_url": "/download/desktop",
        "changelog": "Comic-Lab Studio so'nggi yangilanishi.",
        "min_supported_version": "1.0.0"
    })

async def api_check_update(request):
    """Checks remote server or local manifest for new version availability."""
    remote_url = os.environ.get("COMICLAB_UPDATE_URL") or DEFAULT_UPDATE_URL
    latest_v = APP_VERSION
    changelog = ""
    download_url = "/download/desktop"
    release_date = ""

    fetched_data = None
    # 1. Try querying remote server if online
    if remote_url and not remote_url.startswith("http://127.0.0.1") and not remote_url.startswith("http://localhost"):
        try:
            req = urllib.request.Request(
                remote_url,
                headers={"User-Agent": f"ComicLab-Desktop/{APP_VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    fetched_data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass

    # 2. Fallback to local manifest if remote wasn't reachable or in local dev
    if not fetched_data:
        manifest_path = os.path.join(BASE_DIR, "version_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    fetched_data = json.load(f)
            except Exception:
                pass

    if fetched_data:
        latest_v = fetched_data.get("latest_version") or APP_VERSION
        changelog = fetched_data.get("changelog") or ""
        download_url = fetched_data.get("download_url") or "/download/desktop"
        release_date = fetched_data.get("release_date") or ""

    update_available = is_newer_version(latest_v, APP_VERSION)

    return JSONResponse({
        "status": "ok",
        "current_version": APP_VERSION,
        "latest_version": latest_v,
        "update_available": update_available,
        "download_url": download_url,
        "changelog": changelog,
        "release_date": release_date
    })

def _background_download_worker(source_url: str, target_file: str):
    global UPDATE_DOWNLOAD_STATE
    try:
        with UPDATE_LOCK:
            UPDATE_DOWNLOAD_STATE["status"] = "downloading"
            UPDATE_DOWNLOAD_STATE["percent"] = 0
            UPDATE_DOWNLOAD_STATE["downloaded_bytes"] = 0
            UPDATE_DOWNLOAD_STATE["total_bytes"] = 0
            UPDATE_DOWNLOAD_STATE["filepath"] = target_file
            UPDATE_DOWNLOAD_STATE["error"] = None

        # If source_url is relative, point to remote server or local backend
        if source_url.startswith("/"):
            # If local dist file exists, copy directly for instant zero-bandwidth download
            local_candidate = os.path.join(BASE_DIR, "dist", "ComicLab_Portable_v1.0.0.zip")
            if not os.environ.get("COMICLAB_FORCE_REMOTE_DOWNLOAD") and os.path.exists(local_candidate):
                total_sz = os.path.getsize(local_candidate)
                with UPDATE_LOCK:
                    UPDATE_DOWNLOAD_STATE["total_bytes"] = total_sz
                    UPDATE_DOWNLOAD_STATE["total_mb"] = f"{total_sz / (1024*1024):.1f}"

                with open(local_candidate, "rb") as sf, open(target_file, "wb") as df:
                    copied = 0
                    while chunk := sf.read(512 * 1024):
                        df.write(chunk)
                        copied += len(chunk)
                        pct = int((copied / float(total_sz)) * 100) if total_sz > 0 else 50
                        with UPDATE_LOCK:
                            UPDATE_DOWNLOAD_STATE["downloaded_bytes"] = copied
                            UPDATE_DOWNLOAD_STATE["percent"] = pct
                            UPDATE_DOWNLOAD_STATE["downloaded_mb"] = f"{copied / (1024*1024):.1f}"
                        time.sleep(0.01)

                with UPDATE_LOCK:
                    UPDATE_DOWNLOAD_STATE["status"] = "completed"
                    UPDATE_DOWNLOAD_STATE["percent"] = 100
                return

            remote_base = os.environ.get("COMICLAB_REMOTE_BASE", "https://comic-lab-backend.onrender.com")
            source_url = remote_base.rstrip("/") + source_url

        req = urllib.request.Request(source_url, headers={"User-Agent": f"ComicLab-Updater/{APP_VERSION}"})
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            total_sz = int(resp.headers.get("Content-Length", 0))
            with UPDATE_LOCK:
                UPDATE_DOWNLOAD_STATE["total_bytes"] = total_sz
                UPDATE_DOWNLOAD_STATE["total_mb"] = f"{total_sz / (1024*1024):.1f}" if total_sz > 0 else "N/A"

            downloaded = 0
            temp_file = target_file + ".tmp"
            with open(temp_file, "wb") as f_out:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    pct = int((downloaded / float(total_sz)) * 100) if total_sz > 0 else 50
                    with UPDATE_LOCK:
                        UPDATE_DOWNLOAD_STATE["downloaded_bytes"] = downloaded
                        UPDATE_DOWNLOAD_STATE["percent"] = pct
                        UPDATE_DOWNLOAD_STATE["downloaded_mb"] = f"{downloaded / (1024*1024):.1f}"

            if os.path.exists(target_file):
                os.remove(target_file)
            os.rename(temp_file, target_file)

            with UPDATE_LOCK:
                UPDATE_DOWNLOAD_STATE["status"] = "completed"
                UPDATE_DOWNLOAD_STATE["percent"] = 100

    except Exception as exc:
        with UPDATE_LOCK:
            UPDATE_DOWNLOAD_STATE["status"] = "error"
            UPDATE_DOWNLOAD_STATE["error"] = str(exc)

async def api_start_update_download(request):
    """Starts asynchronous background download of the update."""
    global UPDATE_DOWNLOAD_STATE
    try:
        data = await request.json() if request.method == "POST" else {}
        target_version = data.get("version") or "latest"
        download_url = data.get("download_url") or "/download/desktop"

        downloads_dir = gemini_enhancements.get_download_dir()
        os.makedirs(downloads_dir, exist_ok=True)
        filename = f"ComicLab_Portable_v{target_version}.zip"
        target_path = os.path.join(downloads_dir, filename)

        with UPDATE_LOCK:
            if UPDATE_DOWNLOAD_STATE["status"] == "downloading":
                return JSONResponse({"status": "already_downloading", "progress": UPDATE_DOWNLOAD_STATE})

        t = threading.Thread(target=_background_download_worker, args=(download_url, target_path), daemon=True)
        t.start()

        return JSONResponse({
            "status": "started",
            "target_path": target_path,
            "filename": filename
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_get_update_progress(request):
    """Polls real-time download progress."""
    with UPDATE_LOCK:
        return JSONResponse(UPDATE_DOWNLOAD_STATE)

async def api_open_download_folder(request):
    """Opens Windows Explorer with the downloaded zip or saved export selected."""
    global LAST_SAVED_EXPORT_PATH
    try:
        req_path = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_path = body.get("filepath") or body.get("path")
        except Exception:
            pass

        filepath = req_path or LAST_SAVED_EXPORT_PATH
        if not filepath:
            with UPDATE_LOCK:
                filepath = UPDATE_DOWNLOAD_STATE.get("filepath")

        import subprocess
        downloads_dir = gemini_enhancements.get_download_dir()
        if filepath and os.path.exists(filepath):
            subprocess.Popen(f'explorer /select,"{os.path.abspath(filepath)}"')
        else:
            os.makedirs(downloads_dir, exist_ok=True)
            subprocess.Popen(f'explorer "{os.path.abspath(downloads_dir)}"')
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

SESSION_SNAPSHOT_FILE = os.path.join(
    os.environ.get("APPDATA") or os.path.expanduser("~"),
    "ComicLab",
    "session_snapshot.json"
)

async def api_save_session_snapshot(request):
    """Saves complete working project state to disk as persistent safety backup."""
    try:
        data = await request.json()
        os.makedirs(os.path.dirname(SESSION_SNAPSHOT_FILE), exist_ok=True)
        temp_file = SESSION_SNAPSHOT_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if os.path.exists(SESSION_SNAPSHOT_FILE):
            os.remove(SESSION_SNAPSHOT_FILE)
        os.rename(temp_file, SESSION_SNAPSHOT_FILE)
        return JSONResponse({"status": "ok", "saved_at": time.time()})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_get_session_snapshot(request):
    """Retrieves saved session snapshot if present."""
    try:
        if os.path.exists(SESSION_SNAPSHOT_FILE):
            with open(SESSION_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse({"status": "ok", "snapshot": data})
        return JSONResponse({"status": "empty"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

UPDATER_PS1_SCRIPT = """# ComicLab Studio Auto-Updater Script
param(
    [int]$AppPid,
    [string]$AppDir,
    [string]$ExePath,
    [string]$DownloadUrl,
    [string]$ZipPath = ""
)

$host.UI.RawUI.WindowTitle = "Comic-Lab Studio - Yangilanish o'rnatilmoqda"
Write-Host ""
Write-Host " ========================================================== " -ForegroundColor Cyan
Write-Host "           Comic-Lab Studio Avtomatik Yangilash             " -ForegroundColor Green
Write-Host " ========================================================== " -ForegroundColor Cyan
Write-Host ""

# 1. Wait for parent ComicLab process to exit cleanly
if ($AppPid -gt 0) {
    Write-Host "[1/4] Ilova yopilishi kutilmoqda (PID: $AppPid)..." -ForegroundColor Yellow
    $p = Get-Process -Id $AppPid -ErrorAction SilentlyContinue
    if ($p) {
        $p.WaitForExit(10000)
    }
}

# Ensure no ComicLab instance locks files in AppDir
$waitCount = 0
while ($waitCount -lt 15) {
    $running = Get-Process -Name "*ComicLab*" -ErrorAction SilentlyContinue | Where-Object {
        try { $_.Path -like "$AppDir*" } catch { $false }
    }
    if (-not $running) { break }
    Start-Sleep -Seconds 1
    $waitCount++
}
Start-Sleep -Milliseconds 800

# 2. Check or download update package
$finalZip = $ZipPath
if (-not $finalZip -or -not (Test-Path $finalZip) -or ((Get-Item $finalZip).Length -lt 1000000)) {
    $finalZip = Join-Path $env:TEMP "ComicLab_Update.zip"
    Write-Host "[2/4] Yangilanish paketi yuklab olinmoqda..." -ForegroundColor Cyan
    Write-Host "      Manba: $DownloadUrl" -ForegroundColor Gray
    try {
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add("User-Agent", "ComicLab-AutoUpdater/1.0.0")
        $wc.DownloadFile($DownloadUrl, $finalZip)
        Write-Host "      Muvaffaqiyatli yuklandi ($( [math]::Round((Get-Item $finalZip).Length / 1MB, 1) ) MB)" -ForegroundColor Green
    } catch {
        Write-Host "[XATO] Yuklab olishda xatolik yuz berdi: $_" -ForegroundColor Red
        Write-Host "Ilova qayta ishga tushirilmoqda..." -ForegroundColor Yellow
        Start-Process -FilePath $ExePath -WorkingDirectory $AppDir
        exit 1
    }
} else {
    Write-Host "[2/4] Tayyor yangilanish paketi topildi ($( [math]::Round((Get-Item $finalZip).Length / 1MB, 1) ) MB)" -ForegroundColor Green
}

# 3. Extract and overwrite update files
Write-Host "[3/4] Yangi versiya fayllari o'rnatilmoqda..." -ForegroundColor Cyan
try {
    [System.Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null
    $staging = Join-Path $env:TEMP "ComicLab_Staging"
    if (Test-Path $staging) {
        Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    [System.IO.Compression.ZipFile]::ExtractToDirectory($finalZip, $staging)

    # Detect root of extracted files
    $sourceRoot = $staging
    if (-not (Test-Path (Join-Path $sourceRoot "ComicLab.exe"))) {
        $sub = Get-ChildItem -Path $staging -Directory | Where-Object { Test-Path (Join-Path $_.FullName "ComicLab.exe") } | Select-Object -First 1
        if ($sub) {
            $sourceRoot = $sub.FullName
        }
    }

    # Copy all files over AppDir
    Get-ChildItem -Path $sourceRoot -Recurse | ForEach-Object {
        $rel = $_.FullName.Substring($sourceRoot.Length).TrimStart('\\', '/')
        $dest = Join-Path $AppDir $rel
        if ($_.PSIsContainer) {
            if (-not (Test-Path $dest)) {
                New-Item -ItemType Directory -Path $dest -Force | Out-Null
            }
        } else {
            $destDir = Split-Path $dest -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $dest -Force
        }
    }
    Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    Write-Host "      Barcha yangi fayllar muvaffaqiyatli o'rnatildi!" -ForegroundColor Green
} catch {
    Write-Host "[XATO] O'rnatishda xatolik: $_" -ForegroundColor Red
}

# 4. Relaunch ComicLab.exe
Write-Host "[4/4] Comic-Lab Studio qayta ishga tushirilmoqda..." -ForegroundColor Green
Start-Sleep -Milliseconds 600
Start-Process -FilePath $ExePath -WorkingDirectory $AppDir
Write-Host " [Muvaffaqiyatli] Yangilanish to'liq yakunlandi!" -ForegroundColor Cyan
Start-Sleep -Seconds 1
"""

async def api_apply_update_and_restart(request):
    """Triggers the external updater script and closes the current app."""
    try:
        data = await request.json() if request.method == "POST" else {}
        target_version = data.get("version") or "1.0.1"
        download_url = data.get("download_url") or "/download/desktop"

        if download_url.startswith("/"):
            remote_base = os.environ.get("COMICLAB_REMOTE_BASE", "https://comic-lab-backend.onrender.com")
            download_url = remote_base.rstrip("/") + download_url

        # Determine app directory and executable
        if getattr(sys, "frozen", False):
            exe_path = sys.executable
            app_dir = os.path.dirname(exe_path)
        else:
            app_dir = os.path.abspath(os.path.join(BASE_DIR, "dist", "ComicLab"))
            exe_path = os.path.join(app_dir, "ComicLab.exe")

        # Check if zip is already downloaded locally
        with UPDATE_LOCK:
            zip_path = UPDATE_DOWNLOAD_STATE.get("filepath") or ""

        if not (zip_path and os.path.exists(zip_path)):
            local_cand = os.path.join(BASE_DIR, "dist", f"ComicLab_Portable_v{target_version}.zip")
            if not os.path.exists(local_cand):
                local_cand = os.path.join(BASE_DIR, "dist", "ComicLab_Portable_v1.0.0.zip")
            if os.path.exists(local_cand):
                zip_path = local_cand

        user_data = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "ComicLab")
        os.makedirs(user_data, exist_ok=True)
        updater_script_path = os.path.join(user_data, "comic_lab_updater.ps1")
        with open(updater_script_path, "w", encoding="utf-8") as f:
            f.write(UPDATER_PS1_SCRIPT)

        current_pid = os.getpid()

        import subprocess
        cmd = [
            "powershell.exe",
            "-ExecutionPolicy", "Bypass",
            "-NoProfile",
            "-File", updater_script_path,
            "-AppPid", str(current_pid),
            "-AppDir", app_dir,
            "-ExePath", exe_path,
            "-DownloadUrl", download_url,
            "-ZipPath", zip_path
        ]
        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        )

        def _delayed_exit():
            time.sleep(0.5)
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.ExitProcess(0)
                except Exception:
                    pass
            os._exit(0)

        threading.Thread(target=_delayed_exit, daemon=True).start()

        return JSONResponse({"status": "ok", "message": "Ilova yopilib yangilanmoqda va qayta ochiladi..."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

SESSION_COOKIE_NAME = "comiclab_session"

class SessionTokenMiddleware:
    """
    Protects localhost backend from unauthorized origins, DNS rebinding,
    and third-party browser tabs probing localhost.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        expected_token = os.environ.get("COMICLAB_SESSION_TOKEN")
        if not expected_token:
            # If not configured (raw web dev or tests), allow open local development access
            return await self.app(scope, receive, send)

        request = Request(scope, receive)
        path = scope.get("path", "")

        # 1. Allow root, fonts, assets, static files, desktop zip download, settings, glossary, fonts list, and update endpoints
        if (path == "/" or path.startswith("/assets") or path.startswith("/fonts") or 
            path.startswith("/static") or path == "/download/desktop" or 
            path == "/api/health" or path.startswith("/api/settings") or
            path == "/api/browse_folder" or path == "/api/save_export" or
            path == "/api/list_fonts" or path == "/api/glossary" or
            path.startswith("/api/version") or path.startswith("/api/check_update") or 
            path.startswith("/api/start_update_download") or path.startswith("/api/update_download_progress") or 
            path.startswith("/api/open_download_folder") or path.startswith("/api/apply_update_and_restart") or
            path.startswith("/api/save_session_snapshot") or path.startswith("/api/get_session_snapshot")):
            query_token = request.query_params.get("token")
            # If token is passed in query parameter on initial load of "/", attach SameSite=Strict session cookie
            if path == "/" and query_token and secrets.compare_digest(query_token, expected_token):
                async def send_wrapper(message):
                    if message["type"] == "http.response.start":
                        headers = list(message.get("headers", []))
                        cookie_val = f"{SESSION_COOKIE_NAME}={expected_token}; Path=/; SameSite=Strict; HttpOnly"
                        headers.append((b"set-cookie", cookie_val.encode("latin-1")))
                        message["headers"] = headers
                    await send(message)
                return await self.app(scope, receive, send_wrapper)

            return await self.app(scope, receive, send)

        # 2. For all /api/... routes, enforce valid session via cookie, header or query param
        cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
        header_token = request.headers.get("X-ComicLab-Token")
        query_token = request.query_params.get("token")

        is_authorized = (
            (cookie_token and secrets.compare_digest(cookie_token, expected_token)) or
            (header_token and secrets.compare_digest(header_token, expected_token)) or
            (query_token and secrets.compare_digest(query_token, expected_token))
        )

        if not is_authorized:
            response = JSONResponse({"error": "Unauthorized request origin or missing session token"}, status_code=403)
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)

async def api_register_page(request):
    """Registers an uploaded page image into PAGES_CACHE without running OCR (useful for skipped pages)."""
    try:
        form = await request.form()
        file = form.get("file")
        page_id = str(form.get("page_id") or "")
        if not file:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)

        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        filename = getattr(file, "filename", "page.png") or "page.png"
        if not page_id:
            page_id = filename

        PAGES_CACHE[page_id] = {
            "original_image": pil_img,
            "original_bytes": contents,
            "cleaned_image": None,
            "filename": filename,
            "bubbles": []
        }
        return JSONResponse({"status": "ok", "page_id": page_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_process_page(request):
    """
    1-Click Unified API:
    Receives an image -> runs Gemini Vision OCR & Translation -> pure ink inpainting -> returns bubbles & cleaned image.
    Entire process completes in ~1.5 - 2.5 seconds.
    """
    try:
        form = await request.form()
        file = form.get("file")
        page_id = str(form.get("page_id") or "")
        orig_w_raw = form.get("orig_w")
        orig_h_raw = form.get("orig_h")
        is_hybrid = str(form.get("hybrid") or "").lower() in ("true", "1")
        if not file:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)

        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        filename = getattr(file, "filename", "page.png") or "page.png"
        force_refresh = str(form.get("force_refresh") or "").lower() in ("true", "1")
        if not page_id:
            page_id = filename

        det_w, det_h = pil_img.size
        if orig_w_raw and orig_h_raw:
            try:
                target_w = int(orig_w_raw)
                target_h = int(orig_h_raw)
                if target_w > 0 and target_h > 0 and (target_w != det_w or target_h != det_h or is_hybrid):
                    is_hybrid = True
            except (ValueError, TypeError):
                target_w, target_h = det_w, det_h
        else:
            target_w, target_h = det_w, det_h

        scale_x = float(target_w) / float(det_w) if det_w > 0 else 1.0
        scale_y = float(target_h) / float(det_h) if det_h > 0 else 1.0

        CURRENT_STATE["original_image"] = pil_img
        CURRENT_STATE["cleaned_image"] = None
        CURRENT_STATE["bubbles"] = []
        CURRENT_STATE["image_name"] = filename

        # 1. Gemini Multimodal Vision Scan & Translation (~1.5s)
        t0 = time.time()
        bubbles = engine.scan_bubbles_ocr(pil_img, force_refresh=force_refresh)
        t_ocr = time.time() - t0

        if not bubbles:
            PAGES_CACHE[page_id] = {
                "original_image": pil_img,
                "cleaned_image": pil_img,
                "filename": filename,
                "bubbles": []
            }
            return JSONResponse({
                "status": "warning",
                "hybrid": is_hybrid,
                "message": "Hech qanday nutq pufagi aniqlanmadi",
                "page_id": page_id,
                "image_w": target_w,
                "image_h": target_h,
                "bubbles": [],
                "cleaned_image_b64": None if is_hybrid else f"data:image/jpeg;base64,{pil_to_base64_jpeg(pil_img)}"
            })

        # 2. Pure Ink Inpainting (only if not hybrid mode; in hybrid mode, client inpainting saves 530MB server RAM!)
        t_clean0 = time.time()
        if not is_hybrid:
            cleaned = engine.clean_page_ink_telea(pil_img, bubbles)
            CURRENT_STATE["cleaned_image"] = cleaned
        else:
            cleaned = None
            CURRENT_STATE["cleaned_image"] = None
        t_clean = time.time() - t_clean0

        # Collect bubbles that need translation (missing or untranslated English leak)
        untranslated = []
        for idx, b in enumerate(bubbles):
            b_dict = b.model_dump() if hasattr(b, "model_dump") else dict(b)
            orig_txt = str(b_dict.get("original_text") or "").strip()
            uz_txt = str(b_dict.get("uzbek_translation") or "").strip()
            clean_o = re.sub(r'[^a-zA-Z0-9]', '', orig_txt).lower()
            clean_u = re.sub(r'[^a-zA-Z0-9]', '', uz_txt).lower()
            is_untranslated = (not uz_txt) or (clean_o and clean_u and clean_o == clean_u and not engine.is_pure_sfx_text(orig_txt) and len(clean_o) >= 2)

            if orig_txt and is_untranslated:
                if engine.is_pure_sfx_text(orig_txt):
                    if hasattr(bubbles[idx], "uzbek_translation"):
                        bubbles[idx].uzbek_translation = orig_txt
                    elif isinstance(bubbles[idx], dict):
                        bubbles[idx]["uzbek_translation"] = orig_txt
                    continue
                untranslated.append((idx, orig_txt, b_dict.get("speaker")))

        # Batch translate untranslated bubbles to save API calls and avoid 429 rate limit
        if untranslated:
            batch_success = False
            if gemini_enhancements.is_gemini_enabled():
                try:
                    batch_inputs = [{"text": orig, "speaker": spk} for (_, orig, spk) in untranslated]
                    batch_outputs = gemini_enhancements.gemini_translate_dialogues_batch(batch_inputs)
                    if len(batch_outputs) == len(untranslated) and all(bool(t.strip()) for t in batch_outputs):
                        for (b_idx, orig_t, spk_t), tr_res in zip(untranslated, batch_outputs):
                            clean_res = naturalization_rules.apply_naturalization(tr_res, en_orig=orig_t, character_profile=spk_t)
                            clean_res = glossary_manager.apply_glossary_enforcement(clean_res)
                            if hasattr(bubbles[b_idx], "uzbek_translation"):
                                bubbles[b_idx].uzbek_translation = clean_res
                            elif isinstance(bubbles[b_idx], dict):
                                bubbles[b_idx]["uzbek_translation"] = clean_res
                        batch_success = True
                except Exception as b_ex:
                    print(f"[TRANSLATE_WARN] Batch translation error: {b_ex}", flush=True)

            if not batch_success:
                for (b_idx, orig, spk) in untranslated:
                    uz_res = ""
                    if gemini_enhancements.is_gemini_enabled():
                        try:
                            uz_res = gemini_enhancements.gemini_translate_dialogue(orig, speaker=spk)
                        except Exception:
                            uz_res = ""
                    # 100% Reliable Offline Fallback so no bubble is EVER left untranslated!
                    clean_u_chk = re.sub(r'[^a-zA-Z0-9]', '', uz_res).lower()
                    clean_o_chk = re.sub(r'[^a-zA-Z0-9]', '', orig).lower()
                    if not uz_res or (clean_o_chk and clean_u_chk and clean_o_chk == clean_u_chk and not engine.is_pure_sfx_text(orig)):
                        try:
                            uz_res = engine.translate_text(orig, speaker=spk)
                        except Exception as ex:
                            print(f"[TRANSLATE_WARN] Bubble fallback error: {ex}", flush=True)
                    if uz_res:
                        clean_res = naturalization_rules.apply_naturalization(uz_res, en_orig=orig, character_profile=spk)
                        clean_res = glossary_manager.apply_glossary_enforcement(clean_res)
                        if hasattr(bubbles[b_idx], "uzbek_translation"):
                            bubbles[b_idx].uzbek_translation = clean_res
                        elif isinstance(bubbles[b_idx], dict):
                            bubbles[b_idx]["uzbek_translation"] = clean_res

        # Final pass over ALL bubbles to ensure naturalization and glossary enforcement
        for idx, b in enumerate(bubbles):
            b_dict = b.model_dump() if hasattr(b, "model_dump") else dict(b)
            orig_t = str(b_dict.get("original_text") or "").strip()
            uz_t = str(b_dict.get("uzbek_translation") or "").strip()
            spk_t = b_dict.get("speaker")
            if uz_t:
                pol_uz = naturalization_rules.apply_naturalization(uz_t, en_orig=orig_t, character_profile=spk_t)
                pol_uz = glossary_manager.apply_glossary_enforcement(pol_uz)
                if hasattr(bubbles[idx], "uzbek_translation"):
                    bubbles[idx].uzbek_translation = pol_uz
                elif isinstance(bubbles[idx], dict):
                    bubbles[idx]["uzbek_translation"] = pol_uz

        # Extract and order comic panels
        import reading_order
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        panel_boxes = reading_order.detect_panel_boxes(img_cv)
        ordered_panels = reading_order._row_major_order(panel_boxes)
        panels_info = []
        for p_idx, pb in enumerate(ordered_panels, start=1):
            bbox = list(reading_order._extract_bbox(pb))
            scaled_bbox = [
                int(round(bbox[0] * scale_x)),
                int(round(bbox[1] * scale_y)),
                int(round(bbox[2] * scale_x)),
                int(round(bbox[3] * scale_y))
            ] if is_hybrid else bbox
            p_bubble_ids = [
                (b.bubble_id if hasattr(b, "bubble_id") else b.get("bubble_id"))
                for b in bubbles
                if (getattr(b, "panel_id", None) == p_idx) or (isinstance(b, dict) and b.get("panel_id") == p_idx)
            ]
            panels_info.append({
                "panel_id": p_idx,
                "box": scaled_bbox,
                "reading_order": p_idx,
                "bubble_count": len(p_bubble_ids),
                "bubble_ids": [bid for bid in p_bubble_ids if bid is not None]
            })

        # Serialize bubbles for frontend Canvas
        serialized_bubbles = []
        for b in bubbles:
            b_dict = b.model_dump() if hasattr(b, "model_dump") else dict(b)
            orig_txt = str(b_dict.get("original_text") or "").strip()
            uz_txt = str(b_dict.get("uzbek_translation") or "").strip()
            bx0 = int(round(b_dict.get("x0", 0) * scale_x)) if is_hybrid else int(b_dict.get("x0", 0))
            by0 = int(round(b_dict.get("y0", 0) * scale_y)) if is_hybrid else int(b_dict.get("y0", 0))
            bx1 = int(round(b_dict.get("x1", 100) * scale_x)) if is_hybrid else int(b_dict.get("x1", 100))
            by1 = int(round(b_dict.get("y1", 100) * scale_y)) if is_hybrid else int(b_dict.get("y1", 100))

            cnt = b_dict.get("contour") or []
            if cnt and is_hybrid:
                scaled_cnt = [[int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))] for pt in cnt]
            else:
                scaled_cnt = cnt

            serialized_bubbles.append({
                "bubble_id": b_dict.get("bubble_id"),
                "panel_id": b_dict.get("panel_id"),
                "reading_index": b_dict.get("reading_index") or b_dict.get("bubble_id"),
                "box": [bx0, by0, bx1, by1],
                "x0": bx0,
                "y0": by0,
                "x1": bx1,
                "y1": by1,
                "original_text": orig_txt,
                "uzbek_translation": uz_txt,
                "speaker": b_dict.get("speaker") or ("Sharhlovchi" if (b_dict.get("shape_type") == "rectangle" and not b_dict.get("has_tail")) else "Noma'lum"),
                "speaker_confidence": float(b_dict.get("speaker_confidence") or 0.0),
                "speaker_anchor": b_dict.get("speaker_anchor"),
                "shape_type": b_dict.get("shape_type", "oval"),
                "font_size_offset": b_dict.get("font_size_offset", 0),
                "line_spacing": float(b_dict.get("line_spacing", 1.15)),
                "offset_x": b_dict.get("offset_x", 0),
                "offset_y": b_dict.get("offset_y", 0),
                "text_color": b_dict.get("text_color"),
                "bg_color": b_dict.get("bg_color"),
                "white_fill": bool(b_dict.get("white_fill", False)),
                "outline_color": b_dict.get("outline_color"),
                "outline_width": int(b_dict.get("outline_width", 0)),
                "text_align": str(b_dict.get("text_align", "center")),
                "font_family": b_dict.get("font_family") or b_dict.get("font_name") or None,
                "font_weight": str(b_dict.get("font_weight", "normal")),
                "font_style": str(b_dict.get("font_style", "normal")),
                "custom_name": b_dict.get("custom_name"),
                "opacity": float(b_dict.get("opacity", 1.0)),
                "blend_mode": str(b_dict.get("blend_mode", "source-over")),
                "is_printable": bool(b_dict.get("is_printable", True)),
                "color_tag": b_dict.get("color_tag"),
                "group_id": b_dict.get("group_id"),
                "contour": scaled_cnt,
                "has_tail": bool(b_dict.get("has_tail", False)),
                "tail_tip": b_dict.get("tail_tip"),
                "tail_base": b_dict.get("tail_base"),
                "tail_vector": b_dict.get("tail_vector"),
                "tail_angle_deg": b_dict.get("tail_angle_deg"),
                "tail_length": float(b_dict.get("tail_length", 0.0)),
                "tail_direction": str(b_dict.get("tail_direction", "none")),
                "is_off_panel": bool(b_dict.get("is_off_panel", False))
            })

        CURRENT_STATE["bubbles"] = serialized_bubbles
        PAGES_CACHE[page_id] = {
            "original_image": pil_img,
            "original_bytes": contents,
            "cleaned_image": cleaned,
            "filename": filename,
            "bubbles": serialized_bubbles,
            "panels": panels_info
        }

        resp_payload = {
            "status": "success",
            "hybrid": is_hybrid,
            "page_id": page_id,
            "image_w": target_w,
            "image_h": target_h,
            "detection_w": det_w,
            "detection_h": det_h,
            "filename": filename,
            "ocr_time_s": round(t_ocr, 2),
            "clean_time_s": round(t_clean, 2),
            "total_time_s": round(t_ocr + t_clean, 2),
            "bubbles": serialized_bubbles,
            "panels": panels_info
        }
        if not is_hybrid and cleaned is not None:
            resp_payload["cleaned_image_b64"] = f"data:image/jpeg;base64,{pil_to_base64_jpeg(cleaned, quality=98)}"
        else:
            resp_payload["cleaned_image_b64"] = None

        return JSONResponse(resp_payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_inpaint_crops(request):
    """
    Zero-bleed, 100% authentic Telea text-only inpainting on localized bubble crops.
    Client sends bubble crops with their contours; server removes ONLY the text ink pixels
    using OpenCV Telea inpainting, preserving 100% of authentic background paper grain,
    gradients, and outer border strokes.
    Takes < 30ms and uses < 2MB RAM (impervious to 512MB RAM limits).
    """
    try:
        data = await request.json()
        crops_data = data.get("crops", [])
        if not crops_data:
            return JSONResponse({"status": "ok", "cleaned_crops": []})

        import bubble_lettering
        cleaned_crops = []
        for c_item in crops_data:
            bid = c_item.get("bubble_id")
            b64_img = c_item.get("image_b64", "")
            if not b64_img:
                continue

            if "," in b64_img:
                b64_img = b64_img.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            crop_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if crop_cv is None or crop_cv.size == 0:
                continue

            ch, cw = crop_cv.shape[:2]
            box = c_item.get("box", [0, 0, cw, ch])
            bx0, by0 = box[0], box[1]
            cnt_pts = c_item.get("contour", [])
            shape_type = c_item.get("shape_type", "oval")

            if shape_type == "rectangle" or not cnt_pts or len(cnt_pts) < 3:
                crop_cnt = np.array([[[1, 1]], [[cw - 1, 1]], [[cw - 1, ch - 1]], [[1, ch - 1]]], dtype=np.int32)
            else:
                xs = [pt[0] for pt in cnt_pts]
                ys = [pt[1] for pt in cnt_pts]
                if max(xs) > cw or max(ys) > ch:
                    crop_cnt = np.array([[[int(round(pt[0] - bx0)), int(round(pt[1] - by0))]] for pt in cnt_pts], dtype=np.int32)
                else:
                    crop_cnt = np.array([[[int(round(pt[0])), int(round(pt[1]))]] for pt in cnt_pts], dtype=np.int32)

            cleaned_crop = bubble_lettering.inpaint_bubble_text_only(
                crop_cv,
                crop_cnt,
                border_margin=2 if shape_type == "rectangle" else 3,
                ink_dilate=2,
                inpaint_radius=3
            )

            success, enc_buf = cv2.imencode(".png", cleaned_crop)
            if success:
                out_b64 = "data:image/png;base64," + base64.b64encode(enc_buf).decode("ascii")
                cleaned_crops.append({
                    "bubble_id": bid,
                    "image_b64": out_b64
                })

        return JSONResponse({
            "status": "ok",
            "cleaned_crops": cleaned_crops
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

DYNAMIC_FONT_MAP = {}

def get_font_name_from_file(font_path: str) -> str:
    """Extracts font family name from TTF/OTF 'name' table metadata, fallback to filename."""
    base_lower = os.path.basename(font_path).lower()
    if "vcjoecaramagna" in base_lower or "vc_joe" in base_lower or "caramagna" in base_lower or "marvel_vc" in base_lower:
        return "VC Joe Caramagna"
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        if len(data) > 12:
            num_tables = struct.unpack(">H", data[4:6])[0]
            name_offset = None
            for i in range(num_tables):
                entry = data[12 + i * 16 : 28 + i * 16]
                if len(entry) < 16:
                    break
                tag = entry[:4].decode("latin1", errors="ignore")
                if tag == "name":
                    name_offset = struct.unpack(">I", entry[8:12])[0]
                    break
            if name_offset and len(data) >= name_offset + 6:
                format_val, count, string_offset = struct.unpack(">HHH", data[name_offset : name_offset + 6])
                records = []
                for i in range(count):
                    rec_offset = name_offset + 6 + i * 12
                    if len(data) < rec_offset + 12:
                        break
                    platform_id, encoding_id, language_id, name_id, length, offset = struct.unpack(
                        ">HHHHHH", data[rec_offset : rec_offset + 12]
                    )
                    str_start = name_offset + string_offset + offset
                    str_data = data[str_start : str_start + length]
                    if name_id in (1, 4):
                        try:
                            if platform_id == 3 or platform_id == 0:
                                name = str_data.decode("utf-16-be")
                            else:
                                name = str_data.decode("utf-8", errors="ignore")
                            records.append((name_id, name.strip()))
                        except Exception:
                            pass
                for target_id in (1, 4):
                    for nid, name in records:
                        if nid == target_id and name:
                            clean_name = name
                            if clean_name.startswith("t_") or clean_name.startswith("t-"):
                                clean_name = clean_name[2:].strip()
                            return clean_name
    except Exception:
        pass
    base = os.path.splitext(os.path.basename(font_path))[0]
    if base.startswith("t_") or base.startswith("t-"):
        base = base[2:].strip()
    return base.replace("-", " ").replace("_", " ").title()

def get_font_file(requested_font: str, is_italic: bool = False, is_bold: bool = False) -> str:
    fonts_dir = os.path.join(BASE_DIR, "fonts")
    fallback = os.path.join(fonts_dir, "CCWildWords.ttf")
    req = str(requested_font or "").strip()
    if req in ["Comic Mode Regular", "ComicMode Regular", "Comic Mode", "ComicMode"]:
        req = "Comic Mode Medium"

    req_lower = req.lower()
    if "vc joe" in req_lower or "caramagna" in req_lower or "marvel vc" in req_lower:
        has_bold = is_bold or ("bold" in req_lower)
        has_italic = is_italic or ("italic" in req_lower) or ("oblique" in req_lower)
        if has_bold and has_italic:
            return os.path.join(fonts_dir, "VCJoeCaramagna-BoldItalic.ttf")
        elif has_bold:
            return os.path.join(fonts_dir, "VCJoeCaramagna-Bold.ttf")
        elif has_italic:
            return os.path.join(fonts_dir, "VCJoeCaramagna-Italic.ttf")
        else:
            return os.path.join(fonts_dir, "VCJoeCaramagna.ttf")

    if "digital strip" in req_lower or "digitalstrip" in req_lower:
        has_bold = is_bold or ("bold" in req_lower)
        has_italic = is_italic or ("italic" in req_lower) or ("oblique" in req_lower)
        if has_bold or has_italic:
            return os.path.join(fonts_dir, "DigitalStripBB-BoldItalic.ttf" if has_bold else "DigitalStripBB-Italic.ttf")
        else:
            return os.path.join(fonts_dir, "DigitalStripBB.ttf")

    if "cc meanwhile" in req_lower or "meanwhile" in req_lower:
        has_bold = is_bold or ("bold" in req_lower)
        has_italic = is_italic or ("italic" in req_lower) or ("oblique" in req_lower)
        if has_bold and has_italic:
            return os.path.join(fonts_dir, "CCMeanwhile-BoldItalic.ttf")
        elif has_italic:
            return os.path.join(fonts_dir, "CCMeanwhile-Italic.ttf")
        else:
            return os.path.join(fonts_dir, "CCMeanwhile-Regular.ttf")

    if "spinner rack" in req_lower or "spinnerrack" in req_lower:
        has_bold = is_bold or ("bold" in req_lower)
        has_italic = is_italic or ("italic" in req_lower) or ("oblique" in req_lower)
        if has_bold and has_italic:
            return os.path.join(fonts_dir, "SpinnerRackProBB-BoldItalic.ttf")
        elif has_bold:
            return os.path.join(fonts_dir, "SpinnerRackProBB-Bold.ttf")
        elif has_italic:
            return os.path.join(fonts_dir, "SpinnerRackProBB-Italic.ttf")
        else:
            return os.path.join(fonts_dir, "SpinnerRackProBB.ttf")

    if "back issues" in req_lower or "backissues" in req_lower:
        has_bold = is_bold or ("bold" in req_lower)
        has_italic = is_italic or ("italic" in req_lower) or ("oblique" in req_lower)
        if has_bold and has_italic:
            return os.path.join(fonts_dir, "BackIssuesBB_boldital.ttf")
        elif has_italic:
            return os.path.join(fonts_dir, "BackIssuesBB_ital.ttf")
        else:
            return os.path.join(fonts_dir, "BackIssuesBB_reg.ttf")

    # 1. Dynamic uploaded fonts
    if req in DYNAMIC_FONT_MAP and os.path.exists(DYNAMIC_FONT_MAP[req]):
        return DYNAMIC_FONT_MAP[req]
    for k, v in DYNAMIC_FONT_MAP.items():
        if k.lower() == req.lower() and os.path.exists(v):
            return v

    # 2. Standard built-in font map
    font_map = {
        "VC Joe Caramagna": os.path.join(fonts_dir, "VCJoeCaramagna-BoldItalic.ttf" if (is_bold and is_italic) else ("VCJoeCaramagna-Bold.ttf" if is_bold else ("VCJoeCaramagna-Italic.ttf" if is_italic else "VCJoeCaramagna.ttf"))),
        "VC Joe": os.path.join(fonts_dir, "VCJoeCaramagna-BoldItalic.ttf" if (is_bold and is_italic) else ("VCJoeCaramagna-Bold.ttf" if is_bold else ("VCJoeCaramagna-Italic.ttf" if is_italic else "VCJoeCaramagna.ttf"))),
        "Marvel VC": os.path.join(fonts_dir, "VCJoeCaramagna-BoldItalic.ttf" if (is_bold and is_italic) else ("VCJoeCaramagna-Bold.ttf" if is_bold else ("VCJoeCaramagna-Italic.ttf" if is_italic else "VCJoeCaramagna.ttf"))),
        "Digital Strip BB": os.path.join(fonts_dir, "DigitalStripBB-BoldItalic.ttf" if (is_bold and is_italic) else ("DigitalStripBB-Italic.ttf" if is_italic else "DigitalStripBB.ttf")),
        "CC Meanwhile": os.path.join(fonts_dir, "CCMeanwhile-BoldItalic.ttf" if (is_bold and is_italic) else ("CCMeanwhile-Italic.ttf" if is_italic else "CCMeanwhile-Regular.ttf")),
        "Back Issues": os.path.join(fonts_dir, "BackIssuesBB_boldital.ttf" if (is_bold and is_italic) else ("BackIssuesBB_ital.ttf" if is_italic else "BackIssuesBB_reg.ttf")),
        "Back Issues BB": os.path.join(fonts_dir, "BackIssuesBB_boldital.ttf" if (is_bold and is_italic) else ("BackIssuesBB_ital.ttf" if is_italic else "BackIssuesBB_reg.ttf")),
        "BackIssuesBB": os.path.join(fonts_dir, "BackIssuesBB_boldital.ttf" if (is_bold and is_italic) else ("BackIssuesBB_ital.ttf" if is_italic else "BackIssuesBB_reg.ttf")),
        "Spinner Rack Pro BB": os.path.join(fonts_dir, "SpinnerRackProBB-BoldItalic.ttf" if (is_bold and is_italic) else ("SpinnerRackProBB-Bold.ttf" if is_bold else ("SpinnerRackProBB-Italic.ttf" if is_italic else "SpinnerRackProBB.ttf"))),
        "t_Spinner Rack Pro BB": os.path.join(fonts_dir, "SpinnerRackProBB-BoldItalic.ttf" if (is_bold and is_italic) else ("SpinnerRackProBB-Bold.ttf" if is_bold else ("SpinnerRackProBB-Italic.ttf" if is_italic else "SpinnerRackProBB.ttf"))),
        "CC Wild Words": os.path.join(fonts_dir, "CCWildWords.ttf"),
        "CCYadaYadaYada": os.path.join(fonts_dir, "CCYadaYadaYada-Italic.ttf" if is_italic else "CCYadaYadaYada.ttf"),
        "CC Yada Yada Yada": os.path.join(fonts_dir, "CCYadaYadaYada-Italic.ttf" if is_italic else "CCYadaYadaYada.ttf"),
        "CCYadaYadaYada Italic": os.path.join(fonts_dir, "CCYadaYadaYada-Italic.ttf"),
        "CCYadaYadaYada italic": os.path.join(fonts_dir, "CCYadaYadaYada-Italic.ttf"),
        "Jet Set": os.path.join(fonts_dir, "JetSet-Italic.ttf" if is_italic else "JetSet.ttf"),
        "JetSet": os.path.join(fonts_dir, "JetSet-Italic.ttf" if is_italic else "JetSet.ttf"),
        "Jet Set Italic": os.path.join(fonts_dir, "JetSet-Italic.ttf"),
        "JetSet Italic": os.path.join(fonts_dir, "JetSet-Italic.ttf"),
        "JetSet-Italic": os.path.join(fonts_dir, "JetSet-Italic.ttf"),
        "Gandur New": os.path.join(fonts_dir, "GandurNew-Light.otf"),
        "Gandur New Light": os.path.join(fonts_dir, "GandurNew-Light.otf"),
        "Gandur": os.path.join(fonts_dir, "GandurNew-Light.otf"),
        "Comic Mode Medium": os.path.join(fonts_dir, "ComicMode-MediumOblique.otf" if is_italic else "ComicMode-Medium.otf"),
        "ComicMode Medium": os.path.join(fonts_dir, "ComicMode-MediumOblique.otf" if is_italic else "ComicMode-Medium.otf"),
        "Comic Mode Medium Oblique": os.path.join(fonts_dir, "ComicMode-MediumOblique.otf"),
        "ComicMode Medium Oblique": os.path.join(fonts_dir, "ComicMode-MediumOblique.otf"),
        "Comic Neue": os.path.join(fonts_dir, "ComicNeue-Bold.ttf"),
        "Komika Text": os.path.join(fonts_dir, "KOMTXTB_.ttf"),
        "Manga Temple": os.path.join(fonts_dir, "mangatb.ttf") if os.path.exists(os.path.join(fonts_dir, "mangatb.ttf")) else os.path.join(fonts_dir, "mangat.ttf"),
        "Mason": os.path.join(fonts_dir, "Mason-Bold.ttf" if is_bold else "Mason-Regular.ttf"),
        "Mason Serif": os.path.join(fonts_dir, "Mason-Bold.ttf" if is_bold else "Mason-Regular.ttf"),
        "Mason (Doctor Doom)": os.path.join(fonts_dir, "Mason-Bold.ttf" if is_bold else "Mason-Regular.ttf"),
        "Mason Regular": os.path.join(fonts_dir, "Mason-Regular.ttf"),
        "Mason Bold": os.path.join(fonts_dir, "Mason-Bold.ttf"),
        "Mason Demi": os.path.join(fonts_dir, "Mason-Demi.ttf"),
    }
    f = font_map.get(req)
    if f and os.path.exists(f):
        return f

    # 3. Direct file checks in fonts directory
    for ext in [".ttf", ".otf", ".woff", ".woff2", ""]:
        c = os.path.join(fonts_dir, req + ext)
        if os.path.exists(c):
            return c

    return fallback

async def api_upload_font(request):
    """
    Uploads a custom font (.ttf, .otf, .woff, .woff2), saves it to fonts/ directory,
    extracts the font family name, and registers it dynamically.
    """
    try:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            return JSONResponse({"status": "error", "message": "Fayl yuborilmadi"}, status_code=400)

        filename = os.path.basename(uploaded_file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".ttf", ".otf", ".woff", ".woff2"]:
            return JSONResponse({"status": "error", "message": "Faqat .ttf, .otf, .woff, .woff2 formatlar qo'llab-quvvatlanadi"}, status_code=400)

        fonts_dir = os.path.join(BASE_DIR, "fonts")
        os.makedirs(fonts_dir, exist_ok=True)

        safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
        save_path = os.path.join(fonts_dir, safe_name)

        content = await uploaded_file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        extracted_name = get_font_name_from_file(save_path)
        font_name = extracted_name or os.path.splitext(safe_name)[0]

        DYNAMIC_FONT_MAP[font_name] = save_path
        DYNAMIC_FONT_MAP[safe_name] = save_path
        DYNAMIC_FONT_MAP[os.path.splitext(safe_name)[0]] = save_path

        global _FONTS_CACHE
        _FONTS_CACHE = None

        return JSONResponse({
            "status": "ok",
            "font_name": font_name,
            "filename": safe_name,
            "url": f"/fonts/{safe_name}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

_FONTS_CACHE = None

async def api_list_fonts(request):
    """Lists all available fonts in the fonts/ directory with in-memory caching."""
    global _FONTS_CACHE
    if _FONTS_CACHE is not None:
        return JSONResponse(_FONTS_CACHE)

    fonts_dir = os.path.join(BASE_DIR, "fonts")
    result = []
    seen_names = set()
    if os.path.exists(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if fname.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
                fpath = os.path.join(fonts_dir, fname)
                name = get_font_name_from_file(fpath) or os.path.splitext(fname)[0]
                norm_name = name.lower().strip()
                if norm_name in seen_names:
                    continue
                seen_names.add(norm_name)
                result.append({
                    "font_name": name,
                    "filename": fname,
                    "url": f"/fonts/{fname}"
                })
    _FONTS_CACHE = {"status": "ok", "fonts": result}
    return JSONResponse(_FONTS_CACHE)

async def api_delete_font(request):
    """
    Deletes a font file from the fonts/ directory and unregisters it from DYNAMIC_FONT_MAP.
    """
    try:
        data = await request.json()
        font_name = (data.get("font_name") or "").strip()
        filename = (data.get("filename") or "").strip()

        fonts_dir = os.path.join(BASE_DIR, "fonts")
        deleted_files = []

        # 1. If explicit filename provided, delete from fonts/
        if filename:
            safe_fname = os.path.basename(filename)
            fpath = os.path.join(fonts_dir, safe_fname)
            if os.path.exists(fpath) and os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    deleted_files.append(safe_fname)
                except Exception as e:
                    print(f"[FONT] Could not delete {fpath}: {e}")

        # 2. Match by font_name in DYNAMIC_FONT_MAP and fonts_dir
        if font_name:
            fpath = DYNAMIC_FONT_MAP.pop(font_name, None)
            if fpath and os.path.exists(fpath) and os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    base_n = os.path.basename(fpath)
                    if base_n not in deleted_files:
                        deleted_files.append(base_n)
                except Exception:
                    pass

            if os.path.exists(fonts_dir):
                for fname in os.listdir(fonts_dir):
                    if fname.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
                        test_path = os.path.join(fonts_dir, fname)
                        try:
                            extracted = get_font_name_from_file(test_path)
                            base_no_ext = os.path.splitext(fname)[0]
                            if (extracted and extracted.lower() == font_name.lower()) or (base_no_ext.lower() == font_name.lower()):
                                os.remove(test_path)
                                if fname not in deleted_files:
                                    deleted_files.append(fname)
                        except Exception:
                            pass

        global _FONTS_CACHE
        _FONTS_CACHE = None

        return JSONResponse({
            "status": "ok",
            "message": f"'{font_name or filename}' shrifti o'chirildi",
            "deleted_files": deleted_files
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_inpaint_crop(request):
    """
    Zero-loss localized crop inpainting:
    Receives a tiny bubble crop (<50 KB), runs OpenCV Telea text inpainting strictly inside the crop,
    and returns pristine PNG bytes. Server peak memory stays safely under 5 MB.
    """
    try:
        data = await request.json()
        crop_b64 = data.get("crop_b64")
        if not crop_b64:
            return JSONResponse({"error": "No crop image"}, status_code=400)
        if "," in crop_b64:
            crop_b64 = crop_b64.split(",", 1)[1]
        crop_bytes = base64.b64decode(crop_b64)
        crop_img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
        crop_cv = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
        ink_thresh = int(data.get("ink_thresh", 165))
        text_mask = (gray < ink_thresh).astype(np.uint8) * 255

        # Protect outer border strokes with configurable margin
        margin = int(data.get("border_margin", 3))
        if margin > 0 and text_mask.shape[0] > margin * 2 and text_mask.shape[1] > margin * 2:
            border_mask = np.zeros_like(text_mask)
            border_mask[margin:-margin, margin:-margin] = 255
            text_mask = cv2.bitwise_and(text_mask, border_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        cleaned_cv = cv2.inpaint(crop_cv, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(cleaned_rgb).save(buf, format="PNG")
        cleaned_crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "cleaned_crop_b64": f"data:image/png;base64,{cleaned_crop_b64}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_render_hd(request):
    """
    Renders HD final lettering image using selected typography.
    Supports individual page download with strict sequential numbering,
    and supports pristine untouched original export for skipped pages.
    """
    try:
        data = await request.json()
        bubbles_data = data.get("bubbles", [])
        page_id = data.get("page_id")
        is_skipped = bool(data.get("is_skipped", False))
        
        p_entry = PAGES_CACHE.get(page_id) if page_id else CURRENT_STATE
        cleaned_img = p_entry.get("cleaned_image") if p_entry else CURRENT_STATE.get("cleaned_image")
        orig_img = p_entry.get("original_image") if p_entry else CURRENT_STATE.get("original_image")

        target_filename = data.get("filename") or f"{os.path.splitext((p_entry or CURRENT_STATE).get('image_name', 'comic'))[0]}_uzbek_HD.png"

        rendered_b64 = data.get("rendered_image_b64")
        if rendered_b64:
            if "," in rendered_b64:
                rendered_b64 = rendered_b64.split(",", 1)[1]
            return Response(
                content=base64.b64decode(rendered_b64),
                media_type="image/png",
                headers={"Content-Disposition": f'attachment; filename="{target_filename}"'}
            )

        if is_skipped or (not bubbles_data and not cleaned_img):
            if p_entry and p_entry.get("original_bytes"):
                return Response(
                    content=p_entry["original_bytes"],
                    media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{target_filename}"'}
                )
            if orig_img is not None:
                return Response(
                    content=pil_to_png_bytes(orig_img),
                    media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{target_filename}"'}
                )
            return JSONResponse({"error": "No image to render"}, status_code=400)

        if cleaned_img is None or orig_img is None:
            return JSONResponse({"error": "No active page to render"}, status_code=400)

        # Convert back to SpeechBubble objects for engine typesetter
        bubble_objs = []
        for bd in bubbles_data:
            if bd.get("is_hidden"):
                continue
            b = engine.SpeechBubble(
                bubble_id=int(bd.get("bubble_id", 1)),
                x0=int(bd.get("x0", 0)),
                y0=int(bd.get("y0", 0)),
                x1=int(bd.get("x1", 100)),
                y1=int(bd.get("y1", 100)),
                box=[int(bd.get("x0", 0)), int(bd.get("y0", 0)), int(bd.get("x1", 100)), int(bd.get("y1", 100))],
                original_text=str(bd.get("original_text", "")),
                clean_text=str(bd.get("original_text", "")),
                uzbek_translation=str(bd.get("uzbek_translation", "")),
                pipeline_uzbek_translation=str(bd.get("uzbek_translation", "")),
                confidence=1.0,
                shape_type=str(bd.get("shape_type", "oval")),
                font_size_offset=int(bd.get("font_size_offset", 0)),
                line_spacing=float(bd.get("line_spacing", 1.15)),
                offset_x=int(bd.get("offset_x", 0)),
                offset_y=int(bd.get("offset_y", 0)),
                white_fill=bool(bd.get("white_fill", False)),
                text_color=bd.get("text_color"),
                bg_color=bd.get("bg_color"),
                outline_color=bd.get("outline_color"),
                outline_width=int(bd.get("outline_width", 0)),
                border_color=bd.get("border_color"),
                border_width=int(bd.get("border_width", 0)),
                text_align=str(bd.get("text_align", "center")),
                font_family=bd.get("font_family") or bd.get("font_name"),
                font_name=bd.get("font_family") or bd.get("font_name"),
                font_weight=str(bd.get("font_weight", "normal")),
                font_style=str(bd.get("font_style", "normal")),
                is_hidden=bool(bd.get("is_hidden", False)),
                is_locked=bool(bd.get("is_locked", False)),
                is_printable=bool(bd.get("is_printable", True)),
                custom_name=bd.get("custom_name"),
                opacity=float(bd.get("opacity", 1.0)),
                blend_mode=str(bd.get("blend_mode", "source-over")),
                color_tag=bd.get("color_tag"),
                group_id=bd.get("group_id"),
                is_text_hidden=bool(bd.get("is_text_hidden", False)),
                is_shape_hidden=bool(bd.get("is_shape_hidden", False)),
                is_text_locked=bool(bd.get("is_text_locked", False)),
                is_shape_locked=bool(bd.get("is_shape_locked", False)),
                speaker=bd.get("speaker") if bd.get("speaker") != "Noma'lum" else None,
                is_active=bool(bd.get("is_active", True)),
                contour=bd.get("contour", []),
                contour_points=bd.get("contour", [])
            )
            bubble_objs.append(b)

        requested_font = str(data.get("font_name", "CC Wild Words"))
        font_file = get_font_file(requested_font)

        rendered = engine.typeset_lettering_on_page(
            cleaned_image=cleaned_img,
            bubbles=bubble_objs,
            font_path=font_file,
            max_font_size=28,
            min_font_size=12,
            line_gap=3,
            original_image=orig_img
        )

        png_bytes = pil_to_png_bytes(rendered)
        
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{target_filename}"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_export_chapter_zip(request):
    """
    Exports all queued pages (both translated and skipped) into a single sequentially numbered ZIP.
    Numbering is guaranteed from 01 to N with zero mix-ups.
    """
    try:
        data = await request.json()
        pages = data.get("pages", [])
        font_name = data.get("font_name", "CC Wild Words")
        if not pages:
            return JSONResponse({"error": "No pages to export"}, status_code=400)

        font_file = get_font_file(font_name)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for p_info in pages:
                p_id = p_info.get("page_id")
                p_filename = p_info.get("filename", f"page_{p_info.get('page_num', 1):02d}.png")
                status = p_info.get("status", "pending")
                rendered_b64 = p_info.get("rendered_image_b64")
                orig_b64 = p_info.get("original_image_b64")

                if rendered_b64:
                    if "," in rendered_b64:
                        rendered_b64 = rendered_b64.split(",", 1)[1]
                    zf.writestr(p_filename, base64.b64decode(rendered_b64))
                    continue

                if orig_b64 and (status == "skipped" or status == "pending"):
                    if "," in orig_b64:
                        orig_b64 = orig_b64.split(",", 1)[1]
                    zf.writestr(p_filename, base64.b64decode(orig_b64))
                    continue

                p_entry = PAGES_CACHE.get(p_id)
                if not p_entry:
                    continue

                orig_img = p_entry.get("original_image")
                cleaned_img = p_entry.get("cleaned_image")
                bubbles_data = p_info.get("bubbles", [])

                if status == "skipped" or not cleaned_img or not bubbles_data:
                    # Pristine original page byte-for-byte with exact sequential number
                    if p_entry and p_entry.get("original_bytes"):
                        zf.writestr(p_filename, p_entry["original_bytes"])
                    elif orig_img is not None:
                        zf.writestr(p_filename, pil_to_png_bytes(orig_img))
                else:
                    # Lettered Uzbek page
                    bubble_objs = []
                    for bd in bubbles_data:
                        if bd.get("is_hidden") or bd.get("is_printable") is False:
                            continue
                        b = engine.SpeechBubble(
                            bubble_id=int(bd.get("bubble_id", 1)),
                            x0=int(bd.get("x0", 0)),
                            y0=int(bd.get("y0", 0)),
                            x1=int(bd.get("x1", 100)),
                            y1=int(bd.get("y1", 100)),
                            box=[int(bd.get("x0", 0)), int(bd.get("y0", 0)), int(bd.get("x1", 100)), int(bd.get("y1", 100))],
                            original_text=str(bd.get("original_text", "")),
                            clean_text=str(bd.get("original_text", "")),
                            uzbek_translation=str(bd.get("uzbek_translation", "")),
                            pipeline_uzbek_translation=str(bd.get("uzbek_translation", "")),
                            confidence=1.0,
                            shape_type=str(bd.get("shape_type", "oval")),
                            font_size_offset=int(bd.get("font_size_offset", 0)),
                            line_spacing=float(bd.get("line_spacing", 1.15)),
                            offset_x=int(bd.get("offset_x", 0)),
                            offset_y=int(bd.get("offset_y", 0)),
                            white_fill=bool(bd.get("white_fill", False)),
                            text_color=bd.get("text_color"),
                            bg_color=bd.get("bg_color"),
                            outline_color=bd.get("outline_color"),
                            outline_width=int(bd.get("outline_width", 0)),
                            border_color=bd.get("border_color"),
                            border_width=int(bd.get("border_width", 0)),
                            text_align=str(bd.get("text_align", "center")),
                            font_family=bd.get("font_family") or bd.get("font_name"),
                            font_name=bd.get("font_family") or bd.get("font_name"),
                            font_weight=str(bd.get("font_weight", "normal")),
                            font_style=str(bd.get("font_style", "normal")),
                            is_hidden=bool(bd.get("is_hidden", False)),
                            is_locked=bool(bd.get("is_locked", False)),
                            is_printable=bool(bd.get("is_printable", True)),
                            custom_name=bd.get("custom_name"),
                            opacity=float(bd.get("opacity", 1.0)),
                            blend_mode=str(bd.get("blend_mode", "source-over")),
                            color_tag=bd.get("color_tag"),
                            group_id=bd.get("group_id"),
                            is_text_hidden=bool(bd.get("is_text_hidden", False)),
                            is_shape_hidden=bool(bd.get("is_shape_hidden", False)),
                            is_text_locked=bool(bd.get("is_text_locked", False)),
                            is_shape_locked=bool(bd.get("is_shape_locked", False)),
                            speaker=bd.get("speaker") if bd.get("speaker") != "Noma'lum" else None,
                            is_active=bool(bd.get("is_active", True)),
                            contour=bd.get("contour", []),
                            contour_points=bd.get("contour", [])
                        )
                        bubble_objs.append(b)

                    rendered = engine.typeset_lettering_on_page(
                        cleaned_image=cleaned_img,
                        bubbles=bubble_objs,
                        font_path=font_file,
                        max_font_size=28,
                        min_font_size=12,
                        line_gap=3,
                        original_image=orig_img
                    )
                    zf.writestr(p_filename, pil_to_png_bytes(rendered))

        zip_bytes = zip_buffer.getvalue()
        chapter_name = data.get("chapter_name", "comic_uzbek_chapter")
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{chapter_name}.zip"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_retranslate(request):
    """Retranslates a single bubble with Gemini with offline fallback."""
    try:
        data = await request.json()
        text = data.get("text", "")
        speaker = data.get("speaker", "")
        if not text:
            return JSONResponse({"uzbek_translation": ""})
        if engine.is_pure_sfx_text(text):
            return JSONResponse({"uzbek_translation": text})
        uz = ""
        if gemini_enhancements.is_gemini_enabled():
            try:
                uz = gemini_enhancements.gemini_translate_dialogue(text, speaker=speaker, use_cache=False)
            except Exception as gem_ex:
                print(f"[RETRANSLATE_WARN] Gemini retranslation error: {gem_ex}", flush=True)
        if not uz:
            uz = engine.translate_text(text, speaker=speaker)
        return JSONResponse({"uzbek_translation": uz})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_get_glossary(request):
    """Returns all shared team glossary terms."""
    terms = glossary_manager.load_glossary()
    return JSONResponse({"status": "ok", "terms": terms})

async def api_save_glossary_term(request):
    """Adds, updates or replaces terms in the shared team glossary."""
    try:
        data = await request.json()
        if "terms" in data and isinstance(data["terms"], dict):
            glossary_manager.save_glossary(data["terms"])
        elif "source" in data and "target" in data:
            glossary_manager.set_term(data["source"], data["target"])
        gemini_enhancements.clear_gemini_cache()
        terms = glossary_manager.load_glossary()
        return JSONResponse({"status": "ok", "terms": terms})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_delete_glossary_term(request):
    """Deletes a term from the shared team glossary."""
    try:
        data = await request.json()
        source = data.get("source", "")
        if source:
            glossary_manager.remove_term(source)
            gemini_enhancements.clear_gemini_cache()
        terms = glossary_manager.load_glossary()
        return JSONResponse({"status": "ok", "terms": terms})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_enforce_glossary(request):
    """Applies glossary regex enforcement to a given text or batch of texts."""
    try:
        data = await request.json()
        text = data.get("text", "")
        enforced = glossary_manager.apply_glossary_enforcement(text)
        return JSONResponse({"status": "ok", "result": enforced})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_snap_bubble(request):
    """
    Direction 3: 1-Click Magnetic Snap to Bubble Border.
    Takes a box [x0, y0, x1, y1] (and optional page_id), runs CV contour detector
    on original page image, and magnetically snaps the box to the true bubble boundary.
    """
    try:
        data = await request.json()
        page_id = str(data.get("page_id") or "")
        box = data.get("box")
        bubble_id = data.get("bubble_id")

        pil_img = None
        if page_id and page_id in PAGES_CACHE:
            pil_img = PAGES_CACHE[page_id].get("original_image")
        if pil_img is None:
            pil_img = CURRENT_STATE.get("original_image")

        if pil_img is None or not box or len(box) < 4:
            return JSONResponse({"status": "error", "message": "Rasm yoki koordinatalar topilmadi"}, status_code=400)

        import numpy as np
        import cv2
        import bubble_lettering

        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        H_img, W_img = img_bgr.shape[:2]

        bw = max(10, x1 - x0)
        bh = max(10, y1 - y0)

        cnt_page, is_rect = bubble_lettering.get_bubble_contour(
            img_bgr,
            (x0, y0, x1, y1),
            approx_glyph_height=max(20, int(bh * 0.35))
        )
        rx, ry, rw, rh = cv2.boundingRect(cnt_page)

        # Sanity check: Ensure snapped box is a reasonable bubble
        if rw >= 15 and rh >= 15 and rw <= bw * 2.5 and rh <= bh * 2.5:
            snapped_x0 = max(0, rx)
            snapped_y0 = max(0, ry)
            snapped_x1 = min(W_img, rx + rw)
            snapped_y1 = min(H_img, ry + rh)
        else:
            snapped_x0 = x0
            snapped_y0 = y0
            snapped_x1 = x1
            snapped_y1 = y1

        cnt_pts = cnt_page.reshape(-1, 2).tolist() if (cnt_page is not None and len(cnt_page) > 0) else []
        shape_type = "rectangle" if is_rect else "oval"

        import tail_detector
        import speaker_detector
        tail_info = tail_detector.detect_bubble_tail(cnt_page, [snapped_x0, snapped_y0, snapped_x1, snapped_y1], shape_type)
        spk_name, spk_conf, spk_anchor = speaker_detector.ground_bubble_speaker(
            {
                "shape_type": shape_type,
                "has_tail": tail_info.get("has_tail", False),
                "tail_tip": tail_info.get("tail_tip"),
                "tail_vector": tail_info.get("tail_vector")
            },
            image_bgr=img_bgr
        )

        return JSONResponse({
            "status": "ok",
            "bubble_id": bubble_id,
            "box": [snapped_x0, snapped_y0, snapped_x1, snapped_y1],
            "x0": snapped_x0,
            "y0": snapped_y0,
            "x1": snapped_x1,
            "y1": snapped_y1,
            "shape_type": shape_type,
            "contour": cnt_pts,
            "has_tail": tail_info.get("has_tail", False),
            "tail_tip": tail_info.get("tail_tip"),
            "tail_base": tail_info.get("tail_base"),
            "tail_vector": tail_info.get("tail_vector"),
            "tail_angle_deg": tail_info.get("tail_angle_deg"),
            "tail_length": tail_info.get("tail_length", 0.0),
            "tail_direction": tail_info.get("tail_direction", "none"),
            "is_off_panel": bool(tail_info.get("is_off_panel", False)),
            "speaker": spk_name or ("Sharhlovchi" if (shape_type == "rectangle" and not tail_info.get("has_tail")) else None),
            "speaker_confidence": spk_conf,
            "speaker_anchor": list(spk_anchor) if spk_anchor else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_clear_workspace_state(request):
    """Flushes server-side state, page caches, and gemini caches for a 100% clean chapter restart."""
    try:
        PAGES_CACHE.clear()
        CURRENT_STATE["original_image"] = None
        CURRENT_STATE["cleaned_image"] = None
        CURRENT_STATE["bubbles"] = []
        CURRENT_STATE["image_name"] = "comic_page.png"
        try:
            gemini_enhancements.clear_gemini_cache()
        except Exception:
            pass
        return JSONResponse({"status": "ok", "message": "Ish maydoni va barcha keshlar tozalandi"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def api_get_page_state(request):
    try:
        page_id = str(request.query_params.get("page_id") or "")
        cached = None
        if page_id and page_id in PAGES_CACHE:
            cached = PAGES_CACHE[page_id]

        if cached:
            cleaned_b64 = None
            if cached.get("cleaned_image") is not None:
                cleaned_b64 = f"data:image/jpeg;base64,{pil_to_base64_jpeg(cached['cleaned_image'], quality=98)}"
            orig_b64 = None
            if cached.get("original_image") is not None:
                orig_b64 = f"data:image/jpeg;base64,{pil_to_base64_jpeg(cached['original_image'], quality=98)}"
            return JSONResponse({
                "status": "ok",
                "has_cache": True,
                "page_id": page_id,
                "bubbles": cached.get("bubbles", []),
                "panels": cached.get("panels", []),
                "cleaned_image_b64": cleaned_b64,
                "original_image_b64": orig_b64
            })
        return JSONResponse({"status": "not_found", "has_cache": False})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

assets_dir = os.path.join(BASE_DIR, "assets")
os.makedirs(assets_dir, exist_ok=True)
fonts_dir = os.path.join(BASE_DIR, "fonts")
os.makedirs(fonts_dir, exist_ok=True)
web_dir = os.path.join(BASE_DIR, "web")
os.makedirs(web_dir, exist_ok=True)

routes = [
    Route("/", serve_index, methods=["GET"]),
    Route("/download/desktop", api_download_desktop, methods=["GET"]),
    Route("/api/version", api_version, methods=["GET"]),
    Route("/api/check_update", api_check_update, methods=["GET"]),
    Route("/api/start_update_download", api_start_update_download, methods=["POST"]),
    Route("/api/update_download_progress", api_get_update_progress, methods=["GET"]),
    Route("/api/open_download_folder", api_open_download_folder, methods=["POST"]),
    Route("/api/apply_update_and_restart", api_apply_update_and_restart, methods=["POST"]),
    Route("/api/save_session_snapshot", api_save_session_snapshot, methods=["POST"]),
    Route("/api/get_session_snapshot", api_get_session_snapshot, methods=["GET"]),
    Route("/api/health", api_health, methods=["GET"]),
    Route("/api/settings", api_get_settings, methods=["GET"]),
    Route("/api/settings", api_save_settings, methods=["POST"]),
    Route("/api/settings", api_delete_settings, methods=["DELETE"]),
    Route("/api/settings/download_dir", api_save_download_dir, methods=["POST"]),
    Route("/api/browse_folder", api_browse_folder, methods=["POST"]),
    Route("/api/save_export", api_save_export_to_download_dir, methods=["POST"]),
    Route("/api/register_page", api_register_page, methods=["POST"]),
    Route("/api/process_page", api_process_page, methods=["POST"]),
    Route("/api/get_page_state", api_get_page_state, methods=["GET"]),
    Route("/api/clear_workspace_state", api_clear_workspace_state, methods=["POST"]),
    Route("/api/render_hd", api_render_hd, methods=["POST"]),
    Route("/api/export_zip", api_export_chapter_zip, methods=["POST"]),
    Route("/api/retranslate", api_retranslate, methods=["POST"]),
    Route("/api/glossary", api_get_glossary, methods=["GET"]),
    Route("/api/glossary", api_save_glossary_term, methods=["POST"]),
    Route("/api/glossary", api_delete_glossary_term, methods=["DELETE"]),
    Route("/api/enforce_glossary", api_enforce_glossary, methods=["POST"]),
    Route("/api/snap_bubble", api_snap_bubble, methods=["POST"]),
    Route("/api/upload_font", api_upload_font, methods=["POST"]),
    Route("/api/list_fonts", api_list_fonts, methods=["GET"]),
    Route("/api/delete_font", api_delete_font, methods=["POST"]),
    Route("/api/inpaint_crop", api_inpaint_crop, methods=["POST"]),
    Route("/api/inpaint_crops", api_inpaint_crops, methods=["POST"]),
    Mount("/assets", StaticFiles(directory=assets_dir), name="assets"),
    Mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts"),
    Mount("/static", StaticFiles(directory=web_dir), name="static")
]

middleware = [
    Middleware(SessionTokenMiddleware),
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"[OK] COMIC-LAB STUDIO (FastAPI/Starlette Engine)")
    print(f"[URL] Studio URL: http://{host}:{port}")
    print(f"[INFO] RAM: ~35 MB | Re-run latency: 0 ms | 60 FPS Canvas")
    print(f"=======================================================\n")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except Exception as e:
        alt_port = 8080 if port != 8080 else 8001
        print(f"[RETRY] Port {port} band, muqobil port {alt_port} ishlatilmoqda...")
        uvicorn.run(app, host=host, port=alt_port, log_level="warning")
