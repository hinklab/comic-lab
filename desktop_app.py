import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import json
import socket
import time
import secrets
import threading
import urllib.request
import psutil
import argparse
import webview

def get_base_dir() -> str:
    """Returns bundle base directory (sys._MEIPASS) or script directory."""
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def get_user_data_dir() -> str:
    """Returns persistent AppData directory for ComicLab settings and caches."""
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    target_dir = os.path.join(appdata, "ComicLab")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# Setup persistent logging and avoid NullWriter/NoneType crashes in windowed mode
USER_DATA_DIR = get_user_data_dir()
LOG_FILE = os.path.join(USER_DATA_DIR, "comic_lab.log")
WINDOW_STATE_FILE = os.path.join(USER_DATA_DIR, "window_state.json")

def load_window_state() -> dict:
    """Loads previous window geometry and maximized/fullscreen state."""
    default_state = {
        "width": 1440,
        "height": 920,
        "x": None,
        "y": None,
        "maximized": False,
        "fullscreen": False
    }
    if not os.path.isfile(WINDOW_STATE_FILE):
        return default_state
    try:
        with open(WINDOW_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default_state

        state = default_state.copy()
        if isinstance(data.get("width"), (int, float)) and data["width"] >= 1080:
            state["width"] = int(data["width"])
        if isinstance(data.get("height"), (int, float)) and data["height"] >= 720:
            state["height"] = int(data["height"])
        if isinstance(data.get("maximized"), bool):
            state["maximized"] = data["maximized"]
        if isinstance(data.get("fullscreen"), bool):
            state["fullscreen"] = data["fullscreen"]

        # Validate x, y coordinate bounds on multi-monitor setups
        x = data.get("x")
        y = data.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            x, y = int(x), int(y)
            try:
                import ctypes
                user32 = ctypes.windll.user32
                v_left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
                v_top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
                v_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                v_height = user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
                if v_width <= 0:
                    v_left, v_top = 0, 0
                    v_width = user32.GetSystemMetrics(0)
                    v_height = user32.GetSystemMetrics(1)
                # Ensure at least 150x100 of the window is visible on some monitor
                if v_left <= x <= (v_left + v_width - 150) and v_top <= y <= (v_top + v_height - 100):
                    state["x"] = x
                    state["y"] = y
            except Exception:
                pass
        return state
    except Exception:
        return default_state

def save_window_state(state: dict) -> None:
    """Atomically writes current window geometry to JSON config."""
    try:
        temp_file = WINDOW_STATE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, WINDOW_STATE_FILE)
    except Exception:
        pass

class OutputLogger:
    def __init__(self, filename):
        self.logfile = open(filename, "a", encoding="utf-8")
    def write(self, message):
        try:
            self.logfile.write(message)
            self.logfile.flush()
        except Exception:
            pass
    def flush(self):
        try:
            self.logfile.flush()
        except Exception:
            pass
    def isatty(self):
        return False

# Always redirect stdout/stderr to logfile in frozen app or when needed
file_logger = OutputLogger(LOG_FILE)
sys.stdout = file_logger
sys.stderr = file_logger

# Ensure standalone offline mode to eliminate cloud DNS delays on startup
os.environ["COMICLAB_STANDALONE"] = "1"

def find_free_port() -> int:
    """Finds an available TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

import uvicorn
from server import app, warm_engine_in_background

class ServerThread(threading.Thread):
    """Runs Uvicorn server in a background daemon thread."""
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="warning",
            log_config=None,
            access_log=False
        )
        self.server = uvicorn.Server(config)

    def run(self):
        try:
            self.server.run()
        except Exception as e:
            if sys.stderr:
                print(f"[SERVER_ERROR] {e}", file=sys.stderr, flush=True)

    def stop(self):
        self.server.should_exit = True

class DesktopBridge:
    """Native Windows bridge exposed to JavaScript in pywebview."""
    def __init__(self, window=None):
        # Must use private _window with leading underscore to prevent pywebview
        # recursive object introspection deadlock during window initialization!
        self._window = window

    def open_files_dialog(self):
        """Opens native Windows file dialog to select comic images."""
        if not self._window:
            return []
        file_types = (
            "Comic Pages (*.png;*.jpg;*.jpeg;*.webp)",
            "All Files (*.*)"
        )
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=file_types
        )
        return result or []

    def open_folder_dialog(self):
        """Opens native Windows folder selection dialog."""
        if not self._window:
            return None
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if (result and len(result) > 0) else None

    def get_system_specs(self):
        """Returns machine specifications."""
        try:
            mem = psutil.virtual_memory()
            return {
                "total_ram_gb": round(mem.total / (1024 ** 3), 1),
                "available_ram_gb": round(mem.available / (1024 ** 3), 1),
                "cpu_count": psutil.cpu_count(logical=True)
            }
        except Exception:
            return {"total_ram_gb": 8.0, "available_ram_gb": 4.0, "cpu_count": 4}

    def open_external_url(self, url: str) -> bool:
        """Opens external URL in user's default Windows browser."""
        try:
            import webbrowser
            webbrowser.open(url)
            return True
        except Exception:
            return False

    def open_download_folder(self, filepath: str = "") -> bool:
        """Opens Windows Explorer with downloaded file selected."""
        try:
            import subprocess
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            if filepath and os.path.exists(filepath):
                subprocess.Popen(f'explorer /select,"{os.path.abspath(filepath)}"')
            else:
                subprocess.Popen(f'explorer "{downloads}"')
            return True
        except Exception:
            return False

def wait_for_server(port: int, token: str = "", timeout: float = 10.0) -> bool:
    """Polls server health endpoint until ready."""
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/api/health"
    if token:
        url += f"?token={token}"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=0.3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.02)
    return False

def main():
    try:
        parser = argparse.ArgumentParser(description="ComicLab Standalone Desktop App")
        parser.add_argument("--port", type=int, default=0, help="Local server port (0 = auto)")
        parser.add_argument("--dev", action="store_true", help="Enable developer tools (Inspect)")
        parser.add_argument("--headless", action="store_true", help="Run server only without opening UI window")
        args = parser.parse_args()

        try:
            import pyi_splash
            pyi_splash.update_text("Lokal backend server ishga tushirilmoqda...")
        except ImportError:
            pass

        session_token = secrets.token_hex(24)
        os.environ["COMICLAB_SESSION_TOKEN"] = session_token

        port = args.port if args.port > 0 else find_free_port()
        print(f"[STARTUP] Starting ComicLab backend on 127.0.0.1:{port}", flush=True)

        server_thread = ServerThread("127.0.0.1", port)
        server_thread.start()
        warm_engine_in_background()

        if not wait_for_server(port, token=session_token, timeout=12.0):
            print(f"[FATAL] Backend server failed to start on port {port} within 12 seconds.", file=sys.stderr, flush=True)
            sys.exit(1)

        print(f"[STARTUP] Backend healthy on 127.0.0.1:{port}", flush=True)

        try:
            import pyi_splash
            pyi_splash.update_text("Studio interfeysi tayyorlanmoqda...")
        except ImportError:
            pass

        if args.headless:
            print("[HEADLESS] Server running. Press Ctrl+C to terminate.", flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                server_thread.stop()
            return

        bridge = DesktopBridge()

        win_state = load_window_state()

        window = webview.create_window(
            title="Comic-Lab Studio",
            url=f"http://127.0.0.1:{port}/?token={session_token}&desktop=1",
            js_api=bridge,
            width=win_state.get("width", 1440),
            height=win_state.get("height", 920),
            x=win_state.get("x"),
            y=win_state.get("y"),
            maximized=win_state.get("maximized", False),
            fullscreen=win_state.get("fullscreen", False),
            min_size=(1080, 720),
            resizable=True,
            text_select=True,
            zoomable=True,
            background_color="#121214"
        )
        bridge._window = window

        # Track window state changes in real time
        current_state = {
            "width": win_state.get("width", 1440),
            "height": win_state.get("height", 920),
            "x": win_state.get("x"),
            "y": win_state.get("y"),
            "maximized": win_state.get("maximized", False),
            "fullscreen": win_state.get("fullscreen", False)
        }

        def on_maximized(*args):
            try:
                current_state["maximized"] = True
                current_state["fullscreen"] = False
                save_window_state(current_state)
            except Exception:
                pass

        def on_restored(*args):
            try:
                current_state["maximized"] = False
                current_state["fullscreen"] = False
                save_window_state(current_state)
            except Exception:
                pass

        def on_resized(*args):
            try:
                if current_state["maximized"] or current_state["fullscreen"]:
                    return
                if len(args) >= 2:
                    w, h = int(args[0]), int(args[1])
                    if w >= 1080 and h >= 720:
                        current_state["width"] = w
                        current_state["height"] = h
                        save_window_state(current_state)
            except Exception:
                pass

        def on_moved(*args):
            try:
                if current_state["maximized"] or current_state["fullscreen"]:
                    return
                if len(args) >= 2:
                    x, y = int(args[0]), int(args[1])
                    current_state["x"] = x
                    current_state["y"] = y
                    save_window_state(current_state)
            except Exception:
                pass

        window.events.maximized += on_maximized
        window.events.restored += on_restored
        window.events.resized += on_resized
        window.events.moved += on_moved

        def close_splash():
            try:
                import pyi_splash
                if pyi_splash.is_alive():
                    pyi_splash.close()
            except Exception:
                pass

        window.events.shown += close_splash
        window.events.loaded += close_splash

        def on_closing():
            try:
                if sys.platform == "win32":
                    try:
                        import ctypes
                        hwnd = None
                        if hasattr(window, "native") and window.native and hasattr(window.native, "Handle"):
                            hwnd = window.native.Handle.ToInt32()
                        if not hwnd:
                            hwnd = ctypes.windll.user32.FindWindowW(None, "Comic-Lab Studio")
                        if hwnd:
                            if ctypes.windll.user32.IsZoomed(hwnd):
                                current_state["maximized"] = True
                                current_state["fullscreen"] = False
                            elif not ctypes.windll.user32.IsIconic(hwnd):
                                current_state["maximized"] = False
                    except Exception:
                        pass
                save_window_state(current_state)
            except Exception:
                pass

            try:
                server_thread.stop()
            except Exception:
                pass
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.ExitProcess(0)
                except Exception:
                    pass
            os._exit(0)

        window.events.closing += on_closing
        window.events.closed += on_closing

        # Start pywebview using Edge Chromium (WebView2)
        webview.start(debug=args.dev)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if sys.stderr:
            sys.stderr.flush()

if __name__ == "__main__":
    main()
