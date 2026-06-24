"""Launch script — starts Flask and opens desktop window with window control API."""
import threading, time, subprocess, os
import webview
from flask import jsonify

_main_window = None

def kill_existing(port=5000):
    if os.name == 'nt':
        try:
            subprocess.run(
                f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}.*LISTENING\') do taskkill /F /PID %a',
                shell=True, capture_output=True, text=True, timeout=10)
        except: pass

def main():
    global _main_window
    kill_existing(5000)
    time.sleep(2)

    from app import app
    port = 5000

    @app.route('/api/window/minimize')
    def _win_min():
        if _main_window: _main_window.minimize()
        return jsonify({"ok": True})

    @app.route('/api/window/toggle_max')
    def _win_toggle():
        if _main_window:
            try: _main_window.toggle_fullscreen()
            except: pass
        return jsonify({"ok": True})

    @app.route('/api/window/close')
    def _win_close():
        if _main_window: _main_window.destroy()
        return jsonify({"ok": True})

    threading.Thread(target=lambda: app.run(port=port, debug=False, use_reloader=False), daemon=False).start()
    time.sleep(2)
    print(f"系统已启动: http://127.0.0.1:{port}")

    _main_window = webview.create_window("岩心孔洞裂缝分析系统", f"http://127.0.0.1:{port}",
                                         width=1280, height=800, resizable=True)
    webview.start()

if __name__ == "__main__":
    main()
