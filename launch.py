"""Launch script — starts Flask and opens native desktop window."""
import threading, time, subprocess, os, webview

def kill_existing(port=5000):
    if os.name == 'nt':
        try:
            subprocess.run(
                f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}.*LISTENING\') do taskkill /F /PID %a',
                shell=True, capture_output=True, text=True, timeout=10)
        except: pass

def main():
    kill_existing(5000)
    time.sleep(2)
    from app import app
    port = 5000
    threading.Thread(target=lambda: app.run(port=port, debug=False, use_reloader=False), daemon=False).start()
    time.sleep(2)
    print(f"系统已启动: http://127.0.0.1:{port}")
    webview.create_window("岩心孔洞裂缝分析系统", f"http://127.0.0.1:{port}",
                          width=1280, height=800, resizable=True)
    webview.start()

if __name__ == "__main__":
    main()
