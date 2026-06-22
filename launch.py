"""Launch script — starts Flask and opens browser."""
import webbrowser, threading, time, subprocess, os

def kill_existing(port=5000):
    if os.name == 'nt':
        try:
            result = subprocess.run(
                f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}.*LISTENING\') do taskkill /F /PID %a',
                shell=True, capture_output=True, text=True, timeout=10)
        except: pass

def main():
    kill_existing(5000)
    time.sleep(2)
    from app import app
    port = 5000
    threading.Thread(target=lambda: app.run(port=port, debug=False, use_reloader=False), daemon=True).start()
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:{port}")
    print(f"系统已启动: http://127.0.0.1:{port}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("关闭系统")

if __name__ == "__main__":
    main()
