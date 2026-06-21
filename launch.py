"""Launch script — starts Flask and opens browser."""
import webbrowser, threading, time

def main():
    from app import app
    port = 5000
    threading.Thread(target=lambda: app.run(port=port, debug=False), daemon=True).start()
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}")
    print(f"系统已启动: http://localhost:{port}")
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("关闭系统")

if __name__ == "__main__":
    main()
