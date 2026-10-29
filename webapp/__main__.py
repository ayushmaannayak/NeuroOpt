"""python -m webapp [--host 127.0.0.1] [--port 8000] [--enable-measure]"""
import argparse

from .app import create_app


def main():
    ap = argparse.ArgumentParser(prog="python -m webapp")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--enable-measure", action="store_true",
                    help="allow /api/measure (executes submitted instructions natively; local use only)")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1") and a.enable_measure:
        raise SystemExit("Refusing to enable hardware timing on a non-local address.")
    app = create_app(enable_measure=a.enable_measure)
    print(f"NeuroOpt web app on http://{a.host}:{a.port}  (hardware timing {'ON' if a.enable_measure else 'off'})")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
