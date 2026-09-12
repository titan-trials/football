"""Print what this environment actually has, so a missing package is obvious."""
import sys, importlib
print(f"python   {sys.version.split()[0]}   ({sys.executable})")
need = ["polars", "numpy", "pandas", "pyarrow", "nflreadpy", "scipy", "sklearn",
        "streamlit", "altair", "pytest", "requests"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  {m:12} MISSING")
        missing.append(m)
if missing:
    print(f"\n{len(missing)} missing: {', '.join(missing)}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 12):
        print(f"\nPython {major}.{minor} is below 3.12, and requirements.txt pins")
        print("numpy==2.5.1 and scipy==1.18.0, which require 3.12+. pip resolves a")
        print("requirements file ATOMICALLY -- one unsatisfiable pin installs NOTHING,")
        print("which is exactly what an empty venv looks like.")
        print("\nFix, either:")
        print("  1. install Python 3.12+ and re-run setup.ps1, or")
        print("  2. relax those two lines in requirements.txt to:")
        print("       numpy>=2.0,<2.5")
        print("       scipy>=1.11,<1.18")
    else:
        print("\nPython is new enough; the install just did not run or did not finish.")
        print("  python -m pip install -r requirements.txt")
else:
    print("\nall present")

print()
try:
    from data.env import env_exists, env_path
    from data.odds import resolve_api_key, NFL_ENV
    print(f".env  {'found' if env_exists() else 'MISSING'}  at {env_path()}")
    try:
        _, source = resolve_api_key()
        print(f"key   resolves from {source}")
        if source == "ODDS_API_KEY":
            print("      ^ that is the SHARED process variable "
                  "(baseball_predictor's on this machine).")
            print(f'      Fix: Set-Content -Path .env -Value "{NFL_ENV}=your-key" '
                  f"-Encoding utf8")
    except Exception as e:
        print(f"key   NOT FOUND -- {e}")
except ImportError:
    pass
