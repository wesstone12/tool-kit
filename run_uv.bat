@echo off
echo Running Python script: main.py
echo Press Enter to continue or Ctrl+C to cancel...
pause >nul
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv run test.py %*