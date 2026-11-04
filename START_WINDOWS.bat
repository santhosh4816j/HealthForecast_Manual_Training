@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 or newer is required. Install Python and enable the py launcher.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your connection and try again.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
pause
