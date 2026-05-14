@echo off
title AIES NIDS  v4.0  -  CT-361 AIES CCP
color 0E
set PYTHONNOUSERSITE=

echo ================================================================
echo   AIES NIDS  v4.0  -  Hybrid AI + Expert System NIDS
echo   CT-361 AIES CCP  -  NED University BCIT
echo ================================================================
echo.

REM --- Auto-detect Python (no hardcoded paths) ---
set "PYTHON_EXE="
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
  if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE if exist "C:\Python314\python.exe" set "PYTHON_EXE=C:\Python314\python.exe"
if not defined PYTHON_EXE if exist "C:\Python313\python.exe" set "PYTHON_EXE=C:\Python313\python.exe"
if not defined PYTHON_EXE if exist "C:\Python312\python.exe" set "PYTHON_EXE=C:\Python312\python.exe"
if not defined PYTHON_EXE if exist "C:\Python311\python.exe" set "PYTHON_EXE=C:\Python311\python.exe"
if not defined PYTHON_EXE if exist "C:\Python310\python.exe" set "PYTHON_EXE=C:\Python310\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PYTHON_EXE goto :no_python

echo   Python:  %PYTHON_EXE%
echo   Project: %~dp0
echo   URL:     http://localhost:8501
echo.

REM --- Probe deps ---
"%PYTHON_EXE%" -c "import sklearn,pandas,numpy,joblib,matplotlib,streamlit,plotly,seaborn,docx,streamlit_autorefresh" >nul 2>nul
if not errorlevel 1 goto :launch

echo  [INFO] First-run setup: installing dependencies (~60-120 sec, ~200 MB)
"%PYTHON_EXE%" -m pip install --user --quiet scikit-learn pandas numpy joblib matplotlib seaborn streamlit plotly python-docx streamlit-autorefresh
if errorlevel 1 goto :pip_failed
echo  [OK] Dependencies installed.
goto :launch

:no_python
echo  [ERROR] Python not found.
echo.
echo  Install Python 3.10+ from https://www.python.org/downloads/
echo  IMPORTANT: CHECK "Add Python to PATH" during install.
echo.
pause
exit /b 1

:pip_failed
echo  [ERROR] Dependency install failed. Check internet connection.
pause
exit /b 1

:launch
echo   [*] Starting AIES NIDS v4 dashboard...
echo   [*] Browser auto-opens in 6 sec. Stop: close window or Ctrl+C.
echo.
echo ----------------------------------------------------------------

cd /d "%~dp0"
start "" /b cmd /c "C:\Windows\System32\timeout.exe /t 6 /nobreak >nul && start http://localhost:8501"

REM Use python -m streamlit to bypass any APPDATA path issues (portable)
"%PYTHON_EXE%" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

echo.
echo ----------------------------------------------------------------
echo   Streamlit stopped. Press any key to close.
pause >nul
