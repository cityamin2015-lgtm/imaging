@echo off
chcp 65001 >nul
setlocal

set APP_VERSION=1.0.0
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

echo [1/4] Installing requirements...
python -m pip install -r requirements.txt || goto :error
python -m pip install pyinstaller || goto :error

echo [2/4] Building PatternEngine.exe...
python -m PyInstaller --noconfirm --clean --onefile --windowed --name PatternEngine pattern_gui.py || goto :error

echo [3/4] Building installer with Inno Setup...
if not exist %ISCC% (
  echo Inno Setup 6 not found. Install it from https://jrsoftware.org/isinfo.php
  goto :error
)
%ISCC% /DAppVersion=%APP_VERSION% installer.iss || goto :error

echo [4/4] Done.
echo Installer: installer_output\PatternEngine-Setup-%APP_VERSION%.exe
goto :eof

:error
echo Build failed.
exit /b 1
