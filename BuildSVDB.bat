@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python BuildSVDB.py
echo.
pause
