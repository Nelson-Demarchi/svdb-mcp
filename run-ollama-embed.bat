@echo off
REM === Ollama embedding server for SVDB (models kept inside the SVDB folder) ===
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
set "OLLAMA_MODELS=%~dp0ollama-models"
set "OLLAMA_HOST=127.0.0.1:11434"

REM Machine-specific overrides (GPU pinning etc.) live in ollama-env.bat, which is not published.
REM It is called last, so anything set there wins over the defaults above.
if exist "%~dp0ollama-env.bat" call "%~dp0ollama-env.bat"

echo This script runs a dedicated Ollama instance that serves models from:
echo   %OLLAMA_MODELS%
echo.
echo To do that it first TERMINATES every running Ollama process, including one
echo you may already be using with its own models. Close this window if that is
echo not what you want.
echo.
pause

echo Stopping any running Ollama instances...
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM "ollama.exe" >nul 2>&1
timeout /t 2 /nobreak >nul

echo Starting Ollama serve...
echo   OLLAMA_MODELS = %OLLAMA_MODELS%
echo   OLLAMA_HOST   = %OLLAMA_HOST%
"%OLLAMA_EXE%" serve
