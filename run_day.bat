@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --list
  echo.
  echo Usage: run_day.bat 0
  exit /b 0
)
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --day %1
