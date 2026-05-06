@echo off
cd /d "%~dp0"
set PYTHONPATH=src
.venv\Scripts\python -m pytest tests/ %*
