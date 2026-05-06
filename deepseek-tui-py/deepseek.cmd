@echo off
cd /d "%~dp0deepseek-tui-py"
set PYTHONPATH=src
.venv\Scripts\python -m deepseek_tui.cli %*
