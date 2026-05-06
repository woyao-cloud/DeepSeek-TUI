@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONPATH=src
python -m deepseek_tui.cli %*
