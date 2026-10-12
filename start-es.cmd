@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\local_es.py serve
