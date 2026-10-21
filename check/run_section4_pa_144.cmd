@echo off
setlocal
set "PYTHON_EXE=C:\Users\USER\.conda\envs\younguk\python.exe"
"%PYTHON_EXE%" "%~dp0run_section4_pa_144.py" %*
if errorlevel 1 pause
endlocal
