@echo off
setlocal
set "PYTHON_EXE=C:\Users\USER\.conda\envs\younguk\python.exe"
"%PYTHON_EXE%" "%~dp0visualize_gt_auto_144.py" %*
if errorlevel 1 pause
endlocal
