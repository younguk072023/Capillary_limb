@echo off
setlocal
set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
"%PYTHON_EXE%" "%~dp0run_section4_pa_150.py" %*
if errorlevel 1 pause
endlocal
