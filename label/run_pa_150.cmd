@echo off
setlocal
set "PYTHON_EXE=C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "IMAGE_ROOT=C:\Users\USER\Desktop\Pa_capillary"
set "REVIEW_CSV=%~dp0outputs\section4_review_pa_150\manual_review_150.csv"

"%PYTHON_EXE%" "%~dp0section4_manual_review.py" --image-dir "%IMAGE_ROOT%" --shape-root "%IMAGE_ROOT%" --independent --output-csv "%REVIEW_CSV%" %*
if errorlevel 1 pause
endlocal
