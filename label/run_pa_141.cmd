@echo off
setlocal
set "PYTHON_EXE=C:\Users\USER\.conda\envs\younguk\python.exe"
set "IMAGE_ROOT=C:\Users\USER\Desktop\Pa_capillary"
set "REVIEW_CSV=%~dp0outputs\section4_review_pa_141\manual_review_141.csv"

"%PYTHON_EXE%" "%~dp0section4_manual_review.py" --image-dir "%IMAGE_ROOT%" --shape-root "%IMAGE_ROOT%" --independent --simple --split-by-shape --output-csv "%REVIEW_CSV%" %*
if errorlevel 1 pause
endlocal
