@echo off
setlocal

REM Define the directory containing the .tex files
set TEX_DIR=.

REM Loop through all .tex files in the directory
for %%f in (%TEX_DIR%\*.tex) do (
    echo Processing %%f...
    lualatex --cnf-line=openout_any=a -shell-escape "%%f"
)

echo All diagrams processed.
endlocal
