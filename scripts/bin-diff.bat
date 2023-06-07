@echo on
if [%1]==[] goto usage
if [%2]==[] goto usage
if not exist "%2" (
  echo bin-diff "%1" "%2"
  "%PROGRAMFILES%\Git\bin\sh.exe" --login -i -c ^
    "diff --report-identical-files --binary %1 %2"
)
goto :done
:usage
echo usage:
echo   bin-diff file1 file2
echo   USE FORWARD "/" !!!
exit /b 1
:done