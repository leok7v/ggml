@echo off
if [%1]==[] goto usage
if [%2]==[] goto usage
if not exist "%2" (
  echo sha256 of "%1" to "%2"
  CertUtil -hashfile "%1" SHA256 | findstr /v "hash" > "%2"
)
goto :done
:usage
echo usage:
echo   sha256 file file.sha256
exit /b 1
:done