@echo off
setlocal EnableDelayedExpansion
rem github builds in UTC timezone. To keep local builds
rem in monotonic sync with github version use UTC too.
for /f "tokens=* usebackq" %%i in (`tzutil /g`) do (
  set q="%%i"
)
set "tzrestore=tzutil /s %q%"
tzutil /s UTC
set GIT_REDIRECT_STDOUT=nul
set hash=BADF00D
if exist ".git" ( 
  for /f "delims=" %%a in ('git rev-parse --short HEAD') do (
      set hash=%%a
  )
)
set yy=%date:~12,2%
set mm=%date:~4,2%
set dd=%date:~7,2%
set hh=%time:~0,2%
set mm10=%mm%
set dd10=%dd%
set hh10=%hh%
rem mitigate for 08 and 09 are illegal octal constants in C
if "%mm10%" equ "08" ( set "mm10=8" )
if "%mm10%" equ "09" ( set "mm10=9" )
if "%dd10%" equ "08" ( set "dd10=8" )
if "%dd10%" equ "09" ( set "dd10=9" )
if "%hh10%" equ "08" ( set "hh10=8" )
if "%hh10%" equ "09" ( set "hh10=9" )
rem replace spaces with zeros:
set hh=%hh: =0%
set tag=C0DFEED
if exist ".git" ( 
  for /f "delims=" %%a in ('git describe --tags HEAD') do (
      set tag=%%a
  )
)
echo #pragma once
echo #define version_hash "%hash%"
echo #define version_tag "%tag%"
echo #define version_yy (%yy%)
echo #define version_mm (%mm10%)
echo #define version_dd (%dd10%)
echo #define version_hh (%hh10%)
echo #define version_str "%yy%.%mm%.%dd%.%hh%UTC %hash%"
echo #define version_int32 (0x%yy%%mm%%dd%%hh%)
rem `git rev-parse --short HEAD` spits out 28 bit hash tail now
rem but there is no guarantee that someone decide to make it longer
echo #define version_hash_int64 (0x%hash%LL)
%tzrestore%
