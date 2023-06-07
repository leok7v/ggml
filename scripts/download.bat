@echo off
if not exist "%1" (
  echo downloading "%1" from "%2"
  rem --location is actually "follow redirect"
  curl.exe --silent --location "%2" --create-dirs --output "%1"
)