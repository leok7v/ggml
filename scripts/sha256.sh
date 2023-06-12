#!/bin/bash
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "usage:"
  echo "  sha256 file file.sha256"
  exit 1
fi

if [ ! -e "$2" ]; then
  echo "sha256 of \"$1\" to \"$2\""
  shasum -a 256 "$1" | awk '{ print $1 }' > "$2"
fi

exit 0

