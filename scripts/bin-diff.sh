#!/bin/bash
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "usage:"
  echo "  bin-diff file1 file2"
  exit 1
fi

if [ ! -e "$2" ]; then
  echo "bin-diff \"$1\" \"$2\""
  diff --report-identical-files --binary "$1" "$2"
fi

exit 0

