#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <filename> <URL>"
else
  if [ ! -e "$1" ]; then
    echo "downloading $1 from $2"
    # --location is actually "follow redirect"
    curl --silent --location "$2" --create-dirs --output "$1"
  else
    echo "$1 already exists"
  fi
fi

