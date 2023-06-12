#!/bin/bash

pushd ..
scripts/version.bat > version.h 2>/dev/null

if [ ! -e "models/ggml-model-gpt-2-117M.bin" ]; then
  scripts/download.sh models/ggml-model-gpt-2-117M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-117M.bin
  scripts/sha256.sh models/ggml-model-gpt-2-117M.bin ggml-model-gpt-2-117M.sha256
  scripts/bin-diff.sh ggml-model-gpt-2-117M.sha256 models/ggml-model-gpt-2-117M.sha256
  rm -f ggml-model-gpt-2-117M.sha256
fi

if [ ! -e "models/ggml-model-gpt-2-345M.bin" ]; then
  scripts/download.sh models/ggml-model-gpt-2-345M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-345M.bin
  scripts/sha256.sh models/ggml-model-gpt-2-345M.bin ggml-model-gpt-2-345M.sha256
  scripts/bin-diff.sh ggml-model-gpt-2-345M.sha256 models/ggml-model-gpt-2-345M.sha256
  rm -f ggml-model-gpt-2-345M.sha256
fi

if [ ! -e "models/ggml-model-gpt-2-774M.bin" ]; then
  scripts/download.sh models/ggml-model-gpt-2-774M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-774M.bin
  scripts/sha256.sh models/ggml-model-gpt-2-774M.bin ggml-model-gpt-2-774M.sha256
  scripts/bin-diff.sh ggml-model-gpt-2-774M.sha256 models/ggml-model-gpt-2-774M.sha256
  rm -f ggml-model-gpt-2-774M.sha256
fi

if [ ! -e "models/ggml-model-gpt-2-1558M.bin" ]; then
  scripts/download.sh models/ggml-model-gpt-2-1558M-00of02.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-1558M-00of02.bin
  scripts/download.sh models/ggml-model-gpt-2-1558M-01of02.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-1558M-01of02.bin
  cat models/ggml-model-gpt-2-1558M-00of02.bin models/ggml-model-gpt-2-1558M-01of02.bin > models/ggml-model-gpt-2-1558M.bin
  rm -f models/ggml-model-gpt-2-1558M-*of02.bin
  scripts/sha256.sh models/ggml-model-gpt-2-1558M.bin ggml-model-gpt-2-1558M.sha256
  scripts/bin-diff.sh ggml-model-gpt-2-1558M.sha256 models/ggml-model-gpt-2-1558M.sha256
  rm -f ggml-model-gpt-2-1558M.sha256
fi

if [ ! -e "models/ggml-model-stablelm-base-alpha-3b-q4_0.bin" ]; then
  scripts/download.sh models/ggml-model-stablelm-base-alpha-3b-q4_0.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-q4_0.bin
  scripts/sha256.sh models/ggml-model-stablelm-base-alpha-3b-q4_0.bin ggml-model-stablelm-base-alpha-3b-q4_0.sha256
  scripts/bin-diff.sh ggml-model-stablelm-base-alpha-3b-q4_0.sha256 models/ggml-model-stablelm-base-alpha-3b-q4_0.sha256
  rm -f ggml-model-stablelm-base-alpha-3b-q4_0.sha256
fi

if [ ! -e "models/ggml-model-stablelm-base-alpha-3b-f16.bin" ]; then
  scripts/download.sh models/ggml-model-stablelm-base-alpha-3b-f16-00of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-00of04.bin
  scripts/download.sh models/ggml-model-stablelm-base-alpha-3b-f16-01of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-01of04.bin
  scripts/download.sh models/ggml-model-stablelm-base-alpha-3b-f16-02of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-02of04.bin
  scripts/download.sh models/ggml-model-stablelm-base-alpha-3b-f16-03of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-03of04.bin
  cat models/ggml-model-stablelm-base-alpha-3b-f16-00of04.bin models/ggml-model-stablelm-base-alpha-3b-f16-01of04.bin models/ggml-model-stablelm-base-alpha-3b-f16-02of04.bin models/ggml-model-stablelm-base-alpha-3b-f16-03of04.bin > models/ggml-model-stablelm-base-alpha-3b-f16.bin
  rm -f models/ggml-model-stablelm-base-alpha-3b-f16-*of04.bin
  scripts/sha256.sh models/ggml-model-stablelm-base-alpha-3b-f16.bin ggml-model-stablelm-base-alpha-3b-f16.sha256
  scripts/bin-diff.sh ggml-model-stablelm-base-alpha-3b-f16.sha256 models/ggml-model-stablelm-base-alpha-3b-f16.sha256
  rm -f ggml-model-stablelm-base-alpha-3b-f16.sha256
fi

if [ ! -e "models/ggml-model-gpt-j-6B.bin" ]; then
  scripts/download.sh models/ggml-model-gpt-j-6B-00of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-00of06.bin
  scripts/download.sh models/ggml-model-gpt-j-6B-01of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-01of06.bin
  scripts/download.sh models/ggml-model-gpt-j-6B-02of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-02of06.bin
  scripts/download.sh models/ggml-model-gpt-j-6B-03of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-03of06.bin
  scripts/download.sh models/ggml-model-gpt-j-6B-04of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-04of06.bin
  scripts/download.sh models/ggml-model-gpt-j-6B-05of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-05of06.bin
  cat models/ggml-model-gpt-j-6B-00of06.bin models/ggml-model-gpt-j-6B-01of06.bin models/ggml-model-gpt-j-6B-02of06.bin models/ggml-model-gpt-j-6B-03of06.bin models/ggml-model-gpt-j-6B-04of06.bin models/ggml-model-gpt-j-6B-05of06.bin > models/ggml-model-gpt-j-6B.bin
  rm -f models/ggml-model-gpt-j-6B-*of06.bin
  scripts/sha256.sh models/ggml-model-gpt-j-6B.bin ggml-model-gpt-j-6B.sha256
  scripts/bin-diff.sh ggml-model-gpt-j-6B.sha256 models/ggml-model-gpt-j-6B.sha256
  rm -f ggml-model-gpt-j-6B.sha256
fi



