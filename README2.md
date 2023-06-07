# Windows deployment of some ggml examples

[![build](https://github.com/leok7v/ggml/actions/workflows/build.yml/badge.svg)](https://github.com/leok7v/ggml/actions/workflows/build.yml)
[![test](https://github.com/leok7v/ggml/actions/workflows/test.yml/badge.svg)](https://github.com/leok7v/ggml/actions/workflows/test.yml)

## Features

msvc2022/ggml.sln with example projects

scripts/prebuild.bat script automatically downloads models from github

For now prebuild.bat is not invoked automatically relying on manual invocation instead

## Roadmap

- [x] github CI build and test
- [x] Example of GPT-2 inference bin\Release|Debug\gpt-2.exe
- [x] Example of GPT-2 inference bin\Release|Debug\gpt-j.exe
