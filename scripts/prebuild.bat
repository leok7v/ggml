@echo off
pushd ..
call scripts\version.bat > version.h 2>nul

call scripts\download.bat models\ggml-model-gpt-2-117M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-117M.bin
call scripts\download.bat models\ggml-model-gpt-2-345M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-345M.bin
call scripts\download.bat models\ggml-model-gpt-2-774M.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-774M.bin

if not exist "models\ggml-model-gpt-2-1558M.bin" (
  call scripts\download.bat models\ggml-model-gpt-2-1558M-00of02.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-1558M-00of02.bin
  call scripts\download.bat models\ggml-model-gpt-2-1558M-01of02.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-2-1558M-01of02.bin
  copy /b ^
    models\ggml-model-gpt-2-1558M-00of02.bin + ^
    models\ggml-model-gpt-2-1558M-01of02.bin   ^
	models\ggml-model-gpt-2-1558M.bin
  del /f /q models\ggml-model-gpt-2-1558M-*of02.bin
  call scripts\sha256 models\ggml-model-gpt-2-1558M.bin ggml-model-gpt-2-1558M.sha256
  "%PROGRAMFILES%\Git\bin\sh.exe" --login -i -c ^
    diff --report-identical-files --binary ^
    ggml-model-gpt-2-1558M.sha256 ^
    models\ggml-model-gpt-2-1558M.sha256
  del /f /q ggml-model-gpt-2-1558M.sha256
)

if not exist "models\ggml-model-gpt-j-6B.bin" (
  call scripts\download.bat models\ggml-model-gpt-j-6B-00of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-00of06.bin
  call scripts\download.bat models\ggml-model-gpt-j-6B-01of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-01of06.bin
  call scripts\download.bat models\ggml-model-gpt-j-6B-02of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-02of06.bin
  call scripts\download.bat models\ggml-model-gpt-j-6B-03of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-03of06.bin
  call scripts\download.bat models\ggml-model-gpt-j-6B-04of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-04of06.bin
  call scripts\download.bat models\ggml-model-gpt-j-6B-05of06.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-gpt-j-6B-05of06.bin
  copy /b ^
	models\ggml-model-gpt-j-6B-00of06.bin + ^
	models\ggml-model-gpt-j-6B-01of06.bin + ^
	models\ggml-model-gpt-j-6B-02of06.bin + ^
	models\ggml-model-gpt-j-6B-03of06.bin + ^
	models\ggml-model-gpt-j-6B-04of06.bin + ^
	models\ggml-model-gpt-j-6B-05of06.bin   ^
    models\ggml-model-gpt-j-6B.bin
  del /f /q models\ggml-model-gpt-j-6B-*of06.bin
)

if not exist "models\ggml-model-stablelm-base-alpha-3b-f16.bin" (
  call scripts\download.bat models\ggml-model-stablelm-base-alpha-3b-f16-00of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-00of06.bin
  call scripts\download.bat models\ggml-model-stablelm-base-alpha-3b-f16-01of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-01of06.bin
  call scripts\download.bat models\ggml-model-stablelm-base-alpha-3b-f16-02of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-02of06.bin
  call scripts\download.bat models\ggml-model-stablelm-base-alpha-3b-f16-03of04.bin https://github.com/leok7v/ggml/releases/download/2023-06-06/ggml-model-stablelm-base-alpha-3b-f16-03of06.bin
  copy /b ^
	models\ggml-model-stablelm-base-alpha-3b-f16-00of04.bin + ^
	models\ggml-model-stablelm-base-alpha-3b-f16-01of04.bin + ^
	models\ggml-model-stablelm-base-alpha-3b-f16-02of04.bin + ^
	models\ggml-model-stablelm-base-alpha-3b-f16-03of04.bin   ^
    models\ggml-model-stablelm-base-alpha-3b-f16.bin
  del /f /q models\ggml-model-stablelm-base-alpha-3b-f16-*of06.bin
)

popd
exit /b 0