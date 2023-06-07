"%PROGRAMFILES%\Git\bin\sh.exe" --login -i -c ^
"split ggml-model-stablelm-base-alpha-3b-f16.bin -b 2000m --verbose ; ^
mv xaa ggml-model-stablelm-base-alpha-3b-f16-00of04.bin; ^
mv xab ggml-model-stablelm-base-alpha-3b-f16-01of04.bin; ^
mv xac ggml-model-stablelm-base-alpha-3b-f16-02of04.bin; ^
mv xad ggml-model-stablelm-base-alpha-3b-f16-03of04.bin; ^
cat ^
ggml-model-stablelm-base-alpha-3b-f16-00of04.bin ^
ggml-model-stablelm-base-alpha-3b-f16-01of04.bin ^
ggml-model-stablelm-base-alpha-3b-f16-02of04.bin ^
ggml-model-stablelm-base-alpha-3b-f16-03of04.bin ^
> ggml-model-stablelm-base-alpha-3b-f16.compare ; ^
diff --report-identical-files --binary ^
  ggml-model-stablelm-base-alpha-3b-f16.bin ggml-model-stablelm-base-alpha-3b-f16.compare ; ^
rm ggml-model-stablelm-base-alpha-3b-f16.compare ; ^
ls -l ggml-model-stablelm-base-alpha-3b-f16*"
