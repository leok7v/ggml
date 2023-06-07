"%PROGRAMFILES%\Git\bin\sh.exe" --login -i -c ^
"split ggml-model-gpt-j-6B.bin -b 2000m --verbose ; ^
mv xaa ggml-model-gpt-j-6B-00of06.bin; ^
mv xab ggml-model-gpt-j-6B-01of06.bin; ^
mv xac ggml-model-gpt-j-6B-02of06.bin; ^
mv xad ggml-model-gpt-j-6B-03of06.bin; ^
mv xae ggml-model-gpt-j-6B-04of06.bin; ^
mv xaf ggml-model-gpt-j-6B-05of06.bin; ^
cat ^
ggml-model-gpt-j-6B-00of06.bin ^
ggml-model-gpt-j-6B-01of06.bin ^
ggml-model-gpt-j-6B-02of06.bin ^
ggml-model-gpt-j-6B-03of06.bin ^
ggml-model-gpt-j-6B-04of06.bin ^
ggml-model-gpt-j-6B-05of06.bin ^
> ggml-model-gpt-j-6B.compare ; ^
diff --report-identical-files --binary ^
  ggml-model-gpt-j-6B.bin ggml-model-gpt-j-6B.compare ; ^
rm ggml-model-gpt-j-6B.compare ; ^
ls -l ggml-model-gpt-j-6B*"
