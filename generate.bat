echo === Running GPT-OSS-20B ===
python scripts/run_generate.py --model openai/gpt-oss-20b --max-samples 1000
if errorlevel 1 exit /b 1

echo === Running GLM-4.7-Flash ===
python scripts/run_generate.py --model zai-org/GLM-4.7-Flash --max-samples 300 --resume
if errorlevel 1 exit /b 1
