@echo off

call venv\scripts\activate

python sample_image2video.py  --model HYVideo-T/2 --input-path "examples/case1" --prompt "An old-fashioned European village with thatched roofs on the houses." --i2v-stability --infer-steps 50 --flow-reverse --flow-shift 7.0 --seed 0 --embedded-cfg-scale 6.0 --use-cpu-offload --save-path ./results

pause