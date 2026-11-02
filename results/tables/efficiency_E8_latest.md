| config | precision | adapter | load_seconds | ttft_ms | tpot_ms | tokens_per_second | peak_vram_gb | weights_gb |
|---|---|---|---|---|---|---|---|---|
| M1_fp16 | fp16 | False | 0.86 | 458.1 | 16.4 | 43.17 | 4.0 | 3.09 |
| M2_int8 | int8 | False | 2.67 | 467.5 | 72.6 | 13.38 | 2.69 | 1.78 |
| M3_int4 | int4 | False | 0.78 | 479.1 | 17.2 | 42.98 | 2.05 | 1.12 |
| M4_lora_fp16 | fp16 | True | 1.2 | 673.6 | 25.9 | 29.67 | 4.03 | 3.12 |
| M5_lora_int4 | int4 | True | 1.09 | 683.7 | 40.3 | 20.78 | 2.09 | 1.16 |
