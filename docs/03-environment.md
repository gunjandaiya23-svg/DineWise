# Environment

> Part of the DineWise project plan. Hardware, Python, dependencies, and runtime decisions.
> See [the plan index](README.md) for the full reading order.

---

## 2.1 Environment

| Item | Value | Consequence |
|---|---|---|
| Working directory | `/home/naimish08/Desktop/DineWise` (empty at planning time) | Greenfield repository |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB VRAM, driver CUDA 13.2 | Sufficient for 1.5B FP16 inference, QLoRA, INT8/INT4 |
| System RAM | 22 GiB total, ~15 GiB available | Comfortable for pandas over the full review file; avoid loading multiple large models simultaneously |
| CPU | 20 logical cores | PDF generation and tokenization are not bottlenecks |
| Disk | 179 GiB free on `/` | Dataset (~1.3 GB), models (~6 GB), indexes (<1 GB) all fit comfortably |
| Default Python | 3.14.4 | **Unusable** for the ML stack; torch and bitsandbytes do not publish 3.14 wheels |
| Available Python | 3.12 via `python3.12`, plus `uv` | Build the project environment with Python 3.12 |

**Decision:** create a project-local virtual environment with `uv venv --python 3.12`, install CUDA-enabled PyTorch from the PyTorch wheel index, then freeze exact versions into `requirements.lock`. Never install into the system interpreter. Set `HF_HOME` and the model cache to a persistent project or home directory rather than `/tmp`.

**Decision:** develop locally on the RTX 4060. A Colab notebook may be generated later as a submission artifact, but it is not the primary runtime. Local execution avoids session limits and keeps the dataset and indexes persistent.
