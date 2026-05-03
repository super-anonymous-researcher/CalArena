# Benchmark data

Pre-generated HDF5 files and experiment CSVs are available on HuggingFace:

> **[https://huggingface.co/datasets/super-anonymous-researcher/CalArena](https://huggingface.co/datasets/super-anonymous-researcher/CalArena)**

Download the `.h5` files and place them here. The experiment CSVs (`*-experiments.csv`) are already committed in this directory.

To regenerate the HDF5 files from scratch, run the generation scripts from the repo root:

```bash
python calibration_benchmarks/generate_tabrepo_benchmarks.py
python calibration_benchmarks/generate_tabarena_benchmarks.py
python calibration_benchmarks/generate_cv_benchmarks.py
```

See the main [README](../README.md) for full instructions and data source requirements.
