# Windows Jupyter Quickstart

This branch does not require Google Colab helpers. In a Windows Jupyter
Notebook, upload or place the project zip beside the notebook and run a pure
Python setup cell.

## Zip To Use

```text
uabla_memory_cache_alt_windows_jupyter.zip
```

## Notes

- Use Python `zipfile` to extract instead of shell `unzip`.
- Use `sys.executable -m pip` so packages install into the active notebook
  kernel.
- If PyTorch is already installed, the tests and smoke check will use it.
- If PyTorch is not installed, install the correct PyTorch build for your
  machine first. CPU wheels are fine for the smoke test; CUDA wheels are only
  needed for GPU experiments.

## Smoke Test Scope

The first Windows cell runs:

```text
pip install -e .
pytest
RevivalMemoryCache random-tensor smoke test
```

This verifies the separate memory-cache simulator, not full model training.
