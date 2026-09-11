# DeepJIT stable-ABI CUDA example

This example registers an `add_one(Tensor) -> Tensor` CUDA operator with
`STABLE_TORCH_LIBRARY`. On its first call, DeepJIT compiles the embedded CUDA
kernel with NVCC, caches and loads the CUBIN, and launches it on PyTorch's
current CUDA stream.

Build the stable-ABI shared library:

```bash
python3 setup.py build_ext --inplace
```

Run the end-to-end example on a host with an accessible CUDA device:

```bash
python3 run_example.py
```

Expected output:

```text
input:  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
DeepJIT compiled and launched the add-one CUDA kernel successfully.
```

The first run compiles the kernel. Later runs reuse DeepJIT's disk cache. Set
`DEEP_JIT_STABLE_ABI_PROBE_JIT_CACHE_DIR` to choose the cache directory.

Audit the extension's PyTorch ABI:

```bash
torch-abi-audit -v ./_deep_jit_stable_abi_probe*.so
```

The expected result is `Torch ABI: STABLE` with zero unstable symbols. Because
the example uses stable dispatcher registration rather than a Python module
initializer, the audit reports its CPython ABI as `n/a`. The binary can still
contain CPython symbols pulled in by DeepJIT's pybind11-based GIL helper.
