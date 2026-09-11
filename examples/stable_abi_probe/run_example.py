from pathlib import Path

import torch


library = next(Path(__file__).parent.glob("_deep_jit_stable_abi_probe*.so"))
torch.ops.load_library(library)

input_tensor = torch.arange(8, device="cuda", dtype=torch.float32)
output_tensor = torch.ops.deep_jit_stable_abi_probe.add_one(input_tensor)
expected = input_tensor + 1
torch.testing.assert_close(output_tensor, expected)

print(f"input:  {input_tensor.cpu().tolist()}")
print(f"output: {output_tensor.cpu().tolist()}")
print("DeepJIT compiled and launched the add-one CUDA kernel successfully.")
