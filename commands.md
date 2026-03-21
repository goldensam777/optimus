# k-mamba CPU Commands Reference

## Build Commands

### Initial Build
```bash
cd /home/samuel-yevi/Dev/optimus/cpu
rm -rf build
mkdir -p build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### Recompile After Changes
```bash
cd /home/samuel-yevi/Dev/optimus/cpu/build
cmake --build . -j$(nproc)
```

Or using make directly:
```bash
make -C /home/samuel-yevi/Dev/optimus/cpu/build -j$(nproc)
```

## Training Commands

### Train on conversations.txt (Paper Config - 500K params)
```bash
./cpu/build/kmamba_cpu train ./cpu/data/conversations.txt ./cpu/ckpt_cpu_500k.bin paper_cpu
```

### Train with Custom Data File
```bash
./cpu/build/kmamba_cpu train <data_file> <checkpoint_file> [log_prefix]
```

**Example:**
```bash
./cpu/build/kmamba_cpu train ./my_data.txt ./my_checkpoint.bin my_experiment
```

## Chat/Interactive Mode

### Start Interactive Chat
```bash
./cpu/build/kmamba_cpu chat ./cpu/ckpt_cpu_500k.bin
```

**Controls:**
- Type your message and press Enter
- Type `quit` or `exit` to end session
- Ctrl+D to exit

## Generation Commands

### Generate Text from Prompt
```bash
./cpu/build/kmamba_cpu gen <checkpoint> [prompt]
```

**Example:**
```bash
./cpu/build/kmamba_cpu gen ./cpu/ckpt_cpu_500k.bin "Les systemes"
```

If no prompt is provided, it generates from an empty context.

## Model Configuration (500K Parameters)

| Parameter | Value |
|-----------|-------|
| Vocab Size | 256 (byte-level) |
| Dimension | 256 |
| State Size | 512 |
| Layers | 2 |
| Sequence Length | 128 |
| Batch Size | 64 |
| Epochs | 100 |
| Learning Rate | 5e-4 |
| Optimizer | AdamW |

**Estimated Parameters:** ~661K

**Chinchilla-Optimal Tokens:** ~13.2M (dataset has ~9.9M)

## CMake Build Options

### CPU Build (No Tests)
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKMAMBA_BUILD_TESTS=OFF \
  -DKMAMBA_BUILD_CUDA=OFF
```

### With Tests
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKMAMBA_BUILD_TESTS=ON
```

## File Locations

| Component | Path |
|-----------|------|
| Executable | `./cpu/build/kmamba_cpu` |
| Training Data | `./cpu/data/conversations.txt` |
| Checkpoint | `./cpu/ckpt_cpu_500k.bin` |
| Logs | `./cpu/ckpt_cpu_500k.bin.step.csv` |
| Logs | `./cpu/ckpt_cpu_500k.bin.epoch.csv` |

## CUDA Build (If Needed)

### Build CUDA Version
```bash
cd /home/samuel-yevi/Dev/optimus/cuda
rm -rf build && mkdir -p build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### CUDA Training
```bash
./cuda/build/kmamba_cuda train ./cpu/data/conversations.txt ./cuda/ckpt_cuda.bin
```

## Troubleshooting

### Clean Build
```bash
cd /home/samuel-yevi/Dev/optimus/cpu
rm -rf build
mkdir build && cd build
cmake .. && cmake --build . -j$(nproc)
```

### Check OpenBLAS Installation
```bash
ldconfig -p | grep openblas
```

### Verify Executable
```bash
./cpu/build/kmamba_cpu
# Should print model configuration and show usage
```
