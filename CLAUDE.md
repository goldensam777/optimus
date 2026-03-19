# CLAUDE.md — Projet optimus

Projet ML en C/CUDA : entraîner un modèle Mamba 1.45M params sur du texte français.

## Structure du repo

```
optimus/
├── k-mamba/                   # sous-module : bibliothèque SSM
│   ├── src/mamba_block.c      # dispatch CPU/CUDA (KMAMBA_BUILD_CUDA)
│   ├── cpu/                   # scan AVX2 (scan1d.asm, mamba_scan.c)
│   ├── cuda/
│   │   ├── scan1d.cu          # Blelloch parallel prefix scan (L<=1024)
│   │   ├── scan1d_backward.cu # adjoint séquentiel
│   │   ├── mamba_scan.cu      # wrappers mamba_scan1d_cuda_forward/backward
│   │   └── mamba_block.cu     # gpu_block_forward / gpu_block_backward
│   ├── optimatrix/
│   │   ├── include/optimatrix.h
│   │   └── cuda/optimizer_utils.cu  # AdamW, MUON, Newton-Schulz CUDA
│   └── include/
│       ├── scan.h             # om_scan1d_forward/backward
│       └── mamba_scan_cuda.h  # mamba_scan1d_cuda_forward/backward
├── cpu/
│   ├── CMakeLists.txt
│   ├── main.c                 # instance CPU (AVX2, MUON CPU)
│   └── conversations.txt      # dataset (lien symbolique ou copie)
└── cuda/
    ├── CMakeLists.txt
    ├── main.cu                # instance GPU (Blelloch scan, MUON GPU)
    └── data/
        └── conversations.txt  # dataset 1.6MB, 195 steps/epoch
```

## Config modèle (les deux instances)

```c
VOCAB_SIZE  = 256
DIM         = 256
STATE_SIZE  = 512
N_LAYERS    = 5
SEQ_LEN     = 256
BATCH_SIZE  = 32
N_EPOCHS    = 200
LR          = 2e-4f   // MUON lr
```

→ **1 450 752 paramètres** (1.45M)

## Build

```bash
# CPU
cd ~/Dev/optimus/cpu
mkdir -p build && cd build && cmake .. && cmake --build . -j

# GPU
cd ~/Dev/optimus/cuda
mkdir -p build && cd build && cmake .. && cmake --build . -j
```

## Lancer l'entraînement

```bash
# CPU (background)
nohup ~/Dev/optimus/cpu/build/kmamba_cpu \
  ~/Dev/optimus/cpu/conversations.txt \
  > /tmp/kmamba_cpu_train.log 2>&1 &

# GPU (background)
nohup ~/Dev/optimus/cuda/build/kmamba_cuda \
  ~/Dev/optimus/cuda/data/conversations.txt \
  > /tmp/kmamba_gpu_train.log 2>&1 &

# Suivre
tail -f /tmp/kmamba_gpu_train.log
```

## Checkpoints

- CPU : `cpu/build/kmamba_cpu.bin` (sauvegardé toutes les 5 epochs)
- GPU : `cuda/build/kmamba_gpu.bin`

---

## Bugs corrigés dans cette session

### 1. `main.c` → `main.cu` (compilation NVCC)
Les kernels CUDA (`__global__`, `<<<...>>>`, `dim3`) étaient dans `main.c` compilé par GCC.
Fix : renommer en `main.cu` + `CMakeLists.txt` → `add_executable(kmamba_cuda main.cu)`.

### 2. `cudaMalloc` void** cast
`cudaMalloc(&p, ...)` avec `float *p` → cast en `(void**)&p` requis en C++.

### 3. `extern "C"` manquant dans `mamba_block.cu`
`gpu_block_forward` / `gpu_block_backward` non accessibles depuis `main.cu` (symbols manglés).
Fix : `extern "C" void gpu_block_forward(...)` et `extern "C" void gpu_block_backward(...)`.

### 4. Buffer `d_tokens` trop petit
`d_tokens` alloué pour `SEQ_LEN` mais rempli avec `SEQ_LEN+1` tokens (input + target).
Fix : `gpu_alloc_int(L + 1)` dans `gpu_model_create`.

### 5. `L` vs `L_in` dans `gpu_forward_backward`
Le forward/backward du modèle doit opérer sur `L_in = L - 1 = SEQ_LEN` positions
(les tokens 1..L sont les targets, pas des inputs supplémentaires).
Fix : introduire `int L_in = L - 1` et l'utiliser partout sauf pour le `cudaMemcpy` des tokens.

### 6. `muon_update_mat_cuda` / `muon_update_vec_cuda` : pointeurs CPU vs GPU
**Bug critique** : ces fonctions font `cudaMemcpyHostToDevice` depuis les pointeurs passés,
supposant qu'ils sont sur CPU. Mais dans `gpu_optimizer_step`, tous les pointeurs
(`b->W_in`, `b->g_W_in`, etc.) sont des **device pointers** (VRAM).
Résultat : CUDA sticky error → aucun paramètre n'est mis à jour → loss bloquée à ln(256) = 5.5452.

Fix : ajout de **variantes device-native** dans `optimizer_utils.cu` et `optimatrix.h` :
```c
void muon_update_mat_device(float *d_param, float *d_grad, float *d_m,
                             size_t rows, size_t cols, const MBOptimConfig *conf);
void muon_update_vec_device(float *d_param, float *d_grad, float *d_m,
                             size_t n, const MBOptimConfig *conf);
```
Ces fonctions appellent directement `newton_schulz5_cuda`, `muon_momentum_kernel`,
`gradient_clip_inplace_cuda`, `muon_step_kernel` sans aucun transfert mémoire.
`gpu_optimizer_step` dans `main.cu` utilise ces variantes.

---

## Bug RESTANT à corriger (non fixé)

### Résiduel in-place incorrect dans `gpu_block_forward`

Dans `gpu_block_forward`, forward et output sont le même buffer :
```c
gpu_block_forward(h, ..., m->d_hidden, m->d_hidden, ...);  // x == y
```
Le code fait :
```c
cudaMemcpy(d_y, d_y_proj, L*dim*sizeof(float), cudaMemcpyDeviceToDevice); // d_hidden = y_proj
add_inplace_kernel(d_y, d_x, L*dim);  // d_hidden += d_x = d_hidden -> 2*y_proj !!
```
Résultat : au lieu de `y = y_proj + x_original`, on obtient `y = 2 * y_proj`.
L'original `x` est écrasé par le `cudaMemcpy` avant que `add_inplace_kernel` puisse l'additionner.

**Fix à implémenter** : sauvegarder `d_x` avant la copie, OU éviter l'appel in-place :
```c
// Option A : dans main.cu, utiliser ws->layer_input (déjà sauvegardé) comme x
// au lieu de m->d_hidden. Passer layer_input comme x et d_hidden comme y.
gpu_block_forward(h, ..., w->layer_input, m->d_hidden, ...);  // x != y

// ET supprimer la sauvegarde séparée (elle devient inutile pour le forward)
// mais la garder pour le backward
```

Avec ce fix, le résiduel sera correct : `d_hidden = y_proj + layer_input`.

---

## État actuel

- **CPU** : tourne à 1.45M params sur `conversations.txt`, ~190s/epoch. Processus tué (RAM).
- **GPU** : loss bloquée à 5.5452 = ln(256). Bug #6 (MUON device) fixé. Bug résiduel in-place pas encore fixé.
- **Prochaine étape** : appliquer le fix in-place ci-dessus, rebuild, et lancer l'entraînement GPU proprement.

## Conventions GEMM row-major cuBLAS (prouvées)

```
C[M,N] = A[M,K] @ B[K,N]    →  cublasSgemm(h, N,N, N,M,K, a, B,N, A,K, b, C,N)
C[M,N] = A[M,K] @ B^T[N,K]  →  cublasSgemm(h, T,N, N,M,K, a, B,K, A,K, b, C,N)
C[M,N] = A^T[K,M] @ B[K,N]  →  cublasSgemm(h, N,T, N,M,K, a, B,N, A,M, b, C,N)
```

## Scan SSM (layout mémoire)

```
x  : [L * D]          (D = state_size)
A  : [D * M]          (M = 1 ici)
B  : [L * D * M]      (= [L * state] avec M=1)
C  : [L * D * M]
dt : [L * D]          (broadcast depuis [L] via broadcast_l_to_ld)
h  : [L * D * M]
y  : [L * D]
```

Le scan backward **zeroise** `dx`, `dA`, `ddt` au début — ne pas pré-zeriser ces buffers.
