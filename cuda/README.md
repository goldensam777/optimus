# k-mamba — Instance CUDA (CPU + MX450)

Byte-level language model basé sur k-mamba, un State Space Model (SSM) de type Mamba
implémenté en C pur et assembleur NASM AVX2. Les optimizer steps (MUON, AdamW) sont
déchargés sur GPU via CUDA.

**Architecture hybride :**
- Forward / backward : CPU (scan ASM AVX2 — pas de transfert mémoire GPU)
- Optimizer steps   : GPU (kernels CUDA sur les gradients CPU → pinned → VRAM)

---

## Prérequis

- CPU x86-64 avec AVX2 (`grep avx2 /proc/cpuinfo`)
- GPU NVIDIA sm_75+ (testé : MX450 2GB VRAM)
- CUDA Toolkit >= 11.0 (`nvcc --version`)
- CMake >= 3.18
- GCC >= 9 (C11)
- NASM >= 2.14

---

## Build

```bash
cd ~/Dev/optimus/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

L'exécutable est `build/kmamba_cuda`.

---

## Utilisation

### Entraînement

```bash
# Sur le texte intégré (démo)
make -C build train

# Sur un fichier texte externe
make -C build train-file DATA_FILE=data/conversations.txt CKPT_FILE=ckpt.bin

# Reprendre depuis un checkpoint
make -C build train-file DATA_FILE=data/conversations.txt CKPT_FILE=ckpt.bin
```

Ou directement :

```bash
./build/kmamba_cuda                               # texte intégré
./build/kmamba_cuda train data/corpus.txt         # fichier → checkpoint par défaut
./build/kmamba_cuda train data/corpus.txt ckpt.bin
```

### Génération

```bash
make -C build gen CKPT_FILE=ckpt.bin PROMPT="Bonjour"
# ou
./build/kmamba_cuda gen ckpt.bin "Bonjour"
```

### Chat interactif (REPL)

```bash
make -C build chat CKPT_FILE=ckpt.bin
# ou
./build/kmamba_cuda chat ckpt.bin
```

```
  k-mamba — session interactive
  Ctrl+D ou 'quit' pour quitter
  ─────────────────────────────

you> Comment tu vas ?
CP> Ça va bien, merci !
```

---

## Configuration du modèle

Tous les hyperparamètres sont des `#define` en tête de `main.c` :

| Paramètre       | Valeur   | Description                                      |
|-----------------|----------|--------------------------------------------------|
| `VOCAB_SIZE`    | 256      | Vocabulaire byte-level                           |
| `DIM`           | 128      | Dimension des embeddings et états cachés         |
| `STATE_SIZE`    | 256      | Taille de l'état SSM par couche                  |
| `N_LAYERS`      | 2        | Nombre de MambaBlocks empilés                    |
| `SEQ_LEN`       | 256      | Longueur de séquence (fenêtre de contexte)       |
| `BATCH_SIZE`    | 32       | Séquences traitées simultanément                 |
| `N_EPOCHS`      | 50       | Nombre de passes sur les données                 |
| `LR_BLOCKS`     | 1e-3     | Learning rate des MambaBlocks (MUON)             |
| `LR_EMBED_HEAD` | 1e-3     | Learning rate embedding + LM head (AdamW)        |
| `WEIGHT_DECAY`  | 1e-5     | Régularisation L2 découplée                      |
| `CLIP_NORM`     | 1.0      | Gradient clipping global (norme L2 max)          |
| `TEMPERATURE`   | 0.8      | Température de sampling (0=greedy, 1=aléatoire)  |
| `GEN_LEN`       | 512      | Bytes générés en mode `gen`                      |
| `CHAT_MAX_RESP` | 256      | Longueur max d'une réponse en mode chat          |

**Empreinte mémoire à ces valeurs :**
- Paramètres CPU : ~2 MB RAM
- États optimiseur (×6) : ~12 MB RAM
- Buffers CUDA temporaires : ~5-15 MB VRAM (pas de copie permanente sur GPU)
- Total : ~30 MB RAM + ~15 MB VRAM

La configuration est intentionnellement plus petite que l'instance CPU pour rester
dans les limites du MX450 (2 GB VRAM). Pour un GPU plus puissant, augmenter
`DIM`, `STATE_SIZE`, et `N_LAYERS` dans `main.c`.

---

## Différence avec l'instance CPU

| Aspect              | CPU (`../cpu/`)         | CUDA (`./`)                     |
|---------------------|-------------------------|---------------------------------|
| Scan forward        | ASM AVX2                | ASM AVX2 (identique)            |
| Backward            | C + ASM                 | C + ASM (identique)             |
| MUON step           | C (CPU)                 | CUDA kernel (GPU)               |
| AdamW step          | C (CPU)                 | CUDA kernel (GPU)               |
| gradient_clip       | C (CPU)                 | CUDA kernel (GPU)               |
| VRAM requise        | 0                       | ~15 MB (buffers temporaires)    |
| Tests de diagnostic | `test_scan`, `test_grad`| aucun (utiliser les tests CPU)  |

Le checkpoint est compatible entre les deux instances — un modèle entraîné sur CPU
peut être repris sur CUDA et vice-versa.

---

## Format des données

Même format que l'instance CPU : fichier texte brut `.txt`, niveau byte (pas de tokenizer).

Pour le mode chat, le corpus doit suivre :
```
Human: <question>
Bot: <réponse>
```

Les labels `CHAT_USER` / `CHAT_BOT` dans `main.c` doivent correspondre au format
du corpus d'entraînement.

---

## Entraînement long (nohup)

```bash
nohup ./build/kmamba_cuda train data/corpus.txt ckpt.bin > train_cuda.log 2>&1 &
tail -f train_cuda.log
```

---

## Structure des fichiers

```
cuda/
├── CMakeLists.txt      — build CUDA (sm_75), lie k-mamba, CUDA toolkit
├── main.c              — runner : train / gen / chat, configs #define
└── build/
    └── kmamba_cuda     — exécutable principal
```

La bibliothèque k-mamba et ses kernels sont dans `../k-mamba/`.
