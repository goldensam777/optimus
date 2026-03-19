# k-mamba — Instance CPU (AVX2)

Byte-level language model basé sur k-mamba, un State Space Model (SSM) de type Mamba
implémenté en C pur et assembleur NASM AVX2. Aucune dépendance Python, aucun framework
ML. Juste du C, de l'assembleur, et libc/libm.

---

## Prérequis

- CPU x86-64 avec support AVX2 (vérifier : `grep avx2 /proc/cpuinfo`)
- CMake ≥ 3.18
- GCC ≥ 9 (C11)
- NASM ≥ 2.14
- ~120 MB de RAM libre

---

## Build

```bash
cd ~/Dev/optimus/cpu
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

L'exécutable principal est `build/kmamba_cpu`.

Les outils de diagnostic sont aussi compilés :
- `build/test_scan`  — vérifie le scan1d ASM contre une référence scalaire
- `build/test_grad`  — vérifie le backward pass (gradient check numérique)

---

## Utilisation

### Entraînement

```bash
# Sur le texte intégré (démo, ~1400 bytes)
./build/kmamba_cpu

# Sur un fichier texte
./build/kmamba_cpu train data/conversations.txt

# Avec sauvegarde de checkpoint
./build/kmamba_cpu train data/conversations.txt ckpt.bin

# Reprendre depuis un checkpoint existant
./build/kmamba_cpu train data/conversations.txt ckpt.bin
```

La progression s'affiche en temps réel :

```code
 epoch |   loss   |  ms/epoch
-------+----------+-----------
       step   50/389  loss=5.4821
  ...
     1 |   5.4712 | 9725000.0
```

Un checkpoint est sauvegardé automatiquement tous les 10 epochs (`SAVE_EVERY`).

### Génération libre

```bash
./build/kmamba_cpu gen ckpt.bin "Bonjour"
```

Génère 512 bytes (configurable via `GEN_LEN`) en continuant le prompt.

### Chat interactif (REPL)

```bash
./build/kmamba_cpu chat ckpt.bin
```

```
  k-mamba — session interactive
  Ctrl+D ou 'quit' pour quitter
  ─────────────────────────────

you> Comment tu vas ?
CP> Ça va bien, merci !

you> quit
  [session terminée]
```

Le modèle répond jusqu'au premier `\n` généré (max `CHAT_MAX_RESP` tokens).
Le contexte glisse automatiquement sur les derniers `SEQ_LEN` bytes.

---

## Configuration du modèle

Tous les hyperparamètres sont des `#define` en tête de `main.c` :

| Paramètre      | Valeur   | Description                                      |
|----------------|----------|--------------------------------------------------|
| `VOCAB_SIZE`   | 256      | Vocabulaire byte-level (tous les bytes ASCII)    |
| `DIM`          | 512      | Dimension des embeddings et des états cachés     |
| `STATE_SIZE`   | 1024     | Taille de l'état SSM par couche                  |
| `N_LAYERS`     | 4        | Nombre de MambaBlocks empilés                    |
| `SEQ_LEN`      | 256      | Longueur de séquence (fenêtre de contexte)       |
| `BATCH_SIZE`   | 16       | Séquences traitées simultanément                 |
| `N_EPOCHS`     | 50       | Nombre de passes sur les données                 |
| `LR_BLOCKS`    | 1e-3     | Learning rate des MambaBlocks (AdamW)            |
| `LR_EMBED_HEAD`| 1e-3     | Learning rate de l'embedding et du LM head       |
| `WEIGHT_DECAY` | 1e-5     | Régularisation L2                                |
| `CLIP_NORM`    | 1.0      | Gradient clipping (norme maximale)               |
| `TEMPERATURE`  | 0.8      | Température de sampling (0=greedy, 1=aléatoire)  |
| `dt_scale`     | 1.0      | Facteur d'échelle du pas de temps SSM            |
| `dt_min`       | 0.001    | Borne inférieure de delta (softplus clampé)      |
| `dt_max`       | 1.0      | Borne supérieure de delta                        |
| `SEED`         | 42       | Graine aléatoire (init Xavier + sampling)        |

**Empreinte mémoire à ces valeurs :**
- Paramètres : ~17 MB
- États optimiseur (×6) : ~102 MB
- Total : ~119 MB

---

## Format des données d'entraînement

Fichier texte `.txt` brut. Le modèle opère au niveau byte — il n'y a pas de tokenizer.
Chaque byte du fichier est un token (vocab_size=256).

Les données sont découpées en séquences aléatoires de `SEQ_LEN+1` bytes à chaque step.
La cible est le décalage d'un byte : le modèle apprend à prédire `text[i+1]` depuis `text[0..i]`.

**Format recommandé pour le mode chat :**
```
Human: <question>
Bot: <réponse>
```

Les labels `CHAT_USER` / `CHAT_BOT` dans `main.c` doivent correspondre exactement
au format du corpus d'entraînement pour que le mode chat fonctionne correctement.

---

## Format du checkpoint

Fichier binaire `ckpt.bin`, magic `KMAMBA`, version 1. Contient :
- La configuration du modèle (`KMambaConfig`)
- Tous les tenseurs de paramètres (float32, row-major)
- Les états des optimiseurs (moments Adam)

Le chargement reprend automatiquement l'entraînement là où il s'était arrêté.

---

## Diagnostics

```bash
# Vérifier que le scan1d ASM est correct
./build/test_scan

# Vérifier le backward pass
./build/test_grad
```

`test_scan` compare le kernel ASM (`scan1d.asm`) avec une implémentation scalaire
de référence et affiche l'erreur absolue (doit être 0.000000).

`test_grad` effectue 4 tests :
1. Descente de loss sur un step
2. Cohérence `train_batch` vs `train_step`
3. Gradient check numérique (différences finies) sur W_out
4. Monotonie de la loss sur 10 steps

---

## Structure des fichiers

```
cpu/
├── CMakeLists.txt      — build : lie k-mamba, active AVX2, -no-pie
├── main.c              — runner : train / gen / chat, configs #define
├── test_grad.c         — diagnostic backward pass
├── test_scan.c         — diagnostic kernel scan1d ASM
├── data/
│   ├── conversations.txt   — corpus français (1.6 MB)
│   └── train.txt           — corpus anglais Q&A (320 KB)
└── build/
    ├── kmamba_cpu      — exécutable principal
    ├── test_grad       — diagnostic gradients
    └── test_scan       — diagnostic scan
```

La bibliothèque k-mamba elle-même est dans `../k-mamba/`.
Voir `../k-mamba/CONTEXT.md` pour l'API complète et les règles d'architecture.

---

## Entraînement long (nohup)

```bash
# Lancer en arrière-plan, log dans train.log
nohup ./build/kmamba_cpu train data/conversations.txt ckpt.bin > train.log 2>&1 &

# Suivre la progression
tail -f train.log

# Arrêter proprement (le checkpoint est sauvegardé à chaque SAVE_EVERY epoch)
kill <PID>
```

Temps estimés (CPU x86-64 moyen) :
- `train.txt` (320 KB) : ~32 min/epoch
- `conversations.txt` (1.6 MB) : ~2.7 h/epoch