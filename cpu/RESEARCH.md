# RESEARCH — Quantification des états pour réduire l'empreinte mémoire

**Problème :** vitesse limitée par la mémoire. 119 MB pour le modèle + optimiseur,
dont 96 MB rien que pour les moments Adam. Réduire la mémoire permet d'augmenter
batch_size, state_size, ou n_layers sans changer de machine.

---

## Analyse des coûts mémoire

```
ForwardStore (pendant backward uniquement) :
  x, A_diag, B_bar, u_seq : 4 × seq_len × state_size × float32
  = 4 × 256 × 1024 × 4B = 4 MB/bloc × 4 blocs = ~16 MB (temporaire)

Paramètres :
  W_in, W_out, A_log, B_proj, C_proj, delta_proj × 4 blocs
  + embedding + LM head = ~17 MB (permanent)

Moments Adam (m + v) :
  2 × ~17 MB = ~34 MB × (facteur Adam = 3) = ~96 MB (permanent)  ← DOMINANT
```

---

## Cible 1 — Quantification des ForwardStore states

### Idée

Pendant le forward pass, les 4 tenseurs de ForwardStore sont calculés en float32
puis stockés pour être relus dans le backward. On peut les compresser en int8
avant stockage et décompresser avant usage dans le backward.

### Schéma : int8 symétrique par tenseur

```c
/* Quantification (après calcul, avant stockage) */
float scale = max_abs(x, n) / 127.0f;
for (size_t i = 0; i < n; i++)
    q[i] = (int8_t)clamp((int)roundf(x[i] / scale), -127, 127);

/* Déquantification (avant usage dans backward) */
for (size_t i = 0; i < n; i++)
    x[i] = (float)q[i] * scale;
```

Erreur max par élément : `scale / 2 = max(|x|) / 254`

### Modification de ForwardStore dans mamba_block.c

```c
/* AVANT */
typedef struct {
    float *x;       /* seq_len x state_size */
    float *A_diag;  /* seq_len x state_size */
    float *B_bar;   /* seq_len x state_size */
    float *u_seq;   /* seq_len x state_size */
} ForwardStore;

/* APRÈS */
typedef struct {
    int8_t *x_q;      float x_scale;       /* quantifié int8 */
    int8_t *A_diag_q; float A_diag_scale;
    int8_t *B_bar_q;  float B_bar_scale;
    float  *u_seq;                          /* garde float32 : gradient sensible */
} ForwardStore;
```

`u_seq` reste en float32 car il est utilisé dans le gradient de W_in
(erreur de quantification se propage directement dans dL/dW_in).

### Gain

- ForwardStore : 4 MB/bloc → 1 MB/bloc (×4 réduction)
- Total : ~16 MB → ~4 MB pendant le backward
- Risque sur convergence : nul (bruit borné, similaire au bruit de batch)

### Accélération AVX2 possible

La déquantification int8 → float32 est vectorisable :

```c
/* AVX2 : 32 int8 → 32 float32 en ~4 instructions */
__m256i qi = _mm256_loadu_si256((__m256i *)&q[i]);
__m256  xf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(...));
xf = _mm256_mul_ps(xf, _mm256_set1_ps(scale));
_mm256_storeu_ps(&out[i], xf);
```

---

## Cible 2 — Quantification des moments Adam (8-bit Adam)

### Idée

Les moments `m` et `v` d'AdamW sont des float32 permanents (~96 MB).
On peut les stocker en uint8 avec une échelle dynamique par bloc de 64 valeurs
(blockwise quantization), et dequantifier au moment de la mise à jour des poids.

C'est exactement ce que fait **bitsandbytes 8-bit Adam** (Tim Dettmers, 2022).

### Schéma : blockwise uint8

```
Pour chaque bloc de B=64 valeurs :
  absmax = max(|m[i..i+B]|)
  scale  = absmax / 127.0
  q[j]   = (uint8_t)clamp(round(m[j] / scale + 127), 0, 255)

Déquantification :
  m[j] = ((float)q[j] - 127) * scale
```

Stocker : `uint8_t m_q[n]` + `float m_scales[n/B]`
Surcoût de scale : `n/B × 4B` = négligeable (1/16 des données originales)

### Gain

- Moments Adam : ~96 MB → ~24 MB + ~6 MB de scales = ~30 MB
- **Économie totale : ~66 MB**
- Avec ça : 119 MB → ~53 MB → possible de doubler state_size (1024 → 2048)

### Modification dans mamba_block.c (MBOptimState)

```c
/* AVANT */
typedef struct {
    float *m_W_in, *v_W_in;
    float *m_W_out, *v_W_out;
    /* ... */
} MBOptimState;

/* APRÈS */
#define ADAM_Q_BLOCK 64
typedef struct {
    uint8_t *m_W_in_q;  float *m_W_in_scales;   /* int8 + scales par bloc */
    uint8_t *v_W_in_q;  float *v_W_in_scales;
    uint8_t *m_W_out_q; float *m_W_out_scales;
    /* ... */
} MBOptimState;
```

La mise à jour Adam :
1. Dequantify m, v → float32 temporaire
2. Appliquer la règle Adam normalement
3. Requantify m, v → uint8 + nouvelles scales

### Risque

Précision légèrement réduite pour les paramètres proches de zéro (faibles gradients).
En pratique, la convergence est quasi-identique à float32 selon Dettmers 2022
(testé sur GPT-3 scale). Acceptable pour k-mamba.

---

## Cible 3 — Gradient checkpointing (alternative sans perte de précision)

### Idée

Au lieu de stocker la ForwardStore, on la **recalcule** pendant le backward.
Coût : 2× le nombre de forward passes. Gain : ForwardStore = 0 bytes.

Utile si la précision est critique et qu'on peut se permettre 2× les FLOPs.
Pour k-mamba sur CPU, les FLOPs sont déjà le goulot — **non recommandé**.

---

## Ordre d'implémentation recommandé

1. **Cible 1** (ForwardStore int8) — safe, simple, zero risque convergence
   - Modifier `selective_scan_forward_store` : quantifier x, A_diag, B_bar
   - Modifier `selective_scan_backward` : dequantifier avant usage
   - Tester avec `test_grad` : la loss doit toujours décroître monotoniquement

2. **Cible 2** (Adam 8-bit) — plus impactif mais plus complexe
   - Modifier `MBOptimState` : remplacer m/v par uint8 + scales
   - Modifier `mb_optim_step` : dequant → update → requant
   - Valider sur 10 epochs : courbe de loss identique à float32

3. **Mesure finale**
   - Avant : 119 MB, batch=16, state=1024
   - Après : ~53 MB → batch=32 ou state=2048 avec le même budget mémoire
