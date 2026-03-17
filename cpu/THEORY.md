# THEORY — k-mamba CPU : fondements théoriques et implémentation

Ce document explique, de zéro, le fonctionnement de ce modèle de langage :
la théorie mathématique des State Space Models, l'architecture Mamba,
les choix d'implémentation en C/ASM, et les détails du backward pass.

---

## 1. Le problème : modéliser une séquence de bytes

L'objectif est d'apprendre une distribution de probabilité sur des séquences de bytes.
Étant donné une séquence `x_0, x_1, ..., x_{T-1}`, le modèle apprend à prédire
le prochain byte `x_t` depuis tous les bytes précédents `x_0, ..., x_{t-1}`.

Formellement, on maximise la log-vraisemblance :

```
L = sum_t log P(x_t | x_0, ..., x_{t-1})
```

En pratique, on minimise la cross-entropie négative (équivalent).

Pourquoi byte-level ? Zéro tokenizer, zéro vocabulaire appris, zéro dépendance externe.
Le modèle apprend directement la structure des bytes — lettres, ponctuation, UTF-8, tout.

---

## 2. State Space Models (SSM) — la base

### 2.1 Le système continu

Un SSM linéaire continu en temps est défini par :

```
h'(t) = A h(t) + B u(t)      [état caché]
y(t)  = C h(t) + D u(t)      [sortie]
```

où :
- `u(t)` : entrée scalaire au temps t
- `h(t)` : vecteur d'état de dimension N (la "mémoire" du système)
- `y(t)` : sortie scalaire
- `A ∈ R^{N×N}` : matrice de transition d'état
- `B ∈ R^{N×1}` : matrice d'entrée
- `C ∈ R^{1×N}` : matrice de sortie
- `D ∈ R`        : terme de skip-connection (souvent ignoré)

### 2.2 Discrétisation (Zero-Order Hold)

Pour traiter des séquences discrètes, on discrétise avec un pas `Δ` (delta) :

```
Ā = exp(Δ A)
B̄ = (Δ A)^{-1} (exp(Δ A) - I) Δ B  ≈  Δ B  [quand A est diagonal]
```

La récurrence discrète devient :

```
h_t = Ā h_{t-1} + B̄ u_t
y_t = C h_t
```

C'est le "scan" — le cœur de tout SSM.

### 2.3 Pourquoi diagonal ?

Dans k-mamba, `A` est restreinte à être diagonale : `A = diag(a_1, ..., a_N)`.
Alors `exp(Δ A) = diag(exp(Δ a_1), ..., exp(Δ a_N))`.
Le scan devient N récurrences scalaires indépendantes — parallélisables.

---

## 3. L'innovation Mamba : le scan sélectif

Le problème des SSM classiques : A, B, C sont fixes (ne dépendent pas de l'entrée).
Le modèle ne peut pas choisir quoi retenir en fonction du contenu.

**Mamba rend B, C, et Δ dépendants de l'entrée :**

```
Δ_t = softplus(delta_proj(x_t))    [pas de temps adaptatif]
B_t = B_proj(x_t)                  [entrée sélective]
C_t = C_proj(x_t)                  [lecture sélective]
```

Maintenant le modèle peut décider :
- `Δ` grand → oublier rapidement le passé (fenêtre courte)
- `Δ` petit → intégrer sur une longue durée (mémoire longue)

La récurrence devient :

```
h_t = exp(Δ_t · A) · h_{t-1} + Δ_t · B_t · u_t
y_t = C_t · h_t
```

Ce scan "sélectif" est le **scan1d** implémenté dans `../k-mamba/cpu/scan1d.asm`.

---

## 4. Architecture MambaBlock

Un MambaBlock transforme une séquence `x[0..T-1]` de vecteurs R^D en une sortie
de même forme. Voici le flux complet :

```
Input x ∈ R^{T × D}
    │
    ├──────────────────────────────────┐  [branche résiduelle]
    │                                 │
    ▼                                 │
W_in ∈ R^{D × S}                     │   projection d'entrée
    │                                 │
    ▼                                 │
SiLU(z) · z  [gating]                │   z = W_in @ x_t, u = SiLU(z) * z
    │                                 │
    ▼                                 │
Selective Scan 1D                    │   h_t = exp(Δ·A)·h_{t-1} + Δ·B·u, y=C·h
    │                                 │
    ▼                                 │
W_out ∈ R^{S × D}                    │   projection de sortie
    │                                 │
    ▼                                 │
    + ◄───────────────────────────────┘  output = input + mamba(input)
    │
    ▼
Output ∈ R^{T × D}
```

**Paramètres appris du MambaBlock :**
- `W_in  ∈ R^{D × S}` : projection entrée (D → S)
- `W_out ∈ R^{S × D}` : projection sortie (S → D)
- `A_diag ∈ R^S`      : diagonale de A (initialisée avec un spectre HiPPO-like)
- `B_proj ∈ R^{S × D}`, `C_proj ∈ R^{S × D}` : projections B et C sélectifs
- `delta_proj ∈ R^{D}`  : projection pour calculer Δ

avec D = `DIM` = 512 et S = `STATE_SIZE` = 1024 dans cette instance.

### 4.1 La connexion résiduelle — pourquoi c'est critique

Sans résidu : `output = mamba(input)`

Le scan produit des valeurs ~1000× plus petites que l'embedding (car les poids
sont initialisés proches de zéro). Le gradient qui remonte à travers W_out est
donc ~1000× plus petit que le signal d'entrée. Après 4 couches, le signal
backward est virtuellement nul.

Avec résidu : `output = input + mamba(input)`

Le gradient peut maintenant remonter directement via la connexion identité,
indépendamment de la magnitude du scan. C'est un principe fondamental des
réseaux profonds (ResNet, 2015).

---

## 5. Le modèle de langage complet (KMamba)

```
Bytes bruts x_0, ..., x_{T-1}  ∈ {0..255}
    │
    ▼
Embedding E ∈ R^{256 × D}      lookup : x_t → E[x_t]  ∈ R^D
    │
    ▼
MambaBlock 1                   (D → D, avec résidu)
    │
    ▼
MambaBlock 2
    │
    ...
    │
    ▼
MambaBlock N_LAYERS
    │
    ▼
LM Head W_head ∈ R^{D × 256}   logits_t = W_head @ h_t  ∈ R^{256}
    │
    ▼
Softmax + Cross-Entropy
```

**Taille totale : ~4.5M paramètres (float32 = ~17 MB)**

---

## 6. Forward pass — détail par étape

### 6.1 Embedding lookup

```c
// Pour chaque token x_t dans la séquence :
float *emb = &model->embedding[x_t * dim];
// emb ∈ R^D
```

### 6.2 SiLU gate dans W_in

```
z = W_in @ x_t          ∈ R^S
u = z * σ(z)             [SiLU = x * sigmoid(x)]
```

SiLU est une activation douce, différentiable partout, qui agit comme un gate
contrôlé par la magnitude du signal lui-même.

### 6.3 Calcul de delta

```
raw = delta_proj · x_t   ∈ R
Δ_t = clamp(softplus(raw), dt_min, dt_max)
```

`softplus(x) = log(1 + exp(x))` est toujours positif et lisse.
**Point important :** `softplus(x) ≥ ln(2) ≈ 0.693`. Si `dt_max < 0.693`,
Δ est toujours saturé à dt_max et ne peut pas varier — delta_proj ne reçoit
aucun gradient utile. C'est pourquoi `dt_max = 1.0` dans cette instance.

### 6.4 Le scan sélectif (cœur du modèle)

Pour chaque pas de temps t, pour chaque dimension d ∈ [0, S) :

```
ā_d = exp(Δ_t * A_diag[d])        [decay factor]
b̄_d = Δ_t * B_t[d]               [input scaling]

h_t[d] = ā_d * h_{t-1}[d] + b̄_d * u_t[d]
y_t[d] = C_t[d] * h_t[d]
```

Ce scan est implémenté en AVX2 dans `scan1d.asm`, traitant 8 dimensions
en parallèle via les registres YMM (256 bits = 8 × float32).

### 6.5 Projection de sortie et résidu

```
ybuf = W_out @ h_t              ∈ R^D
output_t = input_t + ybuf        [résidu]
```

### 6.6 LM Head et cross-entropie

```
logits_t = W_head @ output_T     ∈ R^{256}
probs_t  = softmax(logits_t)
loss     = -log(probs_t[x_{t+1}])   [cross-entropie sur le token suivant]
```

---

## 7. Backward pass — propagation des gradients

La backpropagation calcule le gradient de la loss par rapport à chaque paramètre,
en appliquant la règle de dérivation en chaîne à rebours.

### 7.1 LM Head

```
dL/d(logits) = probs - one_hot(x_{t+1})    [gradient de softmax+CE]
dL/d(W_head) = dL/d(logits) @ output^T
dL/d(output) = W_head^T @ dL/d(logits)
```

### 7.2 Résidu

```
dL/d(input_mamba) = dL/d(output)       [gradient identique : output = input + mamba(input)]
dL/d(input_residual) = dL/d(output)    [la connexion identité passe le gradient intact]
```

### 7.3 W_out

```
dL/d(W_out) = dL/d(ybuf) @ h_T^T
dL/d(h_T)   = W_out^T @ dL/d(ybuf)
```

### 7.4 Scan backward (le plus complexe)

On veut `dL/dA`, `dL/dB`, `dL/dC`, `dL/dΔ`, `dL/du`.

La récurrence forward est :
```
h_t = ā_t * h_{t-1} + b̄_t * u_t
y_t = C_t * h_t
```

Le backward se fait à rebours (t = T-1 → 0) :

```
dL/dh_t   = C_t^T * dL/dy_t  +  ā_{t+1} * dL/dh_{t+1}    [adjoint de h]
dL/dC_t   = dL/dy_t * h_t^T
dL/db̄_t  = dL/dh_t * u_t^T
dL/du_t   = b̄_t^T * dL/dh_t
dL/dā_t   = dL/dh_t * h_{t-1}^T
```

Puis remontée vers les paramètres primaires :
```
dL/dΔ_t   = sum_d (dL/dā * dā/dΔ  +  dL/db̄ * db̄/dΔ)
           = sum_d (dL/dā * A[d]*ā  +  dL/db̄ * B[d])
dL/d(raw) = dL/dΔ * dΔ/d(raw)   [straight-through si clampé]
```

**Le straight-through estimator** : quand Δ est clampé, `dΔ/d(raw)` serait
normalement 0 (la fonction clamp est plate aux bornes). On utilise
l'estimateur straight-through : on propage le gradient comme si le clamp
n'existait pas. Sans ça, `delta_proj` ne reçoit jamais de gradient.

### 7.5 SiLU backward

```
dL/dz = dL/du * (σ(z) + z * σ(z) * (1 - σ(z)))
      = dL/du * σ(z) * (1 + z * (1 - σ(z)))
```

### 7.6 W_in backward

```
dL/d(W_in) = dL/dz @ x^T
dL/dx      = W_in^T @ dL/dz    [gradient vers l'embedding]
```

---

## 8. Optimiseur — AdamW

Pour chaque paramètre θ, AdamW maintient deux moments exponentiels :

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t         [moment 1, momentum]
v_t = β2 * v_{t-1} + (1 - β2) * g_t²        [moment 2, variance]

m̂_t = m_t / (1 - β1^t)                       [correction du biais]
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε) - lr * wd * θ_{t-1}
```

Hyperparamètres utilisés :
- `β1 = 0.9`  (MOMENTUM) — lissage du gradient
- `β2 = 0.999` (BETA2) — lissage de la variance
- `ε = 1e-8`  (EPS) — évite la division par zéro
- `lr = 1e-3` pour les MambaBlocks
- `wd = 1e-5` (WEIGHT_DECAY) — régularisation L2

Le gradient clipping (CLIP_NORM = 1.0) est appliqué avant la mise à jour :
si `||g|| > clip_norm`, on rescale `g ← g * clip_norm / ||g||`.

---

## 9. Génération — température sampling

Après entraînement, on génère token par token :

1. Placer un prompt dans le contexte `ctx[0..SEQ_LEN-1]`
2. Forward pass → `logits ∈ R^{256}` pour la dernière position
3. Appliquer la température : `logits_scaled = logits / T`
4. Softmax → distribution de probabilité
5. Tirer un sample → `next_token`
6. Décaler le contexte : `ctx[i] = ctx[i+1]`, `ctx[SEQ_LEN-1] = next_token`
7. Répéter

**Température T :**
- `T → 0` : greedy (toujours le token le plus probable — répétitif)
- `T = 0.8` : légèrement créatif (par défaut)
- `T = 1.0` : distribution brute (le plus aléatoire)
- `T > 1.0` : plus d'entropie que le modèle (incohérent)

---

## 10. Implémentation C/ASM — choix techniques

### 10.1 Architecture dualiste Volontés/Puissance

Le code est séparé en deux parties :

- **k-mamba** (`../k-mamba/src/`) — la logique du modèle (embedding, MambaBlock,
  training loop, checkpoint). C'est la "volonté" : ce que le modèle fait.

- **optimatrix** (`../k-mamba/optimatrix/`) — les kernels de calcul génériques
  (GEMM AVX2, activations, optimiseurs). C'est la "puissance" : comment les
  calculs sont exécutés.

Les scans SSM (scan1d, scan2d) restent dans k-mamba car ils sont spécifiques
à Mamba et ne sont pas des kernels génériques.

### 10.2 scan1d.asm — AVX2 NASM

Le kernel de scan 1D est écrit en assembleur NASM pur pour maximiser les
performances sur CPU AVX2.

```
; Pour chaque pas de temps t :
;   ymm0 = ā[0..7]     (8 decay factors en float32)
;   ymm1 = h[0..7]     (état courant)
;   ymm2 = b̄[0..7] * u[0..7]
;
;   vmulps ymm1, ymm0, ymm1    ; h = ā * h
;   vaddps ymm1, ymm1, ymm2    ; h = h + b̄*u
;   vmovaps [out], ymm1
```

AVX2 traite 8 float32 simultanément → 8× le débit d'une boucle scalaire.

**Contrainte `-no-pie`** : le code ASM utilise des relocations 32-bit pour
adresser les données en section `.data`. Position-Independent Executable
(PIE) requiert des relocations 64-bit. On désactive PIE avec `-no-pie`.

### 10.3 gemm_avx2 — convention accumulation

`gemm_avx2(A, B, C, M, K, N)` calcule `C += A @ B`.
**Attention : C n'est PAS mis à zéro automatiquement.**
L'appelant doit zéroïser C avant d'appeler gemm_avx2 si il veut C = A @ B.

### 10.4 Checkpoint — format binaire

Le checkpoint est un fichier binaire brut :

```
[magic: "KMAMBA" 6 bytes]
[version: uint32]
[KMambaConfig struct]
[tenseurs paramètres : float32, row-major, séquentiels]
[tenseurs moments optimiseur : float32, même ordre]
```

L'entraînement peut être repris exactement depuis un checkpoint car les
moments Adam (m_t, v_t) sont sauvegardés — le scheduler de learning rate
peut continuer correctement.

---

## 11. Complexité computationnelle

| Opération          | Complexité            | Dominante ?  |
|--------------------|-----------------------|--------------|
| Embedding lookup   | O(T · D)              | non          |
| W_in projection    | O(T · D · S)          | oui          |
| Selective scan     | O(T · S)              | non          |
| W_out projection   | O(T · S · D)          | oui          |
| LM Head            | O(T · D · V)          | oui          |

avec T=256, D=512, S=1024, V=256 : la complexité par séquence est O(T · D · S).

**Avantage vs Transformer :** l'attention self-attention est O(T² · D).
Pour T=256, Mamba est ~256× moins coûteux en calcul d'attention.
La complexité linéaire en T est l'avantage principal des SSM pour les longues séquences.

---

## 12. Bugs corrigés — ce qui était cassé et pourquoi

### Bug 1 : delta_proj sans gradient

**Code original :**
```c
float sp = scalar_softplus(raw_t);
if (sp > block->config.dt_min && sp < block->config.dt_max) {
    // ... gradient de delta_proj
}
```

**Problème :** `softplus(x) ≥ ln(2) ≈ 0.693 > dt_max = 0.1` toujours.
La condition était toujours fausse. `delta_proj` n'a jamais reçu de gradient.
Le modèle ne pouvait pas apprendre à moduler le pas de temps Δ.

**Fix :** Straight-through estimator — on supprime la condition.
On propage le gradient même si Δ est clampé.

### Bug 2 : connexion résiduelle manquante

**Code original :**
```c
batch_output[t * dim + j] = ybuf[j];   // output = mamba(input)
```

**Problème :** Le scan produit des valeurs ~1000× plus petites que l'embedding
(poids proches de zéro à l'initialisation Xavier). Sans résidu, le signal
backward à travers le scan est négligeable — le modèle n'apprend pas.

**Fix :**
```c
batch_output[t * dim + j] = batch_input[t * dim + j] + ybuf[j];  // output = input + mamba
```
+ dans le backward : `d_input += dY` (gradient de la connexion identité).

### Bug 3 : dt_max trop petit

**Valeur originale :** `dt_max = 0.1`

**Problème :** `softplus(x) ≥ ln(2) ≈ 0.693 > 0.1`. Δ était toujours clampé
à 0.1 — une constante. Le modèle ne pouvait pas varier la durée d'intégration
en fonction du contexte. La sélectivité de Mamba était désactivée.

**Fix :** `dt_max = 1.0` (supérieur à ln(2)).

**Résultat des 3 fixes :** loss 5.608 → 5.594 en 6 epochs, visible et monotone,
contre stagnation complète à ln(256) = 5.545 avant les corrections.

---

## 13. Limites de cette instance

- **Fenêtre de contexte fixe à 256 bytes.** Le modèle ne "voit" que les 256
  derniers bytes. Les conversations longues perdent le contexte ancien.

- **Byte-level = lent à l'entraînement.** Un mot de 5 lettres = 5 tokens.
  Les vrais LLM utilisent des tokenizers BPE (1 token ≈ 4 bytes en moyenne).

- **Pas de positional encoding.** Mamba encode la position implicitement via
  la récurrence — mais ce n'est pas aussi expressif qu'un encoding explicite
  pour les séquences très courtes.

- **Pas de normalisation de couche (LayerNorm/RMSNorm).** L'ajout de RMSNorm
  entre les couches stabiliserait l'entraînement sur de longs runs.

- **CPU single-threaded.** L'entraînement utilise un seul cœur. Le batch
  processing n'est pas parallélisé entre les séquences du batch.
