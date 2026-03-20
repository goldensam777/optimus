# rsults.md

Gabarit de resultats pour consolider les runs en materiau directement exploitable dans le paper.

## Mode d'emploi

- Remplir une section par experience importante.
- Toujours separer faits mesures, interpretation et limites.
- Ne jamais conclure au-dela de ce que montrent les metriques test.
- Reporter les runs rates ou negatifs s'ils informent une decision.

## Resume executif

| Semaine | Experience cle | Resultat principal | Niveau de confiance | Decision |
|---|---|---|---|---|
| 1 | a remplir | a remplir | faible / moyen / fort | go / no-go |
| 2 | a remplir | a remplir | faible / moyen / fort | go / no-go |
| 3 | a remplir | a remplir | faible / moyen / fort | go / no-go |
| 4 | a remplir | a remplir | faible / moyen / fort | go / no-go |

## Fiche resultat standard

### Identite de l'experience

- ID :
- Date :
- Auteur / machine :
- Commit / etat du code :
- Statut : `draft` / `valide` / `a refaire`

### Objectif

- Question scientifique :
- Hypothese :
- Critere de succes annonce avant run :

### Configuration

- Modele / operateur :
- Taille du modele :
- Donnees :
- Split :
- Batch size :
- Nombre d'epochs :
- Seed :
- Materiel :
- Commande exacte :

### Metriques brutes

| Metrique | Train | Val | Test | Unite | Commentaire |
|---|---|---|---|---|---|
| Loss | a remplir | a remplir | a remplir | nats / CE | |
| Perplexity | a remplir | a remplir | a remplir | ppl | si applicable |
| Accuracy | a remplir | a remplir | a remplir | % | si applicable |
| PSNR | - | a remplir | a remplir | dB | si applicable |
| SSIM | - | a remplir | a remplir | score | si applicable |
| Throughput | a remplir | - | - | samples/s ou tokens/s | |
| Temps par epoch | a remplir | - | - | s | |
| RAM max | a remplir | - | - | MB ou kB | |
| VRAM max | a remplir | - | - | MB | si applicable |

### Comparaison baseline / controle

| Comparaison | Metrique cle | Notre modele | Baseline / controle | Delta | Verdict |
|---|---|---|---|---|---|
| baseline simple | a remplir | a remplir | a remplir | a remplir | meilleur / pire / egal |
| controle negatif | a remplir | a remplir | a remplir | a remplir | signal present / absent |
| ablation | a remplir | a remplir | a remplir | a remplir | utile / non concluant |

### Stabilite

| Seed | Metrique val cle | Metrique test cle | Notes |
|---|---|---|---|
| seed 1 | a remplir | a remplir | |
| seed 2 | a remplir | a remplir | |
| seed 3 | a remplir | a remplir | |

Resume stabilite :

- moyenne :
- ecart-type :
- conclusion :

### Evidence qualitative

- Figures produites :
- Echantillons ou transcripts :
- Exemple le plus convaincant :
- Contre-exemple ou echec notable :

### Interpretation

- Faits etablis par les mesures :
- Ce que cela suggere sans le prouver :
- Ce que cela ne permet pas de conclure :

### Limitations

- Limite 1 :
- Limite 2 :
- Limite 3 :

### Decision

- Verdict : `go` / `no-go` / `rerun`
- Action suivante :

## Resultats a suivre en priorite

### Semaine 1 - Texte / REPL

- Baseline retenue :
- Meilleur checkpoint :
- `test_ppl` :
- Controle negatif :
- Qualite REPL :
- Decision :

### Semaine 2 - 2D images

- Dataset :
- Metrique principale :
- Resultat `2D` natif :
- Resultat rasterise `1D` :
- Ablation :
- Decision :

### Semaine 3 - 3D videos

- Etat `scan3D_ref` :
- Test de correction :
- Dataset :
- Metrique principale :
- Controle ordre temporel :
- Decision :

### Semaine 4 - GPU exploratoire

- Stabilite CUDA :
- VRAM max :
- Gain de `scan1d++ + convND` :
- Resultat exploratoire principal :
- Niveau de prudence du claim :
- Decision :

## Regles de redaction pour le paper

- Utiliser "we observe" quand il s'agit d'une observation empirique.
- Utiliser "preliminary evidence suggests" pour les signaux fragiles.
- Utiliser "under our experimental conditions" quand le materiel ou le dataset sont limitants.
- Eviter "we prove" sauf preuve formelle reelle.
- Ne jamais presenter une figure qualitative seule comme preuve centrale.
