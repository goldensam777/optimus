# ROADMAP_PLAN.md

Plan de travail research-grade pour `optimus` / `k-mamba`, avec criteres minimaux de succes qui restent credibles sous les conditions actuelles.

---

## 1. But du programme

Construire, verifier et demontrer progressivement une famille d'operateurs Mamba-3 natifs :

- `1D` pour les sequences texte
- `2D` pour les images
- `3D` pour les videos
- puis une version GPU plus dense pour des experiences exploratoires multi-echelles

Le fil directeur scientifique est :

- montrer que les scans natifs ND sont calculatoirement corrects
- montrer qu'ils apprennent vraiment
- montrer qu'ils sont utiles par rapport a des alternatives plus triviales

---

## 2. Conditions actuelles

### Materiel et execution

- Machine locale unique
- CPU 8 coeurs, execution OpenMP fixee a `7` threads pour l'instance CPU
- GPU NVIDIA MX450 `sm_75`, environ `2 GB` VRAM utiles
- Pas de multi-GPU
- Pas de cluster

### Etat actuel du code

- Instance CPU actuelle autour de `~497K` parametres
- Logging CSV disponible sur l'instance CPU :
  - `step.csv`
  - `epoch.csv`
- Split train/validation deja supporte dans le runner CPU
- Generateur `wavefront_nd` de reference en C ajoute et teste
- `scan3D` comme operateur n'est pas encore branche dessus

### Contraintes de recherche

- Temps d'entrainement limite sur CPU
- VRAM tres contrainte sur GPU
- Les claims doivent etre proportionnes au niveau de preuve
- Toute affirmation "theorique forte" devra etre separee des experiences exploratoires

---

## 3. Barre minimale d'acceptation scientifique

Ce qu'un chercheur externe serieux acceptera au minimum pour juger une etape credible :

1. **Correction**

- Tests unitaires deterministes
- Verification numerique contre une reference simple quand c'est possible
- Pas seulement des demos qualitatives

2. **Generalisation**

- Split `train / val / test` fige
- Rapport sur `val` et `test`, pas uniquement sur `train`

3. **Baselines ou controles**

- Au moins un baseline simple
- Au moins un controle negatif ou une ablation
- Les exemples :
  - version rasterisee 1D
  - ordre des donnees melange
  - version sans `convND`
  - version sans wavefront natif

4. **Reproductibilite**

- Configs enregistrees
- Commandes reproductibles
- Checkpoints et logs conserves
- Au minimum `3 seeds` pour les experiences clefs, sauf si le cout est prohibitif

5. **Mesures**

- Une metrique d'apprentissage
- Une metrique de generalisation
- Une metrique systeme

Exemples :

- apprentissage : `train_loss`
- generalisation : `val_loss`, `test_loss`, `perplexity`, `accuracy`, `PSNR`, `SSIM`
- systeme : `tokens/s`, `ms/epoch`, `RSS`, VRAM, throughput

6. **Honnetete des claims**

- Les demos qualitatives servent d'illustration, pas de preuve
- Une experience speculative ne doit jamais etre vendue comme resultat etabli

---

## 4. Regles Go / No-Go

On ne passe a la semaine suivante que si la precedente passe sa barre minimale.

### Go

- les tests de correction passent
- la metrique de validation bouge dans le bon sens
- au moins un baseline ou controle est battu
- les resultats sont reproductibles

### No-Go

- seulement des exemples jolis sans mesure
- seulement `train_loss` qui baisse
- pas de baseline
- resultats instables selon la seed
- temps d'execution incompatible avec la suite

---

## 5. Roadmap hebdomadaire

## Semaine 1 - 1-Mamba sequence / chatbot minimal

### Objectif

Verifier qu'un `1-Mamba` texte capte bien la structure sequentielle et permet un REPL minimal base sur checkpoint.

### Scope realiste

- Corpus texte brut byte-level
- Entrainer l'instance CPU actuelle
- Produire un chatbot minimal via `chat ckpt.bin`
- Evaluer sur donnees hors train

### Succes minimum acceptable

- `train_loss` et `val_loss` diminuent de facon stable
- `test_perplexity` est meilleure qu'un controle trivial
- le modele repond en REPL sans casser grossierement le format conversationnel
- au moins `3 seeds` sur une config reduite ou `1 seed complete + 2 seeds courtes` si le cout est trop eleve

### Baselines / controles minimaux

- baseline tres simple :
  - modele de frequence de bytes
  - ou `n-gram` caractere si disponible
- controle negatif :
  - corpus melange au niveau des lignes ou des blocs
  - ou targets decalees aleatoirement

### Metriques minimales

- `train_loss`
- `val_loss`
- `test_loss`
- `perplexity`
- `tokens/s`
- `max_rss_kb`

### Artefacts attendus

- `checkpoint`
- logs CSV
- 5 a 10 transcripts REPL courts sur prompts fixes
- tableau comparatif contre baseline / controle

### Critere Go

- le modele bat clairement le controle trivial sur `test_perplexity`
- les sorties REPL montrent une retention sequentielle minimale

---

## Semaine 2 - 1-Mamba 2D pour images

### Objectif

Verifier que le passage au `2D` natif apporte quelque chose sur une tache image simple et defendable.

### Scope realiste

Ne pas commencer par du VQA ou du "quiz avec images" comme claim principal.
Commencer par une tache propre et mesurable :

- `MNIST` en priorite
- puis `Fashion-MNIST` si le temps le permet

### Succes minimum acceptable

- un modele 2D natif s'entraine sans bug numerique
- il atteint une performance test mesurable sur images
- il est compare a une version `1D rasterisee` a budget proche

### Baselines / controles minimaux

- baseline 1 : meme architecture mais image rasterisee en sequence 1D
- baseline 2 : ordre des pixels perturbe ou scramble fixe
- ablation : wavefront natif desactive / remplace par rasterization

### Metriques minimales

- `train_loss`
- `val_loss`
- `test_accuracy` ou `test_nll`
- `images/s`
- `RAM` / `VRAM`

### Barre d'acceptation mondiale minimale

Un resultat `2D` sera jugable si :

- le protocole est propre
- le split est public
- la comparaison avec rasterization est equitable
- le 2D natif n'est pas seulement "joli", mais utile ou au moins comparable avec un argument systeme clair

### Critere Go

- soit le 2D natif bat la rasterization en qualite
- soit il est comparable en qualite mais meilleur en structure ou en efficience interpretable

---

## Semaine 3 - 1-Mamba 3D pour videos

### Objectif

Verifier qu'un operateur `3D` branche sur le generateur de wavefront ND apprend une dynamique spatio-temporelle non triviale.

### Scope realiste

Commencer par un dataset video simple et raisonnable :

- `Moving MNIST` en priorite
- ensuite seulement un dataset reel plus dur

### Pourquoi

Sur les conditions actuelles, `Moving MNIST` permet :

- verification rapide
- evaluation propre
- comparaison native 3D vs solutions plus simples

### Succes minimum acceptable

- `scan3D_ref` existe et passe des tests de correction
- l'entrainement 3D converge sur une tache video
- le modele generalise au moins sur un split test simple

### Baselines / controles minimaux

- baseline 1 : traitement frame-par-frame en 2D + aggregation temporelle simple
- baseline 2 : flatten `3D -> 1D`
- controle negatif : permutation de l'ordre temporel

### Metriques minimales

Pour prediction video :

- `test_loss`
- `PSNR`
- `SSIM` si possible

Pour classification video :

- `accuracy`
- `macro-F1`

Toujours ajouter :

- `samples/s`
- `RAM/VRAM`

### Barre d'acceptation mondiale minimale

Les chercheurs accepteront un premier resultat 3D si :

- la correction de l'operateur est montree
- la tache est claire
- les controles montrent que l'ordre spatio-temporel compte vraiment

### Critere Go

- le 3D natif bat le controle temporel detruit
- et montre au moins une utilite claire face a une alternative plus triviale

---

## Semaine 4 - Modele dense GPU avec scan1d++ et convND

### Objectif

Passer a une version GPU plus dense pour explorer les capacites multi-echelles de `scan1d++ + convND`.

### Reformulation scientifique necessaire

Le but acceptable scientifiquement n'est **pas** :

- "demontrer la theorie des cordes"

Le but acceptable est :

- tester si un modele dense GPU apprend des regularites multi-echelles complexes
- tester s'il capte des structures sur des donnees physiques ou pseudo-physiques

### Scope realiste

Avant toute speculation physique forte, faire d'abord :

- un benchmark de stabilite
- un benchmark de throughput
- une tache simple sur donnees synthetiques physiques

Exemples :

- champs d'ondes synthetiques
- petits automates cellulaires
- donnees PDE simples

### Succes minimum acceptable

- le modele GPU s'entraine de facon stable sur MX450
- `scan1d++` et `convND` montrent un apport mesurable
- les gains ne sont pas seulement narratifs

### Baselines / controles minimaux

- sans `convND`
- sans scan enrichi
- version CPU reduite
- version rasterisee si applicable

### Metriques minimales

- `train_loss`, `val_loss`, `test_loss`
- throughput
- VRAM max
- temps par epoch
- ablation quality/perf

### Barre d'acceptation mondiale minimale

Un claim exploratoire peut etre garde si :

- il reste prudent
- il est borne par les donnees
- il ne pretend pas conclure sur une theorie physique fondamentale

### Critere Go

- la stack GPU est stable
- les ablations montrent quelque chose de mesurable
- aucun claim excessif n'est fait

---

## 6. Livrables minimaux par semaine

Chaque semaine doit produire :

1. un protocole ecrit
2. une config figee
3. des logs
4. des figures
5. un tableau de resultats
6. au moins un paragraphe "limitations"

---

## 7. Figures minimales a viser

### Semaine 1

- courbe `train_loss / val_loss`
- table `test_perplexity`
- exemple REPL qualitatif

### Semaine 2

- accuracy ou NLL vs epochs
- comparaison `2D natif vs 1D rasterise`
- eventuellement confusion matrix

### Semaine 3

- `PSNR/SSIM` ou accuracy sur video
- comparaison `3D natif vs controle`
- visualisation de quelques predictions

### Semaine 4

- throughput vs qualite
- memoire vs taille modele
- ablation `scan1d++ / convND`

---

## 8. Risques principaux

### Risque 1 - Demo sans preuve

Parade :

- toujours accompagner les demos de mesures test

### Risque 2 - Datasets trop ambitieux trop tot

Parade :

- `Moving MNIST` avant toute video reelle
- `MNIST` avant tout VQA ou image quiz

### Risque 3 - VRAM insuffisante

Parade :

- petites resolutions
- batchs reduits
- checkpoints frequents
- benchmarks de memoire explicites

### Risque 4 - Claims trop forts

Parade :

- separer "exploratoire" et "etabli"
- rester sobre dans le papier

---

## 9. Politique de claims pour le futur paper

Acceptable :

- "we introduce"
- "we validate on"
- "we observe"
- "under our experimental conditions"
- "preliminary evidence suggests"

Inacceptable sans preuve exceptionnelle :

- "we prove"
- "we demonstrate string theory"
- "we solve"
- "universal"
- "state of the art"

---

## 10. Commande CPU actuelle de reference

Depuis `cpu/` :

```bash
./build/kmamba_cpu train ../cuda/data/conversations.txt ckpt_cpu_500k.bin paper_cpu
```

REPL :

```bash
./build/kmamba_cpu chat ckpt_cpu_500k.bin
```

---

## 11. Resume strategique

Le plan est bon si on garde cette discipline :

- S1 : prouver qu'on sait modeliser la sequence
- S2 : prouver que le `2D` natif vaut la peine
- S3 : prouver que le `3D` natif apprend une dynamique
- S4 : explorer le dense GPU sans faire de claim hors de portee

Le point cle a ne jamais perdre :

**chaque semaine doit battre au moins un controle simple sur une metrique test, pas seulement produire une demo impressionnante.**

---

## 12. Checklist d'execution

Cette checklist sert de garde-fou pratique. Si une case critique n'est pas cochee, l'etape n'est pas "paper-ready".

### Checklist globale

- [ ] Tous les runs importants ont une commande reproducible ecrite
- [ ] Tous les runs importants ont un `seed` explicite
- [ ] Les splits `train / val / test` sont figes avant comparaison
- [ ] Les logs bruts sont conserves
- [ ] Les checkpoints associes aux meilleurs runs sont conserves
- [ ] Au moins un baseline simple est execute
- [ ] Au moins un controle negatif ou une ablation est execute
- [ ] Les figures et tableaux peuvent etre regeneres depuis les logs
- [ ] Un paragraphe "limitations" est redige pour l'etape
- [ ] Les claims restent strictement proportionnes au niveau de preuve

### Semaine 1 - Checklist execution texte / REPL

- [ ] Verifier que l'instance CPU compile proprement
- [ ] Lancer au moins `1` run complet de reference
- [ ] Lancer `2` runs supplementaires plus courts avec des `seeds` differentes
- [ ] Produire `step.csv` et `epoch.csv` pour chaque run
- [ ] Garder le meilleur checkpoint selon `val_loss`
- [ ] Evaluer sur un split test fige
- [ ] Mesurer `train_loss`, `val_loss`, `test_loss`, `perplexity`, `tokens/s`, `max_rss_kb`
- [ ] Executer un baseline trivial sur le meme test set
- [ ] Executer un controle negatif avec corpus melange ou targets perturbees
- [ ] Produire `5` a `10` transcripts REPL sur prompts fixes
- [ ] Verifier qualitativement que le REPL ne casse pas grossierement le format des dialogues
- [ ] Rediger une conclusion sobre: "le modele capture / ne capture pas encore la structure sequentielle"

Commande de reference :

```bash
./build/kmamba_cpu train ../cuda/data/conversations.txt ckpt_cpu_500k.bin paper_cpu
```

### Semaine 2 - Checklist execution 2D images

- [ ] Choisir un dataset simple et public en priorite `MNIST`
- [ ] Figer un protocole de split et de preprocessing
- [ ] Verifier la correction numerique de l'operateur 2D contre une reference simple
- [ ] Lancer le modele `2D` natif
- [ ] Lancer la baseline rasterisee `1D` a budget voisin
- [ ] Lancer au moins une ablation ou un controle de destruction de structure spatiale
- [ ] Mesurer `train_loss`, `val_loss`, `test_accuracy` ou `test_nll`, `images/s`, `RAM/VRAM`
- [ ] Repeter sur `3 seeds` si le cout reste supportable, sinon `1 seed complete + 2 seeds courtes`
- [ ] Produire au moins un tableau de comparaison equitable
- [ ] Verifier que toute superiorite annoncee ne vient pas d'un budget cache plus grand
- [ ] Rediger la conclusion en distinguant clairement qualite et efficience

### Semaine 3 - Checklist execution 3D videos

- [ ] Implementer `scan3D_ref` branche sur `wavefront_nd`
- [ ] Ajouter des tests unitaires deterministes pour `scan3D_ref`
- [ ] Verifier la sortie contre une reference naive sur petits tenseurs
- [ ] Commencer avec `Moving MNIST`
- [ ] Figer une tache claire: prediction video ou classification video
- [ ] Lancer le modele `3D` natif
- [ ] Lancer une baseline frame-par-frame `2D`
- [ ] Lancer un controle avec ordre temporel detruit
- [ ] Mesurer `test_loss` et au moins une metrique de qualite video comme `PSNR`
- [ ] Ajouter `SSIM` si le cout d'implementation reste raisonnable
- [ ] Mesurer `samples/s` et `RAM/VRAM`
- [ ] Produire quelques visualisations de predictions ou d'erreurs
- [ ] Rediger ce que le `3D` apporte reellement par rapport a `2D + agregation`

### Semaine 4 - Checklist execution dense GPU exploratoire

- [ ] Figer un perimetre exploratoire prudent avant toute execution
- [ ] Verifier que la stack CUDA tient de facon stable sur MX450
- [ ] Mesurer l'occupation VRAM et le throughput des kernels critiques
- [ ] Lancer un benchmark de stabilite avant les experiences "physiques"
- [ ] Executer une version avec `scan1d++ + convND`
- [ ] Executer les ablations sans `convND` et sans scan enrichi
- [ ] Commencer par des donnees synthetiques ou pseudo-physiques simples
- [ ] Mesurer `train_loss`, `val_loss`, `test_loss`, throughput, VRAM max, temps par epoch
- [ ] Verifier que les gains annonces sont mesurables et repetables
- [ ] Rediger une conclusion explicitement exploratoire, sans claim physique excessif

### Gate final avant redaction du paper

- [ ] Chaque figure principale a une source de donnees claire
- [ ] Chaque tableau principal peut etre retrace a un run et un checkpoint
- [ ] Chaque claim central a au moins une mesure test associee
- [ ] Les limites materiel, taille de modele et taille de datasets sont dites explicitement
- [ ] Les sections "negative results" ou "limitations" ne sont pas cachees
