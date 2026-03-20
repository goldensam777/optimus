# exps.md

Tableau de pilotage des experiences. Ce fichier sert a planifier, lancer et verifier des runs comparables sans perdre le fil scientifique.

## Regles communes

- Utiliser un `seed` explicite pour chaque run.
- Garder le meme split pour toute comparaison d'une meme semaine.
- Noter toute deviation de budget modele, resolution, batch size ou temps d'entrainement.
- Conserver au minimum : commande, config, logs, checkpoint, meilleure metrique `val`.
- Ne pas marquer une experience "complete" si baseline ou controle manquent.

## Tableau d'experiences

| ID | Semaine | Experience | Hypothese testee | Donnees | Comparaison minimale | Seeds | Metriques principales | Barre minimale de succes | Statut |
|---|---|---|---|---|---|---|---|---|---|
| W1-E1 | 1 | `1-Mamba` texte CPU reference | Le modele capte une structure sequentielle non triviale | `cuda/data/conversations.txt` avec split fige | baseline frequence bytes ou `n-gram` | `1` complet + `2` courts | `train_loss`, `val_loss`, `test_loss`, `ppl`, `tokens/s`, `max_rss_kb` | battre le controle trivial sur `test_ppl` | a faire |
| W1-E2 | 1 | Controle negatif texte melange | La structure sequentielle compte vraiment | meme corpus, ordre melange | comparer a W1-E1 | `1` | `test_loss`, `ppl` | degradation nette vs W1-E1 | a faire |
| W1-E3 | 1 | REPL qualitatif sur checkpoint | Le modele garde un minimum de coherence conversationnelle | prompts fixes | comparer sur memes prompts entre checkpoints | meilleur checkpoint | transcripts courts | `5` a `10` sorties lisibles sans rupture grossiere | a faire |
| W2-E1 | 2 | `2D` natif sur `MNIST` | Le `2D` natif apprend une structure spatiale utile | `MNIST` | baseline rasterisee `1D` | `3` si possible | `test_accuracy` ou `test_nll`, `images/s`, memoire | perf test >= baseline ou perf voisine avec gain systeme clair | a faire |
| W2-E2 | 2 | Controle spatial perturbe | L'ordre spatial natif apporte un signal | `MNIST` scramble fixe | comparer a W2-E1 | `1-3` | `test_accuracy`, `test_nll` | degradation nette vs W2-E1 | a faire |
| W2-E3 | 2 | Ablation wavefront natif | L'ordonnancement natif compte | `MNIST` | comparer a W2-E1 | `1-3` | qualite + throughput | difference interpretable et reproductible | a faire |
| W3-E1 | 3 | `scan3D_ref` correction | Le `scan3D_ref` est numeriquement correct | petits tenseurs synthetiques | reference naive | deterministe | erreur max, tests unitaires | erreurs dans la tolerance et tests verts | a faire |
| W3-E2 | 3 | `3D` natif sur `Moving MNIST` | Le `3D` natif apprend une dynamique spatio-temporelle | `Moving MNIST` | baseline `2D` frame-par-frame | `1` complet + `2` courts | `test_loss`, `PSNR`, `SSIM` si disponible, `samples/s` | battre le controle temporel detruit et montrer une utilite claire vs baseline | a faire |
| W3-E3 | 3 | Controle ordre temporel detruit | L'ordre temporel est vraiment utilise | `Moving MNIST` temps permute | comparer a W3-E2 | `1-3` | `test_loss`, `PSNR` | degradation nette vs W3-E2 | a faire |
| W4-E1 | 4 | Stack GPU stable | La version dense GPU tourne sans instabilite critique | donnees synthetiques simples | benchmark interne | `1-3` | throughput, VRAM max, temps/epoch, NaN/Inf | aucun crash majeur, memoire compatible MX450 | a faire |
| W4-E2 | 4 | `scan1d++ + convND` | Les deux briques apportent un gain mesurable | donnees synthetiques ou pseudo-physiques | ablations sans `convND`, sans scan enrichi | `1-3` | qualite test + perf systeme | gain mesurable sur au moins une dimension importante | a faire |
| W4-E3 | 4 | Experience exploratoire physique | Le modele apprend des regularites multi-echelles interessantes | petite tache pseudo-physique | comparer a W4-E2 ablations | `1` complet + `2` courts | `test_loss`, throughput, VRAM, visualisations | resultat presente comme exploratoire et borne | a faire |

## Artefacts par experience

| ID | Commande | Config figee | Logs | Checkpoint | Figure | Tableau | Notes |
|---|---|---|---|---|---|---|---|
| W1-E1 | a remplir | a remplir | `paper_cpu.step.csv`, `paper_cpu.epoch.csv` | `ckpt_cpu_500k.bin` | courbes loss / ppl | comparaison baseline | run reference texte |
| W1-E2 | a remplir | a remplir | a remplir | optionnel | controle negatif | oui | ordre melange ou targets perturbees |
| W1-E3 | a remplir | meilleur checkpoint | transcripts | checkpoint W1-E1 | exemples REPL | resume qualitatif | prompts fixes |
| W2-E1 | a remplir | a remplir | a remplir | a remplir | accuracy/NLL | oui | `MNIST` natif |
| W2-E2 | a remplir | a remplir | a remplir | a remplir | comparaison controle | oui | spatial perturbe |
| W2-E3 | a remplir | a remplir | a remplir | a remplir | ablation wavefront | oui | budget voisin |
| W3-E1 | tests | dimensions fixes | sorties tests | non | non | court tableau d'erreur | correction |
| W3-E2 | a remplir | a remplir | a remplir | a remplir | prediction / courbes | oui | `Moving MNIST` |
| W3-E3 | a remplir | a remplir | a remplir | a remplir | comparaison controle | oui | temps detruit |
| W4-E1 | a remplir | a remplir | a remplir | optionnel | throughput/VRAM | oui | benchmark de stabilite |
| W4-E2 | a remplir | a remplir | a remplir | a remplir | ablations | oui | gains mesures |
| W4-E3 | a remplir | a remplir | a remplir | a remplir | visualisations exploratoires | oui | claims prudents |

## Ordre de priorite recommande

1. W1-E1
2. W1-E2
3. W1-E3
4. W2-E1
5. W2-E2
6. W3-E1
7. W3-E2
8. W3-E3
9. W4-E1
10. W4-E2
11. W4-E3

## Definition de "termine"

Une experience n'est "terminee" que si :

- la commande est ecrite,
- le `seed` est note,
- les logs existent,
- la meilleure metrique `val` est reportee,
- la comparaison minimale a ete executee,
- une phrase d'interpretation honnete a ete ajoutee dans `rsults.md`.
