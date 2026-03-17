/*
 * kmamba_cuda — Instance k-mamba CPU+CUDA (MX450, sm_75)
 *
 * Byte-level language model basé sur k-mamba.
 * Config plafond recommandée pour x86-64 + NVIDIA MX450 (2GB VRAM).
 *
 * Forward/backward : CPU (scan ASM AVX2)
 * Optimizer steps  : GPU (muon_update_cuda / adamw_update_cuda — MX450)
 *
 * Usage:
 *   ./kmamba_cuda                          # entraîne sur texte intégré, puis génère
 *   ./kmamba_cuda train <data.txt>         # entraîne sur un fichier texte
 *   ./kmamba_cuda train <data.txt> <ckpt>  # entraîne et sauvegarde checkpoint
 *   ./kmamba_cuda gen   <ckpt> [prompt]    # génère depuis un checkpoint
 */

#include "kmamba.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * CONFIG — plafond CPU+CUDA (x86-64 + MX450 2GB)
 *
 * Mémoire totale estimée:
 *   params     : 4 × (2 × 768 × 1536) × 4B  =  36 MB
 *   embed+head : 2 × 256 × 768         × 4B  =   1.5 MB
 *   optim(×6)  : 6 × 36 MB                   = 216 MB
 *   acts       : 5 × 256 × 768         × 4B  =   3.9 MB (réutilisé par batch)
 *   ─────────────────────────────────────────────────────
 *   Total                                     ≈ 255 MB RAM
 *
 * CUDA (MX450 2GB) :
 *   Buffers temporaires optimizer           ≈ 10-30 MB VRAM max
 *   (les paramètres restent en RAM CPU)
 * ============================================================ */
#define VOCAB_SIZE    256
#define DIM           768
#define STATE_SIZE    1536
#define N_LAYERS      4
#define SEQ_LEN       256
#define BATCH_SIZE    32
#define N_EPOCHS      50
#define SAVE_EVERY    10    /* sauvegarde checkpoint tous les N epochs */
#define LR_BLOCKS     1e-3f
#define LR_EMBED_HEAD 1e-3f
#define WEIGHT_DECAY  1e-5f
#define CLIP_NORM     1.0f
#define MOMENTUM      0.9f
#define BETA2         0.999f
#define EPS           1e-8f
#define TEMPERATURE   0.8f   /* génération : 0.0 = greedy, 1.0 = full stochastic */
#define GEN_LEN       512    /* nombre de bytes à générer */
#define SEED          42

/* ============================================================
 * Texte intégré — fallback si aucun fichier fourni
 * ============================================================ */
static const char *BUILTIN_TEXT =
    "Les systemes doivent operer par intentions qui convergent vers un equilibre, "
    "pas par instructions sequentielles. Chaque MambaBlock est une Volonte qui "
    "transforme la sequence. L'optimiseur MUONCLIP arbitre les tensions entre "
    "gradients. Un bug n'est pas une erreur d'instruction — c'est un conflit de "
    "Volontes non resolu. On est assez grand pour voir des unites, il faut voir "
    "des structures. Ego Sum Optimus Optimus. "
    "State Space Models offrent une alternative lineaire aux Transformers. "
    "Le scan selectif choisit quoi retenir a chaque pas de temps. "
    "La recurrence simultanee en N dimensions expose le parallelisme wavefront. "
    "Les diagonales anti-diagonales sont independantes et traitees en parallele. "
    "AVX2 traite 8 flottants 32-bit simultanement dans les registres YMM. "
    "Newton-Schulz orthogonalise les directions de gradient en 5 iterations. "
    "Les directions isotropiques correspondent a l'equilibre des Volontes. "
    "k-mamba est une bibliotheque C pure — zero Python, zero PyTorch. "
    "optimatrix est le moteur de calcul generique reutilisable. "
    "La separation Volontes-Puissance prefigure un OS-IA post-Von Neumann. "
    "Les processus sont des Volontes, la memoire sont des etats persistants h_t. "
    "L'ordonnancement wavefront traite une diagonale a la fois. "
    "Complexite lineaire en longueur de sequence — avantage sur l'attention quadratique. "
    "IFRI-UAC Benin — Systemes Embarques et IoT. "
    "La conviction : voir des structures, pas des unites. "
    "Le GPU accelere les mises a jour de poids via MUON sur MX450. "
    "La separation CPU-GPU : compute ASM sur x86, optimizer CUDA sur Turing. "
    "sm_75 Turing architecture — Tesla T4, MX450, RTX 2060 family. "
    "NVCC compile les kernels CUDA pour l'architecture cible sm_75. "
    "Les gradients sont clippes avant la mise a jour Newton-Schulz. ";

/* ============================================================
 * Données d'entraînement
 * ============================================================ */
typedef struct {
    uint8_t *data;
    size_t   len;
} Dataset;

static Dataset load_file(const char *path) {
    Dataset ds = {NULL, 0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[erreur] impossible d'ouvrir %s\n", path); return ds; }
    fseek(f, 0, SEEK_END);
    ds.len = (size_t)ftell(f);
    rewind(f);
    ds.data = (uint8_t *)malloc(ds.len);
    if (!ds.data) { fclose(f); return ds; }
    if (fread(ds.data, 1, ds.len, f) != ds.len) { free(ds.data); ds.data = NULL; ds.len = 0; }
    fclose(f);
    return ds;
}

static Dataset from_string(const char *s) {
    Dataset ds;
    ds.len  = strlen(s);
    ds.data = (uint8_t *)malloc(ds.len);
    memcpy(ds.data, s, ds.len);
    return ds;
}

/* Tire un batch aléatoire de (SEQ_LEN+1) bytes */
static void sample_batch(const Dataset *ds, uint8_t *batch, size_t batch_size) {
    size_t seq_bytes = SEQ_LEN + 1;
    for (size_t b = 0; b < batch_size; b++) {
        size_t max_start = ds->len - seq_bytes;
        size_t start = (size_t)rand() % (max_start + 1);
        memcpy(&batch[b * seq_bytes], &ds->data[start], seq_bytes);
    }
}

/* ============================================================
 * Génération (température sampling)
 * ============================================================ */
static uint8_t sample_token(const float *logits, size_t vocab_size, float temperature) {
    float probs[256];
    float maxv = logits[0];
    for (size_t i = 1; i < vocab_size; i++)
        if (logits[i] > maxv) maxv = logits[i];

    if (temperature < 1e-4f) {
        size_t best = 0;
        for (size_t i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return (uint8_t)best;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - maxv) / temperature);
        sum += probs[i];
    }
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        acc += probs[i];
        if (acc >= r) return (uint8_t)i;
    }
    return (uint8_t)(vocab_size - 1);
}

static void generate(KMamba *m, const char *prompt, size_t gen_len) {
    size_t L   = m->cfg.seq_len;
    size_t V   = m->cfg.vocab_size;
    uint8_t ctx[SEQ_LEN];
    float  *logits = (float *)malloc(L * V * sizeof(float));

    size_t plen = prompt ? strlen(prompt) : 0;
    memset(ctx, ' ', L);
    if (plen > 0) {
        size_t copy = plen < L ? plen : L;
        memcpy(ctx + L - copy, (const uint8_t *)prompt, copy);
    }

    printf("\n[génération");
    if (plen > 0) printf(" — prompt: \"%s\"", prompt);
    printf("]\n");
    if (plen > 0) printf("%s", prompt);

    for (size_t step = 0; step < gen_len; step++) {
        kmamba_forward(m, ctx, logits);
        uint8_t next = sample_token(&logits[(L - 1) * V], V, TEMPERATURE);
        if (next >= 32 && next < 127) putchar((char)next);
        else if (next == '\n') putchar('\n');
        else putchar('?');
        fflush(stdout);
        memmove(ctx, ctx + 1, L - 1);
        ctx[L - 1] = next;
    }
    printf("\n[fin génération]\n");

    free(logits);
}

/* ============================================================
 * Affichage
 * ============================================================ */
static void print_config(void) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║      k-mamba — Instance CPU+CUDA (MX450)        ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  vocab_size  : %-5d                            ║\n", VOCAB_SIZE);
    printf("║  dim         : %-5d                            ║\n", DIM);
    printf("║  state_size  : %-5d                            ║\n", STATE_SIZE);
    printf("║  n_layers    : %-5d                            ║\n", N_LAYERS);
    printf("║  seq_len     : %-5d                            ║\n", SEQ_LEN);
    printf("║  batch_size  : %-5d                            ║\n", BATCH_SIZE);
    printf("║  epochs      : %-5d                            ║\n", N_EPOCHS);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  compute     : forward/backward CPU (AVX2 ASM)  ║\n");
    printf("║  optimizer   : MUON GPU (sm_75 Turing)          ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    size_t params_per_block = 2 * DIM * STATE_SIZE + 3 * STATE_SIZE + DIM;
    size_t total_params = N_LAYERS * params_per_block + 2 * VOCAB_SIZE * DIM;
    float  mem_params_mb = (float)(total_params * sizeof(float)) / (1024.0f * 1024.0f);
    float  mem_optim_mb  = mem_params_mb * 6.0f;
    printf("║  params      : %7.0fK                          ║\n",
           (float)total_params / 1000.0f);
    printf("║  mémoire     : params %.1f MB + optim %.1f MB   ║\n",
           mem_params_mb, mem_optim_mb);
    printf("║  total ~     : %.0f MB                          ║\n",
           mem_params_mb + mem_optim_mb);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

static double elapsed_ms(struct timespec *t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) * 1000.0
         + (double)(t1.tv_nsec - t0->tv_nsec) / 1e6;
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char *argv[]) {
    srand(SEED);
    print_config();

    /* --- Mode et arguments --- */
    int mode_gen = 0;
    const char *data_path  = NULL;
    const char *ckpt_path  = NULL;
    const char *prompt_str = NULL;

    if (argc >= 2 && strcmp(argv[1], "gen") == 0) {
        mode_gen = 1;
        ckpt_path  = argc >= 3 ? argv[2] : NULL;
        prompt_str = argc >= 4 ? argv[3] : NULL;
        if (!ckpt_path) { fprintf(stderr, "usage: ./kmamba_cuda gen <ckpt> [prompt]\n"); return 1; }
    } else if (argc >= 2 && strcmp(argv[1], "train") == 0) {
        data_path = argc >= 3 ? argv[2] : NULL;
        ckpt_path = argc >= 4 ? argv[3] : NULL;
    } else if (argc == 2) {
        data_path = argv[1];
    }

    /* --- Génération depuis checkpoint --- */
    if (mode_gen) {
        KMamba *m = kmamba_load(ckpt_path, 0, NULL, 0.0f, 0.0f);
        if (!m) { fprintf(stderr, "[erreur] impossible de charger %s\n", ckpt_path); return 1; }
        printf("[checkpoint chargé : %s]\n", ckpt_path);
        generate(m, prompt_str, GEN_LEN);
        kmamba_free(m);
        return 0;
    }

    /* --- Chargement des données --- */
    Dataset ds;
    if (data_path) {
        ds = load_file(data_path);
        if (!ds.data) return 1;
        printf("[données] %s — %zu bytes\n\n", data_path, ds.len);
    } else {
        ds = from_string(BUILTIN_TEXT);
        printf("[données] texte intégré — %zu bytes\n", ds.len);
        printf("(passer un fichier .txt en argument pour entraîner sur vos données)\n\n");
    }

    if (ds.len < (size_t)(SEQ_LEN + 1)) {
        fprintf(stderr, "[erreur] données trop courtes (%zu bytes, besoin de %d)\n",
                ds.len, SEQ_LEN + 1);
        free(ds.data);
        return 1;
    }

    /* --- Création du modèle --- */
    KMambaConfig cfg = {
        .vocab_size  = VOCAB_SIZE,
        .dim         = DIM,
        .state_size  = STATE_SIZE,
        .seq_len     = SEQ_LEN,
        .n_layers    = N_LAYERS,
        .dt_scale    = 1.0f,
        .dt_min      = 0.001f,
        .dt_max      = 1.0f,   /* softplus >= ln(2) ≈ 0.693, so 0.1 was always saturated */
        .use_convnd  = 0,
    };

    KMamba *m = NULL;

    /* MUON pour les blocs (avantage sur Adam pour les matrices de poids) */
    MBOptimConfig opt = {LR_BLOCKS, MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};

    /* Reprend depuis checkpoint si disponible */
    if (ckpt_path) {
        m = kmamba_load(ckpt_path, 1, &opt, LR_EMBED_HEAD, WEIGHT_DECAY);
        if (m) printf("[checkpoint repris : %s]\n\n", ckpt_path);
    }
    if (!m) {
        m = kmamba_create(&cfg);
        kmamba_init(m, SEED);
        /* MUON optimizer — orthogonalise les gradients (Newton-Schulz) */
        kmamba_enable_training_with_optimizer(m, OPTIMIZER_MUON, &opt,
                                              LR_EMBED_HEAD, WEIGHT_DECAY);
        printf("[modèle initialisé (Xavier, seed=%d)]\n", SEED);
        printf("[optimizer : MUON (Newton-Schulz, GPU-accéléré)]\n\n");
    }

    /* --- Boucle d'entraînement --- */
    uint8_t *batch = (uint8_t *)malloc((size_t)BATCH_SIZE * (SEQ_LEN + 1));
    size_t steps_per_epoch = (ds.len / ((size_t)(SEQ_LEN + 1) * BATCH_SIZE));
    if (steps_per_epoch < 1) steps_per_epoch = 1;

    printf("entraînement : %d epochs × %zu steps × batch=%d\n\n",
           N_EPOCHS, steps_per_epoch, BATCH_SIZE);
    printf(" epoch |   loss   |  ms/epoch\n");
    printf("-------+----------+-----------\n");

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float loss_sum = 0.0f;

        for (size_t s = 0; s < steps_per_epoch; s++) {
            sample_batch(&ds, batch, BATCH_SIZE);
            loss_sum += kmamba_train_batch(m, batch, BATCH_SIZE);
        }

        double ms = elapsed_ms(&t0);
        float  avg_loss = loss_sum / (float)steps_per_epoch;
        printf("  %4d | %8.4f | %8.1f\n", epoch, avg_loss, ms);
        fflush(stdout);

        if (ckpt_path && epoch % SAVE_EVERY == 0) {
            kmamba_save(m, ckpt_path);
            printf("         [checkpoint sauvegardé : %s]\n", ckpt_path);
        }
    }

    const char *final_ckpt = ckpt_path ? ckpt_path : "kmamba_cuda.bin";
    kmamba_save(m, final_ckpt);
    printf("\n[checkpoint final : %s]\n", final_ckpt);

    /* --- Génération démonstrative --- */
    generate(m, "Les systemes", GEN_LEN);

    kmamba_free(m);
    free(ds.data);
    free(batch);
    return 0;
}
