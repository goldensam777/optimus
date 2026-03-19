/*
 * kmamba_cpu — Instance k-mamba CPU
 *
 * Byte-level language model basé sur k-mamba.
 * Config plafond recommandée pour x86-64 AVX2, ~115 MB RAM.
 *
 * Usage:
 *   ./kmamba_cpu                          # entraîne sur texte intégré, puis génère
 *   ./kmamba_cpu train <data.txt>         # entraîne sur un fichier texte
 *   ./kmamba_cpu train <data.txt> <ckpt>  # entraîne et sauvegarde checkpoint
 *   ./kmamba_cpu gen   <ckpt> [prompt]    # génère depuis un checkpoint
 *   ./kmamba_cpu chat  <ckpt>             # REPL interactif (chatbot)
 */

#include "kmamba.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * CONFIG — plafond CPU (x86-64 AVX2)
 *
 * Config nano — test rapide (~60× plus vite que le plafond)
 * Mémoire totale estimée:
 *   params     : 2 × (2 × 128 × 256) × 4B   =   0.5 MB
 *   embed+head : 2 × 256 × 128        × 4B   =   0.25 MB
 *   optim(×6)  : 6 × 0.5 MB                  =   3 MB
 *   ─────────────────────────────────────────────────────
 *   Total                                     ≈   5 MB
 * ============================================================ */
#define VOCAB_SIZE    256
#define DIM           256
#define STATE_SIZE    512
#define N_LAYERS      5
#define SEQ_LEN       256
#define BATCH_SIZE    32
#define N_EPOCHS      200
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
#define CHAT_MAX_RESP 256    /* longueur max d'une réponse en mode chat */
#define CHAT_USER     "[speaker001:] "   /* format dans le contexte — doit matcher le corpus */
#define CHAT_BOT      "[speaker002:] "   /* idem */
#define REPL_YOU      "speaker001"       /* label affiché à l'utilisateur */
#define REPL_CP       "speaker002"       /* label affiché pour le modèle */
#define SEED          42

/* ============================================================
 * Texte intégré — fallback si aucun fichier fourni
 * (assez long pour un entraînement démonstratif)
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
    "La conviction : voir des structures, pas des unites. ";

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
        /* greedy */
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

    /* Initialise le contexte avec le prompt (tronqué/paddé à SEQ_LEN) */
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
        /* Prend les logits du dernier token */
        uint8_t next = sample_token(&logits[(L - 1) * V], V, TEMPERATURE);
        /* Affiche le byte généré (printable ou '?') */
        if (next >= 32 && next < 127) putchar((char)next);
        else if (next == '\n') putchar('\n');
        else putchar('?');
        fflush(stdout);
        /* Décale le contexte */
        memmove(ctx, ctx + 1, L - 1);
        ctx[L - 1] = next;
    }
    printf("\n[fin génération]\n");

    free(logits);
}

/* ============================================================
 * REPL interactif — mode chat
 * ============================================================ */
static void chat_repl(KMamba *m) {
    size_t L = m->cfg.seq_len;
    size_t V = m->cfg.vocab_size;
    float *logits = (float *)malloc(L * V * sizeof(float));
    if (!logits) { fprintf(stderr, "[erreur] malloc logits\n"); return; }

    /* Contexte glissant initialisé à des espaces */
    uint8_t *ctx = (uint8_t *)malloc(L);
    if (!ctx) { free(logits); return; }
    memset(ctx, ' ', L);

    char line[1024];

    printf("\n");
    printf("  k-mamba — session interactive\n");
    printf("  Ctrl+D ou 'quit' pour quitter\n");
    printf("  ─────────────────────────────\n\n");

    while (1) {
        /* Prompt utilisateur */
        printf("\033[1;32m" REPL_YOU "\033[0m> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;
        }

        /* Retire le \n terminal */
        size_t llen = strlen(line);
        if (llen > 0 && line[llen - 1] == '\n') line[--llen] = '\0';
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
        if (llen == 0) continue;

        /* Formate pour le contexte : "Human: <msg>\nBot: " */
        char inject_buf[sizeof(line) + 32];
        int inject_len = snprintf(inject_buf, sizeof(inject_buf),
                                  "%s%s\n%s", CHAT_USER, line, CHAT_BOT);

        /* Glisse le contexte et injecte à la fin */
        size_t inj = (size_t)inject_len;
        if (inj >= L) {
            memcpy(ctx, (uint8_t *)inject_buf + inj - L, L);
        } else {
            memmove(ctx, ctx + inj, L - inj);
            memcpy(ctx + L - inj, (uint8_t *)inject_buf, inj);
        }

        /* Affiche le label modèle */
        printf("\033[1;34m" REPL_CP "\033[0m> ");
        fflush(stdout);

        /* Génère token par token jusqu'à '\n' ou CHAT_MAX_RESP */
        for (size_t step = 0; step < CHAT_MAX_RESP; step++) {
            kmamba_forward(m, ctx, logits);
            uint8_t next = sample_token(&logits[(L - 1) * V], V, TEMPERATURE);

            /* Fin de réponse à la newline */
            if (next == '\n' || next == '\r') {
                memmove(ctx, ctx + 1, L - 1);
                ctx[L - 1] = '\n';
                break;
            }

            /* Affiche le token généré */
            putchar((next >= 32 && next < 127) ? (char)next : '?');
            fflush(stdout);

            /* Glisse le contexte */
            memmove(ctx, ctx + 1, L - 1);
            ctx[L - 1] = next;
        }
        printf("\n\n");
    }

    printf("  [session terminée]\n");
    free(ctx);
    free(logits);
}

/* ============================================================
 * Affichage
 * ============================================================ */
static void print_config(void) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║         k-mamba — Instance CPU (AVX2)           ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  vocab_size  : %-5d                            ║\n", VOCAB_SIZE);
    printf("║  dim         : %-5d                            ║\n", DIM);
    printf("║  state_size  : %-5d                            ║\n", STATE_SIZE);
    printf("║  n_layers    : %-5d                            ║\n", N_LAYERS);
    printf("║  seq_len     : %-5d                            ║\n", SEQ_LEN);
    printf("║  batch_size  : %-5d                            ║\n", BATCH_SIZE);
    printf("║  epochs      : %-5d                            ║\n", N_EPOCHS);
    printf("╠══════════════════════════════════════════════════╣\n");
    size_t params_per_block = 2 * DIM * STATE_SIZE + 3 * STATE_SIZE + DIM;
    size_t total_params = N_LAYERS * params_per_block + 2 * VOCAB_SIZE * DIM;
    float  mem_params_mb = (float)(total_params * sizeof(float)) / (1024.0f * 1024.0f);
    float  mem_optim_mb  = mem_params_mb * 6.0f;
    printf("║  params      : %7.0fK                          ║\n",
           (float)total_params / 1000.0f);
    printf("║  mémoire     : params %.1f MB + optim %.1f MB   ║\n",
           mem_params_mb, mem_optim_mb);
    printf("║  total ~     : %.0f MB                           ║\n",
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

    if (argc >= 2 && strcmp(argv[1], "chat") == 0) {
        ckpt_path = argc >= 3 ? argv[2] : NULL;
        if (!ckpt_path) { fprintf(stderr, "usage: ./kmamba_cpu chat <ckpt>\n"); return 1; }
        KMamba *m = kmamba_load(ckpt_path, 0, NULL, 0.0f, 0.0f);
        if (!m) { fprintf(stderr, "[erreur] impossible de charger %s\n", ckpt_path); return 1; }
        printf("[checkpoint : %s]\n", ckpt_path);
        chat_repl(m);
        kmamba_free(m);
        return 0;
    } else if (argc >= 2 && strcmp(argv[1], "gen") == 0) {
        mode_gen = 1;
        ckpt_path  = argc >= 3 ? argv[2] : NULL;
        prompt_str = argc >= 4 ? argv[3] : NULL;
        if (!ckpt_path) { fprintf(stderr, "usage: ./kmamba_cpu gen <ckpt> [prompt]\n"); return 1; }
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

    /* Reprend depuis checkpoint si disponible */
    MBOptimConfig opt = {LR_BLOCKS, MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};
    if (ckpt_path) {
        m = kmamba_load(ckpt_path, 1, &opt, LR_EMBED_HEAD, WEIGHT_DECAY);
        if (m) printf("[checkpoint repris : %s]\n\n", ckpt_path);
    }
    if (!m) {
        m = kmamba_create(&cfg);
        kmamba_init(m, SEED);
        kmamba_enable_training(m, &opt, LR_EMBED_HEAD, WEIGHT_DECAY);
        printf("[modèle initialisé (Xavier, seed=%d)]\n\n", SEED);
    }

    /* --- Boucle d'entraînement --- */
    uint8_t *batch = (uint8_t *)malloc((size_t)BATCH_SIZE * (SEQ_LEN + 1));
    size_t steps_per_epoch = (ds.len / ((size_t)(SEQ_LEN + 1) * BATCH_SIZE));
    if (steps_per_epoch < 1) steps_per_epoch = 1;

    printf("entraînement : %d epochs × %zu steps × batch=%d\n\n",
           N_EPOCHS, steps_per_epoch, BATCH_SIZE);
    printf(" epoch |   loss   |  ms/epoch\n");
    printf("-------+----------+-----------\n");
    fflush(stdout);

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float loss_sum = 0.0f;

        for (size_t s = 0; s < steps_per_epoch; s++) {
            sample_batch(&ds, batch, BATCH_SIZE);
            loss_sum += kmamba_train_batch(m, batch, BATCH_SIZE);
            if ((s + 1) % 50 == 0 || s + 1 == steps_per_epoch) {
                printf("       step %4zu/%zu  loss=%.4f\r",
                       s + 1, steps_per_epoch, loss_sum / (float)(s + 1));
                fflush(stdout);
            }
        }
        printf("\n");

        double ms = elapsed_ms(&t0);
        float  avg_loss = loss_sum / (float)steps_per_epoch;
        printf("  %4d | %8.4f | %8.1f\n", epoch, avg_loss, ms);
        fflush(stdout);

        /* Checkpoint périodique */
        if (ckpt_path && epoch % SAVE_EVERY == 0) {
            kmamba_save(m, ckpt_path);
            printf("         [checkpoint sauvegardé : %s]\n", ckpt_path);
        }
    }

    /* Checkpoint final */
    const char *final_ckpt = ckpt_path ? ckpt_path : "kmamba_cpu.bin";
    kmamba_save(m, final_ckpt);
    printf("\n[checkpoint final : %s]\n", final_ckpt);

    /* --- Génération démonstrative --- */
    generate(m, "Les systemes", GEN_LEN);

    kmamba_free(m);
    free(ds.data);
    free(batch);
    return 0;
}
