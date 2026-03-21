#define _POSIX_C_SOURCE 200809L

/*
 * kmamba_cpu — Instance k-mamba CPU
 *
 * Byte-level language model basé sur k-mamba.
 * Config CPU paper : ~500K paramètres, logs CSV et corpus borné.
 *
 * Usage:
 *   ./kmamba_cpu                          # entraîne sur texte intégré, puis génère
 *   ./kmamba_cpu train <data.txt> [ckpt] [log-prefix]
 *   ./kmamba_cpu gen   <ckpt> [prompt]    # génère depuis un checkpoint
 *   ./kmamba_cpu chat  <ckpt>             # REPL interactif (chatbot)
 */

#include "kmamba.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * CONFIG — instance CPU Paper (661K params)
 *
 * Cible : ~661K paramètres pour résultats paper.
 * Param count réel avec l'architecture k-mamba actuelle :
 *   embed+head = 2 * vocab * dim
 *   bloc       = W_in + W_out + A_log + W_B + W_C + b_B + b_C
 *              + delta_proj + lambda_proj + theta
 *   total      ≈ 661K paramètres
 * ============================================================ */
#define VOCAB_SIZE    256
#define DIM           256
#define STATE_SIZE    512
#define N_LAYERS      2
#define SEQ_LEN       128
#define BATCH_SIZE    64
#define N_EPOCHS      100
#define SAVE_EVERY    1     /* sauvegarde checkpoint tous les N epochs */
#define LR            5e-4f
#define LR_EMBED_HEAD 1e-4f
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
#define CHINCHILLA_TOKENS_PER_PARAM 20u
#define BLOCK_PARAM_COUNT_EST \
    (2u * DIM + STATE_SIZE + 2u * STATE_SIZE * DIM + 2u * STATE_SIZE + 2u * DIM + (STATE_SIZE / 2u))
#define MODEL_PARAM_COUNT_EST \
    (N_LAYERS * BLOCK_PARAM_COUNT_EST + 2u * VOCAB_SIZE * DIM)
#define DATASET_BYTES_MAX (MODEL_PARAM_COUNT_EST * CHINCHILLA_TOKENS_PER_PARAM)
#define VAL_PERCENT      5u
#define VAL_EVAL_STEPS   64u
#define PRINT_EVERY      50u
#define OMP_THREADS      7

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

static Dataset load_file_prefix(const char *path, size_t max_bytes, size_t *source_len_out) {
    Dataset ds = {NULL, 0};
    size_t file_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[erreur] impossible d'ouvrir %s\n", path); return ds; }
    fseek(f, 0, SEEK_END);
    file_len = (size_t)ftell(f);
    rewind(f);
    if (source_len_out) *source_len_out = file_len;
    ds.len = (max_bytes > 0 && file_len > max_bytes) ? max_bytes : file_len;
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

static char *xstrdup_local(const char *s) {
    size_t n;
    char *copy;
    if (!s) return NULL;
    n = strlen(s) + 1;
    copy = (char *)malloc(n);
    if (!copy) return NULL;
    memcpy(copy, s, n);
    return copy;
}

static char *make_log_path(const char *prefix, const char *kind) {
    size_t n;
    char *path;
    if (!prefix || !kind) return NULL;
    n = strlen(prefix) + 1 + strlen(kind) + 4 + 1;
    path = (char *)malloc(n);
    if (!path) return NULL;
    snprintf(path, n, "%s.%s.csv", prefix, kind);
    return path;
}

static FILE *open_csv_log(const char *path, const char *header) {
    FILE *f;
    long pos = 0;

    if (!path || !header) return NULL;

    f = fopen(path, "a+");
    if (!f) {
        fprintf(stderr, "[warning] impossible d'ouvrir %s\n", path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) == 0) pos = ftell(f);
    if (pos == 0) {
        fprintf(f, "%s\n", header);
        fflush(f);
    }
    return f;
}

static size_t current_rss_kb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return (size_t)usage.ru_maxrss;
}

static size_t block_param_count(size_t dim, size_t state_size) {
    size_t theta_size = state_size / 2;
    if (theta_size == 0) theta_size = 1;
    return 2 * dim
         + state_size
         + 2 * state_size * dim
         + 2 * state_size
         + 2 * dim
         + theta_size;
}

static size_t model_param_count(size_t vocab_size, size_t dim,
                                size_t state_size, size_t n_layers) {
    return n_layers * block_param_count(dim, state_size)
         + 2 * vocab_size * dim;
}

static size_t append_tensor(float *dst, size_t offset,
                            const float *src, size_t n) {
    if (dst && src && n > 0) memcpy(dst + offset, src, n * sizeof(float));
    return offset + n;
}

static size_t kmamba_flatten_params(const KMamba *m, float *dst) {
    size_t offset = 0;

    if (!m) return 0;

    offset = append_tensor(dst, offset, m->embedding, m->cfg.vocab_size * m->cfg.dim);
    offset = append_tensor(dst, offset, m->head,      m->cfg.dim * m->cfg.vocab_size);

    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        const MambaBlock *b = m->layers[i];
        size_t theta_size = b->config.state_size / 2;
        if (theta_size == 0) theta_size = 1;

        offset = append_tensor(dst, offset, b->W_in.data,       b->W_in.rows * b->W_in.cols);
        offset = append_tensor(dst, offset, b->W_out.data,      b->W_out.rows * b->W_out.cols);
        offset = append_tensor(dst, offset, b->A_log.data,      b->A_log.rows * b->A_log.cols);
        offset = append_tensor(dst, offset, b->W_B.data,        b->W_B.rows * b->W_B.cols);
        offset = append_tensor(dst, offset, b->W_C.data,        b->W_C.rows * b->W_C.cols);
        offset = append_tensor(dst, offset, b->b_B,             b->config.state_size);
        offset = append_tensor(dst, offset, b->b_C,             b->config.state_size);
        offset = append_tensor(dst, offset, b->delta_proj.data, b->delta_proj.rows * b->delta_proj.cols);
        offset = append_tensor(dst, offset, b->lambda_proj.data, b->lambda_proj.rows * b->lambda_proj.cols);
        offset = append_tensor(dst, offset, b->theta,           theta_size);
    }

    return offset;
}

static float l2_norm_f32(const float *x, size_t n) {
    double acc = 0.0;
    if (!x) return 0.0f;
    for (size_t i = 0; i < n; i++) {
        double v = (double)x[i];
        acc += v * v;
    }
    return sqrtf((float)acc);
}

static float l2_diff_norm_f32(const float *a, const float *b, size_t n) {
    double acc = 0.0;
    if (!a || !b) return 0.0f;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return sqrtf((float)acc);
}

static float safe_perplexity(float loss) {
    if (!isfinite(loss)) return NAN;
    if (loss > 20.0f) loss = 20.0f;
    return expf(loss);
}

static float compute_loss_from_logits(const float *logits,
                                      const uint8_t *targets,
                                      size_t seq_len,
                                      size_t vocab_size) {
    float loss = 0.0f;

    for (size_t t = 0; t < seq_len; t++) {
        const float *row = &logits[t * vocab_size];
        float maxv = row[0];
        float sum = 0.0f;

        for (size_t v = 1; v < vocab_size; v++)
            if (row[v] > maxv) maxv = row[v];
        for (size_t v = 0; v < vocab_size; v++)
            sum += expf(row[v] - maxv);

        {
            float p = expf(row[(size_t)targets[t]] - maxv) / sum;
            if (p < 1e-20f) p = 1e-20f;
            loss += -logf(p);
        }
    }

    return loss / (float)seq_len;
}

static float evaluate_dataset(KMamba *m, const Dataset *ds, size_t max_windows) {
    size_t seq_bytes;
    size_t available;
    size_t windows;
    size_t stride;
    float *logits;
    float loss_sum = 0.0f;

    if (!m || !ds || !ds->data) return NAN;

    seq_bytes = m->cfg.seq_len + 1;
    if (ds->len < seq_bytes) return NAN;

    available = ds->len - seq_bytes + 1;
    windows = available < max_windows ? available : max_windows;
    if (windows == 0) windows = 1;
    stride = windows > 1 ? (available - 1) / (windows - 1) : 0;

    logits = (float *)malloc(m->cfg.seq_len * m->cfg.vocab_size * sizeof(float));
    if (!logits) return NAN;

    for (size_t i = 0; i < windows; i++) {
        size_t start = i * stride;
        const uint8_t *window = &ds->data[start];
        kmamba_forward(m, window, logits);
        loss_sum += compute_loss_from_logits(logits, window + 1,
                                             m->cfg.seq_len, m->cfg.vocab_size);
    }

    free(logits);
    return loss_sum / (float)windows;
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
    size_t total_params = model_param_count(VOCAB_SIZE, DIM, STATE_SIZE, N_LAYERS);
    size_t embed_head_params = 2 * VOCAB_SIZE * DIM;
    size_t block_params_total = N_LAYERS * block_param_count(DIM, STATE_SIZE);
    size_t optimizer_params = 3 * block_params_total + 2 * embed_head_params;
    float  mem_params_mb = (float)(total_params * sizeof(float)) / (1024.0f * 1024.0f);
    float  mem_optim_mb = (float)(optimizer_params * sizeof(float)) / (1024.0f * 1024.0f);

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║         k-mamba — Instance CPU (AVX2)            ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  vocab_size  : %-5d                            ║\n", VOCAB_SIZE);
    printf("║  dim         : %-5d                            ║\n", DIM);
    printf("║  state_size  : %-5d                            ║\n", STATE_SIZE);
    printf("║  n_layers    : %-5d                            ║\n", N_LAYERS);
    printf("║  seq_len     : %-5d                            ║\n", SEQ_LEN);
    printf("║  batch_size  : %-5d                            ║\n", BATCH_SIZE);
    printf("║  epochs      : %-5d                            ║\n", N_EPOCHS);
    printf("║  omp threads : %-5d                            ║\n", OMP_THREADS);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  params      : %7.0fK                          ║\n",
           (float)total_params / 1000.0f);
    printf("║  corpus max  : %5.1f MiB                        ║\n",
           (float)DATASET_BYTES_MAX / (1024.0f * 1024.0f));
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

/* ── Learning Rate Scheduler ──────────────────────────────────── */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static float lr_schedule(float lr_max, size_t step,
                           size_t warmup_steps, size_t total_steps) {
    if (step == 0) return 0.0f;
    if (step < warmup_steps)
        return lr_max * (float)step / (float)warmup_steps;
    float progress = (float)(step - warmup_steps)
                   / (float)(total_steps - warmup_steps);
    float cosine   = 0.5f * (1.0f + cosf(M_PI * progress));
    float lr_min   = lr_max * 0.1f;
    return lr_min + (lr_max - lr_min) * cosine;
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char *argv[]) {
    const size_t seq_bytes = (size_t)(SEQ_LEN + 1);
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(OMP_THREADS);
#endif
    srand(SEED);
    print_config();

    /* --- Mode et arguments --- */
    int mode_gen = 0;
    const char *data_path  = NULL;
    const char *ckpt_path  = NULL;
    const char *prompt_str = NULL;
    const char *log_prefix_arg = NULL;

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
        log_prefix_arg = argc >= 5 ? argv[4] : NULL;
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
    Dataset train_ds = {NULL, 0};
    Dataset val_ds = {NULL, 0};
    size_t source_len = 0;
    size_t val_len = 0;
    if (data_path) {
        ds = load_file_prefix(data_path, DATASET_BYTES_MAX, &source_len);
        if (!ds.data) return 1;
        printf("[données] %s — %zu / %zu bytes utilisés\n",
               data_path, ds.len, source_len);
        if (source_len > ds.len) {
            printf("[corpus] instance CPU bornée aux %zu premiers bytes (%.1f MiB)\n",
                   ds.len, (float)ds.len / (1024.0f * 1024.0f));
        }
    } else {
        ds = from_string(BUILTIN_TEXT);
        printf("[données] texte intégré — %zu bytes\n", ds.len);
        printf("(passer un fichier .txt en argument pour entraîner sur vos données)\n");
    }

    if (ds.len < seq_bytes) {
        fprintf(stderr, "[erreur] données trop courtes (%zu bytes, besoin de %d)\n",
                ds.len, SEQ_LEN + 1);
        free(ds.data);
        return 1;
    }

    if (ds.len >= 20 * seq_bytes) {
        val_len = (ds.len * VAL_PERCENT) / 100u;
        if (val_len < seq_bytes) val_len = seq_bytes;
        if (val_len >= ds.len) val_len = 0;
    }

    train_ds = ds;
    if (val_len > 0 && ds.len > val_len + seq_bytes) {
        train_ds.len = ds.len - val_len;
        val_ds.data = ds.data + train_ds.len;
        val_ds.len = val_len;
        printf("[split] train=%zu bytes | val=%zu bytes (%u%%)\n\n",
               train_ds.len, val_ds.len, VAL_PERCENT);
    } else {
        printf("[split] train seul (validation désactivée, corpus trop court)\n\n");
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
    float lr_max = LR;
    float lr_max_embed = LR_EMBED_HEAD;
    MBOptimConfig opt = {lr_max, MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};
    if (ckpt_path) {
        m = kmamba_load(ckpt_path, 1, &opt, lr_max_embed, WEIGHT_DECAY);
        if (m) printf("[checkpoint repris : %s]\n\n", ckpt_path);
    }
    if (!m) {
        m = kmamba_create(&cfg);
        kmamba_init(m, SEED);
        kmamba_enable_training_with_optimizer(m, OPTIMIZER_ADAMW, &opt, lr_max_embed, WEIGHT_DECAY);
        printf("[modèle initialisé (Xavier, seed=%d)]\n\n", SEED);
    }

    size_t total_params = kmamba_flatten_params(m, NULL);
    char *owned_log_prefix = NULL;
    const char *log_prefix = log_prefix_arg;
    char *step_log_path = NULL;
    char *epoch_log_path = NULL;
    FILE *step_log = NULL;
    FILE *epoch_log = NULL;
    float *prev_params = NULL;
    float *curr_params = NULL;
    unsigned long long run_id = (unsigned long long)time(NULL);

    if (!log_prefix) {
        owned_log_prefix = xstrdup_local(ckpt_path ? ckpt_path : "kmamba_cpu");
        log_prefix = owned_log_prefix;
    }
    step_log_path = make_log_path(log_prefix, "step");
    epoch_log_path = make_log_path(log_prefix, "epoch");
    step_log = open_csv_log(step_log_path,
        "run_id,epoch,step_in_epoch,global_step,train_loss,train_ppl,grad_norm,grad_over_clip,would_clip,step_ms,tokens_per_sec,max_rss_kb,bad_loss,lr");
    epoch_log = open_csv_log(epoch_log_path,
        "run_id,epoch,steps_in_epoch,total_tokens,train_loss,train_ppl,train_eval_loss,train_eval_ppl,val_loss,val_ppl,grad_norm_last,grad_over_clip_last,would_clip_last,epoch_ms,step_ms_avg,tokens_per_sec,param_count,param_norm,update_norm,max_rss_kb,train_bytes,val_bytes");
    if (step_log) setvbuf(step_log, NULL, _IOLBF, 0);
    if (epoch_log) setvbuf(epoch_log, NULL, _IOLBF, 0);
    if (step_log || epoch_log) {
        printf("[logs] step=%s | epoch=%s\n\n",
               step_log_path ? step_log_path : "(désactivé)",
               epoch_log_path ? epoch_log_path : "(désactivé)");
    }

    prev_params = (float *)malloc(total_params * sizeof(float));
    curr_params = (float *)malloc(total_params * sizeof(float));
    if (!prev_params || !curr_params) {
        fprintf(stderr, "[erreur] impossible d'allouer les snapshots de paramètres\n");
        free(ds.data);
        free(owned_log_prefix);
        free(step_log_path);
        free(epoch_log_path);
        if (step_log) fclose(step_log);
        if (epoch_log) fclose(epoch_log);
        kmamba_free(m);
        free(prev_params);
        free(curr_params);
        return 1;
    }
    kmamba_flatten_params(m, prev_params);

    /* --- Boucle d'entraînement --- */
    uint8_t *batch = (uint8_t *)malloc((size_t)BATCH_SIZE * seq_bytes);
    size_t steps_per_epoch = (train_ds.len / (seq_bytes * BATCH_SIZE));
    size_t global_step = 0;
    if (steps_per_epoch < 1) steps_per_epoch = 1;

    /* LR Scheduler: warmup + cosine decay (2-3% warmup for Adam) */
    size_t total_steps = (size_t)N_EPOCHS * steps_per_epoch;
    size_t warmup_steps = total_steps / 40;  /* 2.5% warmup */
    float current_lr = lr_max;
    float current_lr_embed = lr_max_embed;

    printf("entraînement : %d epochs × %zu steps × batch=%d\n\n",
           N_EPOCHS, steps_per_epoch, BATCH_SIZE);
    printf(" epoch | train_bt | train_ev |   val    |  tok/s   |  ms/epoch |    lr\n");
    printf("-------+----------+----------+----------+----------+-----------+-----------\n");
    fflush(stdout);

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float loss_sum = 0.0f;
        double step_ms_sum = 0.0;
        int bad_loss = 0;

        for (size_t s = 0; s < steps_per_epoch; s++) {
            struct timespec step_t0;
            double step_ms;
            double step_tokens_s;
            float batch_loss;

            global_step++;

            /* Update learning rate */
            current_lr = lr_schedule(lr_max, global_step, warmup_steps, total_steps);
            current_lr_embed = lr_schedule(lr_max_embed, global_step, warmup_steps, total_steps);
            opt.lr = current_lr;
            /* Note: embed LR is handled via kmamba_update_training_state or direct access */

            clock_gettime(CLOCK_MONOTONIC, &step_t0);
            sample_batch(&train_ds, batch, BATCH_SIZE);
            batch_loss = kmamba_train_batch(m, batch, BATCH_SIZE);
            step_ms = elapsed_ms(&step_t0);
            step_tokens_s = step_ms > 0.0
                ? ((double)BATCH_SIZE * (double)SEQ_LEN * 1000.0) / step_ms
                : 0.0;

            loss_sum += batch_loss;
            step_ms_sum += step_ms;
            if (!isfinite(batch_loss)) bad_loss = 1;

            if (step_log) {
                fprintf(step_log, "%llu,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%d,%.3f,%.3f,%zu,%d,%.6f\n",
                        run_id, epoch, s + 1, global_step,
                        batch_loss, safe_perplexity(batch_loss),
                        m->last_grad_norm, m->last_grad_over_clip, m->last_grad_would_clip,
                        step_ms, step_tokens_s, current_rss_kb(), bad_loss, current_lr);
            }

            if ((s + 1) % PRINT_EVERY == 0 || s + 1 == steps_per_epoch) {
                printf("       step %4zu/%zu  loss=%8.4f  grad=%12.4f  lr=%.2e\r",
                       s + 1, steps_per_epoch,
                       loss_sum / (float)(s + 1), m->last_grad_norm, current_lr);
                fflush(stdout);
            }
        }
        printf("\n");

        double ms = elapsed_ms(&t0);
        double step_ms_avg = steps_per_epoch > 0 ? step_ms_sum / (double)steps_per_epoch : 0.0;
        double tokens_total = (double)steps_per_epoch * (double)BATCH_SIZE * (double)SEQ_LEN;
        double tokens_s = ms > 0.0 ? (tokens_total * 1000.0) / ms : 0.0;
        float  avg_loss = loss_sum / (float)steps_per_epoch;
        float  train_eval_loss = (train_ds.len >= seq_bytes)
                               ? evaluate_dataset(m, &train_ds, VAL_EVAL_STEPS)
                               : NAN;
        float  val_loss = (val_ds.len >= seq_bytes)
                        ? evaluate_dataset(m, &val_ds, VAL_EVAL_STEPS)
                        : NAN;
        float  param_norm;
        float  update_norm;

        kmamba_flatten_params(m, curr_params);
        param_norm = l2_norm_f32(curr_params, total_params);
        update_norm = l2_diff_norm_f32(curr_params, prev_params, total_params);
        memcpy(prev_params, curr_params, total_params * sizeof(float));

        printf("  %4d | %8.4f | %8.4f | %8.4f | %8.0f | %8.1f | %.2e\n",
               epoch, avg_loss, train_eval_loss, val_loss, tokens_s, ms, current_lr);
        fflush(stdout);

        if (epoch_log) {
            fprintf(epoch_log,
                    "%llu,%d,%zu,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.3f,%.3f,%.3f,%zu,%.6f,%.6f,%zu,%zu,%zu\n",
                    run_id, epoch, steps_per_epoch, tokens_total,
                    avg_loss, safe_perplexity(avg_loss),
                    train_eval_loss, safe_perplexity(train_eval_loss),
                    val_loss, safe_perplexity(val_loss),
                    m->last_grad_norm, m->last_grad_over_clip, m->last_grad_would_clip,
                    ms, step_ms_avg, tokens_s,
                    total_params, param_norm, update_norm,
                    current_rss_kb(), train_ds.len, val_ds.len);
        }

        /* Checkpoint périodique */
        if (ckpt_path && epoch % SAVE_EVERY == 0) {
            kmamba_save(m, ckpt_path);
            printf("         [checkpoint sauvegardé : %s]\n", ckpt_path);
        }

        if (bad_loss) {
            printf("         [warning] loss non finie détectée pendant l'epoch %d\n", epoch);
        }
    }

    /* Checkpoint final */
    const char *final_ckpt = ckpt_path ? ckpt_path : "kmamba_cpu.bin";
    kmamba_save(m, final_ckpt);
    printf("\n[checkpoint final : %s]\n", final_ckpt);

    /* --- Génération démonstrative --- */
    generate(m, "Les systemes", GEN_LEN);

    if (step_log) fclose(step_log);
    if (epoch_log) fclose(epoch_log);
    kmamba_free(m);
    free(ds.data);
    free(batch);
    free(prev_params);
    free(curr_params);
    free(owned_log_prefix);
    free(step_log_path);
    free(epoch_log_path);
    return 0;
}
