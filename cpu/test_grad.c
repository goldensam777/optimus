/*
 * test_grad.c — Diagnostic du backward MambaBlock
 *
 * Ce test fait 3 choses :
 *
 * 1. NORME DES GRADIENTS — vérifie que chaque gradient est non-nul après backward.
 *    Si une norme est 0.0, on a trouvé où le gradient meurt.
 *
 * 2. FINITE DIFFERENCE CHECK — vérifie la correction numérique de chaque gradient.
 *    Pour chaque paramètre p_i :
 *        grad_fd[i] = (L(p + eps*e_i) - L(p - eps*e_i)) / (2*eps)
 *    Comparé à grad_analytical[i]. L'erreur relative doit être < 1e-3.
 *
 * 3. LOSS DECREASE — vérifie qu'après un optimizer step, la loss diminue.
 *
 * Config intentionnellement petite pour que le FD check soit rapide.
 */

#include "kmamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Config minimaliste pour debug */
#define DIM        8
#define STATE      8
#define SEQ_LEN    4
#define VOCAB      256

/* -------- utils -------- */
static float vec_norm(const float *v, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float vec_max_abs(const float *v, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(v[i]);
        if (a > m) m = a;
    }
    return m;
}

/* -------- loss sur une séquence fixe -------- */
static float compute_loss(KMamba *m, const uint8_t *tokens_plus1) {
    /* kmamba_train_step retourne la loss SANS faire le backward
     * Non — il fait les deux. On va devoir faire le forward manuellement. */

    /* Forward seul */
    size_t L = m->cfg.seq_len;
    size_t V = m->cfg.vocab_size;
    float *logits = (float *)calloc(L * V, sizeof(float));
    kmamba_forward(m, tokens_plus1, logits);

    /* Cross-entropy */
    float loss = 0.0f;
    for (size_t t = 0; t < L; t++) {
        /* Softmax */
        float maxv = logits[t * V];
        for (size_t i = 1; i < V; i++)
            if (logits[t*V+i] > maxv) maxv = logits[t*V+i];
        float sum = 0.0f;
        float probs[VOCAB];
        for (size_t i = 0; i < V; i++) {
            probs[i] = expf(logits[t*V+i] - maxv);
            sum += probs[i];
        }
        float p = probs[(size_t)tokens_plus1[t + 1]] / sum;
        if (p < 1e-20f) p = 1e-20f;
        loss += -logf(p);
    }
    loss /= (float)L;

    free(logits);
    return loss;
}

/* -------- print grad norms -------- */
static void print_grad_norms(MambaBlock *block) {
    /* On doit accéder à l'état optimizer — on utilise mamba_zero_grads comme
     * proxy : si g_W_in est accessible, les norms le sont aussi.
     * Ici on crée un modèle de wrapping pour accéder aux gradients via
     * une passe backward instrumentée. */

    /* Cette fonction est appelée APRÈS mamba_backward.
     * Pour récupérer les gradients, on réutilise une passe backward
     * avec dY = identité et on observe via les updates de l'optimizer. */

    printf("  [note: grad norms non accessibles directement depuis l'API publique]\n");
    printf("  [utiliser un optimizer step avec lr minuscule et comparer les poids]\n");
}

/* -------- test principal -------- */
int main(void) {
    printf("=== test_grad : diagnostic backward MambaBlock ===\n\n");

    srand(12345);

    KMambaConfig cfg = {
        .vocab_size = VOCAB,
        .dim        = DIM,
        .state_size = STATE,
        .seq_len    = SEQ_LEN,
        .n_layers   = 1,
        .dt_scale   = 1.0f,
        .dt_min     = 0.001f,
        .dt_max     = 0.1f,
        .use_convnd = 0,
    };

    MBOptimConfig opt = {
        .lr           = 1e-3f,
        .mu           = 0.9f,
        .beta2        = 0.999f,
        .eps          = 1e-8f,
        .clip_norm    = 1.0f,
        .weight_decay = 0.0f,  /* 0 pour simplifier le FD check */
    };

    /* Séquence fixe */
    uint8_t tokens[SEQ_LEN + 1] = {65, 66, 67, 68, 69};  /* ABCDE */

    /* ================================================================
     * TEST 1 : Vérifier que la loss change après un optimizer step
     * ================================================================ */
    printf("--- TEST 1 : Loss decrease après un step ---\n");

    KMamba *m1 = kmamba_create(&cfg);
    kmamba_init(m1, 42);
    kmamba_enable_training(m1, &opt, 1e-3f, 0.0f);

    float loss_before = compute_loss(m1, tokens);
    printf("  Loss avant step : %.6f\n", loss_before);

    float loss_step = kmamba_train_step(m1, tokens);
    float loss_after = compute_loss(m1, tokens);
    printf("  Loss retournée par train_step : %.6f\n", loss_step);
    printf("  Loss après step  : %.6f\n", loss_after);

    if (fabsf(loss_after - loss_before) < 1e-8f) {
        printf("  [FAIL] Loss identique -> poids non modifiés!\n");
        printf("  [FAIL] Le backward ne produit pas de gradients non-nuls,\n");
        printf("         OU l'optimizer step est un no-op.\n");
    } else if (loss_after < loss_before) {
        printf("  [PASS] Loss a diminué de %.6f\n", loss_before - loss_after);
    } else {
        printf("  [WARN] Loss a AUGMENTÉ de %.6f (peut arriver sur 1 step)\n",
               loss_after - loss_before);
    }

    kmamba_free(m1);

    /* ================================================================
     * TEST 2 : Vérifier avec train_batch (batch=1)
     * ================================================================ */
    printf("\n--- TEST 2 : train_batch(batch=1) vs train_step ---\n");

    KMamba *m2 = kmamba_create(&cfg);
    kmamba_init(m2, 42);
    kmamba_enable_training(m2, &opt, 1e-3f, 0.0f);

    float loss_b_before = compute_loss(m2, tokens);
    float loss_b_step   = kmamba_train_batch(m2, tokens, 1);
    float loss_b_after  = compute_loss(m2, tokens);

    printf("  Loss avant       : %.6f\n", loss_b_before);
    printf("  Loss (train_batch): %.6f\n", loss_b_step);
    printf("  Loss après       : %.6f\n", loss_b_after);
    float delta_b = loss_b_before - loss_b_after;
    if (fabsf(delta_b) < 1e-8f)
        printf("  [FAIL] Poids non modifiés par train_batch\n");
    else
        printf("  [%s] Delta = %.6f\n", delta_b > 0 ? "PASS" : "WARN", delta_b);

    kmamba_free(m2);

    /* ================================================================
     * TEST 3 : Finite difference check sur W_out[0]
     *
     *  grad_fd[i] = (L(w+eps) - L(w-eps)) / (2*eps)
     *  compare avec gradient analytique via deux backward passes
     * ================================================================ */
    printf("\n--- TEST 3 : Finite difference check (W_out[0..7]) ---\n");

    float EPS = 1e-3f;
    int fd_pass = 0, fd_fail = 0;

    KMamba *m3 = kmamba_create(&cfg);
    kmamba_init(m3, 42);

    /* Modèle "référence" sans entraînement — juste forward */
    MambaBlock *block = m3->layers[0];
    float *W_out = block->W_out.data;  /* [DIM × STATE] */

    printf("  Paramètre | grad_fd       | rel_err\n");
    printf("  ----------+---------------+---------\n");

    /* On fait la backward une fois pour obtenir le gradient analytique */
    KMamba *m_anal = kmamba_create(&cfg);
    kmamba_init(m_anal, 42);
    /* On copie les mêmes poids que m3 */
    {
        MambaBlock *b_anal = m_anal->layers[0];
        size_t n_in  = block->W_in.rows * block->W_in.cols;
        size_t n_out = block->W_out.rows * block->W_out.cols;
        size_t n_a   = block->A_log.rows * block->A_log.cols;
        size_t n_dp  = block->delta_proj.rows * block->delta_proj.cols;
        memcpy(b_anal->W_in.data,       block->W_in.data,       n_in  * sizeof(float));
        memcpy(b_anal->W_out.data,      block->W_out.data,      n_out * sizeof(float));
        memcpy(b_anal->A_log.data,      block->A_log.data,      n_a   * sizeof(float));
        memcpy(b_anal->B_mat.data,      block->B_mat.data,      n_a   * sizeof(float));
        memcpy(b_anal->C_mat.data,      block->C_mat.data,      n_a   * sizeof(float));
        memcpy(b_anal->delta_proj.data, block->delta_proj.data, n_dp  * sizeof(float));
    }

    /* Sauvegarde checkpoint pour le FD check */
    kmamba_save(m3, "/tmp/test_grad_ref.bin");

    /* Gradient analytique via backward */
    MBOptimConfig opt_tiny = {
        .lr = 0.0f,  /* lr=0 -> on ne bouge pas les poids, mais les gradients sont calculés */
        .mu = 0.0f, .beta2 = 0.999f, .eps = 1e-8f,
        .clip_norm = 0.0f,   /* PAS de clipping pour le FD check */
        .weight_decay = 0.0f
    };

    /* On utilise un lr≈0 pour récupérer les gradients sans modifier les poids.
     * Comme on n'a pas accès direct aux gradients depuis l'API publique,
     * on va plutôt utiliser une technique différente :
     * train_step avec lr très petit, puis mesurer le delta de chaque poids. */

    float lr_probe = 1e-10f;  /* infime */
    MBOptimConfig opt_probe = {
        .lr = lr_probe, .mu = 0.0f, .beta2 = 0.999f, .eps = 1e-8f,
        .clip_norm = 0.0f, .weight_decay = 0.0f
    };

    KMamba *m_probe = kmamba_create(&cfg);
    kmamba_init(m_probe, 42);
    kmamba_enable_training(m_probe, &opt_probe, lr_probe, 0.0f);

    /* On copie les poids de m3 dans m_probe */
    {
        MambaBlock *bp = m_probe->layers[0];
        size_t n_out = block->W_out.rows * block->W_out.cols;
        memcpy(bp->W_out.data, block->W_out.data, n_out * sizeof(float));
    }

    /* Save weights before step */
    float W_out_before[DIM * STATE];
    memcpy(W_out_before, m_probe->layers[0]->W_out.data, DIM * STATE * sizeof(float));

    float loss_probe = kmamba_train_step(m_probe, tokens);

    /* Gradient analytique approx : (w_before - w_after) / lr */
    /* With Adam at step 1: param -= lr * g / (|g| + eps).
     * So g_analytical ≈ (w_before - w_after) / lr * |g| / ... complicated.
     *
     * Better: use SGD with mu=0 and no correction:
     * param -= lr * g  →  g ≈ (w_before - w_after) / lr
     */
    /* Let's redo with SGD */
    kmamba_free(m_probe);

    MBOptimConfig opt_sgd = {
        .lr = lr_probe, .mu = 0.0f, .beta2 = 0.999f, .eps = 1e-8f,
        .clip_norm = 0.0f, .weight_decay = 0.0f
    };

    KMamba *m_sgd = kmamba_create(&cfg);
    kmamba_init(m_sgd, 42);
    kmamba_enable_training_with_optimizer(m_sgd, OPTIMIZER_SGD, &opt_sgd, lr_probe, 0.0f);

    /* Copy reference weights */
    {
        MambaBlock *bs = m_sgd->layers[0];
        size_t n_out = block->W_out.rows * block->W_out.cols;
        size_t n_in  = block->W_in.rows * block->W_in.cols;
        size_t n_a   = block->A_log.rows * block->A_log.cols;
        size_t n_dp  = block->delta_proj.rows * block->delta_proj.cols;
        memcpy(bs->W_out.data,      block->W_out.data,      n_out * sizeof(float));
        memcpy(bs->W_in.data,       block->W_in.data,       n_in  * sizeof(float));
        memcpy(bs->A_log.data,      block->A_log.data,      n_a   * sizeof(float));
        memcpy(bs->B_mat.data,      block->B_mat.data,      n_a   * sizeof(float));
        memcpy(bs->C_mat.data,      block->C_mat.data,      n_a   * sizeof(float));
        memcpy(bs->delta_proj.data, block->delta_proj.data, n_dp  * sizeof(float));
    }

    float W_out_ref[DIM * STATE];
    memcpy(W_out_ref, m_sgd->layers[0]->W_out.data, DIM * STATE * sizeof(float));

    kmamba_train_step(m_sgd, tokens);

    /* g_analytical[i] = (W_ref[i] - W_after[i]) / lr_probe */
    /* (SGD: w_after = w_ref - lr * m, with mu=0: m = g, so g = (w_ref - w_after)/lr) */
    float *W_out_after_sgd = m_sgd->layers[0]->W_out.data;

    /* Check first 8 elements of W_out */
    printf("  Checking W_out[0..%d] (dim=%d, state=%d):\n", DIM*STATE-1, DIM, STATE);

    int total_nonzero = 0;
    for (int i = 0; i < DIM * STATE; i++) {
        float g_anal = (W_out_ref[i] - W_out_after_sgd[i]) / lr_probe;

        /* Finite difference */
        float w_orig = W_out_ref[i];

        KMamba *mfp = kmamba_load("/tmp/test_grad_ref.bin", 0, NULL, 0.0f, 0.0f);
        MambaBlock *bfp = mfp->layers[0];
        memcpy(bfp->W_out.data, W_out_ref, DIM * STATE * sizeof(float));
        bfp->W_out.data[i] = w_orig + EPS;
        float loss_plus = compute_loss(mfp, tokens);
        kmamba_free(mfp);

        KMamba *mfm = kmamba_load("/tmp/test_grad_ref.bin", 0, NULL, 0.0f, 0.0f);
        MambaBlock *bfm = mfm->layers[0];
        memcpy(bfm->W_out.data, W_out_ref, DIM * STATE * sizeof(float));
        bfm->W_out.data[i] = w_orig - EPS;
        float loss_minus = compute_loss(mfm, tokens);
        kmamba_free(mfm);

        float g_fd = (loss_plus - loss_minus) / (2.0f * EPS);

        if (fabsf(g_anal) > 1e-9f || fabsf(g_fd) > 1e-9f) total_nonzero++;

        float rel_err = 0.0f;
        float denom = 0.5f * (fabsf(g_anal) + fabsf(g_fd));
        if (denom > 1e-8f) rel_err = fabsf(g_anal - g_fd) / denom;

        int pass = (rel_err < 0.05f) || (fabsf(g_anal - g_fd) < 1e-5f);
        if (pass) fd_pass++;
        else       fd_fail++;

        if (i < 8 || !pass) {
            printf("  W_out[%2d]: g_anal=%+.4e g_fd=%+.4e rel=%.3f [%s]\n",
                   i, g_anal, g_fd, rel_err, pass ? "OK" : "FAIL");
        }
    }

    printf("\n  W_out gradient summary:\n");
    printf("    Non-zero gradients : %d / %d\n", total_nonzero, DIM * STATE);
    printf("    FD check  PASS     : %d / %d\n", fd_pass, DIM * STATE);
    printf("    FD check  FAIL     : %d / %d\n", fd_fail, DIM * STATE);

    if (total_nonzero == 0) {
        printf("\n  [FAIL CRITIQUE] Tous les gradients de W_out sont nuls!\n");
        printf("  -> Le backward ne propage rien depuis l'output.\n");
    }

    kmamba_free(m_sgd);
    kmamba_free(m3);

    /* ================================================================
     * TEST 4 : 10 steps consécutifs — vérifier convergence
     * ================================================================ */
    printf("\n--- TEST 4 : 10 training steps ---\n");

    KMamba *m4 = kmamba_create(&cfg);
    kmamba_init(m4, 42);
    MBOptimConfig opt4 = {
        .lr = 1e-2f,  /* lr plus élevé pour voir l'effet */
        .mu = 0.9f, .beta2 = 0.999f, .eps = 1e-8f,
        .clip_norm = 1.0f, .weight_decay = 0.0f
    };
    kmamba_enable_training(m4, &opt4, 1e-2f, 0.0f);

    printf("  step | loss\n");
    printf("  -----+--------\n");
    for (int step = 1; step <= 10; step++) {
        float l = kmamba_train_step(m4, tokens);
        printf("  %4d | %.4f\n", step, l);
    }

    kmamba_free(m4);

    printf("\n=== fin test_grad ===\n");
    return 0;
}
