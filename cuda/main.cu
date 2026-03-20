/*
 * main.c — Instance GPU (MX450 / sm_75)
 *
 * Architecture : TOUT sur GPU.
 *   embed   : lookup sur GPU
 *   forward : N × gpu_block_forward (Blelloch scan + cuBLAS GEMM)
 *   backward: N × gpu_block_backward
 *   optimizer: MUON (Newton-Schulz) + AdamW — déjà sur GPU (optimatrix-cuda)
 *
 * Aucun aller-retour CPU↔GPU pendant l'entraînement.
 * Seule la loss (scalaire) remonte sur CPU pour affichage.
 *
 * Usage :
 *   ./kmamba_cuda <data.txt>          — entraîne et sauvegarde
 *   ./kmamba_cuda gen  <ckpt> <prompt>
 *   ./kmamba_cuda chat <ckpt>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "optimatrix.h"   /* MBOptimConfig, adamw_update_cuda, muon_update_*_cuda */

/* ── Config modèle ────────────────────────────────────────────── */
#define VOCAB_SIZE    256
#define DIM           256
#define STATE_SIZE    512
#define N_LAYERS      5
#define SEQ_LEN       256
#define BATCH_SIZE    64
#define N_EPOCHS      1000
#define SAVE_EVERY    5

#define LR_BLOCKS     1e-4f
#define LR_EMBED_HEAD 1e-4f
#define WEIGHT_DECAY  1e-5f
#define CLIP_NORM     1.0f
#define MOMENTUM      0.9f
#define BETA2         0.999f
#define EPS           1e-8f
#define TEMPERATURE   0.8f
#define GEN_LEN       512
#define CHAT_MAX_RESP 256

/* ── Déclarations externes (mamba_block.cu) ─────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_block_forward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out,
    const float *d_A_log, const float *d_W_B, const float *d_W_C,
    const float *d_delta_proj, const float *d_theta, const float *d_lambda_proj,
    const float *d_x, float *d_y,
    float *d_u_raw, float *d_u,
    float *d_dt_raw, float *d_dt,
    float *d_B_exp, float *d_C_exp, float *d_dt_exp,
    float *d_h_store, float *d_y_scan, float *d_y_proj,
    float *d_lambda_raw, float *d_lambda,
    int L, int state, int dim);

void gpu_block_backward(
    cublasHandle_t cublas,
    const float *d_W_in, const float *d_W_out,
    const float *d_A_log,
    const float *d_W_B, const float *d_W_C,
    const float *d_delta_proj, const float *d_theta, const float *d_lambda_proj,
    const float *d_x,
    const float *d_u_raw, const float *d_u,
    const float *d_dt_raw, const float *d_dt,
    const float *d_B_exp, const float *d_C_exp, const float *d_dt_exp,
    const float *d_h_store, const float *d_y_scan,
    const float *d_lambda,
    const float *d_dy,
    float *d_dW_in, float *d_dW_out, float *d_dA_log,
    float *d_dW_B, float *d_dW_C, float *d_ddelta_proj,
    float *d_g_theta, float *d_g_lambda_proj,
    float *d_dx,
    float *d_dy_scan, float *d_du, float *d_du_raw,
    float *d_ddt, float *d_ddt_raw,
    float *d_dB_scan, float *d_dC_scan, float *d_ddt_scan,
    float *d_dA_tmp, float *d_dlambda, float *d_dlambda_raw,
    int L, int state, int dim);

#ifdef __cplusplus
}
#endif

/* ── Macros CUDA ─────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { cudaError_t _e=(x); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(_e));exit(1);} } while(0)

#define CUBLAS_CHECK(x) do { cublasStatus_t _s=(x); \
    if(_s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error %d at %s:%d\n", \
    _s,__FILE__,__LINE__);exit(1);} } while(0)

/* ── Paramètres d'un bloc (tout en VRAM) ────────────────────── */
typedef struct {
    /* Paramètres */
    float *W_in,  *W_out;
    float *A_log, *W_B, *W_C;
    float *delta_proj;
    float *theta;          /* [state/2] rotation angles */
    float *lambda_proj;    /* [dim] exp-trapezoidal lambda projection */
    /* Gradients */
    float *g_W_in,  *g_W_out;
    float *g_A_log, *g_W_B, *g_W_C;
    float *g_delta_proj;
    float *g_theta;        /* [state/2] */
    float *g_lambda_proj;  /* [dim] */
    /* Momentum (MUON) */
    float *m_W_in,  *m_W_out;
    float *m_A_log, *m_W_B, *m_W_C;
    float *m_delta_proj;
    float *m_lambda_proj;  /* [dim] */
    float *m_theta;        /* [state/2] */
    /* Variance (AdamW pour embed/head) */
    float *v_W_in,  *v_W_out;
} GpuBlock;

/* ── Workspace forward/backward par layer ───────────────────── */
typedef struct {
    float *u_raw, *u;
    float *dt_raw, *dt;
    float *B_exp, *C_exp, *dt_exp;
    float *h_store, *y_scan, *y_proj;
    float *lambda_raw, *lambda;  /* [L] for exp-trapezoidal */
    /* backward */
    float *dy_scan, *du, *du_raw;
    float *ddt, *ddt_raw;
    float *dB_scan, *dC_scan, *ddt_scan;
    float *dA_tmp;
    float *dlambda, *dlambda_raw;  /* [L] */
    /* input de la couche sauvé pour backward */
    float *layer_input;
} GpuWorkspace;

/* ── Modèle GPU complet ──────────────────────────────────────── */
typedef struct {
    int vocab, dim, state, n_layers, seq_len;
    /* Embedding + head */
    float *d_embed;    /* [vocab, dim] */
    float *d_head;     /* [vocab, dim] */
    float *d_g_embed, *d_g_head;
    float *d_m_embed,  *d_m_head;
    float *d_v_embed,  *d_v_head;
    /* Blocs */
    GpuBlock     *blocks;
    GpuWorkspace *ws;
    /* Buffers partagés (réutilisés par batch element) */
    float *d_hidden;   /* [seq_len, dim] sortie courante */
    float *d_dy;       /* [seq_len, dim] gradient remonté */
    /* Logits + loss */
    float *d_logits;   /* [seq_len, vocab] */
    float *d_dlogits;  /* [seq_len, vocab] */
    float *d_loss;     /* [1] scalaire sur GPU */
    /* Tokens sur GPU */
    int   *d_tokens;   /* [seq_len] */
    cublasHandle_t cublas;
    size_t step;
} GpuModel;

/* ── Kernels GPU pour embed, loss ────────────────────────────── */

/* Embedding lookup forward : hidden[t,d] = embed[token[t], d] */
static __global__ void embed_fwd_kernel(
    const float *embed, const int *tokens, float *out,
    int L, int D)
{
    int t = blockIdx.x, d = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= L || d >= D) return;
    out[t * D + d] = embed[tokens[t] * D + d];
}

/* Embedding backward : grad_embed[token[t], d] += dy[t, d] */
static __global__ void embed_bwd_kernel(
    float *g_embed, const int *tokens, const float *dy,
    int L, int D)
{
    int t = blockIdx.x, d = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= L || d >= D) return;
    atomicAdd(&g_embed[tokens[t] * D + d], dy[t * D + d]);
}

/* Softmax + cross-entropy loss forward + backward (fused)
 * logits [L, V] → probs → loss scalaire
 * dlogits[t, v] = (probs[t,v] - (v==target[t])) / L  */
static __global__ void softmax_ce_kernel(
    const float *logits, const int *targets,
    float *dlogits, float *loss_acc,
    int L, int V)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;

    const float *row = logits  + t * V;
    float       *drow = dlogits + t * V;

    /* max pour stabilité numérique */
    float mx = row[0];
    for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];

    float sum = 0.0f;
    for (int v = 0; v < V; v++) sum += expf(row[v] - mx);

    int tgt = targets[t];
    float log_prob = row[tgt] - mx - logf(sum);
    atomicAdd(loss_acc, -log_prob);

    float inv_sum = 1.0f / sum;
    for (int v = 0; v < V; v++) {
        float p = expf(row[v] - mx) * inv_sum;
        drow[v] = (p - (float)(v == tgt)) / (float)L;
    }
}

/* head backward : d_dy [L, dim] = dlogits [L, V] @ head [V, dim] */
/* head grad : d_g_head [V, dim] += dlogits^T @ hidden [L, dim]   */
/* (ces deux GEMMs sont faits via cuBLAS dans la fonction training) */

/* ── Utilitaires ─────────────────────────────────────────────── */

static float *gpu_alloc(size_t n) {
    float *p; CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(float))); return p;
}
static int *gpu_alloc_int(size_t n) {
    int *p; CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(int))); return p;
}
static void gpu_zero(float *p, size_t n) {
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
}

/* Xavier init sur CPU, copie sur GPU */
static void xavier_init_gpu(float *d_p, int fan_in, int fan_out, size_t n,
                             unsigned int *seed) {
    float *h = (float *)malloc(n * sizeof(float));
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (size_t i = 0; i < n; i++) {
        float u1 = ((float)(*seed = *seed * 1664525u + 1013904223u) + 1.0f)
                   / 4294967296.0f;
        float u2 = ((float)(*seed = *seed * 1664525u + 1013904223u) + 1.0f)
                   / 4294967296.0f;
        h[i] = scale * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    CUDA_CHECK(cudaMemcpy(d_p, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

/* Spacing négatif pour A_log (stabilité SSM) */
static void a_log_init_gpu(float *d_p, int state, unsigned int *seed) {
    float *h = (float *)malloc(state * sizeof(float));
    for (int i = 0; i < state; i++) {
        float r = ((float)(*seed = *seed * 1664525u + 1013904223u) + 1.0f)
                  / 4294967296.0f;
        h[i] = -(1.0f + r);  /* A_log in [-2, -1] */
    }
    CUDA_CHECK(cudaMemcpy(d_p, h, state * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

/* cuBLAS GEMM row-major : C[M,N] = alpha*A[M,K]@B[K,N] + beta*C */
static void gemm_gpu(cublasHandle_t h, int M, int N, int K,
                     float alpha, const float *A, const float *B,
                     float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}
/* C[M,N] = alpha * A^T @ B  (A=[K,M], B=[K,N]) */
static void gemm_gpu_at(cublasHandle_t h, int M, int N, int K,
                        float alpha, const float *A, const float *B,
                        float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

/* ── Allocation du modèle ─────────────────────────────────────── */

static GpuModel *gpu_model_create(void) {
    GpuModel *m = (GpuModel *)calloc(1, sizeof(GpuModel));
    m->vocab    = VOCAB_SIZE;
    m->dim      = DIM;
    m->state    = STATE_SIZE;
    m->n_layers = N_LAYERS;
    m->seq_len  = SEQ_LEN;

    CUBLAS_CHECK(cublasCreate(&m->cublas));

    int V = m->vocab, D = m->dim, S = m->state, L = m->seq_len;

    /* Embedding + head */
    m->d_embed   = gpu_alloc(V * D); m->d_g_embed = gpu_alloc(V * D);
    m->d_m_embed = gpu_alloc(V * D); m->d_v_embed = gpu_alloc(V * D);
    m->d_head    = gpu_alloc(V * D); m->d_g_head  = gpu_alloc(V * D);
    m->d_m_head  = gpu_alloc(V * D); m->d_v_head  = gpu_alloc(V * D);
    gpu_zero(m->d_m_embed, V*D); gpu_zero(m->d_v_embed, V*D);
    gpu_zero(m->d_m_head,  V*D); gpu_zero(m->d_v_head,  V*D);

    /* Blocs */
    m->blocks = (GpuBlock *)calloc(m->n_layers, sizeof(GpuBlock));
    m->ws     = (GpuWorkspace *)calloc(m->n_layers, sizeof(GpuWorkspace));

    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock     *b = &m->blocks[i];
        GpuWorkspace *w = &m->ws[i];

        int TS = S / 2; if (TS < 1) TS = 1; /* theta size = state/2 */
        b->W_in  = gpu_alloc(S * D); b->g_W_in  = gpu_alloc(S * D);
        b->m_W_in= gpu_alloc(S * D); b->v_W_in  = gpu_alloc(S * D);
        b->W_out = gpu_alloc(D * S); b->g_W_out = gpu_alloc(D * S);
        b->m_W_out= gpu_alloc(D*S);  b->v_W_out = gpu_alloc(D * S);
        b->A_log = gpu_alloc(S);     b->g_A_log = gpu_alloc(S);
        b->m_A_log= gpu_alloc(S);
        b->W_B   = gpu_alloc(S*D);   b->g_W_B   = gpu_alloc(S*D);
        b->m_W_B = gpu_alloc(S*D);
        b->W_C   = gpu_alloc(S*D);   b->g_W_C   = gpu_alloc(S*D);
        b->m_W_C = gpu_alloc(S*D);
        b->delta_proj = gpu_alloc(D); b->g_delta_proj = gpu_alloc(D);
        b->m_delta_proj = gpu_alloc(D);
        b->theta       = gpu_alloc(TS);  b->g_theta       = gpu_alloc(TS);
        b->m_theta     = gpu_alloc(TS);
        b->lambda_proj = gpu_alloc(D);   b->g_lambda_proj = gpu_alloc(D);
        b->m_lambda_proj = gpu_alloc(D);

        /* Zéro des moments */
        gpu_zero(b->m_W_in, S*D); gpu_zero(b->v_W_in, S*D);
        gpu_zero(b->m_W_out, D*S); gpu_zero(b->v_W_out, D*S);
        gpu_zero(b->m_A_log, S); gpu_zero(b->m_W_B, S*D);
        gpu_zero(b->m_W_C, S*D); gpu_zero(b->m_delta_proj, D);
        gpu_zero(b->m_theta, TS); gpu_zero(b->m_lambda_proj, D);

        /* Workspace forward */
        w->u_raw    = gpu_alloc(L * S); w->u       = gpu_alloc(L * S);
        w->dt_raw   = gpu_alloc(L);     w->dt      = gpu_alloc(L);
        w->B_exp    = gpu_alloc(L * S); w->C_exp   = gpu_alloc(L * S);
        w->dt_exp   = gpu_alloc(L * S); w->h_store = gpu_alloc(L * S);
        w->y_scan   = gpu_alloc(L * S); w->y_proj  = gpu_alloc(L * D);
        /* Workspace backward */
        w->dy_scan  = gpu_alloc(L * S); w->du      = gpu_alloc(L * S);
        w->du_raw   = gpu_alloc(L * S); w->ddt     = gpu_alloc(L);
        w->ddt_raw  = gpu_alloc(L);
        w->dB_scan  = gpu_alloc(L * S); w->dC_scan = gpu_alloc(L * S);
        w->ddt_scan = gpu_alloc(L * S); w->dA_tmp  = gpu_alloc(S);
        w->dlambda      = gpu_alloc(L); w->dlambda_raw = gpu_alloc(L);
        w->lambda_raw   = gpu_alloc(L); w->lambda      = gpu_alloc(L);
        /* Input de chaque couche (sauvé pour backward) */
        w->layer_input = gpu_alloc(L * D);
    }

    /* Buffers partagés */
    m->d_hidden  = gpu_alloc(L * D);
    m->d_dy      = gpu_alloc(L * D);
    m->d_logits  = gpu_alloc(L * V);
    m->d_dlogits = gpu_alloc(L * V);
    CUDA_CHECK(cudaMalloc((void**)&m->d_loss, sizeof(float)));
    m->d_tokens  = gpu_alloc_int(L + 1); /* +1 : on copie seq_len+1 tokens (input+target) */

    return m;
}

static void gpu_model_init(GpuModel *m) {
    unsigned int seed = 42;
    int V = m->vocab, D = m->dim, S = m->state;

    /* Embedding : Xavier [V, D] */
    xavier_init_gpu(m->d_embed, V, D, V * D, &seed);
    xavier_init_gpu(m->d_head,  D, V, V * D, &seed);
    gpu_zero(m->d_g_embed, V*D); gpu_zero(m->d_g_head, V*D);

    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        int TS = S / 2; if (TS < 1) TS = 1;
        xavier_init_gpu(b->W_in,       S, D, S * D, &seed);
        xavier_init_gpu(b->W_out,      D, S, D * S, &seed);
        a_log_init_gpu(b->A_log, S, &seed);
        xavier_init_gpu(b->W_B,        S, D, S*D,   &seed);
        xavier_init_gpu(b->W_C,        S, D, S*D,   &seed);
        xavier_init_gpu(b->delta_proj, 1, D, D,     &seed);
        /* Init theta small ~ 2π/S */
        {
            float *h_theta = (float *)malloc(TS * sizeof(float));
            float base = 6.2831853f / (float)S;
            for (int ti = 0; ti < TS; ti++) {
                float r = ((float)(*seed = *seed * 1664525u + 1013904223u) + 1.0f) / 4294967296.0f;
                h_theta[ti] = r * base;
            }
            CUDA_CHECK(cudaMemcpy(b->theta, h_theta, TS * sizeof(float),
                                  cudaMemcpyHostToDevice));
            free(h_theta);
        }
        xavier_init_gpu(b->lambda_proj, 1, D, D, &seed);
        /* Zéro des gradients */
        gpu_zero(b->g_W_in, S*D); gpu_zero(b->g_W_out, D*S);
        gpu_zero(b->g_A_log, S);  gpu_zero(b->g_W_B, S*D);
        gpu_zero(b->g_W_C, S*D);  gpu_zero(b->g_delta_proj, D);
        gpu_zero(b->g_theta, TS);  gpu_zero(b->g_lambda_proj, D);
    }
}

static void gpu_zero_grads(GpuModel *m) {
    int V = m->vocab, D = m->dim, S = m->state;
    int TS = S / 2; if (TS < 1) TS = 1;
    gpu_zero(m->d_g_embed, V*D); gpu_zero(m->d_g_head, V*D);
    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        gpu_zero(b->g_W_in,  S*D); gpu_zero(b->g_W_out,  D*S);
        gpu_zero(b->g_A_log, S);   gpu_zero(b->g_W_B,   S*D);
        gpu_zero(b->g_W_C,  S*D);  gpu_zero(b->g_delta_proj, D);
        gpu_zero(b->g_theta, TS);   gpu_zero(b->g_lambda_proj, D);
    }
}

/* ── Forward + backward sur un exemple (loss accumulée) ────────── */

static float gpu_forward_backward(GpuModel *m,
                                   const uint8_t *tokens_cpu, int L) {
    int V = m->vocab, D = m->dim, S = m->state;
    cublasHandle_t h = m->cublas;

    /* Copier les tokens sur GPU */
    int *h_tokens_int = (int *)malloc(L * sizeof(int));
    for (int i = 0; i < L; i++) h_tokens_int[i] = (int)tokens_cpu[i];
    CUDA_CHECK(cudaMemcpy(m->d_tokens, h_tokens_int, L * sizeof(int),
                          cudaMemcpyHostToDevice));
    free(h_tokens_int);

    /* L_in = nb de tokens en entrée du modèle (= seq_len)
     * d_tokens[0..L_in-1] = inputs, d_tokens[1..L_in] = targets */
    int L_in = L - 1;

    /* ── FORWARD ─────────────────────────────────────────────── */

    /* Embedding lookup : L_in positions */
    dim3 bg(L_in, (D + 255) / 256), tg(256, 1);
    embed_fwd_kernel<<<bg, tg>>>(m->d_embed, m->d_tokens, m->d_hidden, L_in, D);

    /* N blocs Mamba */
    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock     *b = &m->blocks[i];
        GpuWorkspace *w = &m->ws[i];
        /* Sauvegarder l'entrée de la couche pour le backward ET le résiduel */
        CUDA_CHECK(cudaMemcpy(w->layer_input, m->d_hidden, L_in * D * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        /* x = layer_input (copie), y = d_hidden (résultat)
         * IMPORTANT : x != y pour que le résiduel y = y_proj + x soit correct */
        gpu_block_forward(
            h, b->W_in, b->W_out, b->A_log, b->W_B, b->W_C, b->delta_proj,
            b->theta, b->lambda_proj,
            w->layer_input, m->d_hidden,
            w->u_raw, w->u, w->dt_raw, w->dt,
            w->B_exp, w->C_exp, w->dt_exp,
            w->h_store, w->y_scan, w->y_proj,
            w->lambda_raw, w->lambda,
            L_in, S, D);
    }

    /* Head : logits [L_in, V] = hidden [L_in, D] @ head^T [D, V] */
    {
        float a1 = 1.0f, b0 = 0.0f;
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                                 V, L_in, D, &a1, m->d_head, D,
                                 m->d_hidden, D, &b0, m->d_logits, V));
    }

    /* Loss + dlogits */
    CUDA_CHECK(cudaMemset(m->d_loss, 0, sizeof(float)));
    /* targets = tokens décalés de 1 */
    int *d_targets = m->d_tokens + 1;  /* tokens[1..L_in] comme targets[0..L_in-1] */
    int L_loss = L_in;
    {
        int blk = (L_loss + 255) / 256;
        softmax_ce_kernel<<<blk, 256>>>(m->d_logits, d_targets,
                                        m->d_dlogits, m->d_loss,
                                        L_loss, V);
    }
    /* Récupérer la loss */
    float loss_h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&loss_h, m->d_loss, sizeof(float),
                          cudaMemcpyDeviceToHost));
    loss_h /= (float)L_loss;

    /* ── BACKWARD ────────────────────────────────────────────── */

    /* Gradient sur head : g_head [V,D] += dlogits^T @ hidden [L_in,D] */
    gemm_gpu_at(h, V, D, L_in, 1.0f, m->d_dlogits, m->d_hidden, 1.0f, m->d_g_head);

    /* Gradient vers hidden : d_dy [L_in,D] = dlogits [L_in,V] @ head [V,D] */
    gemm_gpu(h, L_in, D, V, 1.0f, m->d_dlogits, m->d_head, 0.0f, m->d_dy);

    /* N blocs Mamba (backward, ordre inverse) */
    for (int i = m->n_layers - 1; i >= 0; i--) {
        GpuBlock     *b = &m->blocks[i];
        GpuWorkspace *w = &m->ws[i];
        gpu_block_backward(
            h,
            b->W_in, b->W_out, b->A_log, b->W_B, b->W_C, b->delta_proj,
            b->theta, b->lambda_proj,
            w->layer_input,
            w->u_raw, w->u, w->dt_raw, w->dt,
            w->B_exp, w->C_exp, w->dt_exp,
            w->h_store, w->y_scan,
            w->lambda,
            m->d_dy,
            b->g_W_in, b->g_W_out, b->g_A_log,
            b->g_W_B, b->g_W_C, b->g_delta_proj,
            b->g_theta, b->g_lambda_proj,
            m->d_hidden,
            w->dy_scan, w->du, w->du_raw,
            w->ddt, w->ddt_raw,
            w->dB_scan, w->dC_scan, w->ddt_scan,
            w->dA_tmp, w->dlambda, w->dlambda_raw,
            L_in, S, D);
        /* Préparer le gradient pour la couche suivante */
        CUDA_CHECK(cudaMemcpy(m->d_dy, m->d_hidden, L_in * D * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    /* Backward embedding */
    embed_bwd_kernel<<<bg, tg>>>(m->d_g_embed, m->d_tokens, m->d_dy, L_in, D);

    return loss_h;
}

/* ── Optimizer step ────────────────────────────────────────────── */

static void gpu_optimizer_step(GpuModel *m, const MBOptimConfig *conf_blk,
                                const MBOptimConfig *conf_eh) {
    m->step++;
    int V = m->vocab, D = m->dim, S = m->state;

    /* Embed + head : AdamW */
    adamw_update_cuda(m->d_embed, m->d_g_embed, m->d_m_embed, m->d_v_embed,
                      V * D, conf_eh, m->step);
    adamw_update_cuda(m->d_head,  m->d_g_head,  m->d_m_head,  m->d_v_head,
                      V * D, conf_eh, m->step);

    /* Blocs : MUON device-native (paramètres déjà en VRAM) */
    int TS = S / 2; if (TS < 1) TS = 1;
    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        muon_update_mat_device(b->W_in,  b->g_W_in,  b->m_W_in,  S, D, conf_blk);
        muon_update_mat_device(b->W_out, b->g_W_out, b->m_W_out, D, S, conf_blk);
        muon_update_vec_device(b->A_log,      b->g_A_log,      b->m_A_log,      S, conf_blk);
        muon_update_mat_device(b->W_B,  b->g_W_B,  b->m_W_B,  S, D, conf_blk);
        muon_update_mat_device(b->W_C,  b->g_W_C,  b->m_W_C,  S, D, conf_blk);
        muon_update_vec_device(b->delta_proj, b->g_delta_proj, b->m_delta_proj, D, conf_blk);
        muon_update_vec_device(b->theta,      b->g_theta,      b->m_theta,      TS, conf_blk);
    }
}

/* ── Checkpoint save/load ─────────────────────────────────────── */

#define CKPT_MAGIC 0x4B4D4755  /* "KMGU" */
#define CKPT_VER   4           /* v4: adds theta (complex SSM angles) */

static void save_param(FILE *f, float *d_p, size_t n) {
    float *h = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h, d_p, n * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h, sizeof(float), n, f);
    free(h);
}
static int load_param(FILE *f, float *d_p, size_t n) {
    float *h = (float *)malloc(n * sizeof(float));
    if (fread(h, sizeof(float), n, f) != n) { free(h); return 0; }
    CUDA_CHECK(cudaMemcpy(d_p, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    return 1;
}

static void gpu_model_save(GpuModel *m, const char *path, int epoch) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[erreur] impossible d'écrire %s\n", path); return; }
    int V = m->vocab, D = m->dim, S = m->state;
    uint32_t magic = CKPT_MAGIC, ver = CKPT_VER;
    fwrite(&magic, 4, 1, f);
    fwrite(&ver,   4, 1, f);
    fwrite(&epoch, 4, 1, f);
    int cfg[5] = {V, D, S, m->n_layers, m->seq_len};
    fwrite(cfg, 4, 5, f);
    save_param(f, m->d_embed, V * D);
    save_param(f, m->d_head,  V * D);
    int TS = S / 2; if (TS < 1) TS = 1;
    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        save_param(f, b->W_in,  S * D); save_param(f, b->W_out,  D * S);
        save_param(f, b->A_log, S);     save_param(f, b->W_B,   S * D);
        save_param(f, b->W_C,  S * D); save_param(f, b->delta_proj, D);
        save_param(f, b->theta, TS);
        save_param(f, b->m_W_in, S*D);  save_param(f, b->m_W_out, D*S);
        save_param(f, b->m_A_log, S);   save_param(f, b->m_W_B, S*D);
        save_param(f, b->m_W_C, S*D);   save_param(f, b->m_delta_proj, D);
        save_param(f, b->m_theta, TS);
    }
    save_param(f, m->d_m_embed, V*D); save_param(f, m->d_m_head, V*D);
    fwrite(&m->step, sizeof(size_t), 1, f);
    fclose(f);
}

static int gpu_model_load(GpuModel *m, const char *path, int *epoch_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int V = m->vocab, D = m->dim, S = m->state;
    uint32_t magic, ver; int epoch;
    if (fread(&magic, 4, 1, f) != 1 || magic != CKPT_MAGIC) { fclose(f); return 0; }
    fread(&ver, 4, 1, f);
    if (ver != CKPT_VER) {
        fprintf(stderr, "[erreur checkpoint] version %u != attendue %u\n", ver, CKPT_VER);
        fclose(f); return 0;
    }
    fread(&epoch, 4, 1, f);
    int cfg[5]; fread(cfg, 4, 5, f);
    if (cfg[0]!=V || cfg[1]!=D || cfg[2]!=S || cfg[3]!=m->n_layers) {
        fprintf(stderr, "[erreur checkpoint] architecture incompatible\n");
        fclose(f); return 0;
    }
    if (!load_param(f, m->d_embed, V*D) || !load_param(f, m->d_head, V*D)) {
        fclose(f); return 0;
    }
    int TS = S / 2; if (TS < 1) TS = 1;
    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        if (!load_param(f,b->W_in,S*D)   ||!load_param(f,b->W_out,D*S)||
            !load_param(f,b->A_log,S)    ||!load_param(f,b->W_B,S*D)  ||
            !load_param(f,b->W_C,S*D)    ||!load_param(f,b->delta_proj,D)||
            !load_param(f,b->theta,TS)   ||
            !load_param(f,b->m_W_in,S*D) ||!load_param(f,b->m_W_out,D*S)||
            !load_param(f,b->m_A_log,S)  ||!load_param(f,b->m_W_B,S*D)||
            !load_param(f,b->m_W_C,S*D)  ||!load_param(f,b->m_delta_proj,D)||
            !load_param(f,b->m_theta,TS)) {
            fclose(f); return 0;
        }
    }
    load_param(f, m->d_m_embed, V*D); load_param(f, m->d_m_head, V*D);
    fread(&m->step, sizeof(size_t), 1, f);
    fclose(f);
    if (epoch_out) *epoch_out = epoch;
    return 1;
}

/* ── Génération d'un token ─────────────────────────────────────── */

/* Sampler avec température */
static int sample_token(float *logits_h, int V, float temp) {
    float mx = logits_h[0];
    for (int i = 1; i < V; i++) if (logits_h[i] > mx) mx = logits_h[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        logits_h[i] = expf((logits_h[i] - mx) / temp);
        sum += logits_h[i];
    }
    double r = (double)rand() / (RAND_MAX + 1.0) * sum;
    for (int i = 0; i < V; i++) {
        r -= logits_h[i];
        if (r <= 0.0) return i;
    }
    return V - 1;
}

static int gpu_generate_token(GpuModel *m, const uint8_t *ctx, int ctx_len,
                               float temperature) {
    int L = ctx_len < m->seq_len ? ctx_len : m->seq_len;
    int D = m->dim, V = m->vocab, S = m->state;
    cublasHandle_t h = m->cublas;

    int *h_tok = (int *)malloc(L * sizeof(int));
    for (int i = 0; i < L; i++) h_tok[i] = ctx[ctx_len - L + i];
    CUDA_CHECK(cudaMemcpy(m->d_tokens, h_tok, L * sizeof(int),
                          cudaMemcpyHostToDevice));
    free(h_tok);

    dim3 bg(L, (D + 255) / 256), tg(256, 1);
    embed_fwd_kernel<<<bg, tg>>>(m->d_embed, m->d_tokens, m->d_hidden, L, D);

    for (int i = 0; i < m->n_layers; i++) {
        GpuBlock *b = &m->blocks[i];
        GpuWorkspace *w = &m->ws[i];
        CUDA_CHECK(cudaMemcpy(w->layer_input, m->d_hidden, L*D*sizeof(float),
                              cudaMemcpyDeviceToDevice));
        gpu_block_forward(
            h, b->W_in, b->W_out, b->A_log, b->W_B, b->W_C, b->delta_proj,
            b->theta, b->lambda_proj,
            w->layer_input, m->d_hidden,
            w->u_raw, w->u, w->dt_raw, w->dt,
            w->B_exp, w->C_exp, w->dt_exp,
            w->h_store, w->y_scan, w->y_proj,
            w->lambda_raw, w->lambda,
            L, S, D);
    }

    /* Logit du dernier token : hidden[(L-1)*D ..] @ head^T */
    float *d_last = m->d_hidden + (L - 1) * D;
    {
        float a1 = 1.0f, b0 = 0.0f;
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                                 V, 1, D, &a1, m->d_head, D,
                                 d_last, D, &b0, m->d_logits, V));
    }

    float *logits_h = (float *)malloc(V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(logits_h, m->d_logits, V * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int tok = sample_token(logits_h, V, temperature);
    free(logits_h);
    return tok;
}

/* ── Chargement des données ────────────────────────────────────── */

static uint8_t *load_data(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[erreur] impossible d'ouvrir %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); *out_len = ftell(f); rewind(f);
    uint8_t *buf = (uint8_t *)malloc(*out_len);
    fread(buf, 1, *out_len, f); fclose(f);
    return buf;
}

/* ── Entraînement ──────────────────────────────────────────────── */

static void train(GpuModel *m, const char *data_path, const char *ckpt_path) {
    size_t data_len = 0;
    uint8_t *data = load_data(data_path, &data_len);
    if (!data) return;

    int L  = m->seq_len + 1;   /* +1 pour le token cible */
    int steps = (int)((data_len - 1) / m->seq_len);
    if (steps == 0) { fprintf(stderr, "[erreur] données trop courtes\n"); free(data); return; }
    int steps_per_epoch = steps / BATCH_SIZE;
    if (steps_per_epoch == 0) steps_per_epoch = 1;

    printf("\n[données] %s — %zu bytes\n\n", data_path, data_len);
    printf("[modèle initialisé (Xavier, seed=42)]\n");
    printf("[optimizer : MUON GPU (Newton-Schulz, MX450 sm_75)]\n\n");
    printf("entraînement : %d epochs × %d steps × batch=%d\n\n",
           N_EPOCHS, steps_per_epoch, BATCH_SIZE);
    printf(" epoch |   loss   |  ms/epoch\n");
    printf("-------+----------+-----------\n");
    fflush(stdout);

    MBOptimConfig conf_blk = {LR_BLOCKS,     MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};
    MBOptimConfig conf_eh  = {LR_EMBED_HEAD, MOMENTUM, BETA2, EPS, CLIP_NORM, WEIGHT_DECAY};

    int start_epoch = 0;
    if (ckpt_path && gpu_model_load(m, ckpt_path, &start_epoch)) {
        printf("[checkpoint : %s (epoch %d)]\n", ckpt_path, start_epoch);
    }

    uint8_t *seq = (uint8_t *)malloc(L * sizeof(uint8_t));

    for (int epoch = start_epoch + 1; epoch <= N_EPOCHS; epoch++) {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        double epoch_loss = 0.0;

        for (int step = 0; step < steps_per_epoch; step++) {
            gpu_zero_grads(m);
            float batch_loss = 0.0f;

            for (int b = 0; b < BATCH_SIZE; b++) {
                size_t off = (size_t)rand() % (data_len - L);
                memcpy(seq, data + off, L);
                batch_loss += gpu_forward_backward(m, seq, L);
            }
            batch_loss /= (float)BATCH_SIZE;
            epoch_loss += batch_loss;

            gpu_optimizer_step(m, &conf_blk, &conf_eh);

            if ((step + 1) % 50 == 0 || step == steps_per_epoch - 1) {
                printf("       step %4d/%d  loss=%.4f\r",
                       step + 1, steps_per_epoch, batch_loss);
                fflush(stdout);
            }
        }
        epoch_loss /= steps_per_epoch;

        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

        printf(" %5d | %8.4f | %9.1f\n", epoch, epoch_loss, ms);
        fflush(stdout);

        if (ckpt_path && epoch % SAVE_EVERY == 0)
            gpu_model_save(m, ckpt_path, epoch);
    }

    if (ckpt_path) gpu_model_save(m, ckpt_path, N_EPOCHS);
    free(seq); free(data);
}

/* ── REPL chat ─────────────────────────────────────────────────── */

static void chat_repl(GpuModel *m) {
    printf("\n  k-mamba GPU — session interactive\n");
    printf("  Ctrl+D ou 'quit' pour quitter\n");
    printf("  Format dataset : [speaker001:] / [speaker002:]\n");
    printf("  ─────────────────────────────\n\n");

    /* Buffer linéaire — quand plein, on décale d'une fenêtre */
    int CTX_MAX = m->seq_len;
    uint8_t *ctx = (uint8_t *)malloc(CTX_MAX);
    int ctx_len = 0;
    char line[512];

    /* Injecter dans ctx (décale si plus de place) */
    #define CTX_PUSH(byte) do { \
        if (ctx_len < CTX_MAX) { ctx[ctx_len++] = (uint8_t)(byte); } \
        else { memmove(ctx, ctx + CTX_MAX/2, CTX_MAX/2); \
               ctx_len = CTX_MAX/2; ctx[ctx_len++] = (uint8_t)(byte); } \
    } while(0)

    while (1) {
        printf("[speaker001:] "); fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t llen = strlen(line);
        if (llen > 0 && line[llen-1] == '\n') line[--llen] = '\0';
        if (strcmp(line, "quit") == 0) break;
        if (llen == 0) continue;

        /* Format du dataset : "[speaker001:] <message>\n[speaker002:] " */
        const char *pfx = "[speaker001:] ";
        const char *sfx = "\n[speaker002:] ";
        for (const char *p = pfx; *p; p++) CTX_PUSH(*p);
        for (size_t i = 0; i < llen; i++) CTX_PUSH(line[i]);
        for (const char *p = sfx; *p; p++) CTX_PUSH(*p);

        printf("[speaker002:] ");
        for (int i = 0; i < CHAT_MAX_RESP; i++) {
            int tok = gpu_generate_token(m, ctx, ctx_len, TEMPERATURE);
            if (tok == '\n' && i > 10) break;
            putchar(tok); fflush(stdout);
            CTX_PUSH(tok);
        }
        printf("\n\n");

        /* Ajouter le séparateur de tour */
        CTX_PUSH('\n');
    }
    #undef CTX_PUSH
    free(ctx);
}

/* ── main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    srand(42);
    int V = VOCAB_SIZE, D = DIM, S = STATE_SIZE, NL = N_LAYERS;
    long params = (long)NL * (2*S*D + 3*S + D) + 2L * V * D;

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║      k-mamba — Instance GPU (MX450 sm_75)       ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  vocab_size  : %-5d                            ║\n", V);
    printf("║  dim         : %-5d                            ║\n", D);
    printf("║  state_size  : %-5d                            ║\n", S);
    printf("║  n_layers    : %-5d                            ║\n", NL);
    printf("║  seq_len     : %-5d                            ║\n", SEQ_LEN);
    printf("║  batch_size  : %-5d                            ║\n", BATCH_SIZE);
    printf("║  epochs      : %-5d                            ║\n", N_EPOCHS);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  compute     : forward/backward GPU (Blelloch)  ║\n");
    printf("║  optimizer   : MUON GPU (Newton-Schulz sm_75)   ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  params      : %6ldK                           ║\n", params / 1000);
    printf("╚══════════════════════════════════════════════════╝\n");

    GpuModel *m = gpu_model_create();

    /* Mode train (argument = fichier texte) */
    if (argc == 2) {
        const char *data_path = argv[1];
        const char *ckpt_path = "kmamba_gpu.bin";
        gpu_model_init(m);
        train(m, data_path, ckpt_path);
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s <data.txt>              — entraîne\n", argv[0]);
        fprintf(stderr, "  %s gen  <ckpt> [prompt]    — génère\n", argv[0]);
        fprintf(stderr, "  %s chat <ckpt>             — chat REPL\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *ckpt = argv[2];

    gpu_model_init(m);
    int epoch = 0;
    if (!gpu_model_load(m, ckpt, &epoch)) {
        fprintf(stderr, "[erreur] impossible de charger %s\n", ckpt);
        return 1;
    }
    printf("[checkpoint : %s (epoch %d)]\n", ckpt, epoch);

    if (strcmp(mode, "chat") == 0) {
        chat_repl(m);
    } else if (strcmp(mode, "gen") == 0) {
        const char *prompt = (argc >= 4) ? argv[3] : "";
        size_t plen = strlen(prompt);
        uint8_t ctx[512] = {0};
        memcpy(ctx, prompt, plen < 512 ? plen : 512);
        int ctx_len = (int)(plen < 512 ? plen : 512);
        printf("%s", prompt);
        for (int i = 0; i < GEN_LEN; i++) {
            int tok = gpu_generate_token(m, ctx, ctx_len > 0 ? ctx_len : 1, TEMPERATURE);
            putchar(tok); fflush(stdout);
            if (ctx_len < 512) ctx[ctx_len++] = (uint8_t)tok;
            else { memmove(ctx, ctx + 1, 511); ctx[511] = (uint8_t)tok; }
        }
        printf("\n");
    }

    return 0;
}
