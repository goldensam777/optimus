/*
 * test_scan.c — Vérifie que scan1d.asm produit un output non-nul
 *
 * Test minimal : appelle mamba_scan1d_forward directement avec
 * des paramètres connus et vérifie le résultat analytiquement.
 */

#include "kmamba.h"
#include "mamba_scan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Calcule la somme absolue d'un vecteur */
static float sum_abs(const float *v, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += fabsf(v[i]);
    return s;
}

/* ============================================================
 * Référence scalaire du scan 1D (pour vérification)
 *
 * h_t[d] = exp(dt[t,d] * A[d]) * h_{t-1}[d]  +  dt[t,d] * B[t,d] * x[t,d]
 * y_t[d] = C[t,d] * h_t[d]
 * ============================================================ */
static void scan1d_ref(
        const float *x,     /* [L, D] */
        const float *A,     /* [D]    */
        const float *B,     /* [L, D] */
        const float *C,     /* [L, D] */
        const float *delta, /* [L, D] */
        float       *y,     /* [L, D] out */
        float       *h_out, /* [D]    out — état final */
        long L, long D)
{
    float *h = (float *)calloc((size_t)D, sizeof(float));

    for (long t = 0; t < L; t++) {
        for (long d = 0; d < D; d++) {
            long td = t * D + d;
            float dt  = delta[td];
            float dA  = expf(dt * A[d]);
            float bbar = dt * B[td];
            h[d] = dA * h[d] + bbar * x[td];
            y[td] = C[td] * h[d];
        }
    }
    if (h_out) memcpy(h_out, h, (size_t)D * sizeof(float));
    free(h);
}

int main(void) {
    printf("=== test_scan : vérification scan1d ASM ===\n\n");

    /* --- Cas 1 : paramètres trivials (tout = 1) --- */
    printf("--- Cas 1 : L=4, D=4, A=1, B=1, C=1, x=1, delta=0.1 ---\n");
    {
        long L = 4, D = 4;
        float x[16], A[4], B[16], C[16], delta[16];
        float y_asm[16] = {0}, y_ref[16] = {0};
        float h_asm[4] = {0}, h_ref[4] = {0};

        for (int i = 0; i < L*D; i++) { x[i] = 1.0f; B[i] = 1.0f; C[i] = 1.0f; delta[i] = 0.1f; }
        for (int i = 0; i < D;   i++) A[i] = 1.0f;

        /* Référence */
        scan1d_ref(x, A, B, C, delta, y_ref, h_ref, L, D);

        /* ASM */
        ScanParams sp = {
            .x = x, .A = A, .B = B, .C = C, .delta = delta,
            .h = h_asm, .y = y_asm,
            .L = L, .D = D, .M = 1
        };
        mamba_scan1d_forward(&sp);

        printf("  y_ref  : ");
        for (int i = 0; i < L*D; i++) printf("%.4f ", y_ref[i]);
        printf("\n  y_asm  : ");
        for (int i = 0; i < L*D; i++) printf("%.4f ", y_asm[i]);
        printf("\n");

        float err = 0.0f;
        for (int i = 0; i < L*D; i++) err += fabsf(y_asm[i] - y_ref[i]);
        printf("  |err|  : %.6f\n", err);
        printf("  sum_abs(y_asm) = %.6f  sum_abs(y_ref) = %.6f\n",
               sum_abs(y_asm, L*D), sum_abs(y_ref, L*D));

        if (sum_abs(y_asm, L*D) < 1e-9f)
            printf("  [FAIL] y_asm est entièrement zéro!\n");
        else if (err < 1e-4f)
            printf("  [PASS] résultats concordent\n");
        else
            printf("  [FAIL] erreur trop grande : %.6f\n", err);
    }

    /* --- Cas 2 : A négatif (comme dans k-mamba : A_log ≈ -1) --- */
    printf("\n--- Cas 2 : A=-1 (valeur réelle de k-mamba), D=8, L=8 ---\n");
    {
        long L = 8, D = 8;
        float x[64], A[8], B[64], C[64], delta[64];
        float y_asm[64] = {0}, y_ref[64] = {0};
        float h_asm[8] = {0}, h_ref[8] = {0};

        float B_val = 1.0f / sqrtf((float)D);
        float C_val = 1.0f / sqrtf((float)D);

        for (int d = 0; d < D; d++) A[d] = -1.0f;
        for (int i = 0; i < L*D; i++) {
            x[i]     = 0.5f;
            B[i]     = B_val;
            C[i]     = C_val;
            delta[i] = 0.01f;
        }

        scan1d_ref(x, A, B, C, delta, y_ref, h_ref, L, D);

        ScanParams sp = {
            .x = x, .A = A, .B = B, .C = C, .delta = delta,
            .h = h_asm, .y = y_asm,
            .L = L, .D = D, .M = 1
        };
        mamba_scan1d_forward(&sp);

        printf("  y_ref[0..7]  : ");
        for (int i = 0; i < D; i++) printf("%.6f ", y_ref[i]);
        printf("\n  y_asm[0..7]  : ");
        for (int i = 0; i < D; i++) printf("%.6f ", y_asm[i]);
        printf("\n");

        float err = 0.0f;
        for (int i = 0; i < L*D; i++) err += fabsf(y_asm[i] - y_ref[i]);
        printf("  |err|  : %.6f\n", err);

        if (sum_abs(y_asm, L*D) < 1e-9f)
            printf("  [FAIL] y_asm est entièrement zéro!\n");
        else if (err < 1e-4f)
            printf("  [PASS] résultats concordent\n");
        else
            printf("  [FAIL] erreur = %.6f\n", err);
    }

    /* --- Cas 3 : Vérifie que mamba_block_forward produit du non-zéro --- */
    printf("\n--- Cas 3 : mamba_block_forward output non-nul? ---\n");
    {
        MBConfig bc = {
            .dim = 8, .state_size = 8, .seq_len = 4,
            .dt_scale = 1.0f, .dt_min = 0.001f, .dt_max = 0.1f
        };
        MambaBlock *block = mamba_block_create(&bc);
        mamba_block_init(block);

        float input[4 * 8];
        for (int i = 0; i < 4*8; i++) input[i] = 0.1f * (float)(i+1);

        float output[4 * 8] = {0};
        mamba_block_forward(block, output, input, 1);

        printf("  output[0..31] :\n  ");
        for (int i = 0; i < 4*8; i++) {
            printf("%.5f ", output[i]);
            if ((i+1) % 8 == 0) printf("\n  ");
        }
        printf("\n");
        printf("  sum_abs(output) = %.6f\n", sum_abs(output, 4*8));
        printf("  sum_abs(input)  = %.6f\n", sum_abs(input, 4*8));

        if (sum_abs(output, 4*8) < 1e-9f) {
            printf("  [FAIL] output = 0 — mamba_block_forward est cassé!\n");
            printf("  -> Le scan ou la projection W_out sort du zéro.\n");
        } else {
            printf("  [PASS] output non-nul\n");
        }

        /* Aussi vérifier u_seq directement */
        printf("\n  Vérif u_seq (SiLU(W_in @ x)) sur t=0:\n");
        float z[8], u[8];
        gemv_avx2(block->W_in.data, input, z, (long)bc.state_size, (long)bc.dim);
        silu_f32(z, u, (long)bc.state_size);
        printf("  z[0..7] : ");
        for (int d = 0; d < 8; d++) printf("%.5f ", z[d]);
        printf("\n  u[0..7] : ");
        for (int d = 0; d < 8; d++) printf("%.5f ", u[d]);
        printf("\n  sum_abs(u) = %.6f\n", sum_abs(u, 8));

        /* Scan direct sur u */
        float B_s[4*8], C_s[4*8], d_s[4*8], h_s[8] = {0}, y_s[4*8] = {0};
        for (int t = 0; t < 4; t++) {
            float dt = 0.01f;
            for (int d = 0; d < 8; d++) {
                B_s[t*8+d] = block->B_mat.data[d];
                C_s[t*8+d] = block->C_mat.data[d];
                d_s[t*8+d] = dt;
            }
        }

        /* u_seq pour tous les timesteps */
        float u_all[4*8];
        float ztmp[8];
        for (int t = 0; t < 4; t++) {
            gemv_avx2(block->W_in.data, &input[t*8], ztmp, 8, 8);
            silu_f32(ztmp, &u_all[t*8], 8);
        }

        ScanParams sp = {
            .x = u_all, .A = block->A_log.data,
            .B = B_s, .C = C_s, .delta = d_s,
            .h = h_s, .y = y_s,
            .L = 4, .D = 8, .M = 1
        };
        mamba_scan1d_forward(&sp);

        printf("\n  Scan direct sur u_all:\n");
        printf("  y_scan[0..31] :\n  ");
        for (int i = 0; i < 32; i++) {
            printf("%.5f ", y_s[i]);
            if ((i+1) % 8 == 0) printf("\n  ");
        }
        printf("  sum_abs(y_scan) = %.6f\n", sum_abs(y_s, 32));

        if (sum_abs(y_s, 32) < 1e-9f)
            printf("  [FAIL] scan output = 0\n");
        else
            printf("  [PASS] scan output non-nul\n");

        mamba_block_free(block);
    }

    printf("\n=== fin test_scan ===\n");
    return 0;
}
