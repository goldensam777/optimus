with open('/home/samuel-yevi/Dev/optimus/cuda/main.cu', 'r') as f:
    mc = f.read()

# ── 1. Batch size 16 → 32 ─────────────────────────────────────────
mc = mc.replace('#define BATCH_SIZE    16', '#define BATCH_SIZE    32')

# ── 2. SAVE_EVERY → 1 ────────────────────────────────────────────
mc = mc.replace('#define SAVE_EVERY    5', '#define SAVE_EVERY    1')

# ── 3. LR constant ───────────────────────────────────────────────
mc = mc.replace('#define LR_BLOCKS     2e-5f', '#define LR_BLOCKS     3e-5f')
mc = mc.replace('#define LR_EMBED_HEAD 2e-5f', '#define LR_EMBED_HEAD 3e-5f')

# ── 4. Scheduler cosine → constant après warmup ──────────────────
old_sched = '''static float lr_schedule(float lr_max, size_t step,
                         size_t warmup_steps, size_t total_steps) {
    if (step == 0) return 0.0f;
    if (step < warmup_steps)
        return lr_max * (float)step / (float)warmup_steps;
    float progress = (float)(step - warmup_steps)
                   / (float)(total_steps - warmup_steps);
    float cosine   = 0.5f * (1.0f + cosf(M_PI * progress));
    float lr_min   = lr_max * 0.1f;
    return lr_min + (lr_max - lr_min) * cosine;
}'''

new_sched = '''static float lr_schedule(float lr_max, size_t step,
                         size_t warmup_steps, size_t total_steps) {
    (void)total_steps;
    if (step == 0) return 0.0f;
    if (step < warmup_steps)
        return lr_max * (float)step / (float)warmup_steps;
    return lr_max;  /* constant après warmup */
}'''

assert old_sched in mc, "ERREUR: pattern lr_schedule non trouvé"
mc = mc.replace(old_sched, new_sched)

# ── 5. compute_val_loss → forward-only sans backward ─────────────
old_val = '''static float compute_val_loss(GpuModel *m, const uint8_t *data,
                               size_t data_len, int L, int n_val) {
    double vl = 0.0;
    for (int i = 0; i < n_val; i++) {
        size_t off = (size_t)rand() % (data_len - L);
        uint8_t *seq = (uint8_t *)malloc(L * sizeof(uint8_t));
        memcpy(seq, data + off, L);
        vl += gpu_forward_backward(m, seq, L);
        free(seq);
    }
    return (float)(vl / n_val);
}'''

new_val = '''static float compute_val_loss(GpuModel *m, const uint8_t *data,
                               size_t data_len, int L, int n_val) {
    double vl = 0.0;
    for (int i = 0; i < n_val; i++) {
        size_t off = (size_t)rand() % (data_len - L);
        uint8_t *seq = (uint8_t *)malloc(L * sizeof(uint8_t));
        memcpy(seq, data + off, L);
        vl += gpu_forward_loss(m, seq, L);  /* forward-only, pas de backward */
        free(seq);
    }
    return (float)(vl / n_val);
}'''

assert old_val in mc, "ERREUR: pattern compute_val_loss non trouvé"
mc = mc.replace(old_val, new_val)

# ── 6. Réactiver val et train_eval ───────────────────────────────
old_eval = '''        /* Val loss sur 64 séquences aléatoires */
        float val_loss = compute_val_loss(m, data, data_len, L, 64);

        /* Train eval sur 64 séquences */
        float train_eval = compute_val_loss(m, data, data_len, L, 64);'''

new_eval = '''        /* Val loss sur 32 séquences — forward-only, propre */
        float val_loss   = compute_val_loss(m, data, data_len, L, 32);

        /* Train eval sur 32 séquences — forward-only, propre */
        float train_eval = compute_val_loss(m, data, data_len, L, 32);'''

assert old_eval in mc, "ERREUR: pattern val/train_eval non trouvé"
mc = mc.replace(old_eval, new_eval)

# ── Écriture finale ───────────────────────────────────────────────
with open('/home/samuel-yevi/Dev/optimus/cuda/main.cu', 'w') as f:
    f.write(mc)

print("✓ Patch appliqué avec succès.")
print("  - BATCH_SIZE    : 16 → 32")
print("  - SAVE_EVERY    : 5 → 1")
print("  - LR_BLOCKS     : 2e-5 → 3e-5")
print("  - LR_EMBED_HEAD : 2e-5 → 3e-5")
print("  - lr_schedule   : cosine → constant après warmup")
print("  - compute_val_loss : forward-only (gpu_forward_loss)")
print("  - val/train_eval   : réactivés, 32 séquences chacun")
