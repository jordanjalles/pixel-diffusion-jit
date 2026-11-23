"""Multi-scale attention separation (user, 2026-10-15): cfg["msa"]. Three forwards of the same weights per train step.

  dense  (D): full attention over the 64 patches. It is the teacher of the alignment terms and the model that samples.
  split  (S): a random 42 / 22 partition of the patches (65 / 35), attention only inside each group.
  sparse (P): 32 single patches and 16 two-patch groups. "pairing": "spatial" = the pairs are neighbors (a domino
              tiling, horizontal or vertical per image), "long" = the pairs are 4 rows and 4 columns apart (distance 5.66).
              16 of the 32 candidate pairs are drawn per image and step. The patches of the other 16 are single.

Packed attention, no dense mask:
  split:  the tokens are put in group order once per forward (the rope tables follow them), every block runs one
          attention on the first 42 tokens and one on the last 22, and the order is undone for the readouts and the output.
  sparse: a group of two has a softmax over two scores, so each token mixes its own value and its partner's value with
          weight sigmoid(q.k_own - q.k_partner) / sqrt(head size). A single patch is its own partner: its output is its value.
The three forwards of a step share one rope table (rope_tables draws a random scale per call in training).

"timesteps": "shared"    one t per image, the same noised input for all three forwards.
             "staggered" t = 1 is the clean image. Teacher t_D from broad15. Split: each group g gets t_D * u_g.
                         Sparse: each group h gets (min over its patches of the split t) * v_h. u, v ~ U(0, 1). So at every
                         patch position the teacher is cleaner than the split student, which is cleaner than the sparse one.
                         Every forward gets its time per token. The noise sample is the same for the three forwards.
             "mixed"     (user 2026-10-15, Self-Flow's two levels made three, option A) three t per image from broad15,
                         sorted t1 >= t2 >= t3 (t1 cleanest). Dense: t1 at every token. One random mask of MIX_N = 16 of
                         64 tokens (noisy_share 0.25): split gets t2 there, sparse t3, both get t1 elsewhere. So no student
                         token is cleaner than its teacher's, for all three alignment pairs.

Loss: the flow loss of each forward (x prediction with the clamp01 velocity weight per patch, plus the readouts, as the
dense baseline), all weight 1, plus weight * (d(S3, D6) + d(P3, D6) + d(P3, S6)). d = mse of the RMS-normalized tokens
at the same patch positions (teach_distance), the targets detached. No projection heads (same width). No EMA teacher.
"""
import torch
import torch.nn.functional as F

R = C = 8                      # 8 x 8 patches
T = R * C
N_SPLIT_A = 42                 # round(0.65 * 64)
MIX_N = 16                     # mixed timesteps: tokens at the students' higher noise levels (Self-Flow noisy_share 0.25)
N_PAIRS, N_CAND = 16, 32       # 16 of 32 candidate pairs are pairs, the other 32 patches are single


def candidate_pairs(kind):
    """(orientations, 32, 2) token indices, row-major (r * 8 + c)."""
    idx = torch.arange(T).reshape(R, C)
    if kind == "spatial":
        hor = torch.stack([idx[:, 0::2].reshape(-1), idx[:, 1::2].reshape(-1)], 1)      # (r, 2j) - (r, 2j + 1)
        ver = torch.stack([idx[0::2, :].reshape(-1), idx[1::2, :].reshape(-1)], 1)      # (2i, c) - (2i + 1, c)
        return torch.stack([hor, ver])
    assert kind == "long", kind
    top = idx[:4, :].reshape(-1)                                                        # (r, c), r < 4
    far = idx[(torch.arange(4)[:, None] + 4), (torch.arange(C)[None, :] + 4) % C].reshape(-1)
    return torch.stack([top, far], 1)[None]


def take(x, idx):
    """x (B, T, ...) or a table (T, D); idx (B, T): the tokens in the order of idx."""
    if x.ndim == 2:
        return x[idx]
    return x.gather(1, idx.reshape(*idx.shape, *([1] * (x.ndim - 2))).expand(*idx.shape, *x.shape[2:]))


def make(ns):
    """ns: model, device, apply_2d_rope, weight_maps, teach_distance, pyramid_weight, cfg, selfteach (main.py's make_selfteach() or None)."""
    model, device, cfg = ns["model"], ns["device"], ns["cfg"]
    MSA = cfg["msa"]
    pairing, tsched = MSA.get("pairing", "spatial"), MSA.get("timesteps", "shared")
    assert pairing in ("spatial", "long") and tsched in ("shared", "staggered", "mixed"), MSA
    lam, pw = float(MSA.get("weight", 1.0)), float(ns["pyramid_weight"])
    # cfg["selfteach"] (2026-10-15): main.py's self-teaching term (frozen copy as judge), on the dense forward: its noised
    # input at t_D and its final output. None = off
    st = ns.get("selfteach")
    # "split_form": how the split forward runs its attention (all exact): "split" = two calls (42, 22), "split_pad" = one call
    # (group B padded to 42, stacked with A), "mask" = one call over 64 tokens with a same-group mask. Chosen by the bench
    split_form = MSA.get("split_form", "split")
    assert split_form in ("split", "split_pad", "mask"), split_form
    rope2d, weight_maps, dist = ns["apply_2d_rope"], ns["weight_maps"], ns["teach_distance"]
    cand_all = candidate_pairs(pairing).to(device)                                     # (O, 32, 2)
    rr, cc = torch.arange(T, device=device) // C, torch.arange(T, device=device) % C

    def mha(m, x, rope, form, idx):
        """form "dense"; "split" (x in group order); "sparse" (idx = partner per token); "mask" (idx = same-group mask, check only)."""
        B, T_, _ = x.shape
        nh, hs = m.n_heads, m.head_size
        q, k, v = m.QKV(x).split(nh * hs, -1)
        q, k, v = (z.reshape(B, T_, nh, hs).permute(0, 2, 1, 3) for z in (q, k, v))
        q, k = rope2d(m.q_norm(q), m.k_norm(k), rope=rope)
        if form == "dense":
            o = F.scaled_dot_product_attention(q, k, v)
        elif form == "split":
            a = N_SPLIT_A
            o = torch.cat([F.scaled_dot_product_attention(q[:, :, :a], k[:, :, :a], v[:, :, :a]),
                           F.scaled_dot_product_attention(q[:, :, a:], k[:, :, a:], v[:, :, a:])], 2)
        elif form == "split_pad":
            # one call: group B (22) padded to 42 and stacked with group A along the batch. An extra q/k dimension (q 1,
            # k 0 for real keys and -6e4 for pad keys) puts -1e4 on the scores of the pad keys, so they get no weight
            a, b = N_SPLIT_A, T_ - N_SPLIT_A
            ext = (-hs) % 8 or 8                                                         # head dim to a multiple of 8
            pad = lambda z: F.pad(z, (0, 0, 0, a - b))
            qq = torch.cat([q[:, :, :a], pad(q[:, :, a:])], 0)
            kk = torch.cat([k[:, :, :a], pad(k[:, :, a:])], 0)
            vv = torch.cat([v[:, :, :a], pad(v[:, :, a:])], 0)
            bias = torch.zeros(2 * B, 1, a, 1, device=x.device, dtype=q.dtype)
            bias[B:, :, b:] = -6e4
            qq = torch.cat([qq, torch.ones_like(qq[..., :1]), qq.new_zeros(*qq.shape[:-1], ext - 1)], -1)
            kk = torch.cat([kk, bias.expand(2 * B, nh, a, 1), kk.new_zeros(*kk.shape[:-1], ext - 1)], -1)
            vv = F.pad(vv, (0, ext))
            oo = F.scaled_dot_product_attention(qq, kk, vv, scale=hs ** -0.5)[..., :hs]
            o = torch.cat([oo[:B], oo[B:, :, :b]], 2)
        elif form == "sparse":
            ix = idx[:, None, :, None].expand(B, nh, T_, hs)
            kp, vp = k.gather(2, ix), v.gather(2, ix)
            w = torch.sigmoid((q.float() * (k.float() - kp.float())).sum(-1, keepdim=True) * hs ** -0.5)
            o = (vp.float() + w * (v.float() - vp.float())).to(v.dtype)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=idx[:, None])
        return m.out_proj(o.permute(0, 2, 1, 3).flatten(-2))

    def forward(images, t_tok, taps, rope, form="dense", idx=None):
        """forward_taps with grouped attention. t_tok: (B, T). idx: split = perm (B, T) (group A first), sparse = partner (B, T),
        mask = (B, T, T). Returns (outputs [readouts..., final], {block: hidden}) in patch order."""
        m = getattr(model, "_orig_mod", model)
        core = m.core_transformer
        rows, cols = images.shape[-2] // m.patch_size, images.shape[-1] // m.patch_size
        x = core.norm_in(m.embed(images))
        temb = m.timestep_embedder(t_tok.reshape(-1)).reshape(t_tok.shape[0], t_tok.shape[1], -1)
        att_idx, back = idx, None
        if form in ("split", "split_pad"):                   # group order for the whole forward; the rope tables follow
            x, temb, rope, back = take(x, idx), take(temb, idx), tuple(take(r, idx) for r in rope), idx.argsort(1)
        unperm = (lambda z: take(z, back)) if back is not None else (lambda z: z)
        outputs, hidden = [], {}
        for i, blk in enumerate(core.blocks):
            sa, sm, ha, hm, ga, gm = blk.scale_shift_gate(temb).chunk(6, dim=-1)
            x = x + ga * blk.post_norm1(mha(blk.MHA, blk.pre_norm1(x) * (1 + sa) + ha, rope, form, att_idx))
            x = x + gm * blk.post_norm2(blk.SwiGLU(blk.pre_norm2(x) * (1 + sm) + hm))
            if i + 1 in taps or str(i + 1) in m.readout_heads:
                xp = unperm(x)
                if i + 1 in taps:
                    hidden[i + 1] = xp
                if str(i + 1) in m.readout_heads:
                    outputs.append(m.readout_heads[str(i + 1)](xp, rows, cols))
        outputs.append(torch.sigmoid(m.un_embed(core.norm_out(unperm(x)), rows, cols)))
        return outputs, hidden

    def partitions(B):
        """split: perm (group A = its first 42) + group id per token. sparse: partner per token + group id
        (singles 0..31, pairs 32..47) + mean pair distance."""
        perm_s = torch.rand(B, T, device=device).argsort(1)
        gid_s = torch.zeros(B, T, dtype=torch.long, device=device).scatter_(1, perm_s[:, N_SPLIT_A:], 1)
        orient = torch.randint(0, cand_all.shape[0], (B,), device=device)
        cand = cand_all[orient]                                                          # (B, 32, 2)
        order = torch.rand(B, N_CAND, device=device).argsort(1)
        cand = cand.gather(1, order[:, :, None].expand(B, N_CAND, 2))                    # first 16 = the pairs
        pr, singles = cand[:, :N_PAIRS], cand[:, N_PAIRS:].reshape(B, -1)
        partner = torch.arange(T, device=device).repeat(B, 1)
        partner = partner.scatter(1, pr[..., 0], pr[..., 1]).scatter(1, pr[..., 1], pr[..., 0])
        gid_p = torch.empty(B, T, dtype=torch.long, device=device)
        gid_p = gid_p.scatter(1, singles, torch.arange(N_CAND, device=device).expand(B, N_CAND))
        pair_id = (N_CAND + torch.arange(N_PAIRS, device=device)).expand(B, N_PAIRS)
        gid_p = gid_p.scatter(1, pr[..., 0], pair_id).scatter(1, pr[..., 1], pair_id)
        pair_dist = ((rr[pr[..., 0]] - rr[pr[..., 1]]).float().square() + (cc[pr[..., 0]] - cc[pr[..., 1]]).float().square()).sqrt().mean()
        return perm_s, gid_s, partner, gid_p, pair_dist

    def times(B, gid_s, gid_p):
        t_d = torch.sigmoid(torch.randn(B, device=device) * 1.5)                        # broad15 (no generator kwarg: compiles with any B)
        if tsched == "shared":
            t = t_d[:, None].expand(B, T)
            return t, t, t, None
        if tsched == "mixed":
            t3 = torch.sigmoid(torch.randn(B, 3, device=device) * 1.5).sort(1, descending=True).values
            mask = torch.zeros(B, T, dtype=torch.bool, device=device).scatter_(1, torch.rand(B, T, device=device).argsort(1)[:, :MIX_N], True)
            t1 = t3[:, :1].expand(B, T)
            return t1, torch.where(mask, t3[:, 1:2], t1), torch.where(mask, t3[:, 2:3], t1), mask
        t_s = (t_d[:, None] * torch.rand(B, 2, device=device)).gather(1, gid_s)          # per split group
        n_grp = N_CAND + N_PAIRS
        m_grp = torch.full((B, n_grp), 2.0, device=device).scatter_reduce(1, gid_p, t_s, reduce="amin")
        t_p = (m_grp * torch.rand(B, n_grp, device=device)).gather(1, gid_p)             # per sparse group
        return t_d[:, None].expand(B, T), t_s, t_p, None

    def flow(outputs, images, t_tok, rows, cols):
        w_maps = weight_maps(t_tok, rows, cols, outputs)
        losses = []
        for x_pred, w in zip(outputs, w_maps):
            goal = images if x_pred.shape[-2:] == images.shape[-2:] else F.interpolate(images, size=x_pred.shape[-2:], mode="bilinear", antialias=True)
            losses.append(((x_pred.float() - goal.float()).square() * w).mean())
        return losses

    def train_step(images, optimizer):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        m = getattr(model, "_orig_mod", model)
        p = m.patch_size
        rows, cols = images.shape[-2] // p, images.shape[-1] // p
        rope = m.core_transformer.rope_tables(rows, cols)                                # one table (one random scale) for the three forwards
        perm_s, gid_s, partner, gid_p, pair_dist = partitions(B)
        t_d, t_s, t_p, mask = times(B, gid_s, gid_p)
        noise = torch.randn_like(images)
        noised = lambda t_tok: (lambda m_: m_ * images + (1.0 - m_) * noise)(
            t_tok.reshape(B, 1, rows, cols).repeat_interleave(p, 2).repeat_interleave(p, 3))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_d = noised(t_d)
            out_d, h_d = forward(x_d, t_d, (6,), rope)
            out_s, h_s = forward(noised(t_s), t_s, (3, 6), rope, split_form, gid_s[:, :, None] == gid_s[:, None, :] if split_form == "mask" else perm_s)
            out_p, h_p = forward(noised(t_p), t_p, (3,), rope, "sparse", partner)
            l_d, l_s, l_p = (flow(o, images, t, rows, cols) for o, t in ((out_d, t_d), (out_s, t_s), (out_p, t_p)))
            f_s, f_p = l_s[-1] + pw * sum(l_s[:-1]), l_p[-1] + pw * sum(l_p[:-1])
            a_ds, a_dp, a_sp = dist(h_s[3], h_d[6].detach()), dist(h_p[3], h_d[6].detach()), dist(h_p[3], h_s[6].detach())
            teach = a_ds + a_dp + a_sp
            loss = l_d[-1] + pw * sum(l_d[:-1]) + f_s + f_p + lam * teach
            if st is not None:
                st_loss, st_out, st_noised = st["terms"](images, x_d, t_d[:, 0], out_d[-1], rope)
                loss = loss + st_loss
        loss.backward()
        optimizer.step()
        order_bad = ((t_s > t_d) | (t_p > t_s)).float().mean()
        # mixed: share of tokens at a higher noise level than the teacher's, and unmasked tokens whose t is not the teacher's (want 0)
        noisy = (t_p < t_d) if mask is None else mask
        mix_bad = torch.zeros((), device=device) if mask is None else (~mask & ((t_s != t_d) | (t_p != t_d))).float().mean()
        extra = {"st_out": st_out.detach(), "st_noised": st_noised.detach()} if st is not None else {}
        return {**extra, "loss": loss.detach(), "loss_final": l_d[-1].detach(), "readout_losses": torch.stack(l_d[:-1]).detach(), "teach": teach.detach(),
                "msa_flow_split": f_s.detach(), "msa_flow_sparse": f_p.detach(), "msa_final_split": l_s[-1].detach(), "msa_final_sparse": l_p[-1].detach(),
                "msa_d6s3": a_ds.detach(), "msa_d6p3": a_dp.detach(), "msa_s6p3": a_sp.detach(),
                "msa_t_dense": t_d.mean(), "msa_t_split": t_s.mean(), "msa_t_sparse": t_p.mean(), "msa_order_bad": order_bad,
                "msa_pair_dist": pair_dist, "msa_split_share": (gid_s == 0).float().mean(), "msa_noisy_share": noisy.float().mean(), "msa_mix_bad": mix_bad}

    p_size = lambda: getattr(model, "_orig_mod", model).patch_size

    @torch.no_grad()
    def check():
        """At start: (1) the dense form is the model's forward. (2) The packed split and sparse forms equal a forward with
        dense attention under a same-group mask (outputs and the hidden states of blocks 3 and 6). (3) Planned group sizes."""
        m = getattr(model, "_orig_mod", model)
        was = m.training
        m.eval()
        x = torch.rand(4, 3, R * p_size(), C * p_size(), device=device)
        t = torch.rand(4, device=device)
        rope = m.core_transformer.rope_tables(R, C)
        own = m(x, t, all_readouts=True)
        mine, _ = forward(x, t[:, None].expand(4, T), (), rope)
        e0 = max((a.float() - b.float()).abs().max().item() for a, b in zip(own, mine))
        perm_s, gid_s, partner, gid_p, pair_dist = partitions(4)
        t_tok = torch.rand(4, T, device=device)
        errs = []
        for form, idx, gid in (("split", perm_s, gid_s), ("split_pad", perm_s, gid_s), ("sparse", partner, gid_p)):
            o1, h1 = forward(x, t_tok, (3, 6), rope, form, idx)
            o2, h2 = forward(x, t_tok, (3, 6), rope, "mask", gid[:, :, None] == gid[:, None, :])
            errs.append(max([(a - b).abs().max().item() for a, b in zip(o1, o2)] + [(h1[k] - h2[k]).abs().max().item() for k in (3, 6)]))
        sizes_s = torch.bincount(gid_s[0], minlength=2).tolist()
        sizes_p = torch.bincount(gid_p[0], minlength=N_CAND + N_PAIRS)
        m.train(was)
        print(f"CHECK msa at start: dense form vs the model: max difference {e0:.2e} | packed vs masked attention (outputs, blocks 3 and 6): "
              f"split {errs[0]:.2e}, split one call {errs[1]:.2e}, sparse {errs[2]:.2e} | split group sizes {sizes_s} | sparse: {(sizes_p == 1).sum().item()} single, "
              f"{(sizes_p == 2).sum().item()} pairs | pairing {pairing}: mean pair distance {pair_dist.item():.3f} patches | timesteps {tsched} | weight {lam}", flush=True)
        assert e0 < 1e-3 and max(errs) < 1e-4, (e0, errs)
        assert sizes_s == [N_SPLIT_A, T - N_SPLIT_A] and (sizes_p == 1).sum() == N_CAND and (sizes_p == 2).sum() == N_PAIRS
        assert abs(pair_dist.item() - (1.0 if pairing == "spatial" else 32 ** 0.5)) < 1e-4, pair_dist

    def bench(B=64, reps=30):
        """cfg msa "bench": true. Time of the dense, split and sparse forms, compiled, bf16: (1) the attention of one block,
        (2) the whole forward plus backward."""
        import time as _time
        m = getattr(model, "_orig_mod", model)
        blk = m.core_transformer.blocks[0]
        rope = m.core_transformer.rope_tables(R, C)
        perm_s, gid_s, partner, _, _ = partitions(B)
        forms = (("dense", None), ("split", perm_s), ("split_pad", perm_s), ("mask", gid_s[:, :, None] == gid_s[:, None, :]), ("sparse", partner))
        h = torch.randn(B, T, blk.MHA.emb_dim, device=device)
        x = torch.rand(B, 3, R * p_size(), C * p_size(), device=device)
        t = torch.rand(B, T, device=device)

        def timed(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return 1000 * (_time.perf_counter() - t0) / reps

        att, full = {}, {}
        for name, idx in forms:
            rope_f = tuple(take(r, idx) for r in rope) if name in ("split", "split_pad") else rope
            f_att = torch.compile(lambda h_, idx=idx, name=name, rope_f=rope_f: mha(blk.MHA, h_, rope_f, name, idx))
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                att[name] = timed(lambda: f_att(h))
            f_full = torch.compile(lambda x_, idx=idx, name=name: forward(x_, t, (), rope, name, idx)[0][-1])

            def fb():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = f_full(x)
                out.float().square().mean().backward()
            full[name] = timed(fb)
        m.zero_grad(set_to_none=True)
        print("BENCH msa (batch %d, compiled, ms): attention of one block: %s | whole forward + backward: %s" % (
            B, " / ".join(f"{k} {v:.3f}" for k, v in att.items()), " / ".join(f"{k} {v:.2f}" for k, v in full.items())), flush=True)

    return train_step, (lambda: (check(), bench() if MSA.get("bench") else None))
