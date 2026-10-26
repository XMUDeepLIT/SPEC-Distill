import torch
import math
from collections import defaultdict, deque

def _rand_like_compat(t: torch.Tensor, seed: int | None = None):
    if seed is None:
        return torch.rand_like(t)
    g = torch.Generator(device=t.device if t.device.type == "cpu" else "cpu")
    g.manual_seed(seed)
    try:
        return torch.rand(t.shape, device=t.device, dtype=t.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.rand_like(t)

@torch.no_grad()
def spec_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
):

    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    # Δlogp = new - old
    log_ratio = (new_logp - old_logp).masked_fill(~valid, 0.0)

    U = _rand_like_compat(new_logp, seed=seed).clamp_min(1e-12)
    logU = torch.log(U)

    # accept condition: (Δ>=0) or (logU <= Δ)
    accept_mask = (~valid) | (log_ratio >= 0) | (logU <= log_ratio)

    # optional absolute threshold: require new_logp >= log(p_abs_thresh)
    if p_abs_thresh is not None:
        import math
        log_th = math.log(p_abs_thresh)
        accept_mask &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept_mask)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)  # no bad = 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = (resp_len - cut_idx).clamp(min=0)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


@torch.no_grad()
def spec_cut_with_knobs(
    old_logp: torch.Tensor,         # [B,R]
    new_logp: torch.Tensor,         # [B,R]
    response_mask: torch.Tensor,    # [B,R] 1=valid response token
    *,
    bias: float = 0.0,              # b:  >0 more strict; <0 more lenient
    scale: float = 1.0,             # s:  <1 more lenient; >1 more strict
    p_abs_thresh: float | None = None,  # optional: new >= 0.3
    seed: int | None = None,
):
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim()==2
    B, R   = new_logp.shape
    valid  = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    delta  = (new_logp - old_logp).masked_fill(~valid, 0.0)  # Δ

    delta2 = scale * (delta - bias)

    logU = torch.log(_rand_like_compat(new_logp, seed=seed).clamp_min(1e-12))
    accept = (~valid) | (delta2 >= 0) | (logU <= delta2)

    if p_abs_thresh is not None:
        log_th = math.log(p_abs_thresh)
        accept &= ((new_logp >= log_th) | (~valid))

    bad = valid & (~accept)
    has_bad   = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)       # Return 0 when there are no bad tokens -> 0
    cut_idx   = torch.where(has_bad, first_bad, resp_len)     # [B]

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)

    # note: saved tokens ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/bias": float(bias), "spec/scale": float(scale),

    }
    return {
        "cut_idx": cut_idx,
        "idx_reuse": idx_reuse, "idx_need": idx_need,
        "resp_len": resp_len,
        "per_request_max_new_tokens": (R - cut_idx).to(torch.long),
        "metrics": metrics
        
    }


@torch.no_grad()
def rand_reuse_cut(
    old_logp: torch.Tensor,          # [B,R] 
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # ∈[0,1]
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select rows with probability reuse_prob ----
    # only need per-row random number: reuse -> cut_idx=resp_len; otherwise -> cut_idx=0
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    reuse_mask = (row_rand < reuse_prob)                                    # [B] bool
    need_mask  = ~reuse_mask

    # ---- calculate cut_idx / idx_* / per_request_max_new_tokens----
    zero_like = torch.zeros_like(resp_len)
    cut_idx   = torch.where(reuse_mask, resp_len, zero_like)                # [B]

    # ---- additional check: ensure cut_idx is in valid range ----
    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)       # [Nr]
    idx_need  = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)       # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)               # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    # "saved tokens" use the same definition as spec_cut_with_knobs: saved ≈ accepted prefix = cut_idx
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_p":   float(reuse_prob),
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }

@torch.no_grad()
def rand_reuse_all_cut(
    old_logp: torch.Tensor,          # [B,R]
    new_logp: torch.Tensor,          # [B,R]
    response_mask: torch.Tensor,     # [B,R]
    *,
    reuse_prob: float,               # probability to randomly select truncation for all data
    seed: int | None = None,
):
    # ---- basic shape check ----
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))  # clamp to [0,1]

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)          # [B]

    # ---- randomly select truncation point with probability reuse_prob ----
    # use new_logp's dtype/device to produce random number, ensure device/precision consistency
    row_rand  = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)  # [B]
    cut_idx   = torch.floor(row_rand * resp_len.to(torch.float32)).to(torch.long)  # [B]
    
    # ensure cut_idx is in valid range

    cut_idx = cut_idx.clamp(min=torch.tensor([0] * resp_len.shape[0]), max=resp_len)  # ensure cut_idx is in valid range

    # ---- calculate idx_* / per_request_max_new_tokens (align with spec_*)----
    reuse_mask = (cut_idx == resp_len)                       # [B] bool
    need_mask  = ~reuse_mask                                 # [B] bool
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)  # [Nr]
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)  # [Nn]
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)  # [B]

    # ---- calculate metrics (keep spec/* names, avoid downstream changes)----
    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio":       reuse_mask.float().mean().item(),
        "spec/cont_ratio":       need_mask.float().mean().item(),
        "spec/avg_cut_idx":      cut_idx.float().mean().item(),
        "spec/avg_resp_len":     resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_all_p":   float(reuse_prob),  # new: random truncation probability
    }

    return {
        "cut_idx": cut_idx,                           # [B] long
        "resp_len": resp_len,                         # [B] long
        "idx_reuse": idx_reuse, "idx_need": idx_need, # 1D long
        "per_request_max_new_tokens": per_request_max_new_tokens,  # [B] long
        "metrics": metrics,
    }



@torch.no_grad()
def spec_cut_multi_reject(
    old_logp: torch.Tensor,         # [B,R]
    new_logp: torch.Tensor,         # [B,R]
    response_mask: torch.Tensor,    # [B,R] 1=valid response token
    *,
    bias: float = 0.0,
    scale: float = 1.0,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
    group_size: int = 8,
    n_reverse_per_group: int = 1,
):
    """
    Multi-rejection speculative cut with mixed first-bad / last-bad strategy.

    Identifies ALL rejection points per sequence.  Within each group of
    ``group_size`` rows, the ``n_reverse_per_group`` rows that have the most
    rejections are assigned the *reverse* strategy (cut at ``last_bad + 1``,
    maximising prefix reuse); the remaining rows with rejections use the
    default strategy (cut at ``first_bad``).

    Returns the same dict schema as ``spec_cut_with_knobs`` plus extra
    diagnostics (``bad_mask``, ``use_reverse``, ``num_rejections``).
    """
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    assert B % group_size == 0, (
        f"Batch size {B} must be divisible by group_size {group_size}"
    )

    valid    = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    delta  = (new_logp - old_logp).masked_fill(~valid, 0.0)
    delta2 = scale * (delta - bias)

    logU   = torch.log(_rand_like_compat(new_logp, seed=seed).clamp_min(1e-12))
    accept = (~valid) | (delta2 >= 0) | (logU <= delta2)

    if p_abs_thresh is not None:
        log_th = math.log(p_abs_thresh)
        accept &= ((new_logp >= log_th) | (~valid))

    bad     = valid & (~accept)                       # [B, R]
    has_bad = bad.any(dim=1)                           # [B]
    num_bad = bad.sum(dim=1)                           # [B]

    first_bad = torch.argmax(bad.to(torch.int8), dim=1)                  # [B]
    last_bad_from_end = torch.argmax(bad.flip(dims=[1]).to(torch.int8), dim=1)
    last_bad  = R - 1 - last_bad_from_end                                # [B]

    default_cut = torch.where(has_bad, first_bad, resp_len)
    reverse_cut = torch.where(has_bad,
                              (last_bad + 1).clamp(max=resp_len),
                              resp_len)

    # --- per-group selection: pick rows that benefit most from reverse ---
    use_reverse = torch.zeros(B, dtype=torch.bool, device=old_logp.device)
    num_groups  = B // group_size

    for g in range(num_groups):
        s, e = g * group_size, (g + 1) * group_size
        grp_bad_idx = torch.nonzero(has_bad[s:e], as_tuple=False).squeeze(-1)
        if grp_bad_idx.numel() == 0:
            continue
        n_pick = min(n_reverse_per_group, grp_bad_idx.numel())
        _, topk_local = torch.topk(num_bad[s:e][grp_bad_idx], n_pick)
        use_reverse[s + grp_bad_idx[topk_local]] = True

    cut_idx = torch.where(use_reverse, reverse_cut, default_cut)

    reuse_mask = (cut_idx == resp_len)
    need_mask  = ~reuse_mask
    idx_reuse  = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need   = torch.nonzero(need_mask,  as_tuple=False).squeeze(-1)

    saved_tokens   = cut_idx.to(torch.float32)
    n_rev_used     = int(use_reverse.sum().item())
    n_default_used = int(has_bad.sum().item()) - n_rev_used

    metrics = {
        "spec/skip_ratio":           reuse_mask.float().mean().item(),
        "spec/cont_ratio":           need_mask.float().mean().item(),
        "spec/avg_cut_idx":          cut_idx.float().mean().item(),
        "spec/avg_resp_len":         resp_len.float().mean().item(),
        "spec/avg_saved_tokens":     saved_tokens.mean().item(),
        "spec/bias":                 float(bias),
        "spec/scale":                float(scale),
        "spec/avg_num_rejections":   num_bad.float().mean().item(),
        "spec/n_reverse_strategy":   float(n_rev_used),
        "spec/n_default_strategy":   float(n_default_used),
        "spec/avg_first_bad":        (first_bad[has_bad].float().mean().item()
                                      if has_bad.any() else 0.0),
        "spec/avg_last_bad":         (last_bad[has_bad].float().mean().item()
                                      if has_bad.any() else 0.0),
        "spec/reverse_avg_cut_idx":  (cut_idx[use_reverse].float().mean().item()
                                      if use_reverse.any() else 0.0),
        "spec/default_avg_cut_idx":  (cut_idx[has_bad & ~use_reverse].float().mean().item()
                                      if (has_bad & ~use_reverse).any() else 0.0),
    }

    return {
        "cut_idx": cut_idx,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "resp_len": resp_len,
        "per_request_max_new_tokens": (R - cut_idx).to(torch.long),
        "bad_mask": bad,
        "use_reverse": use_reverse,
        "num_rejections": num_bad,
        "metrics": metrics,
    }


@torch.no_grad()
def spec_think_reuse_cut(
    response_ids: torch.Tensor,      # [B, R] token IDs of teacher responses
    response_mask: torch.Tensor,     # [B, R] 1=valid response token
    tokenizer,
    *,
    think_end_tag: str = "</think>",
    no_think_action: str = "regenerate",  # "regenerate" | "reuse"
):
    """
    Reuse the teacher's <think>...</think> prefix and regenerate the rest.

    For each row, finds the first occurrence of ``think_end_tag`` in the
    response token IDs and sets ``cut_idx`` to right after it.  Content up
    to (and including) the tag is kept; everything after is re-generated.

    Parameters
    ----------
    no_think_action : str
        What to do when ``</think>`` is not found in a response:
        - ``"regenerate"``: cut_idx=0  → regenerate the whole response
        - ``"reuse"``:      cut_idx=resp_len → keep the whole response
    """
    assert response_ids.shape == response_mask.shape and response_ids.dim() == 2
    B, R = response_ids.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    end_tokens = tokenizer.encode(think_end_tag, add_special_tokens=False)
    end_len = len(end_tokens)
    end_tensor = torch.tensor(end_tokens, device=response_ids.device, dtype=response_ids.dtype)

    cut_idx = torch.zeros(B, dtype=torch.long, device=response_ids.device)

    if end_len == 1:
        tag_id = end_tokens[0]
        matches = (response_ids == tag_id) & valid
        has_match = matches.any(dim=1)
        first_match = torch.argmax(matches.to(torch.int8), dim=1)
        fallback = resp_len if no_think_action == "reuse" else torch.zeros_like(resp_len)
        cut_idx = torch.where(has_match, first_match + 1, fallback)
    else:
        fallback_val = "resp_len" if no_think_action == "reuse" else 0
        for i in range(B):
            rl = resp_len[i].item()
            found = False
            for j in range(max(0, rl - end_len + 1)):
                if torch.equal(response_ids[i, j:j + end_len], end_tensor):
                    cut_idx[i] = j + end_len
                    found = True
                    break
            if not found:
                cut_idx[i] = resp_len[i] if fallback_val == "resp_len" else 0

    cut_idx = cut_idx.clamp(max=resp_len)

    reuse_mask = (cut_idx == resp_len)
    need_mask = ~reuse_mask
    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need = torch.nonzero(need_mask, as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = cut_idx.to(torch.float32)
    n_has_think = int((cut_idx > 0).sum().item())
    n_full_reuse = int(reuse_mask.sum().item())

    metrics = {
        "spec/skip_ratio":              reuse_mask.float().mean().item(),
        "spec/cont_ratio":              need_mask.float().mean().item(),
        "spec/avg_cut_idx":             cut_idx.float().mean().item(),
        "spec/avg_resp_len":            resp_len.float().mean().item(),
        "spec/avg_saved_tokens":        saved_tokens.mean().item(),
        "spec/think_reuse_has_think":   float(n_has_think),
        "spec/think_reuse_full_reuse":  float(n_full_reuse),
        "spec/think_reuse_total":       float(B),
    }

    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


def build_ctx(p_ids, p_msk, p_pos,
                      response_ids,          # [N, R]
                      cut_idx,               # [N]
                      pad_id: int):
    # convention: p_ids/msk/pos shape are [N, P] (sub-batch of idx_need from gen_batch)
    N, P = p_ids.shape
    R    = response_ids.shape[1]
    k_vec = cut_idx.clamp(min=0, max=R)          # [N]
    max_k = int(k_vec.max().item()) if N > 0 else 0
    ctx_len = P + max_k

    # calculate actual length of each prompt (right-aligned non-pad segment)
    Lp = p_msk.sum(dim=1)                         # [N]
    start = ctx_len - (Lp + k_vec)                # [N] length of left pad

    # prepare column index [ctx_len]
    col = torch.arange(ctx_len, device=p_ids.device).unsqueeze(0)   # [1, ctx_len]
    start_ = start.unsqueeze(1)                                     # [N,1]
    Lp_    = Lp.unsqueeze(1)
    k_     = k_vec.unsqueeze(1)

    # 1) boolean mask and source column for prompt segment
    prom_mask = (col >= start_) & (col < start_ + Lp_)              # [N, ctx_len]
    # map columns of ctx back to columns of p_ids: right-aligned with Lp
    src_prom_col = (P - Lp_ ) + (col - start_)                      # [N, ctx_len]
    src_prom_col = src_prom_col.clamp(0, P-1)

    # 2) boolean mask and source column for prefix segment
    pref_mask = (col >= start_ + Lp_) & (col < start_ + Lp_ + k_)   # [N, ctx_len]
    # first cut old responses of each row to max_k, for gather
    resp_cut = response_ids[:, :max_k]                              # [N, max_k]
    src_pref_col = (col - (start_ + Lp_)).clamp(0, max_k-1)         # [N, ctx_len]

    # 3) assemble ctx_ids (default full pad, then fill two segments with where)
    ctx_ids = torch.full((N, ctx_len), pad_id, dtype=p_ids.dtype, device=p_ids.device)
    # prompt segment from the tail of p_ids
    prom_vals = torch.gather(p_ids, 1, src_prom_col)                # [N, ctx_len]
    ctx_ids = torch.where(prom_mask, prom_vals, ctx_ids)
    # prefix segment from response prefix
    pref_vals = torch.gather(resp_cut, 1, src_pref_col)             # [N, ctx_len]
    ctx_ids = torch.where(pref_mask, pref_vals, ctx_ids)

    # 4) attention_mask
    ctx_msk = (prom_mask | pref_mask).to(p_msk.dtype)               # [N, ctx_len]

    # 5) position_ids (prompt segment copy original position, prefix segment increment)
    ctx_pos = torch.zeros((N, ctx_len), dtype=p_pos.dtype, device=p_pos.device)
    prom_pos_vals = torch.gather(p_pos, 1, src_prom_col)            # [N, ctx_len]
    ctx_pos = torch.where(prom_mask, prom_pos_vals, ctx_pos)

    last_pos = p_pos[:, -1].unsqueeze(1)                            # [N,1] original right end position
    ar = torch.arange(1, max_k+1, device=p_pos.device).unsqueeze(0) # [1, max_k]
    pref_pos_table = last_pos + ar                                  # [N, max_k]
    pref_pos_vals = torch.gather(pref_pos_table, 1, src_pref_col.clamp(0, max_k-1))
    ctx_pos = torch.where(pref_mask, pref_pos_vals, ctx_pos)

    return ctx_ids, ctx_msk, ctx_pos, max_k


def align_prev_to_gen(
        *,
        prev_data: dict,
        gen_batch,
        tokenizer,
        n_repeat: int,
        log_prob_key: str = "log_probs",
):
    """
    Align tensors from a *previous rollout file* (`prev_data`) to the row-order
    of the *current* `gen_batch`, where `gen_batch` has already been
    `repeat(interleave=True)` so that **each prompt appears `n_repeat` times
    consecutively.**

    Parameters
    ----------
    prev_data : dict
        Loaded torch object, must contain:
            • prev_data["input"]       – list[str] prompt texts
            • prev_data[log_prob_key]  – Tensor [B, …]   (old log-probs, etc.)
            • prev_data[resp_len_key]  – Tensor [B]
        Row count `B` == len(gen_batch) (after repeat).
    gen_batch : DataProto
        Current batch after `.repeat(interleave=True)`.
    tokenizer : transformers.PreTrainedTokenizer
        Same tokenizer used during rollout.
    n_repeat : int
        Number of consecutive duplicates for every prompt (e.g. 8).
    log_prob_key : str
        Keys of tensors inside `prev_data` you want to align and return.

    Returns
    -------
    aligned : dict
        {
            log_prob_key : Tensor aligned to gen_batch row order,
            "perm"       : LongTensor row-index permutation that did the trick
        }
        If you later need `input_ids`, re-encode on-the-fly from
        `[prev_data["input"][i] for i in perm]`.
    """
    # ---------- reference order (deduped) ----------
    ref_prompts = tokenizer.batch_decode(
        gen_batch.batch["input_ids"][::n_repeat],  # take first row of each block
        skip_special_tokens=True,
    )

    # ---------- build prompt -> deque[row_ids] -----

    bucket = defaultdict(deque)
    for idx, txt in enumerate(prev_data["input"]):
        bucket[txt].append(idx)

    # ---------- pick rows block-by-block -----------
    perm_rows = []
    for txt in ref_prompts:
        if len(bucket[txt]) < n_repeat:
            raise ValueError(
                f"Prompt \n«{txt[:]}…»\n appears fewer than {n_repeat} times "
                "in prev_data – cannot align."
            )
        for _ in range(n_repeat):
            perm_rows.append(bucket[txt].popleft())

    perm = torch.tensor(perm_rows, dtype=torch.long)

    # ---------- index-select tensors ---------------
    aligned = {
        log_prob_key: prev_data[log_prob_key].index_select(0, perm),
        "perm": perm,
    }
    return aligned




def align_prev_to_gen_few_to_n(
    *,
    prev_data: dict,
    gen_batch,
    tokenizer,
    n_prev: int,            # pilot 次数，例如 8
    n_target: int,          # 当前步重复次数，例如 64（假设 gen_batch 已按块排列）
    log_prob_key: str = "log_probs",
    extra_keys: tuple = ("responses", "response_masks", "position_ids"),
):
    """
    非 interleave、按块连续的对齐（块状复制）：
    对每个 prompt，从 pilot 取到 n_prev 条 core = [c0, c1, ..., c_{n_prev-1}]，
    构造 block 为：
        [c0]*k0 + [c1]*k1 + ... + [c_{n_prev-1}]*k_{n_prev-1}
    其中 k_j 尽量均分 n_target（当 n_target % n_prev != 0 时，前面的若干块多 1）。

    要求：
      - prev_data['input'] 中每个 prompt 至少出现 n_prev 次。
      - 文本与 tokenizer.batch_decode(..., skip_special_tokens=True) 的结果完全一致。
    """
    if "input" not in prev_data:
        raise ValueError("prev_data missing 'input' for alignment.")

    total = gen_batch.batch["input_ids"].size(0)
    if total % n_target != 0:
        raise ValueError(f"gen_batch size ({total}) must be divisible by n_target ({n_target}).")
    num_prompts = total // n_target

    # 每组第一条是该 prompt 的代表（非 interleave）
    first_rows = torch.arange(0, total, n_target)
    ref_prompts = tokenizer.batch_decode(
        gen_batch.batch["input_ids"][first_rows],
        skip_special_tokens=True,
    )

    # pilot 侧建桶（按纯文本）
    bucket = defaultdict(deque)
    for i, txt in enumerate(prev_data["input"]):
        bucket[txt].append(i)

    # 块状复制参数：尽量均分 n_target 到 n_prev 块
    base = n_target // n_prev         # 每块最少重复次数
    rem  = n_target %  n_prev         # 有 rem 块需要多 1 次

    perm_rows = []
    for txt in ref_prompts:
        if len(bucket[txt]) < n_prev:
            raise ValueError(
                f"Prompt «{txt[:60]}…» appears fewer than {n_prev} times in prev_data – cannot align."
            )
        # 取出 pilot 的 n_prev 条
        core = [bucket[txt].popleft() for _ in range(n_prev)]

        # 按块状复制：c0 连续 k0 次，c1 连续 k1 次，...
        block = []
        for j, idx in enumerate(core):
            reps = base + (1 if j < rem else 0)   # 前 rem 块多 1 次
            block.extend([idx] * reps)

        # 保险：长度可能因为四舍五入多/少 1，裁剪到 n_target
        block = block[:n_target]
        perm_rows.extend(block)

    perm = torch.tensor(perm_rows, dtype=torch.long)

    aligned = {
        log_prob_key: prev_data[log_prob_key].index_select(0, perm),
        "perm": perm,
    }
    for k in extra_keys:
        if k in prev_data:
            aligned[k] = prev_data[k].index_select(0, perm)

    return aligned