import numpy as np
import re
from collections import Counter
from typing import Dict, Any, List

EPS = 1e-12
LAMBDA = 1e-2

def _is_nonsense(text: str) -> bool:
    if text is None:
        return True
    t = text.strip().lower()
    if len(t) < 10:
        return True
    if any(p in t for p in ("i don't know", "i do not know", "idk", "no idea", "not sure", "blah", "asdf")):
        return True
    tokens = re.findall(r"\w+", t)
    if len(tokens) <= 2:
        return True
    if len(tokens) > 0:
        counts = Counter(tokens)
        if counts.most_common(1)[0][1] / float(len(tokens)) >= 0.6:
            return True
    return False

def _ledoit_wolf_shrinkage_cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    m, d = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    sample = (Xc.T @ Xc) / float(max(1, m))
    trace = np.trace(sample)
    F = (trace / float(d)) * np.eye(d)
    beta = 0.1
    Sigma = (1.0 - beta) * sample + beta * F
    return Sigma

def _regularize_G(C: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    d = C.shape[0]
    return (1.0 - lam) * C + lam * np.eye(d)

def _mahalanobis_root(delta: np.ndarray, G: np.ndarray) -> float:
    val = float(delta.T @ G @ delta)
    if val < 0:
        val = max(val, 0.0)
    return float(np.sqrt(val))

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return 0.0
    return float((u @ v) / (nu * nv))

def score_question(canonical_text: str,
                   candidate_text: str,
                   O_k_candidate,
                   O_k_canonical,
                   O_r_candidate,
                   O_r_canonical,
                   role_config: Dict[str, Any]) -> Dict[str, Any]:

    if _is_nonsense(candidate_text):
        return {
            "CCI_local": 0.0,
            "s_IG": 0.0,
            "s_geo": 0.0,
            "aspect_scores": [],
            "raw": {"nonsense": True}
        }

    O_k_candidate = np.asarray(O_k_candidate, dtype=float)
    O_k_canonical = np.asarray(O_k_canonical, dtype=float)
    O_r_candidate = np.asarray(O_r_candidate, dtype=float)
    O_r_canonical = np.asarray(O_r_canonical, dtype=float)

    if O_k_candidate.shape != O_k_canonical.shape:
        return {
            "CCI_local": 0.0,
            "s_IG": 0.0,
            "s_geo": 0.0,
            "aspect_scores": [],
            "raw": {"shape_mismatch": True}
        }

    delta = O_k_candidate - O_k_canonical
    d = delta.shape[0]

    micro = role_config.get("canonical_micro_corpus_embeddings", None)
    if micro is None or len(micro) == 0:
        micro = O_k_canonical.reshape(1, -1)
    micro = np.asarray(micro, dtype=float)
    try:
        C = _ledoit_wolf_shrinkage_cov(micro)
    except Exception:
        C = np.cov(micro.T) if micro.shape[0] > 1 else np.eye(d)

    G = _regularize_G(C, lam=LAMBDA)

    dG_candidate = _mahalanobis_root(delta, G)

    offtopic_embed = role_config.get("offtopic_embedding", None)
    if offtopic_embed is None:
        centroid = micro.mean(axis=0)
        try:
            eigvals, eigvecs = np.linalg.eigh(C + 1e-9 * np.eye(d))
            pc = eigvecs[:, -1]
            offtopic_embed = centroid + 5.0 * pc
        except Exception:
            offtopic_embed = centroid + np.ones(d)
    offtopic_embed = np.asarray(offtopic_embed, dtype=float)
    delta_off = offtopic_embed - O_k_canonical
    dG_off = _mahalanobis_root(delta_off, G)

    s_IG = max(0.0, 1.0 - (dG_candidate / (dG_off + EPS)))

    R = role_config.get("R_frame", None)
    if R is None:
        norm = np.linalg.norm(delta) + EPS
        R = (delta / norm).reshape(d, 1)
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    n_aspects = R.shape[1]

    energies: List[float] = []
    for i in range(n_aspects):
        Ri = R[:, i]
        proj = float(delta @ Ri)
        energies.append(proj ** 2)

    E_i_max = role_config.get("E_i_max", None)
    if E_i_max is None:
        deltas_micro = micro - O_k_canonical.reshape(1, -1)
        E_i_max = []
        for i in range(n_aspects):
            Ri = R[:, i]
            projs = (deltas_micro @ Ri) ** 2
            try:
                emax = float(max(np.percentile(projs, 95), EPS))
            except Exception:
                emax = float(np.max(projs) if projs.size > 0 else EPS)
            E_i_max.append(emax)

    aspect_scores: List[float] = []
    for i, E_i in enumerate(energies):
        denom = float(E_i_max[i] if i < len(E_i_max) else (max(energies) + EPS))
        s_i = 1.0 - (E_i / (denom + EPS))
        s_i = min(max(float(s_i), 0.0), 1.0)
        aspect_scores.append(s_i)

    s_geo = float((_cosine(O_r_candidate, O_r_canonical) + 1.0) / 2.0)
    s_geo = min(max(s_geo, 0.0), 1.0)

    try:
        sims = [_cosine(O_k_candidate, row) for row in micro]
        mean_sim = float(np.mean(sims)) if len(sims) > 0 else _cosine(O_k_candidate, O_k_canonical)
        s_consistency = float((mean_sim + 1.0) / 2.0)
    except Exception:
        s_consistency = 1.0

    try:
        noise_scale = float(role_config.get("robustness_noise_scale", 1e-3))
        trials = int(role_config.get("robustness_trials", 8))
        s_vals = []
        for _ in range(max(1, trials)):
            noise = np.random.normal(scale=noise_scale, size=delta.shape)
            dG_pert = _mahalanobis_root(delta + noise, G)
            s_vals.append(max(0.0, 1.0 - (dG_pert / (dG_off + EPS))))
        var = float(np.var(s_vals))
        s_robustness = float(max(0.0, 1.0 - var / (var + EPS)))
    except Exception:
        s_robustness = 1.0

    weights = role_config.get("weights", {})
    w_ig = float(weights.get("w_ig", 0.4))
    w_i_list = weights.get("w_i", [(1.0 - w_ig) / max(1, n_aspects)] * n_aspects)
    if len(w_i_list) != n_aspects:
        w_i_list = (w_i_list * n_aspects)[:n_aspects]
    w_geo = float(weights.get("w_geo", 0.1))
    w_cons = float(weights.get("w_cons", 0.05))
    w_rob = float(weights.get("w_rob", 0.05))

    weight_vec = np.array([w_ig] + w_i_list + [w_geo, w_cons, w_rob], dtype=float)
    if weight_vec.sum() <= 0:
        weight_vec = np.ones_like(weight_vec)
    weight_vec = weight_vec / float(weight_vec.sum())

    w_ig_n = float(weight_vec[0])
    w_i_n = weight_vec[1:1 + n_aspects].tolist()
    w_geo_n = float(weight_vec[1 + n_aspects])
    w_cons_n = float(weight_vec[2 + n_aspects])
    w_rob_n = float(weight_vec[3 + n_aspects])

    cci_local = w_ig_n * s_IG
    for wi, si in zip(w_i_n, aspect_scores):
        cci_local += float(wi * si)
    cci_local += w_geo_n * s_geo + w_cons_n * s_consistency + w_rob_n * s_robustness
    cci_local = float(min(max(cci_local, 0.0), 1.0))

    return {
        "CCI_local": cci_local,
        "s_IG": float(s_IG),
        "s_geo": float(s_geo),
        "aspect_scores": [float(x) for x in aspect_scores],
        "raw": {
            "dG_candidate": float(dG_candidate),
            "dG_off": float(dG_off),
            "energies": energies,
            "E_i_max": E_i_max,
            "s_consistency": s_consistency,
            "s_robustness": s_robustness,
            "weights_normalized": {
                "w_ig": w_ig_n,
                "w_i": w_i_n,
                "w_geo": w_geo_n,
                "w_cons": w_cons_n,
                "w_rob": w_rob_n
            }
        }
    }
