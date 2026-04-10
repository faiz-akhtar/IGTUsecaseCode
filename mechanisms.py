import numpy as np

def fee_schedule_progressive(j, F0, beta, omega=1.0):
    """Fee for j-th submission by an author."""
    return omega * (F0 + beta * (j - 1))

def fee_schedule_ijcai_style(j, free_first=True, fee=100.0, cap=8, omega=1.0):
    """
    IJCAI-like stylized: first free (if enabled), then flat fee.
    cap is enforced outside.
    """
    if free_first and j == 1:
        return 0.0
    return omega * fee

def assign_reviews(num_papers, num_reviewers, R=3, rng=None):
    """
    Random assignment with approximate uniform load.
    Returns: list of arrays reviewers_for_paper[p] of length R.
    """
    rng = np.random.default_rng() if rng is None else rng
    # Create a multiset of reviewer IDs with repetition to match total reviews
    total_reviews = num_papers * R
    reviewers = np.arange(num_reviewers)
    # approximate balance by repeating and shuffling
    reps = int(np.ceil(total_reviews / num_reviewers))
    pool = np.tile(reviewers, reps)[:total_reviews]
    rng.shuffle(pool)
    return pool.reshape(num_papers, R)

def reviewer_signal_continuous(q, effort, rng, sigma_hi=0.7, sigma_lo=1.6):
    """
    Continuous-score proxy: observed score = 1 + 4*q + noise, then clipped to [1,5] and rounded.
    Higher effort => smaller noise.
    """
    mu = 1.0 + 4.0 * q
    sigma = sigma_hi if effort == 1 else sigma_lo
    x = mu + rng.normal(0.0, sigma)
    x = np.clip(x, 1.0, 5.0)
    # report discrete score in {1,2,3,4,5}
    return int(np.rint(x))

def estimate_conditional_matrix(scores_self, scores_peer, num_levels=5, eps=1e-12):
    """
    Estimate Pr(peer=y | self=x) from paired samples.
    scores_self, scores_peer: arrays in {1..num_levels}
    Returns matrix M[x-1, y-1].
    """
    M = np.zeros((num_levels, num_levels), dtype=float)
    for x, y in zip(scores_self, scores_peer):
        M[x-1, y-1] += 1.0
    row_sums = M.sum(axis=1, keepdims=True)
    M = (M + eps) / (row_sums + eps * num_levels)
    return M

def peer_prediction_payment(score_self, score_peer, M_cond, a, b, eps=1e-12):
    """
    Log-likelihood peer prediction payment:
      payment = a + b * log Pr(peer=score_peer | self=score_self)
    """
    prob = M_cond[score_self-1, score_peer-1]
    return a + b * np.log(max(prob, eps))

def accept_top_k(posterior_scores, K):
    """Accept top K by posterior_scores."""
    idx = np.argsort(-posterior_scores)
    accepted = np.zeros(len(posterior_scores), dtype=bool)
    accepted[idx[:K]] = True
    return accepted