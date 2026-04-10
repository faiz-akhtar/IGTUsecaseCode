import numpy as np
import pandas as pd
from dataclasses import dataclass
from mechanisms import (
    fee_schedule_progressive,
    fee_schedule_ijcai_style,
    assign_reviews,
    reviewer_signal_continuous,
    estimate_conditional_matrix,
    peer_prediction_payment,
    accept_top_k,
)

@dataclass
class SimParams:
    num_potential_papers: int = 10_000
    acceptance_capacity: int = 2_500
    reviews_per_paper: int = 3
    num_reviewers: int = 5_000
    reviewer_cost: float = 2.0  # cost units for high effort per review

    # Quality generation
    quality_beta_a: float = 2.0
    quality_beta_b: float = 5.0

    # Author groups for fairness
    p_waiver: float = 0.35
    omega_waiver: float = 0.35  # 65% waiver
    omega_regular: float = 1.0

    # Author decision model (stylized)
    author_value: float = 200.0
    submission_private_cost: float = 5.0  # non-monetary time cost
    # acceptance probability proxy: increases with q; we model expected accept value as author_value*q
    # so submit if author_value*q - fee - submission_private_cost >= 0

    # Review model
    sigma_hi: float = 0.7
    sigma_lo: float = 1.6

def generate_population(params: SimParams, rng):
    q = rng.beta(params.quality_beta_a, params.quality_beta_b, size=params.num_potential_papers)
    waiver = rng.random(params.num_potential_papers) < params.p_waiver
    omega = np.where(waiver, params.omega_waiver, params.omega_regular)
    return q, waiver, omega

def author_submit_decisions(q, omega, fee_fn):
    """
    fee_fn takes (j, omega_i) but here we treat each paper as independently owned with j=1 (conservative),
    OR we can simulate per-author multiple submissions. For realism, we implement a per-author model in run().
    """
    raise NotImplementedError

def run_mechanism(params: SimParams, mech_name: str, rng, fee_F0=60.0, fee_beta=40.0,
                  ijcai_fee=100.0, ijcai_cap=8,
                  bonus_b=8.0, payment_shift_a=8.0,
                  flat_honorarium=0.0):
    """
    Returns dict metrics and a dataframe with per-review payments if needed.
    """

    # --- Build authors with multiple potential papers (realistic) ---
    # We generate ~5000 authors, each with Poisson(2) papers, total approx 10k.
    num_authors = 5_000
    papers_per_author = rng.poisson(2.0, size=num_authors) + 1  # at least 1
    # scale to match approx target
    scale = params.num_potential_papers / papers_per_author.sum()
    papers_per_author = np.maximum(1, np.floor(papers_per_author * scale).astype(int))
    # adjust rounding
    diff = params.num_potential_papers - papers_per_author.sum()
    if diff > 0:
        papers_per_author[:diff] += 1
    elif diff < 0:
        take = np.where(papers_per_author > 1)[0][:(-diff)]
        papers_per_author[take] -= 1

    # author group
    waiver_author = rng.random(num_authors) < params.p_waiver
    omega_author = np.where(waiver_author, params.omega_waiver, params.omega_regular)

    # allocate papers to authors
    author_id = np.repeat(np.arange(num_authors), papers_per_author)
    omega_paper = omega_author[author_id]
    waiver_paper = waiver_author[author_id]

    # paper quality
    q = rng.beta(params.quality_beta_a, params.quality_beta_b, size=params.num_potential_papers)

    # --- Determine submission decisions paper-by-paper with progressive fees per author ---
    submitted = np.zeros(params.num_potential_papers, dtype=bool)
    fees_paid = np.zeros(params.num_potential_papers, dtype=float)

    # Count submissions per author to apply progressive fee
    sub_count = np.zeros(num_authors, dtype=int)

    for p in range(params.num_potential_papers):
        i = author_id[p]
        j = sub_count[i] + 1  # would be j-th if submitted
        omega_i = omega_author[i]

        if mech_name == "baseline_no_fee":
            fee = 0.0
            cap_ok = True
        elif mech_name == "ijcai_style":
            cap_ok = (j <= ijcai_cap)
            fee = fee_schedule_ijcai_style(j, free_first=True, fee=ijcai_fee, cap=ijcai_cap, omega=omega_i)
        elif mech_name == "ours_progressive_pp":
            fee = fee_schedule_progressive(j, F0=fee_F0, beta=fee_beta, omega=omega_i)
            cap_ok = True
        else:
            raise ValueError(mech_name)

        # submit if expected private benefit >= costs
        # expected accept value proxy: author_value * q
        utility_submit = params.author_value * q[p] - fee - params.submission_private_cost

        if cap_ok and utility_submit >= 0:
            submitted[p] = True
            fees_paid[p] = fee
            sub_count[i] += 1

    # Filter to submitted papers
    idx = np.where(submitted)[0]
    S = len(idx)
    if S == 0:
        return {"mech": mech_name, "S": 0}

    q_sub = q[idx]
    waiver_sub = waiver_paper[idx]
    omega_sub = omega_paper[idx]
    author_sub = author_id[idx]
    fees_sub = fees_paid[idx]

    # --- Assign reviews ---
    R = params.reviews_per_paper
    assignments = assign_reviews(S, params.num_reviewers, R=R, rng=rng)  # shape (S,R)

    # --- Reviewer effort response (stylized best response) ---
    # Baseline: no incentives => mostly low effort
    # IJCAI-style: flat honorarium might increase some effort but weaker than peer prediction
    # Ours: peer prediction with bonus_b -> higher effort if expected gain > cost
    #
    # For simulation, we implement probabilistic best response:
    # effort_prob = sigmoid(gain - cost)
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    if mech_name == "baseline_no_fee":
        expected_gain = 0.0
    elif mech_name == "ijcai_style":
        expected_gain = flat_honorarium
    elif mech_name == "ours_progressive_pp":
        # expected informational gain scales with b; approximate
        expected_gain = 0.35 * bonus_b  # tuned so effort responds reasonably
    else:
        expected_gain = 0.0

    effort_prob = sigmoid(expected_gain - params.reviewer_cost)

    # --- Generate review reports ---
    scores = np.zeros((S, R), dtype=int)
    effort = np.zeros((S, R), dtype=int)
    for p in range(S):
        for t in range(R):
            e = 1 if rng.random() < effort_prob else 0
            effort[p, t] = e
            scores[p, t] = reviewer_signal_continuous(
                q_sub[p], e, rng, sigma_hi=params.sigma_hi, sigma_lo=params.sigma_lo
            )

    # --- Decision rule: posterior proxy from mean score ---
    # Convert mean score to a monotone "posterior score"
    mean_score = scores.mean(axis=1)
    posterior_score = (mean_score - 1.0) / 4.0  # in [0,1] roughly
    K = min(params.acceptance_capacity, S)
    accepted = accept_top_k(posterior_score, K=K)

    # --- Reviewer payments ---
    total_payout = 0.0
    if mech_name == "ours_progressive_pp":
        # Build conditional matrix based on all paired reviews in this run (anchored estimation)
        # Pair each (p,t) with random peer t'
        self_list, peer_list = [], []
        for p in range(S):
            for t in range(R):
                t2 = rng.integers(0, R-1)
                if t2 >= t:
                    t2 += 1
                self_list.append(scores[p, t])
                peer_list.append(scores[p, t2])
        M_cond = estimate_conditional_matrix(np.array(self_list), np.array(peer_list), num_levels=5)

        for p in range(S):
            for t in range(R):
                t2 = rng.integers(0, R-1)
                if t2 >= t:
                    t2 += 1
                pay = peer_prediction_payment(scores[p, t], scores[p, t2], M_cond, a=payment_shift_a, b=bonus_b)
                total_payout += pay
    elif mech_name == "ijcai_style":
        total_payout = flat_honorarium * S * R
    else:
        total_payout = 0.0

    # --- Metrics ---
    fee_revenue = fees_sub.sum()
    accepted_quality_mean = q_sub[accepted].mean() if accepted.any() else 0.0
    submitted_quality_mean = q_sub.mean()

    # Decision "accuracy": rank correlation between posterior_score and true q
    # (proxy for how well decisions track true quality)
    from scipy.stats import spearmanr
    rho = spearmanr(posterior_score, q_sub).correlation
    rho = float(rho) if rho is not None else np.nan

    # fairness: submission rate by waiver status among potential papers
    # (computed at paper level)
    # We compute among potential papers: but here we only have waiver_paper for all papers.
    waiver_all = waiver_paper
    sub_rate_waiver = submitted[waiver_all].mean() if waiver_all.any() else np.nan
    sub_rate_regular = submitted[~waiver_all].mean() if (~waiver_all).any() else np.nan

    # reviewer effort stats
    effort_rate = effort.mean()

    out = dict(
        mech=mech_name,
        potential_papers=params.num_potential_papers,
        submitted=S,
        accepted=int(accepted.sum()),
        acc_rate=float(accepted.mean()),
        accepted_quality_mean=float(accepted_quality_mean),
        submitted_quality_mean=float(submitted_quality_mean),
        decision_rank_corr=float(rho),
        reviewer_effort_rate=float(effort_rate),
        fee_revenue=float(fee_revenue),
        reviewer_payout=float(total_payout),
        budget_surplus=float(fee_revenue - total_payout),
        sub_rate_waiver=float(sub_rate_waiver),
        sub_rate_regular=float(sub_rate_regular),
    )
    return out

def run_all(seed=0):
    rng = np.random.default_rng(seed)
    params = SimParams()

    results = []
    results.append(run_mechanism(params, "baseline_no_fee", rng))
    results.append(run_mechanism(params, "ijcai_style", rng, ijcai_fee=100.0, ijcai_cap=8, flat_honorarium=1.0))
    results.append(run_mechanism(params, "ours_progressive_pp", rng,
                                 fee_F0=60.0, fee_beta=40.0, bonus_b=8.0, payment_shift_a=8.0))
    return pd.DataFrame(results)