import numpy as np
import pandas as pd
from tqdm import trange
from simulate import run_mechanism, SimParams
from plotting import make_all_plots, save_summary_table_tex, ensure_dirs

def run_multi_seed(num_seeds=10):
    params = SimParams()
    all_rows = []
    for seed in trange(num_seeds, desc="seeds"):
        rng = np.random.default_rng(seed)
        all_rows.append(run_mechanism(params, "baseline_no_fee", rng))
        all_rows.append(run_mechanism(params, "ijcai_style", rng, ijcai_fee=100.0, ijcai_cap=8, flat_honorarium=1.0))
        all_rows.append(run_mechanism(params, "ours_progressive_pp", rng,
                                      fee_F0=60.0, fee_beta=40.0, bonus_b=8.0, payment_shift_a=25.0))
    df = pd.DataFrame(all_rows)
    # average over seeds
    g = df.groupby("mech", as_index=False).mean(numeric_only=True)
    return g

def main():
    ensure_dirs()
    df = run_multi_seed(num_seeds=15)
    df.to_csv("tables/summary_metrics.csv", index=False)
    make_all_plots(df)
    save_summary_table_tex(df, "tables/summary_table.tex")
    print(df)

if __name__ == "__main__":
    main()