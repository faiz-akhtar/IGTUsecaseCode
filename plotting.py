import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def ensure_dirs():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

def save_summary_table_tex(df: pd.DataFrame, path="tables/summary_table.tex"):
    cols = [
        "mech", "submitted", "accepted", "acc_rate",
        "accepted_quality_mean", "decision_rank_corr",
        "reviewer_effort_rate", "fee_revenue", "reviewer_payout", "budget_surplus",
        "sub_rate_waiver", "sub_rate_regular",
    ]
    d = df[cols].copy()
    d["acc_rate"] = (100 * d["acc_rate"]).map(lambda x: f"{x:.1f}\\%")
    for c in ["accepted_quality_mean","decision_rank_corr","reviewer_effort_rate"]:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    for c in ["fee_revenue","reviewer_payout","budget_surplus"]:
        d[c] = d[c].map(lambda x: f"{x:,.0f}")
    for c in ["sub_rate_waiver","sub_rate_regular"]:
        d[c] = d[c].map(lambda x: f"{100*x:.1f}\\%")

    d = d.rename(columns={
        "mech":"Mechanism",
        "submitted":"Submitted",
        "accepted":"Accepted",
        "acc_rate":"Acceptance",
        "accepted_quality_mean":"Mean $q$ (accepted)",
        "decision_rank_corr":"Spearman($\\hat q,q$)",
        "reviewer_effort_rate":"Effort rate",
        "fee_revenue":"Fee revenue",
        "reviewer_payout":"Reviewer payout",
        "budget_surplus":"Surplus",
        "sub_rate_waiver":"Submit rate (waiver)",
        "sub_rate_regular":"Submit rate (regular)",
    })

    tex = d.to_latex(index=False, escape=False, column_format="lrrrrrrrrrrr",
                     caption="Summary metrics (single-seed run; regenerate by averaging over multiple seeds for final numbers).",
                     label="tab:summary",
                     longtable=False)
    with open(path, "w") as f:
        f.write(tex)

def plot_bar(df, y, title, ylabel, path):
    plt.figure(figsize=(7.2, 4.0))
    ax = sns.barplot(data=df, x="mech", y=y, hue="mech", legend=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_fairness(df, path="figures/fairness_submission_rates.pdf"):
    d = df.melt(id_vars=["mech"], value_vars=["sub_rate_waiver","sub_rate_regular"],
                var_name="group", value_name="rate")
    d["group"] = d["group"].map({
        "sub_rate_waiver":"Waiver group",
        "sub_rate_regular":"Regular group"
    })
    plt.figure(figsize=(7.2, 4.0))
    ax = sns.barplot(data=d, x="mech", y="rate", hue="group")
    ax.set_title("Submission rates by resource group")
    ax.set_xlabel("")
    ax.set_ylabel("Submission rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def make_all_plots(df):
    ensure_dirs()
    plot_bar(df, "accepted_quality_mean",
             "Accepted-paper mean quality", "Mean true quality of accepted papers",
             "figures/accepted_quality_vs_mechanism.pdf")
    plot_bar(df, "reviewer_effort_rate",
             "Reviewer effort induced", "Fraction high-effort reviews",
             "figures/reviewer_effort_vs_mechanism.pdf")
    plot_fairness(df)