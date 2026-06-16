from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd

def run_statistical_analysis(df, feature_cols, cluster_col="cluster"):
    results = []

    N = len(df)
    
    for feature in feature_cols:
        groups = [df[df[cluster_col] == c][feature].dropna() for c in sorted(df[cluster_col].unique())]
        h_stat, p_value = kruskal(*groups)

        # 효과 크기(Epsilon-squared)
        # 공식: ε² = H / ((N² - 1) / (N + 1))
        epsilon_sq = h_stat / ((N**2 - 1) / (N + 1))

        results.append({
            "feature": feature,
            "H_statistic": h_stat,
            "epsilon_squared": epsilon_sq,  # 추가된 지표
            "p_value": p_value,
            "is_significant": p_value < 0.001
        })
    
    stat_df = pd.DataFrame(results)

    posthoc_results = {}
    for feature in feature_cols:
        if stat_df[stat_df['feature'] == feature]['is_significant'].iloc[0]:
            posthoc = sp.posthoc_dunn(df, val_col=feature, group_col=cluster_col, p_adjust="bonferroni")
            posthoc_results[feature] = posthoc
            
    return stat_df, posthoc_results