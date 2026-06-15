import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from statannotations.Annotator import Annotator


def plot_stat_boxplot(df, feature, result_dir):
 
    plt.figure(figsize=(8, 6))
    
    ax = sns.boxplot(x='cluster', y=feature, data=df, palette='Set2')
    
    clusters = sorted(df['cluster'].unique())
    pairs = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pairs.append((clusters[i], clusters[j]))

    annotator = Annotator(ax, pairs, data=df, x='cluster', y=feature)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    
    plt.title(f"Group Comparison: {feature}", fontsize=14)
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    
    plt.savefig(os.path.join(result_dir, f"stat_plot_{feature}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_dimension_reduction_plots(df, X_scaled, result_dir):
  
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    _plot_2d(pca_result, df['cluster'], "PCA", result_dir)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    _plot_2d(tsne_result, df['cluster'], "t-SNE", result_dir)

    # UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    _plot_2d(umap_result, df['cluster'], "UMAP", result_dir)


def _plot_2d(data, labels, title, result_dir):

    plt.figure(figsize=(8, 6))
    
    scatter = plt.scatter(data[:, 0], data[:, 1], 
                          c=labels, 
                          cmap='Set1', 
                          alpha=0.7, 
                          s=30, 
                          edgecolors='white', 
                          linewidth=0.5)
    
    handles, _ = scatter.legend_elements()
    unique_labels = sorted(list(set(labels)))
    plt.legend(handles, [f"Cluster {c}" for c in unique_labels], 
               title="Clusters", loc='best', fontsize=10)
    
    plt.title(f"{title} Projection", fontsize=14, fontweight='bold')
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--') 
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 파일 저장
    plt.savefig(os.path.join(result_dir, f"{title.lower()}_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()