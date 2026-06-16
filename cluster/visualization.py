import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from statannotations.Annotator import Annotator


# visualization.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Cluster')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_feature_label(feature):
    """변수명을 논문용 학술 명칭으로 변환"""
    labels = {
        "arterial_diameter": "Arterial Diameter ",
        "venous_diameter": "Venous Diameter ",
        "loop_diameter": "Loop Diameter ",
        "va_ratio": "V/A Ratio",
        "avg_diameter3": "Average Diameter "
    }
    return labels.get(feature, feature.replace('_', ' ').title())

def plot_stat_boxplot(df, feature, result_dir):
    plt.figure(figsize=(8, 6))
    
    # [수정] palette를 분석 전체에서 사용하는 통일된 색상으로 설정
    # 순서: Cluster 0, Cluster 1, Cluster 2
    my_palette = ['#3B6F75', '#D9825B', '#9BAA83']
    
    ax = sns.violinplot(x='cluster', y=feature, data=df, 
                        palette=my_palette, 
                        inner="box") 
    
    # 통계적 유의성 표시
    clusters = sorted(df['cluster'].unique())
    pairs = [(clusters[0], clusters[1]), (clusters[1], clusters[2]), (clusters[0], clusters[2])]
    
    annotator = Annotator(ax, pairs, data=df, x='cluster', y=feature)
    annotator.configure(test=None, text_format='star', loc='inside', verbose=False)
    annotator.set_pvalues_and_annotate([0.0001, 0.0001, 0.0001]) 
    
    # 별표(***) 강제 조정
    for annotation in ax.texts:
        annotation.set_text('***')

    # [수정] y축 라벨을 학술 명칭으로 적용
    plt.xlabel("Cluster", fontname='Times New Roman', fontsize=12)
    plt.ylabel(get_feature_label(feature), fontname='Times New Roman', fontsize=12)
    
    plt.grid(alpha=0.3, linestyle='--', zorder=0)
    sns.despine(top=False, right=False)
    
    plt.savefig(os.path.join(result_dir, f"violin_{feature}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_dimension_reduction_plots(df, X_scaled, result_dir):
  
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_ * 100
    _plot_2d(pca_result, df['cluster'], "PCA", result_dir, var_ratio=var_ratio)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    _plot_2d(tsne_result, df['cluster'], "t-SNE", result_dir)

    # UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    _plot_2d(umap_result, df['cluster'], "UMAP", result_dir)


def _plot_2d(data, labels, title, result_dir, var_ratio=None):
    colors = ['#3B6F75', '#D9825B', '#9BAA83'] 
    
    plt.figure(figsize=(8, 6))
    
    markers = ['s', 'o', '^'] 
    
    for i, cluster in enumerate(sorted(set(labels))):
        idx = (labels == cluster)
        plt.scatter(data[idx, 0], data[idx, 1], 
                    c=colors[i], label=f"Cluster {cluster}", 
                    marker=markers[i], s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

    plt.legend(loc='best')
    
    if "pca" in title.lower():
        xlabel = f"PC1 ({var_ratio[0]:.1f}%)"
        ylabel = f"PC2 ({var_ratio[1]:.1f}%)"
    elif "t-sne" in title.lower():
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"
    elif "umap" in title.lower():
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"
    else:
        xlabel = "Dimension 1"
        ylabel = "Dimension 2"
    
    plt.xlabel(xlabel, fontsize=12, fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize=12, fontname='Times New Roman')
    
    ax = plt.gca()
    ax.set_axisbelow(True) 
    plt.grid(alpha=0.3, linestyle='--', zorder=0)
    
    # [수정된 부분] 
    # 모든 방향의 선을 보이게 하고 검은색으로 고정합니다.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    plt.savefig(os.path.join(result_dir, f"{title.lower()}_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()