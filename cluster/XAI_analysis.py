import os
import shap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from visualization import plot_confusion_matrix

def run_xai_analysis(df, feature_cols, result_dir):
    xai_dir = os.path.join(result_dir, "xai_plots")
    os.makedirs(xai_dir, exist_ok=True)

    feature_name_map = {
        "arterial_diameter": "Arterial Diameter ",
        "venous_diameter": "Venous Diameter ",
        "loop_diameter": "Loop Diameter ",
        "va_ratio": "V/A Ratio",
        "avg_diameter3": "Average Diameter "
    }
    
    df_xai = df[feature_cols].rename(columns=feature_name_map)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_cols], df['cluster'])

    y_pred = rf.predict(df[feature_cols])
    
    # 1. 텍스트 성적표 출력
    print("--- Model Performance Report ---")
    print(classification_report(df['cluster'], y_pred))
    
    # 2. 혼동 행렬 시각화 (논문용 이미지)
    cm = confusion_matrix(df['cluster'], y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Cluster')
    plt.savefig(os.path.join(xai_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df[feature_cols])

    # 1. SHAP Bar Plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, df_xai, plot_type="bar", show=False)
    
    # 범례 재설정
    leg = plt.legend()
    # 버전 호환성을 고려하여 핸들 가져오기
    handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
    
    # [중요] SHAP은 보통 역순으로 범례를 생성하므로 2, 1, 0 순서를 0, 1, 2로 배치
    new_labels = ["Cluster 0", "Cluster 1", "Cluster 2"]
    plt.legend(handles=[handles[2], handles[1], handles[0]], labels=new_labels, loc='best', prop={'family': 'Times New Roman'},frameon=False)
    
    plt.savefig(os.path.join(xai_dir, "shap_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 클래스별 상세 그래프 (도트 차트)
    classes = sorted(df['cluster'].unique())
    for c_idx in classes:
        plt.figure(figsize=(8, 6))
        class_shap_vals = shap_values[c_idx] if isinstance(shap_values, list) else shap_values[:, :, c_idx]
        
        shap.summary_plot(class_shap_vals, df_xai, show=False)
        plt.savefig(os.path.join(xai_dir, f"shap_dot_cluster{c_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"XAI 분석 완료: {xai_dir} 폴더 확인")