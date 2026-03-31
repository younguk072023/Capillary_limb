import pandas as pd
import plotly.express as px

# 1. 데이터 로드
FINAL_RESULT_CSV = r"C:\Users\park_younguk\Desktop\Capillary_limb\limb\clustering_results\final_clustering_result.csv"
df = pd.read_csv(FINAL_RESULT_CSV)

# 2. 3D 산점도 생성
# x, y, z축에 각각 영욱님이 뽑은 3가지 직경을 넣습니다.
fig = px.scatter_3d(
    df, 
    x='A_Diameter', 
    y='V_Diameter', 
    z='Loop_Diameter',
    color='Cluster',           # 군집별 색상
    symbol='Cluster',          # 군집별 모양 다르게
    opacity=0.7,               # 투명도
    title='3D Visualization of Capillary Clusters',
    labels={'A_Diameter': 'Arterial (px)', 
            'V_Diameter': 'Venous (px)', 
            'Loop_Diameter': 'Loop (px)'},
    hover_data=['Filename']    # 마우스 올리면 파일명 보이게
)

# 3. 그래프 설정 (회전 및 크기)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# 4. 브라우저에서 열기 (또는 html로 저장)
fig.show()
# fig.write_html("capillary_3d_clusters.html") # 파일로 저장하고 싶을 때