import pandas as pd
import numpy as np
import os


def load_and_preprocess_data(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff","", regex=False)

    required_cols = ["filename", "loop_length", "arterial_diameter", "venous_diameter"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 csv에 없습니다: {col}")
    
    for col in required_cols[1:]:
        df[col]=pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    #feature Engineering
    eps=1e-6    #0으로 나누기 에러 방지
    df["loop_diameter"] = df["loop_length"]
    df["va_ratio"] = df["venous_diameter"] / (df["arterial_diameter"] + eps)
    df["avg_diameter3"] = (df["arterial_diameter"] + df["venous_diameter"] + df["loop_diameter"]) / 3.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"data.py: {len(df)}개의 유효 샘플 전처리 완료.")
    return df


