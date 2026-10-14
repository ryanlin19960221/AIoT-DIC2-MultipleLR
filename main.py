"""
================================================================================
專案名稱：50 Startups 多元線性迴歸專案 (Multiple Linear Regression)
方法論：CRISP-DM (Cross-Industry Standard Process for Data Mining)
作者：資深資料科學家 / Python 軟體工程師
================================================================================
"""

import os
import sys
import io
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 設定終端機標準輸出為 UTF-8 編碼
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# Helper Functions: 資料下載與檢核
# ==============================================================================
DATASET_PATH = "50_Startups.csv"
DATASET_URL = "https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv"
MODEL_PATH = "startup_profit_model.pkl"
TRANSFORMER_PATH = "startup_column_transformer.pkl"


def ensure_dataset(file_path: str = DATASET_PATH, download_url: str = DATASET_URL) -> str:
    """確認資料集是否存在，若不存在則自公開來源下載"""
    if not os.path.exists(file_path):
        print(f"[系統訊息] 找不到 {file_path}，正從公開儲存庫下載標準 50_Startups.csv ...")
        urllib.request.urlretrieve(download_url, file_path)
        print(f"[系統訊息] 下載成功，檔案大小: {os.path.getsize(file_path)} bytes。\n")
    return file_path


# ==============================================================================
# Stage 1: Business Understanding (商業理解)
# ==============================================================================
def stage_1_business_understanding():
    """
    Stage 1: 商業理解 (Business Understanding)
    - 專案目標：透過各項經營指標預測新創公司獲利，並辨識關鍵影響因子。
    - 決策價值：為創投機構（VC）與管理層提供資源配置與資本分配的量化決策依據。
    """
    print("\n" + "=" * 80)
    print("【Stage 1: Business Understanding (商業理解)】".center(75))
    print("=" * 80)
    business_goal = (
        "【專案核心目標】\n"
        "本專案旨在透過「研發支出 (R&D Spend)」、「行政管理支出 (Administration)」、\n"
        "「行銷支出 (Marketing Spend)」及「所在州別 (State)」等關鍵運營指標，建立多元線性迴歸模型\n"
        "準確預測新創公司的「獲利 (Profit)」，並找出對公司獲利最具影響力的關鍵驅動因素，\n"
        "以作為創投機構評估投資標的與新創團隊資本預算配置之決策依據。"
    )
    print(business_goal)
    print("-" * 80)


# ==============================================================================
# Stage 2: Data Understanding (資料理解)
# ==============================================================================
def stage_2_data_understanding(file_path: str) -> pd.DataFrame:
    """
    Stage 2: 資料理解 (Data Understanding)
    - 讀取資料集
    - 檢查前五筆 (head)
    - 檢查資料型態與缺失值 (info)
    - 檢視數值欄位敘述統計 (describe)
    """
    print("\n" + "=" * 80)
    print("【Stage 2: Data Understanding (資料理解)】".center(75))
    print("=" * 80)

    df = pd.read_csv(file_path)
    print(f"資料集維度 (Shape): {df.shape[0]} 筆樣本, {df.shape[1]} 個欄位\n")

    print("[1] 前 5 筆資料 (head):")
    print(df.head().to_string(index=False))
    print("\n" + "-" * 50)

    print("[2] 資料結構與缺失值檢查 (info):")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue().strip())
    print("-" * 50)

    print("[3] 數值欄位敘述性統計 (describe):")
    describe_func = getattr(df.describe(), 'map', getattr(df.describe(), 'applymap', None))
    print(describe_func(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)))
    print("-" * 50)

    print("[4] 類別欄位分佈 (State value counts):")
    state_counts = df['State'].value_counts()
    for state, count in state_counts.items():
        print(f"  - {state}: {count} 家新創公司 ({count/len(df):.1%})")

    return df


# ==============================================================================
# Stage 3: Data Preparation (資料準備)
# ==============================================================================
def stage_3_data_preparation(df: pd.DataFrame):
    """
    Stage 3: 資料準備 (Data Preparation)
    - 特徵與目標分離：X 與 y
    - 獨熱編碼 (OneHotEncoder with drop='first') 防止虛擬變數陷阱 (Dummy Variable Trap)
    - 保留轉換後之特徵名稱 (ct.get_feature_names_out)
    - 資料集切分：80% 訓練集, 20% 測試集 (random_state=42)
    """
    print("\n" + "=" * 80)
    print("【Stage 3: Data Preparation (資料準備)】".center(75))
    print("=" * 80)

    # 1. 特徵與目標分離
    feature_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    target_col = 'Profit'

    X = df[feature_cols]
    y = df[target_col]
    print(f"特徵特徵矩陣 X 維度: {X.shape}")
    print(f"目標變數 y 維度: {y.shape}")

    # 2. 類別變數編碼：使用 ColumnTransformer + OneHotEncoder(drop='first')
    # drop='first' 能有效消除共線性，規避虛擬變數陷阱 (Dummy Variable Trap)
    column_transformer = ColumnTransformer(
        transformers=[
            ('state_encoder', OneHotEncoder(drop='first', sparse_output=False), ['State'])
        ],
        remainder='passthrough'
    )

    # 3. 資料集切分 (80% Train, 20% Test)
    # 為防止資料洩漏 (Data Leakage)，切分後再進行 fit_transform 與 transform
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = column_transformer.fit_transform(X_train_raw)
    X_test = column_transformer.transform(X_test_raw)

    # 4. 取得編碼轉換後的特徵名稱
    feature_names = column_transformer.get_feature_names_out()

    print(f"訓練集維度: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"測試集維度: X_test={X_test.shape}, y_test={y_test.shape}")
    print("\n轉換後特徵名稱列表 (get_feature_names_out):")
    for idx, fname in enumerate(feature_names, start=1):
        print(f"  {idx}. {fname}")

    return X_train, X_test, y_train, y_test, column_transformer, feature_names


# ==============================================================================
# Stage 4: Modeling (建立模型)
# ==============================================================================
def stage_4_modeling(X_train: np.ndarray, y_train: pd.Series, feature_names: np.ndarray):
    """
    Stage 4: 建立模型 (Modeling)
    - 初始化與訓練多元線性迴歸模型 (LinearRegression)
    - 提取截距 (Intercept) 與各特徵迴歸係數 (Coefficients)
    """
    print("\n" + "=" * 80)
    print("【Stage 4: Modeling (建立模型)】".center(75))
    print("=" * 80)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"模型截距 (Intercept, β0): {model.intercept_:,.4f}\n")

    # 整理各特徵迴歸係數為 DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient (β)': model.coef_,
        'Absolute Impact': np.abs(model.coef_)
    }).sort_values(by='Absolute Impact', ascending=False).reset_index(drop=True)

    print("各特徵迴歸係數 (Coefficients):")
    print(coef_df[['Feature', 'Coefficient (β)']].to_string(index=False))

    return model, coef_df


# ==============================================================================
# Stage 5: Evaluation & Business Insights (模型評估與商業洞察)
# ==============================================================================
def stage_5_evaluation_and_insights(
    model: LinearRegression,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: np.ndarray,
    coef_df: pd.DataFrame
):
    """
    Stage 5: 模型評估與商業洞察 (Evaluation & Business Insights)
    - 計算 R^2, Adjusted R^2, MAE, RMSE
    - 呈現測試集 Actual vs Predicted vs Residual 對照表
    - 自動依係數輸出商業洞察摘要
    """
    print("\n" + "=" * 80)
    print("【Stage 5: Evaluation & Business Insights (模型評估與商業洞察)】".center(75))
    print("=" * 80)

    # 1. 預測測試集
    y_pred = model.predict(X_test)

    # 2. 計算評估指標
    n = len(y_test)
    p = X_test.shape[1]

    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("[1] 測試集整體績效指標:")
    print(f"  • R² Score (決定係數)      : {r2:.4f} (模型解釋了約 {r2*100:.2f}% 的獲利變異)")
    print(f"  • Adjusted R² (調整後 R²)  : {adj_r2:.4f} (考量特徵數後的解釋力)")
    print(f"  • MAE (平均絕對誤差)       : ${mae:,.2f}")
    print(f"  • RMSE (均方根誤差)        : ${rmse:,.2f}")
    print("-" * 80)

    # 3. 測試集對照表 (Actual vs Predicted vs Residual)
    comparison_df = pd.DataFrame({
        'Actual Profit': y_test.values,
        'Predicted Profit': y_pred,
        'Residual (Error)': y_test.values - y_pred,
        'Abs % Error': np.abs((y_test.values - y_pred) / y_test.values) * 100
    })

    print("[2] 測試集預測明細 (Actual vs. Predicted vs. Residual):")
    format_func = getattr(comparison_df, 'map', getattr(comparison_df, 'applymap', None))
    formatted_comparison = format_func(
        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
    )
    print(formatted_comparison.to_string(index=True))
    print("-" * 80)

    # 4. 商業洞察摘要 (Business Insights)
    rd_coef = float(coef_df.loc[coef_df['Feature'].str.contains('R&D Spend'), 'Coefficient (β)'].iloc[0])
    mkt_coef = float(coef_df.loc[coef_df['Feature'].str.contains('Marketing Spend'), 'Coefficient (β)'].iloc[0])
    admin_coef = float(coef_df.loc[coef_df['Feature'].str.contains('Administration'), 'Coefficient (β)'].iloc[0])

    print("[3] 商業洞察摘要 (Executive Business Insights):")
    insights_text = (
        f"1. 核心獲利引擎 (R&D Spend, β ≈ {rd_coef:.2f})：\n"
        f"   - 研發支出係數高達約 {rd_coef:.2f}，代表在其他條件不變下，每額外投入 $1 美元的研發資金，\n"
        f"     能為公司帶來約 ${rd_coef:.2f} 美元的獲利回報，是驅動新創獲利最顯著且不可或缺的核心引擎。\n"
        f"2. 微幅正向動能 (Marketing Spend, β ≈ {mkt_coef:.4f})：\n"
        f"   - 行銷支出呈現微幅正向效益（約每投入 $1 美元增加 ${mkt_coef:.2f} 美元獲利），\n"
        f"     顯示市場拓展具備輔助擴大獲利規模的效果。\n"
        f"3. 影響極小/次要因子 (Administration & State)：\n"
        f"   - 行政管理費用 (β ≈ {admin_coef:.4f}) 對整體利潤影響極微甚至略呈負相關，可能反映官僚成本；\n"
        f"   - 所在州別 (State) 虛擬變數相較於基準州 (California) 的係數差異極低，\n"
        f"     表明地理區位並非決定獲利水準的關鍵驅動力。\n"
        f"4. 戰略建議：\n"
        f"   - 投資機構與經營團隊應優先將資本集中於研發技術突破，並輔以精準行銷，同時嚴格控制行政間接成本。"
    )
    print(insights_text)
    print("-" * 80)

    return comparison_df


# ==============================================================================
# Stage 6: Deployment & Inference (部署與推論)
# ==============================================================================
def stage_6_deployment(model: LinearRegression, column_transformer: ColumnTransformer):
    """
    Stage 6: 部署與推論 (Deployment & Inference)
    - 序列化儲存模型與特徵轉換器
    """
    print("\n" + "=" * 80)
    print("【Stage 6: Deployment & Inference (部署與推論)】".center(75))
    print("=" * 80)

    # 1. 模型與轉換器持久化
    joblib.dump(model, MODEL_PATH)
    joblib.dump(column_transformer, TRANSFORMER_PATH)
    print(f"已成功儲存模型檔案至: {MODEL_PATH}")
    print(f"已成功儲存特徵轉換器至: {TRANSFORMER_PATH}\n")


def predict_startup_profit(
    rd_spend: float,
    admin_spend: float,
    marketing_spend: float,
    state: str,
    model_path: str = MODEL_PATH,
    transformer_path: str = TRANSFORMER_PATH
) -> float:
    """
    新創公司獲利推論函式 (Inference API)
    支援單筆資料輸入，自動完成特徵編碼轉換並進行利潤預測。

    參數：
    - rd_spend (float): 研發支出金額
    - admin_spend (float): 行政管理支出金額
    - marketing_spend (float): 行銷支出金額
    - state (str): 所在州別 ('California', 'Florida', 'New York')

    回傳：
    - predicted_profit (float): 預估獲利金額
    """
    # 載入模型與轉換器
    loaded_transformer: ColumnTransformer = joblib.load(transformer_path)
    loaded_model: LinearRegression = joblib.load(model_path)

    # 構建單筆輸入 DataFrame
    input_df = pd.DataFrame([{
        'R&D Spend': float(rd_spend),
        'Administration': float(admin_spend),
        'Marketing Spend': float(marketing_spend),
        'State': str(state)
    }])

    # 特徵轉換與預測
    transformed_features = loaded_transformer.transform(input_df)
    predicted_profit = float(loaded_model.predict(transformed_features)[0])

    return predicted_profit


def run_deployment_test_case():
    """執行測試案例推論"""
    test_case = {
        'rd_spend': 160000.0,
        'admin_spend': 130000.0,
        'marketing_spend': 300000.0,
        'state': 'California'
    }

    print("執行推論測試案例 (Inference Test Case):")
    print(f"  • 研發支出 (R&D Spend)       : ${test_case['rd_spend']:,.2f}")
    print(f"  • 行政管理支出 (Administration): ${test_case['admin_spend']:,.2f}")
    print(f"  • 行銷支出 (Marketing Spend)  : ${test_case['marketing_spend']:,.2f}")
    print(f"  • 所在州別 (State)            : {test_case['state']}")

    pred_profit = predict_startup_profit(**test_case)
    print(f"\n>>> 預估獲利金額 (Estimated Profit): ${pred_profit:,.2f} <<<")
    print("=" * 80 + "\n")


# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    # 0. 環境與資料檢查
    file_path = ensure_dataset()

    # Stage 1: Business Understanding
    stage_1_business_understanding()

    # Stage 2: Data Understanding
    df = stage_2_data_understanding(file_path)

    # Stage 3: Data Preparation
    X_train, X_test, y_train, y_test, ct, feature_names = stage_3_data_preparation(df)

    # Stage 4: Modeling
    model, coef_df = stage_4_modeling(X_train, y_train, feature_names)

    # Stage 5: Evaluation & Business Insights
    stage_5_evaluation_and_insights(model, X_test, y_test, feature_names, coef_df)

    # Stage 6: Deployment & Inference
    stage_6_deployment(model, ct)
    run_deployment_test_case()


if __name__ == "__main__":
    main()
