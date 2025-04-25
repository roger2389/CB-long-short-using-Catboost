import pandas as pd
import cvxpy as cp
def model_fit(w_target:pd.Series,prices:pd.Series,AUM:int,lot_size:int = 1000)->pd.Series:
    w_target = AUM/lot_size*w_target
    USE_AUM = w_target.sum()
    index = w_target.index
    num_stocks = len(w_target)
    # 定義變數：每檔股票購買的張數 (整數)
    n = cp.Variable(num_stocks, integer=True)

    # 計算每個股票的實際投資
    investment = cp.multiply(n,prices.loc[index].values)

    # 定義目標函數：最小化投資金額與目標權重對應的投資金額之間的偏差
    總資金誤差 = cp.abs(USE_AUM - cp.sum(investment))
    標的資金誤差 = cp.sum(cp.abs(w_target.values - investment))
    objective = cp.Minimize(總資金誤差 + 標的資金誤差)

    # 添加約束條件
    constraints = [
        cp.sum(investment) <= AUM/lot_size,  # 總投資不得超過預算
        n >= 0  # 購買張數必須為非負
    ]

    # 定義並求解問題
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC)
    if problem.status == cp.OPTIMAL:
        return pd.Series(n.value,index = index).astype(int)

def model_v2_fit(w_target: pd.Series, prices: pd.Series, AUM: int, lot_size: int = 1000, 最低手續費: int = 20) -> pd.Series:
    w_target = AUM / lot_size * w_target
    USE_AUM = w_target.sum()
    index = w_target.index
    num_stocks = len(w_target)

    # 定義變數：每檔股票購買的張數 (整數)
    n = cp.Variable(num_stocks, integer=True)

    # 計算每個股票的實際投資
    investment = cp.multiply(n, prices.loc[index].values)

    # 計算手續費
    手續費 = cp.multiply(investment * lot_size, 0.001425 * 0.16)

    # 定義目標函數：
    # 1. 最小化投資金額與目標權重對應的投資金額之間的偏差
    總資金誤差 = cp.abs(USE_AUM - cp.sum(investment))
    標的資金誤差 = cp.sum(cp.abs(w_target.values - investment))

    # 2. 添加手續費的懲罰項 (若低於最低手續費，加入懲罰)
    手續費懲罰 = 10000*cp.sum(cp.pos(最低手續費 - 手續費))

    # 合併目標函數
    objective = cp.Minimize(總資金誤差 + 標的資金誤差 + 手續費懲罰)

    # 添加約束條件
    constraints = [
        cp.sum(investment) <= AUM / lot_size,  # 總投資不得超過預算
        n >= 0,  # 購買張數必須為非負
    ]

    # 定義並求解問題
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC)

    if problem.status == cp.OPTIMAL:
        return pd.Series(n.value, index=index).astype(int)
    else:
        raise ValueError('无法找到最优解')
    
def model_v3_fit(w_target: pd.Series, prices: pd.Series, AUM: int, lot_size: int = 1000, 最低手續費: int = 20, 有限購入張數:dict = {}) -> pd.Series:
    w_target = AUM / lot_size * w_target
    USE_AUM = w_target.sum()
    index = w_target.index
    num_stocks = len(w_target)

    # 定義變數：每檔股票購買的張數 (整數)
    n = cp.Variable(num_stocks, integer=True)

    # 計算每個股票的實際投資
    investment = cp.multiply(n, prices.loc[index].values)

    # 計算手續費
    手續費 = cp.multiply(investment * lot_size, 0.001425 * 0.16)

    # 定義目標函數：
    # 1. 最小化投資金額與目標權重對應的投資金額之間的偏差
    總資金誤差 = cp.abs(USE_AUM - cp.sum(investment))
    標的資金誤差 = cp.sum(cp.abs(w_target.values - investment))

    # 2. 添加手續費的懲罰項 (若低於最低手續費，加入懲罰)
    手續費懲罰 = 10000*cp.sum(cp.pos(最低手續費 - 手續費))

    # 合併目標函數
    objective = cp.Minimize(總資金誤差 + 標的資金誤差 + 手續費懲罰)

    # 添加約束條件
    constraints = [
        cp.sum(investment) <= AUM / lot_size,  # 總投資不得超過預算
        n >= 0,  # 購買張數必須為非負
    ]
    for 標的代號,最大限制張數 in 有限購入張數.items():
        if 標的代號 in index:
            constraints.append(n[index.get_loc(標的代號)] <= 最大限制張數)
        else:
            print(f'警告：{標的代號}不在標的清單中')

    # 定義並求解問題
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC)

    if problem.status == cp.OPTIMAL:
        return pd.Series(n.value, index=index).astype(int)
    else:
        raise ValueError('无法找到最优解')
