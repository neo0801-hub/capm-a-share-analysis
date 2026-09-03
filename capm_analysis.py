# -*- coding: utf-8 -*-
"""
capm_analysis.py — CAPM 回归分析（读本地 CSV，无需联网）
模型: R_i - rf = alpha + beta * (R_m - rf) + epsilon
      其中 R_m 为沪深300 日收益率, rf 为常数无风险利率(2% 年化, 日化折算)

用法: python capm_analysis.py
输入: data/{symbol}.csv 与 data/sh000300_index.csv（由 capm_fetch.py 生成）
输出:
  results/capm_results.csv   8 只股票回归系数汇总表
  results/fig1_beta.png      beta 横向条形图（按大小排序）
  results/fig2_scatter.png   个股超额收益 vs 市场超额收益 散点+拟合线（2x4）
  results/fig3_sml.png       证券市场线 SML 检验
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------- 基础设置 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

# 8 只股票: 代码 -> (名称, 行业)
STOCKS = {
    "sh600519": ("Moutai", "Liquor"),
    "sh600036": ("CMB", "Banking"),
    "sz300750": ("CATL", "New Energy"),
    "sh600276": ("Hengrui", "Pharma"),
    "sh600030": ("CITIC Sec", "Brokerage"),
    "sz002594": ("BYD", "EV"),
    "sh601318": ("Ping An", "Insurance"),
    "sz000002": ("Vanke", "Real Estate"),
}
INDEX_FILE = "sh000300_index"

RF_ANNUAL = 0.02          # 无风险利率 2% 年化（课设简化）
TRADING_DAYS = 252
RF_DAILY = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1   # 日化无风险利率


def load_close(code: str) -> pd.Series:
    """读取收盘价序列（按日期排序）"""
    df = pd.read_csv(os.path.join(DATA, f"{code}.csv"), parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def main():
    # ---------- 1. 构造日收益率 ----------
    mkt_close = load_close(INDEX_FILE)
    mkt_ret = mkt_close.pct_change().dropna()          # 市场日收益
    mkt_excess = mkt_ret - RF_DAILY                    # 市场超额收益

    # 与市场交易日对齐
    mkt_excess = mkt_excess.loc[mkt_excess.index.isin(mkt_close.index)]

    rows = []      # 汇总表记录
    fitted = {}    # 每只股票的 (beta, alpha_daily, r2)

    for code, (name, industry) in STOCKS.items():
        close = load_close(code)
        # 与市场对齐（同一交易日），丢弃停牌缺失
        joint = pd.concat([close, mkt_close], axis=1, keys=["stock", "mkt"]).dropna()
        stock_ret = joint["stock"].pct_change().dropna()
        mkt_r = joint["mkt"].pct_change().dropna()
        idx = stock_ret.index.intersection(mkt_r.index)
        stock_ret, mkt_r = stock_ret.loc[idx], mkt_r.loc[idx]

        y = stock_ret - RF_DAILY                        # 个股超额收益
        x = mkt_r - RF_DAILY                            # 市场超额收益

        # OLS: y = alpha + beta * x
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        beta = model.params["mkt"]
        alpha_daily = model.params["const"]
        alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1   # alpha 年化
        t_beta = model.tvalues["mkt"]
        t_alpha = model.tvalues["const"]
        r2 = model.rsquared
        n = model.nobs

        ann_vol = stock_ret.std() * np.sqrt(TRADING_DAYS)      # 年化波动率
        ann_ret = (1 + stock_ret.mean()) ** TRADING_DAYS - 1   # 年化收益

        rows.append({
            "code": code, "name": name, "industry": industry,
            "beta": round(beta, 3), "t_beta": round(t_beta, 2),
            "alpha_annual": round(alpha_annual, 4),
            "t_alpha": round(t_alpha, 2),
            "r_squared": round(r2, 3), "n_obs": int(n),
            "ann_return": round(ann_ret, 4), "ann_vol": round(ann_vol, 4),
        })
        fitted[code] = (name, industry, beta, alpha_daily, r2, x, y)

    result = pd.DataFrame(rows).sort_values("beta", ascending=False)
    result.to_csv(os.path.join(RES, "capm_results.csv"), index=False)
    print(result.to_string(index=False))

    # ---------- 2. 图1: beta 横向条形图 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = result.sort_values("beta")
    labels = [f"{r['name']} ({r['industry']})" for _, r in plot_df.iterrows()]
    bars = ax.barh(labels, plot_df["beta"], color="#4C72B0")
    for bar, b, t in zip(bars, plot_df["beta"], plot_df["t_beta"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{b:.2f}  (t={t:.1f})", va="center", fontsize=8)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="beta = 1 (market)")
    ax.set_xlabel("Beta (systematic risk)")
    ax.set_title("CAPM Beta of 8 A-Share Stocks (Jan 2023 - Oct 2024)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig1_beta.png"), dpi=150)
    plt.close(fig)

    # ---------- 3. 图2: 个股 vs 市场 散点 + 拟合线 (2x4) ----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    codes = list(STOCKS.keys())
    for ax_i, code in enumerate(codes):
        name, industry, beta, alpha_d, r2, x, y = fitted[code]
        ax = axes.flat[ax_i]
        ax.scatter(x * 100, y * 100, s=6, alpha=0.5, color="#4C72B0")
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs * 100, (alpha_d + beta * xs) * 100, color="#C44E52", lw=1.5)
        ax.set_title(f"{name} ({industry})\nbeta={beta:.2f}, R2={r2:.2f}", fontsize=9)
        ax.set_xlabel("Market excess return (%)", fontsize=8)
        ax.set_ylabel("Stock excess return (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Stock vs Market Excess Returns with OLS Fit (daily, %)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 4. 图3: 证券市场线 SML ----------
    # 理论 SML: E(R_i) = R_f + beta * (E(R_m) - R_f)
    mkt_premium = (1 + mkt_ret.mean()) ** TRADING_DAYS - 1 - RF_ANNUAL   # 市场风险溢价(年化)
    betas = result["beta"].values
    ann_rets = result["ann_return"].values

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(betas, ann_rets * 100, s=60, color="#4C72B0", zorder=3)
    for _, r in result.iterrows():
        ax.annotate(f"{r['name']}", (r["beta"], r["ann_return"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    x_line = np.linspace(betas.min() - 0.1, betas.max() + 0.1, 100)
    sml_y = (RF_ANNUAL + x_line * mkt_premium) * 100
    ax.plot(x_line, sml_y, "r--", lw=1.5,
            label=f"SML: Rf + beta*{mkt_premium*100:.1f}%")

    # 样本平均 beta=1 的市场组合点
    ax.scatter([1.0], [(RF_ANNUAL + mkt_premium) * 100], marker="*",
               s=150, color="#C44E52", zorder=4, label="Market (beta=1)")
    ax.axhline(RF_ANNUAL * 100, color="gray", ls=":", lw=1)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Average annualized return (%)")
    ax.set_title("Security Market Line: Risk-Return Tradeoff")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_sml.png"), dpi=150)
    plt.close(fig)

    print(f"\n市场风险溢价(年化): {mkt_premium*100:.2f}%")
    print(f"图表已保存至: {RES}/")


if __name__ == "__main__":
    main()
