# -*- coding: utf-8 -*-
# CAPM 回归: R_i - rf = alpha + beta * (R_m - rf)
# 读 data/*.csv（capm_fetch.py 生成），输出 results/ 下的汇总表和 3 张图

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")

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

RF_ANNUAL = 0.02   # 简化处理: 常数无风险利率 2% 年化（对 beta 影响很小）
TRADING_DAYS = 252
RF_DAILY = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1


def load_close(code):
    df = pd.read_csv(os.path.join(DATA, f"{code}.csv"), parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def main():
    os.makedirs(RES, exist_ok=True)
    mkt_close = load_close(INDEX_FILE)
    mkt_ret = mkt_close.pct_change().dropna()

    rows, fitted = [], {}
    for code, (name, industry) in STOCKS.items():
        close = load_close(code)
        # 个股与市场按交易日对齐，去掉停牌缺失
        joint = pd.concat([close, mkt_close], axis=1, keys=["stock", "mkt"]).dropna()
        stock_ret = joint["stock"].pct_change().dropna()
        mkt_r = joint["mkt"].pct_change().dropna()
        idx = stock_ret.index.intersection(mkt_r.index)
        stock_ret, mkt_r = stock_ret.loc[idx], mkt_r.loc[idx]

        X = sm.add_constant(mkt_r - RF_DAILY)
        model = sm.OLS(stock_ret - RF_DAILY, X).fit()

        beta = model.params["mkt"]
        alpha_daily = model.params["const"]
        rows.append({
            "code": code, "name": name, "industry": industry,
            "beta": round(beta, 3),
            "t_beta": round(model.tvalues["mkt"], 2),
            "alpha_annual": round((1 + alpha_daily) ** TRADING_DAYS - 1, 4),
            "t_alpha": round(model.tvalues["const"], 2),
            "r_squared": round(model.rsquared, 3),
            "n_obs": int(model.nobs),
            "ann_return": round((1 + stock_ret.mean()) ** TRADING_DAYS - 1, 4),
            "ann_vol": round(stock_ret.std() * np.sqrt(TRADING_DAYS), 4),
        })
        fitted[code] = (name, industry, beta, alpha_daily,
                        model.rsquared, mkt_r - RF_DAILY, stock_ret - RF_DAILY)

    result = pd.DataFrame(rows).sort_values("beta", ascending=False)
    result.to_csv(os.path.join(RES, "capm_results.csv"), index=False)
    print(result.to_string(index=False))

    # 图1: beta 条形图（按大小排序）
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

    # 图2: 个股 vs 市场超额收益散点 + 拟合线
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax_i, code in enumerate(STOCKS):
        name, industry, beta, alpha_d, r2, x, y = fitted[code]
        ax = axes.flat[ax_i]
        ax.scatter(x * 100, y * 100, s=6, alpha=0.5, color="#4C72B0")
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs * 100, (alpha_d + beta * xs) * 100, color="#C44E52", lw=1.5)
        ax.set_title(f"{name} ({industry})\nbeta={beta:.2f}, R2={r2:.2f}", fontsize=9)
        ax.set_xlabel("Market excess return (%)", fontsize=8)
        ax.set_ylabel("Stock excess return (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Stock vs Market Excess Returns with OLS Fit (daily, %)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图3: 证券市场线，检验 beta 是否真的带来更高回报
    mkt_premium = (1 + mkt_ret.mean()) ** TRADING_DAYS - 1 - RF_ANNUAL

    fig, ax = plt.subplots(figsize=(8, 5.5))
    betas, ann_rets = result["beta"].values, result["ann_return"].values
    ax.scatter(betas, ann_rets * 100, s=60, color="#4C72B0", zorder=3)
    for _, r in result.iterrows():
        ax.annotate(r["name"], (r["beta"], r["ann_return"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    x_line = np.linspace(betas.min() - 0.1, betas.max() + 0.1, 100)
    ax.plot(x_line, (RF_ANNUAL + x_line * mkt_premium) * 100, "r--", lw=1.5,
            label=f"SML: Rf + beta*{mkt_premium*100:.1f}%")
    ax.scatter([1.0], [(RF_ANNUAL + mkt_premium) * 100], marker="*", s=150,
               color="#C44E52", zorder=4, label="Market (beta=1)")
    ax.axhline(RF_ANNUAL * 100, color="gray", ls=":", lw=1)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Average annualized return (%)")
    ax.set_title("Security Market Line: Risk-Return Tradeoff")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_sml.png"), dpi=150)
    plt.close(fig)

    print(f"\n市场风险溢价(年化): {mkt_premium*100:.2f}%")


if __name__ == "__main__":
    main()
