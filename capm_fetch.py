# -*- coding: utf-8 -*-
"""
capm_fetch.py — 下载 A 股个股与沪深300 日线数据，落盘为 CSV
样本期: 2023-01-01 ~ 2024-10-31（CAPM 课设项目，Oct 2024 – Dec 2024）
数据源: 腾讯财经公开行情接口（web.ifzq.gtimg.cn，个股前复权 qfq；指数不复权 day）
用法:   python capm_fetch.py
输出:   data/{symbol}.csv，列: date,open,close,high,low,volume
"""

import os
import time
import json
import urllib.request
import pandas as pd

# 8 只跨行业 A 股（腾讯代码格式: sh/sz 前缀）+ 市场指数
STOCKS = {
    "sh600519": "贵州茅台(白酒)",
    "sh600036": "招商银行(银行)",
    "sz300750": "宁德时代(新能源)",
    "sh600276": "恒瑞医药(医药)",
    "sh600030": "中信证券(券商)",
    "sz002594": "比亚迪(新能源车)",
    "sh601318": "中国平安(保险)",
    "sz000002": "万科A(地产)",
}
INDEX = "sh000300"   # 沪深300
INDEX_NAME = "沪深300指数"

START = "2023-01-01"
END = "2024-10-31"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_kline(symbol: str, name: str, is_index: bool = False) -> pd.DataFrame:
    """拉取日线。个股用 qfq（前复权），指数用 day（点位本身无需复权）。"""
    # 指数请求需以逗号结尾标记非复权；个股追加 ,qfq 表示前复权
    fq = "," if is_index else ",qfq"
    # 请求 800 条，覆盖 2023-01 ~ 2024-10 约 440 个交易日
    url = f"{URL}?param={symbol},day,{START},{END},800{fq}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if data.get("code") != 0:
        raise RuntimeError(f"{name}: 接口返回 code={data.get('code')}")

    node = data["data"][symbol]
    rows = node.get("qfqday") or node.get("day")
    if not rows:
        raise RuntimeError(f"{name}: 无数据")

    # 除权除息日会附带第 7 列分红说明 dict（如 10派259.11元），只取前 6 列
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df


def main():
    ok, fail = [], []

    # 指数
    try:
        idx = fetch_kline(INDEX, INDEX_NAME, is_index=True)
        p = os.path.join(OUT_DIR, f"{INDEX}_index.csv")
        idx.to_csv(p, index=False)
        print(f"[OK] {INDEX_NAME}: {len(idx)} 行 -> {p}")
        ok.append(INDEX)
    except Exception as e:
        print(f"[FAIL] {INDEX_NAME}: {e}")
        fail.append(INDEX)
    time.sleep(0.5)

    # 个股
    for code, name in STOCKS.items():
        try:
            df = fetch_kline(code, name, is_index=False)
            p = os.path.join(OUT_DIR, f"{code}.csv")
            df.to_csv(p, index=False)
            print(f"[OK] {name}: {len(df)} 行 -> {p}")
            ok.append(code)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            fail.append(code)
        time.sleep(0.5)

    print(f"\n完成: 成功 {len(ok)}/{len(ok) + len(fail)}")
    if fail:
        print(f"失败: {fail}")


if __name__ == "__main__":
    main()
