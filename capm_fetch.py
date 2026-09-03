# -*- coding: utf-8 -*-
# 下载 8 只 A 股 + 沪深300 日线数据，存为 CSV
# 数据源: 腾讯财经公开行情接口 https://web.ifzq.gtimg.cn

import os
import time
import json
import urllib.request
import pandas as pd

STOCKS = {
    "sh600519": "贵州茅台",
    "sh600036": "招商银行",
    "sz300750": "宁德时代",
    "sh600276": "恒瑞医药",
    "sh600030": "中信证券",
    "sz002594": "比亚迪",
    "sh601318": "中国平安",
    "sz000002": "万科A",
}
INDEX = "sh000300"
START = "2023-01-01"
END = "2024-10-31"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_kline(symbol, name, is_index=False):
    # 个股用 qfq 前复权；指数是点位不用复权，URL 结尾要带逗号
    fq = "," if is_index else ",qfq"
    url = f"{URL}?param={symbol},day,{START},{END},800{fq}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        raise RuntimeError(f"{name}: code={data.get('code')}")

    node = data["data"][symbol]
    rows = node.get("qfqday") or node.get("day")
    if not rows:
        raise RuntimeError(f"{name}: 无数据")

    # 除权日会多出第 7 列（分红信息），只取前 6 列
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fetch_kline(INDEX, "沪深300", is_index=True).to_csv(
        os.path.join(OUT_DIR, f"{INDEX}_index.csv"), index=False)
    for code, name in STOCKS.items():
        df = fetch_kline(code, name)
        df.to_csv(os.path.join(OUT_DIR, f"{code}.csv"), index=False)
        print(f"{name}: {len(df)} rows")
        time.sleep(0.5)  # 控制请求频率


if __name__ == "__main__":
    main()
