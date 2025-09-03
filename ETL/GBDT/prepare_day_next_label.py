import os
import argparse
import pandas as pd


DATE_COL = "date"
PRICE_COL_DEFAULT = "期货收盘价(活跃合约):阴极铜"


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if DATE_COL not in df.columns:
        raise ValueError(f"CSV 缺少日期列: {DATE_COL}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def build_day_session_return(df_cu888: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df_cu888.columns or "day_session_open" not in df_cu888.columns:
        cols = ", ".join(df_cu888.columns)
        raise ValueError(f"CU888 缺少必要列: close 或 day_session_open. 可用列: {cols}")
    dn = df_cu888[[DATE_COL, "close", "day_session_open"]].copy()
    dn["dn"] = dn["close"].astype(float) - dn["day_session_open"].astype(float)
    return dn[[DATE_COL, "dn"]]


def main():
    parser = argparse.ArgumentParser(description="根据 CU888 生成次日日盘差(DN)并覆盖表B价格列")
    parser.add_argument("--table_a", type=str, default=os.path.join(os.path.dirname(__file__), "data", "CU", "CU888.csv"), help="表A: CU888.csv 路径")
    parser.add_argument("--table_b", type=str, default=os.path.join(os.path.dirname(__file__), "data", "CU", "以收盘价为label.csv"), help="表B: 特征数据 CSV 路径")
    parser.add_argument("--price_col", type=str, default=PRICE_COL_DEFAULT, help="表B中需要被覆盖的列名")
    parser.add_argument("--output", type=str, default="", help="输出路径。留空则在表B同目录下生成 *_day_next.csv")
    args = parser.parse_args()

    df_a = load_csv(args.table_a)
    df_b = load_csv(args.table_b)

    dn = build_day_session_return(df_a)

    df_merged = df_b.merge(dn, on=DATE_COL, how="left")
    if "dn" not in df_merged.columns:
        raise ValueError("未能从 CU888 计算出 dn 列。")

    if args.price_col not in df_merged.columns:
        raise ValueError(f"表B缺少目标列: {args.price_col}")

    df_merged[args.price_col] = df_merged["dn"]
    df_merged.drop(columns=["dn"], inplace=True)

    if not args.output:
        base, ext = os.path.splitext(args.table_b)
        out_path = f"{base}_day_next{ext or '.csv'}"
    else:
        out_path = args.output

    df_merged.sort_values(DATE_COL, inplace=True)
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()


