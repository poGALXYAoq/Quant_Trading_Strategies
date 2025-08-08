import pandas as pd
import os

def prepare_master_data():
    """
    将期权日行情、期权静态信息、标的日行情三个数据源合并，
    并计算衍生特征，生成用于回测的“大宽表”。
    """
    # --- 1. 定义文件路径 ---
    data_dir = 'data/RB'
    path_A = os.path.join(data_dir, 'rb_option_data_2023_2024.csv')
    path_B = os.path.join(data_dir, 'option_data_RB.csv')
    path_C = os.path.join(data_dir, 'RB888.csv')
    output_path = os.path.join(data_dir, 'master_option_data_RB.csv')

    print("开始加载数据...")
    try:
        # 加载主表A，并清理列名中可能存在的空格
        # 增加 low_memory=False 参数以避免DtypeWarning，一次性读入，对内存有要求但能确保类型一致
        df_A = pd.read_csv(path_A, low_memory=False)
        df_A.columns = df_A.columns.str.strip()

        # 加载信息表B和标的表C
        df_B = pd.read_csv(path_B)
        df_C = pd.read_csv(path_C)
        print("所有数据文件加载成功。")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请确保文件路径正确。")
        return

    # --- 2. 准备并合并表B (期权静态信息) ---
    print("处理并合并期权静态信息表 (表B)...")
    # 只选择我们需要的列
    df_B_prep = df_B[['order_book_id', 'maturity_date']].copy()
    # 重命名列以匹配主表A
    df_B_prep.rename(columns={'order_book_id': '合约名称', 'maturity_date': '到期日'}, inplace=True)
    # 静态信息表，按品种代码去重，防止合并时产生重复行
    df_B_prep.drop_duplicates(subset=['合约名称'], inplace=True)

    # 使用左连接（left join）将到期日添加到主表A中
    # `how='left'` 保证主表A的所有行都保留
    df_merged = pd.merge(df_A, df_B_prep, on='合约名称', how='left')
    print(f"合并后，主表新增 '到期日' 列。")

    # --- 3. 准备并合并表C (标的期货行情) ---
    print("处理并合并标的期货行情表 (表C)...")
    df_C_prep = df_C[['时间', '开盘', '收盘']].copy()
    # 重命名列，清晰地指明这是“标的”数据
    df_C_prep.rename(columns={'时间': '交易日期', '开盘': '标的开盘价', '收盘': '标的收盘价'}, inplace=True)

    # 再次使用左连接，将标的行情按日期匹配到主表
    df_merged = pd.merge(df_merged, df_C_prep, on='交易日期', how='left')
    print(f"合并后，主表新增 '标的开盘价' 和 '标的收盘价' 列。")


    # --- 4. 日期类型转换与特征工程 ---
    print("\n进行日期类型转换和特征计算...")
    # 将日期字符串转换为datetime对象，如果转换失败则填充为NaT (Not a Time)
    df_merged['交易日期_dt'] = pd.to_datetime(df_merged['交易日期'], errors='coerce')
    df_merged['到期日_dt'] = pd.to_datetime(df_merged['到期日'], errors='coerce')

    # 计算剩余到期日 (DTE)
    # 仅在日期有效的情况下计算
    valid_dates_mask = df_merged['到期日_dt'].notna() & df_merged['交易日期_dt'].notna()
    df_merged['剩余到期日'] = None # 先初始化列
    df_merged.loc[valid_dates_mask, '剩余到期日'] = (df_merged.loc[valid_dates_mask, '到期日_dt'] - df_merged.loc[valid_dates_mask, '交易日期_dt']).dt.days

    # 删除临时的datetime列
    df_merged.drop(columns=['交易日期_dt', '到期日_dt'], inplace=True)
    print("核心特征 '剩余到期日' (DTE) 计算完成。")

    # --- 4.5. 最终数据清洗和数值类型转换 ---
    print("\n进行最终数据清洗和数值类型转换...")
    
    # 定义需要转换为数值类型的列名列表
    cols_to_numeric = [
        '昨结算', '今开盘', '最高价', '最低价', '今收盘', '今结算',
        '成交量(手)', '持仓量', '增减量', '成交额(万元)', '行权量'
    ]
    
    for col in cols_to_numeric:
        if col in df_merged.columns:
            # 使用 pd.to_numeric 进行转换，无法转换的值会变成 NaN
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        else:
            print(f"警告：列 '{col}' 在DataFrame中不存在，跳过类型转换。")

    # DTE列也应是数值型
    if '剩余到期日' in df_merged.columns:
        df_merged['剩余到期日'] = pd.to_numeric(df_merged['剩余到期日'], errors='coerce')
            
    print("所有目标列已强制转换为数值类型。")

    # --- 5. 保存结果 ---
    print(f"\n正在将处理完成的数据保存到新文件: {output_path}")
    # 使用 utf-8-sig 编码以确保在Excel中打开CSV文件时中文不会乱码
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n数据准备工作全部完成！")
    print(f"合并后的“大宽表”已保存至 {output_path}")
    print("\n新文件的前5行内容预览:")
    print(df_merged.head())
    print("\n新文件的列信息和数据类型概览:")
    df_merged.info()


if __name__ == '__main__':
    prepare_master_data()
