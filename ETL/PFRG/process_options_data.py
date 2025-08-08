import pandas as pd
import os
import re

def transform_options_data(source_path, futures_data_path, output_path, target_contract_prefix):
    """
    将原始期权数据(格式A)转换为回测脚本所需的格式(格式B)。

    - 根据 target_contract_prefix 筛选合约
    - 解析期权类型和行权价
    - 从期货数据中匹配每日标的价格

    参数:
    source_path (str): 源期权数据CSV的路径 (格式A)。
    futures_data_path (str): 已处理好的期货数据CSV的路径 (用于获取标的价格)。
    output_path (str): 转换后文件的保存路径 (格式B)。
    target_contract_prefix (str): 要筛选的品种代码前缀, 例如 'TA409'。
    """
    # --- 1. 参数校验 ---
    if not os.path.exists(source_path):
        print(f"错误: 源期权文件未找到: {source_path}")
        return
    if not os.path.exists(futures_data_path):
        print(f"错误: 期货数据文件未找到: {futures_data_path}。请先运行期货数据处理脚本。")
        return
    if not target_contract_prefix:
        print("错误: 必须指定 target_contract_prefix。")
        return

    print(f"--- 开始处理期权数据 ---")
    print(f"源文件: {source_path}")
    print(f"目标合约前缀: {target_contract_prefix}")

    # --- 2. 读取并预处理数据 ---
    df_options = pd.read_csv(source_path, skipinitialspace=True)
    df_futures = pd.read_csv(futures_data_path)

    # 清理列名中的空格
    df_options.columns = [col.strip() for col in df_options.columns]
    
    # --- 3. 筛选与解析 ---
    # 筛选出目标合约
    df_filtered = df_options[df_options['品种代码'].str.startswith(target_contract_prefix, na=False)].copy()

    if df_filtered.empty:
        print(f"警告: 在源文件中找不到任何以 '{target_contract_prefix}' 开头的品种代码。")
        return

    print(f"找到 {len(df_filtered)} 条与 '{target_contract_prefix}' 相关的期权记录。")

    # 解析期权类型和行权价
    # 正则表达式: (C|P)(\d+)$ 匹配末尾的 C或P 及后面的数字
    extraction = df_filtered['品种代码'].str.extract(r'(C|P)(\d+)$')
    df_filtered['option_type'] = extraction[0].map({'C': 'call', 'P': 'put'})
    df_filtered['strike_price'] = pd.to_numeric(extraction[1])

    # 删除无法解析的行
    df_filtered.dropna(subset=['option_type', 'strike_price'], inplace=True)

    # --- 4. 选取、重命名并合并 ---
    # 选取并重命名所需列
    df_transformed = df_filtered[['交易日期', '今收盘', 'option_type', 'strike_price']].copy()
    df_transformed.rename(columns={'交易日期': 'datetime', '今收盘': 'close'}, inplace=True)
    
    # 转换日期格式以便合并
    df_transformed['datetime'] = pd.to_datetime(df_transformed['datetime']).dt.strftime('%Y-%m-%d')
    df_futures['datetime'] = pd.to_datetime(df_futures['datetime']).dt.strftime('%Y-%m-%d')
    
    # 准备期货数据用于合并
    df_futures_for_merge = df_futures[['datetime', 'close']].rename(columns={'close': 'underlying_price'})

    # 将标的价格(underlying_price)合并到期权数据中
    df_final = pd.merge(df_transformed, df_futures_for_merge, on='datetime', how='left')
    
    # 删除没有对应标的价格的期权数据
    df_final.dropna(subset=['underlying_price'], inplace=True)
    
    # --- 5. 保存结果 ---
    # 保证最终列的顺序
    output_columns = ['datetime', 'underlying_price', 'strike_price', 'option_type', 'close']
    df_final = df_final[output_columns]
    
    # 排序
    df_final.sort_values(by=['datetime', 'strike_price'], inplace=True)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_final.to_csv(output_path, index=False)
    print(f"转换成功！文件已保存至: {output_path}")
    print(f"--- 期权数据处理完成 ---")


if __name__ == '__main__':
    # --- 全局设置 ---
    
    # 1. 指定您想处理的合约前缀
    # 例如, 对于 PTA 2409 合约, 其代码通常以 'TA409' 开头
    TARGET_CONTRACT = 'TA409' # 这是一个示例, 请根据您的文件名和数据内容进行修改

    # 2. 定义文件路径
    SOURCE_OPTIONS_FILE = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/ta_option_data_2020_2024.csv'
    
    # 这个文件必须是已存在的、格式化后的期货数据
    FUTURES_DATA_FILE = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/futures_data.csv' 
    
    # 这是最终生成的、可用于回测的期权数据文件
    OUTPUT_OPTIONS_FILE = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/options_data.csv'
    
    # --- 执行转换 ---
    # 在运行前, 请确保已有一个格式正确的 futures_data.csv 文件存在
    # 如果没有, 请先运行上一个脚本 (process_futures_data.py)
    transform_options_data(
        source_path=SOURCE_OPTIONS_FILE,
        futures_data_path=FUTURES_DATA_FILE,
        output_path=OUTPUT_OPTIONS_FILE,
        target_contract_prefix=TARGET_CONTRACT
    )

