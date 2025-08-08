import pandas as pd
import os

# --- 请在这里配置 ---
# 输入文件路径 (例如: 'data/SHFE_opt_2023_2024.csv')
INPUT_CSV_PATH = 'SHFE_opt_2023_2024.csv'
# 输出文件路径 (例如: 'data/filtered_rb_data.csv')
OUTPUT_CSV_PATH = 'data/RB/rb_option_data_2023_2024.csv'
# 要筛选的前缀
PREFIX_TO_FILTER = 'rb'
# --- 配置结束 ---

def filter_csv_by_prefix(input_path, output_path, prefix):
    """
    读取CSV文件，根据第一列的值是否以指定前缀开头来筛选行，
    然后将结果（包括表头）保存到新的CSV文件中。

    参数:
    input_path (str): 输入CSV文件的路径。
    output_path (str): 保存筛选结果的CSV文件的路径。
    prefix (str): 用于筛选的前缀字符串。
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：输入文件不存在 -> {input_path}")
            return

        # 使用pandas读取CSV文件
        df = pd.read_csv(input_path, dtype=str) # 将所有列都作为字符串读取以避免类型问题

        # 检查DataFrame是否为空
        if df.empty:
            print("输入文件为空，没有数据可以处理。")
            return

        # 获取第一列的列名
        first_column_name = df.columns[0]

        # 筛选第一列以指定前缀开头的数据 (na=False确保NaN值不会导致错误)
        filtered_df = df[df[first_column_name].str.startswith(prefix, na=False)]

        # 创建输出文件所在的目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")

        # 将筛选后的DataFrame保存到新的CSV文件，不包括pandas的行索引
        filtered_df.to_csv(output_path, index=False, encoding='utf_8_sig')

        print(f"筛选完成！")
        print(f"共找到 {len(filtered_df)} 条匹配 '{prefix}...' 的记录。")
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == '__main__':
    # 检查路径是否已修改
    if INPUT_CSV_PATH == 'path/to/your/input.csv' or OUTPUT_CSV_PATH == 'path/to/your/output.csv':
        print("请先在脚本顶部修改 'INPUT_CSV_PATH' 和 'OUTPUT_CSV_PATH' 的值。")
    else:
        filter_csv_by_prefix(INPUT_CSV_PATH, OUTPUT_CSV_PATH, PREFIX_TO_FILTER)
