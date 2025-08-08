import pandas as pd
import os

# 1. 请在这里设置您的文件路径
# **************************************************
input_file_path = 'SHFE_2024_12.csv'
# 处理后文件的保存路径
output_file_path = '上期所合约行情报表2024/SHFE_2024_12.csv'
# **************************************************


def filter_rows(input_path, output_path):
    """
    根据第一列的模式过滤CSV文件中的行。
    删除第一列内容为“两个字母 + 四个数字”格式的行。
    
    :param input_path: str, 输入CSV文件的路径。
    :param output_path: str, 输出CSV文件的路径。
    """
    try:
        # 读取CSV文件。假设文件没有表头。
        # 由于文件可能是上一步生成的，编码应为默认的utf-8。
        df = pd.read_csv(input_path, header=None)

        # 定义正则表达式，用于匹配“两个英文字母 + 四个数字”的格式。
        # ^ 表示字符串开始，$ 表示字符串结束。
        # [a-zA-Z]{2} 匹配两个大小写英文字母。
        # \d{4} 匹配四个数字。
        pattern = r'^[a-zA-Z]{2}\d{4}$'

        # 筛选数据：保留第一列内容不匹配该模式的行。
        # 我们需要将第一列转换为字符串类型以使用.str访问器。
        # `~` 操作符用于取反，保留不匹配的行。
        original_rows = len(df)
        # na=False确保空值不被匹配
        filtered_df = df[~df[0].astype(str).str.match(pattern, na=False)]
        retained_rows = len(filtered_df)
        
        # 保存处理后的数据到新的CSV文件。
        filtered_df.to_csv(output_path, index=False, header=False)

        print(f"文件处理成功！")
        print(f"原始行数: {original_rows}")
        print(f"处理后行数: {retained_rows}")
        print(f"已删除 {original_rows - retained_rows} 行。")
        print(f"已保存至: {os.path.abspath(output_path)}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{os.path.abspath(input_path)}'")
    except Exception as e:
        print(f"处理文件时发生了一个未知错误: {e}")

if __name__ == "__main__":
    # 确保输出文件所在的目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filter_rows(input_file_path, output_file_path)
