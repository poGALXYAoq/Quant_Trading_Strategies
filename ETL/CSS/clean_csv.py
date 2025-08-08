import pandas as pd
import os

# 1. 请在这里设置您的文件路径
# **************************************************
input_file_path = '上期所合约行情报表2024/所内合约行情报表2024.12.csv'
# 处理后文件的保存路径
output_file_path = 'SHFE_2024_12.csv'
# **************************************************


def process_csv_file(input_path, output_path):
    """
    处理给定的CSV文件，具体操作如下：
    1. 删除文件的前3行和后5行。
    2. 填充第一列中的空白单元格，使用其上方最近的非空单元格的值。
    
    :param input_path: str, 输入CSV文件的路径。
    :param output_path: str, 输出CSV文件的路径。
    """
    try:
        # 2. 对于csv文件删除前三行和后五行
        # 使用skiprows和skipfooter参数在读取时跳过指定的行。
        # engine='python'是使用skipfooter所必需的。
        # 考虑到文件名包含中文，文件编码很可能是'gbk'。
        try:
            df = pd.read_csv(input_path, header=None, skiprows=3, skipfooter=5, engine='python', encoding='gbk')
        except UnicodeDecodeError:
            print(f"使用 'gbk' 编码读取失败，尝试使用 'utf-8' 编码...")
            df = pd.read_csv(input_path, header=None, skiprows=3, skipfooter=5, engine='python', encoding='utf-8')

        # 3. 对于第一列，检查是否有值，若没有则往上找有值的列把遇到的第一个有值的情况来填充
        # pandas会将第一列的空值读取为NaN (Not a Number)。
        # 使用 forward fill (ffill) 方法来用前一个非空值填充NaN。
        df.iloc[:, 0].ffill(inplace=True)

        # (可选) 移除完全是空值的列，这通常是因为CSV每行末尾有多余的逗号。
        df.dropna(axis=1, how='all', inplace=True)

        # 保存处理后的数据到新的CSV文件，不包含索引和表头。
        df.to_csv(output_path, index=False, header=False)

        print(f"文件处理成功！已保存至: {os.path.abspath(output_path)}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{os.path.abspath(input_path)}'")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{os.path.abspath(input_path)}' 是空的，或者在删除指定行后变为空。")
    except Exception as e:
        print(f"处理文件时发生了一个未知错误: {e}")

if __name__ == "__main__":
    # 确保输出文件所在的目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_csv_file(input_file_path, output_file_path)
