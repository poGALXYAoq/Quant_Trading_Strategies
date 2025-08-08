import pandas as pd
import os

# --- 用户配置 ---
# 请将你的CSV文件路径填在这里
# 例如: 'data/some_data.csv'
file_path = 'data\RB\\rb_option_data_2023_2024.csv'

# 请将你要修改的日期列的名称填在这里
# 例如: '交易日'
column_name = 'Date'

# 输出文件的路径 (可选, 如果留空, 默认在原文件同目录下生成一个新文件)
output_file_path = ''
# --- /用户配置 ---

def format_date_string(date_val):
    """
    将YYYYMMDD格式的日期字符串或数字转换为YYYY-MM-DD格式。
    - 能够处理 '20230115', 20230115, 20230115.0 等形式。
    - 如果格式不正确, 则返回原值。
    例如: '20230115' -> '2023-01-15'
    """
    if pd.isna(date_val):
        return date_val

    # 统一处理成字符串
    s = str(date_val)
    
    # 处理来自Excel的浮点数, 如 '20230115.0'
    if '.' in s:
        s = s.split('.')[0]

    # 检查是否为8位数字
    if len(s) == 8 and s.isdigit():
        year = s[0:4]
        month = s[4:6]
        day = s[6:8]
        return f"{year}-{month}-{day}"
    
    # 如果格式不匹配, 返回原始值
    return date_val

def process_csv_file(input_path, col_name, output_path=''):
    """
    读取CSV文件，对指定列应用格式化函数，并保存到新文件。
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 '{input_path}'。请检查路径是否正确。")
        return

    try:
        # 使用dtype=str确保数据按字符串读取，以处理包含字母和数字的合约代码
        df = pd.read_csv(input_path, encoding='utf-8-sig', dtype={col_name: str})
    except UnicodeDecodeError:
        print("使用 'utf-8-sig' 编码读取失败, 尝试使用 'gbk' 编码...")
        try:
            df = pd.read_csv(input_path, encoding='gbk', dtype={col_name: str})
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")
        return

    if col_name not in df.columns:
        print(f"错误: 列名 '{col_name}' 不存在于文件中。")
        print(f"文件包含的列有: {df.columns.tolist()}")
        return
        
    # 对指定列的每一个单元格应用格式化函数
    df[col_name] = df[col_name].apply(format_date_string)

    # 决定输出路径
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_formatted{ext}"

    # 保存修改后的DataFrame到新的CSV文件
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"文件已成功处理并保存到: {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == '__main__':
    # 检查用户是否已修改默认的路径和列名
    if 'your_file.csv' in file_path or 'Date01' in column_name:
         print("!"*30)
         print("请注意：您需要先在脚本中修改 `file_path` 和 `column_name` 变量。")
         print("`file_path` 应设为源CSV文件的完整路径。")
         print("`column_name` 应设为您要修改的日期列的名称。")
         print("!"*30)
    else:
         process_csv_file(file_path, column_name, output_file_path)
