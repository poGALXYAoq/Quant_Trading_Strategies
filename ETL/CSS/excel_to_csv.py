import pandas as pd
import os

def convert_excel_to_csv(directory):
    """
    将目录中所有的 .xls 和 .xlsx 文件转换为 CSV 格式。
    Excel 文件中的每个工作表都会被保存为一个单独的 CSV 文件。
    """
    # 确保目标目录存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在。")
        return

    print(f"开始扫描目录: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith((".xlsx", ".xls")):
            excel_path = os.path.join(directory, filename)
            print(f"找到 Excel 文件: {excel_path}")
            try:
                # 使用 pandas.ExcelFile 可以更高效地处理多工作表的 Excel 文件
                xls = pd.ExcelFile(excel_path)
                for sheet_name in xls.sheet_names:
                    print(f"  正在处理工作表: {sheet_name}")
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    # 创建 CSV 文件名
                    base_filename = os.path.splitext(filename)[0]
                    csv_filename = f"{base_filename}.csv"
                    
                    csv_path = os.path.join(directory, csv_filename)
                    
                    # 保存为 CSV
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"  成功将 '{filename}' (工作表: '{sheet_name}') 转换为 '{csv_filename}'")

            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    # 设置包含 Excel 文件的目标目录
    target_directory = "上期所合约行情报表2024"
    convert_excel_to_csv(target_directory)
