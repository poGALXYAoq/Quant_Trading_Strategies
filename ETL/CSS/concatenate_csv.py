import pandas as pd
import os

def concatenate_csv_files(input_files, output_file_path):
    """
    将多个CSV文件垂直拼接，并保存到一个新的CSV文件中。

    Args:
        input_files (list): 包含要拼接的CSV文件路径的列表。
        output_file_path (str): 保存拼接后CSV文件的路径。
    """
    try:
        if len(input_files) < 2:
            print("错误：需要至少两个文件进行拼接。")
            return

        dataframes = []
        first_file_columns = None

        # 遍历所有输入文件
        for file_path in input_files:
            # 检查输入文件是否存在
            if not os.path.exists(file_path):
                print(f"错误：找不到文件 {file_path}")
                return
            
            # 读取CSV文件
            print(f"正在读取 {file_path}...")
            df = pd.read_csv(file_path)

            # 检查表头是否与第一个文件相同
            if first_file_columns is None:
                first_file_columns = list(df.columns)
            elif list(df.columns) != first_file_columns:
                print(f"错误：文件 {file_path} 的表头与第一个文件的表头不完全相同。")
                print(f"第一个文件的表头: {first_file_columns}")
                print(f"文件 {file_path} 的表头: {list(df.columns)}")
                return
            
            dataframes.append(df)

        # 拼接DataFrame
        print("正在拼接文件...")
        concatenated_df = pd.concat(dataframes, ignore_index=True)

        # 将结果保存到新的CSV文件
        print(f"正在将结果保存到 {output_file_path}...")
        concatenated_df.to_csv(output_file_path, index=False)
        
        input_basenames = [os.path.basename(f) for f in input_files]
        print(f"成功将 {', '.join(input_basenames)} 拼接为 {os.path.basename(output_file_path)}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # --- 在这里修改文件路径 ---
    # 请将下面的列表替换为您需要拼接的实际文件路径
    
    # 输入文件路径列表 (可以包含任意数量的文件)
    INPUT_FILES = [
        "D:\code\Quant_Trading_Strategies\Volatility_Prediction_Strategy\data\option_CSI300_2020.csv",
        "D:\code\Quant_Trading_Strategies\Volatility_Prediction_Strategy\data\option_CSI300_2021-22.csv",
        "D:\code\Quant_Trading_Strategies\Volatility_Prediction_Strategy\data\option_CSI300_2023-24.csv"
    ]
    
    # 合并后输出文件的路径
    OUTPUT_FILE = "D:\code\Quant_Trading_Strategies\Volatility_Prediction_Strategy\data\option_CSI300_2020-24.csv"
    # -------------------------

    concatenate_csv_files(INPUT_FILES, OUTPUT_FILE)
