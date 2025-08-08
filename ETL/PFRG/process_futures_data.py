import pandas as pd
import os

def transform_futures_data(source_path, output_path):
    """
    将原始期货数据CSV从格式A转换为回测脚本所需的格式B。

    格式A (源):
    order_book_id,date,volume,limit_down,high,total_turnover,day_session_open,
    prev_settlement,prev_close,low,open,open_interest,settlement,limit_up,close

    格式B (输出):
    datetime,open,high,low,close,volume

    参数:
    source_path (str): 源CSV文件的路径。
    output_path (str): 转换后文件的保存路径。
    """
    if not os.path.exists(source_path):
        print(f"错误: 源文件未找到: {source_path}")
        return

    print(f"正在读取源文件: {source_path}")
    # 读取源CSV
    df = pd.read_csv(source_path)

    # 定义格式A和格式B所需的列
    required_columns_a = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 检查源文件是否包含所有必需的列
    if not all(col in df.columns for col in required_columns_a):
        missing_cols = [col for col in required_columns_a if col not in df.columns]
        print(f"错误: 源文件缺少以下必需列: {', '.join(missing_cols)}")
        return

    # 选取并重排格式B所需的列
    df_transformed = df[required_columns_a].copy()

    # 重命名列
    df_transformed.rename(columns={'date': 'datetime'}, inplace=True)
    
    # 确保datetime列是标准的YYYY-MM-DD格式
    df_transformed['datetime'] = pd.to_datetime(df_transformed['datetime']).dt.strftime('%Y-%m-%d')
    
    # 确保数据按日期升序排列
    df_transformed.sort_values(by='datetime', inplace=True)

    # 保存为新的CSV文件
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_transformed.to_csv(output_path, index=False)
    print(f"转换成功！文件已保存至: {output_path}")


if __name__ == '__main__':
    # --- 使用示例 ---
    # 定义源文件和目标文件的路径
    # 您可以修改这些路径来处理不同的文件
    
    # 示例1: 处理您选择的PTA数据
    source_file = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/PTA2409.csv'
    # 将其转换为回测脚本默认读取的 futures_data.csv
    output_file = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2409/futures_data.csv'
    
    print("--- 开始转换任务 ---")
    transform_futures_data(source_file, output_file)
    print("--- 任务结束 ---")

    # 示例2: 如果您有另一个文件,可以像这样调用
    # source_file_2 = 'path/to/your/other_data.csv'
    # output_file_2 = 'path/to/your/formatted_other_data.csv'
    # transform_futures_data(source_file_2, output_file_2)

