import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools

warnings.filterwarnings('ignore')

# --- 1. 参数设置 ---

# 策略参数
K_VALUES = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08] # 虚值程度k的取值范围
N_RANGE = range(5, 21) # DTE目标值N的取值范围
DTE_TOLERANCE = 3 # DTE容忍度,如果真实DTE与目标N的差距大于此值,则不交易

# 资金管理参数
INITIAL_CAPITAL = 1_000_000  # 初始资金
CAPITAL_USAGE_RATIO = 0.8  # 动用资金比例

# 交易成本参数
COMMISSION_PER_CONTRACT = 0.5  # 每手期权合约的单边手续费（元）

# 保证金计算参数 (根据文档中的简化公式)
MARGIN_RATE_FUTURES = 0.12 # 标的期货合约保证金率的估算值
MIN_MARGIN_FACTOR = 0.07 # 最低保障系数的估算值 (可以参考交易所标准)

# 并行计算设置
N_JOBS = 30 # 使用所有可用的CPU核心,可以设置为具体数字,如4

# 文件路径
DATA_PATH = 'data\RB\master_option_data_RB.csv'
RESULTS_DIR = 'backtest_results/RB'

# master_option_data_SA.csv中第二列表头为合约代码
col2 = '合约名称'


# --- 2. 数据准备与预处理 ---

def prepare_data(file_path):
    """
    加载并预处理数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")

    df = pd.read_csv(file_path, thousands=',') # 使用thousands=','来处理带逗号的数字

    # 数据清洗与转换
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df['到期日'] = pd.to_datetime(df['到期日'])
    df.drop_duplicates(subset=['交易日期', col2], keep='first', inplace=True)
    df.set_index('交易日期', inplace=True)
    df.sort_index(inplace=True)

    # 特征工程
    df['期权类型'] = df[col2].str.extract(r'(C|P)')
    df['行权价'] = df[col2].str.extract(r'(\d+)$').astype(int)

    # 重命名列以方便访问
    df.rename(columns={
        '剩余到期日': 'DTE',
        '今收盘': 'close_price',
        '今结算': 'settlement_price', # 使用结算价作为核心价格
        '成交量(手)': '成交量',
        '标的开盘价': 'underlying_open',
        '标的收盘价': 'underlying_close'
    }, inplace=True)

    # 确保价格和成交量是数值型
    numeric_cols = ['close_price', 'settlement_price', 'underlying_open', 'underlying_close', '成交量']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['settlement_price', 'underlying_open', 'underlying_close'], inplace=True)
    
    print(f"数据准备完成，共 {len(df)} 条记录。")
    return df

# --- 3. 策略逻辑模块 ---

def generate_trade_schedule(df, N):
    trading_days = df.index.unique().sort_values()
    potential_near_expiries = df[df['DTE'] >= N]['到期日'].unique()
    
    schedule = []
    last_close_date = pd.Timestamp.min
    
    for expiry in sorted(potential_near_expiries):
        if expiry <= last_close_date:
            continue
            
        target_open_date = expiry - pd.Timedelta(days=N)
        open_date_idx = trading_days.searchsorted(target_open_date, side='left')
        if open_date_idx >= len(trading_days):
            continue
        open_date = trading_days[open_date_idx]

        close_date_idx = trading_days.searchsorted(expiry, side='right') - 1
        if close_date_idx < 0:
            continue
        close_date = trading_days[close_date_idx]

        if open_date >= close_date or open_date <= last_close_date:
            continue

        # 新增逻辑：检查实际DTE是否在容忍范围内
        # 在计算出的开仓日，获取对应到期日合约的实际DTE
        options_on_open_date = df.loc[df.index == open_date]
        near_month_options_on_open_date = options_on_open_date[options_on_open_date['到期日'] == expiry]
        
        # 如果当天没有该到期日的合约数据，则跳过
        if near_month_options_on_open_date.empty:
            continue
            
        # 检查实际DTE与目标N的差距是否超过容忍度
        actual_dte = near_month_options_on_open_date['DTE'].iloc[0]
        if abs(actual_dte - N) > DTE_TOLERANCE:
            continue # 如果差距过大，则放弃这个交易计划

        schedule.append((open_date, close_date))
        last_close_date = close_date
        
    return schedule

def find_opening_trade_event(open_date_options, near_month_expiry, k, underlying_open_price):
    near_month_options = open_date_options[open_date_options['到期日'] == near_month_expiry]
    if near_month_options.empty:
        return []

    unique_expiries = sorted(open_date_options['到期日'].unique())
    far_month_candidates = [exp for exp in unique_expiries if exp > near_month_expiry]
    if not far_month_candidates:
        return []
    far_month_expiry = min(far_month_candidates)
    
    far_month_options = open_date_options[open_date_options['到期日'] == far_month_expiry]
    if far_month_options.empty:
        return []

    target_strike_call = underlying_open_price * (1 + k)
    target_strike_put = underlying_open_price * (1 - k)

    near_calls = near_month_options[near_month_options['期权类型'] == 'C']
    near_puts = near_month_options[near_month_options['期权类型'] == 'P']
    if near_calls.empty or near_puts.empty:
        return []

    sell_call_near = near_calls.iloc[(near_calls['行权价'] - target_strike_call).abs().argsort()[:1]]
    sell_put_near = near_puts.iloc[(near_puts['行权价'] - target_strike_put).abs().argsort()[:1]]
    if sell_call_near.empty or sell_put_near.empty: 
        return []

    buy_call_far = far_month_options[
        (far_month_options['期权类型'] == 'C') &
        (far_month_options['行权价'] == sell_call_near['行权价'].iloc[0])
    ]
    buy_put_far = far_month_options[
        (far_month_options['期权类型'] == 'P') &
        (far_month_options['行权价'] == sell_put_near['行权价'].iloc[0])
    ]
    if buy_call_far.empty or buy_put_far.empty: 
        return []

    return [
        (sell_call_near.iloc[0], 'short'),
        (sell_put_near.iloc[0], 'short'),
        (buy_call_far.iloc[0], 'long'),
        (buy_put_far.iloc[0], 'long')
    ]

# --- 4. 回测执行与分析模块 ---

def calculate_margin(option_info, underlying_price):
    settlement_price = option_info['settlement_price']
    strike_price = option_info['行权价']
    option_type = option_info['期权类型']
    out_of_the_money_value = max(0, strike_price - underlying_price) if option_type == 'C' else max(0, underlying_price - strike_price)
    margin = settlement_price + max(underlying_price * MARGIN_RATE_FUTURES - out_of_the_money_value, underlying_price * MIN_MARGIN_FACTOR)
    return margin

def run_backtest_for_params(params, df):
    k, N = params
    return run_backtest_event_driven(df, k, N)

def run_backtest_event_driven(df, k, N):
    
    trade_schedule = generate_trade_schedule(df, N)
    cash = INITIAL_CAPITAL
    total_assets = INITIAL_CAPITAL
    equity_curve = []
    trade_log = []
    detailed_trade_log = []
    trade_group_id = 0
    
    all_trading_days = df.index.unique().sort_values()
    if len(all_trading_days) == 0: return None
    
    schedule_idx = 0
    portfolio = {}
    current_close_date = pd.Timestamp.min

    for today in all_trading_days:
        if portfolio and today == current_close_date:
            close_date_str = today.strftime('%Y-%m-%d')
            close_date_options = df.loc[df.index == close_date_str]
            underlying_price_at_close = close_date_options['underlying_close'].iloc[0] if not close_date_options.empty else df['underlying_close'].asof(today)

            for code, pos_to_close in portfolio.items():
                close_price = -1
                if not close_date_options.empty and code in close_date_options.set_index(col2).index:
                    close_price = close_date_options.set_index(col2).loc[code]['settlement_price']
                else:
                    strike_price = pos_to_close['info']['行权价']
                    option_type = pos_to_close['info']['期权类型']
                    close_price = max(0, underlying_price_at_close - strike_price) if option_type == 'C' else max(0, strike_price - underlying_price_at_close)
                

                pnl = (close_price - pos_to_close['entry_price'] if pos_to_close['direction'] == 'long' else pos_to_close['entry_price'] - close_price) * pos_to_close['lots']
                
                cash_before_this_trade = cash
                if pos_to_close['direction'] == 'long':
                    cash += close_price * pos_to_close['lots']
                else:
                    cash += pos_to_close['margin']
                    cash -= close_price * pos_to_close['lots']

                commission_cost = COMMISSION_PER_CONTRACT * pos_to_close['lots']
                cash -= commission_cost
                net_pnl = pnl - commission_cost
                total_assets += net_pnl
                trade_log.append({'pnl': net_pnl})

                detailed_trade_log.append({
                    'date': close_date_str, 'action': 'close', 'contract': code,
                    'direction': pos_to_close['direction'], 'lots': pos_to_close['lots'],
                    'entry_price': pos_to_close['entry_price'], 'exit_price': close_price,
                    'pnl': pnl, 'commission': commission_cost, 'net_pnl': net_pnl,
                    'margin_released': pos_to_close['margin'] if pos_to_close['direction'] == 'short' else 0,
                    'cash_before': cash_before_this_trade, 'cash_after': cash,
                    'total_assets_after': total_assets, 'k_param': k, 'N_param': N,
                    'underlying_price': underlying_price_at_close,
                    'strategy_group_id': pos_to_close.get('group_id', 'unknown'),
                    'dte_at_close': 0
                })
            portfolio.clear()

        if not portfolio and schedule_idx < len(trade_schedule):
            open_date, close_date_from_schedule = trade_schedule[schedule_idx]
            if today == open_date:
                current_close_date = close_date_from_schedule
                schedule_idx += 1
                
                open_date_options = df.loc[df.index == today.strftime('%Y-%m-%d')]
                if not open_date_options.empty:
                    underlying_open_price = open_date_options['underlying_open'].iloc[0]
                    trade_signals = find_opening_trade_event(open_date_options, current_close_date, k, underlying_open_price)

                    if trade_signals:
                        # 新增逻辑: 检查并修正开仓价格为0的合约，统一替换为0.5，以模拟滑点或最小报价
                        processed_signals = []
                        for trade_info, direction in trade_signals:
                            # 使用副本以安全地修改价格
                            modified_info = trade_info.copy()
                            if modified_info['settlement_price'] == 0:
                                modified_info['settlement_price'] = 0.5
                            processed_signals.append((modified_info, direction))
                        trade_signals = processed_signals

                        trade_group_id += 1
                        short_call_info, short_put_info, long_call_info, long_put_info = (s[0] for s in trade_signals)
                        underlying_close_price_for_margin = open_date_options['underlying_close'].iloc[0]
                        short_margin_call = calculate_margin(short_call_info, underlying_close_price_for_margin)
                        short_margin_put = calculate_margin(short_put_info, underlying_close_price_for_margin)
                        
                        net_cash_outflow_per_lot = (short_margin_call + short_margin_put + 
                                                   long_call_info['settlement_price'] + long_put_info['settlement_price'] - 
                                                   short_call_info['settlement_price'] - short_put_info['settlement_price'] + 
                                                   COMMISSION_PER_CONTRACT * 4)
                        
                        target_capital_usage = cash * CAPITAL_USAGE_RATIO
                        if net_cash_outflow_per_lot > 0:
                            num_lots = int(target_capital_usage // net_cash_outflow_per_lot)
                        else:
                            total_margin_per_lot = short_margin_call + short_margin_put
                            num_lots = int(target_capital_usage // total_margin_per_lot) if total_margin_per_lot > 0 else 0
                        
                        if num_lots > 0:
                            for trade_info, direction in trade_signals:
                                code = trade_info[col2]
                                entry_price = trade_info['settlement_price']
                                cash_before_this_trade = cash
                                margin_required = 0
                                
                                if direction == 'short':
                                    margin_required = calculate_margin(trade_info, underlying_close_price_for_margin) * num_lots
                                    cash += entry_price * num_lots
                                    cash -= margin_required
                                else:
                                    cash -= entry_price * num_lots
                                cash -= COMMISSION_PER_CONTRACT * num_lots
                                
                                portfolio[code] = {'info': trade_info.copy(), 'lots': num_lots, 'entry_price': entry_price, 'margin': margin_required, 'direction': direction, 'group_id': trade_group_id}
                                
                                detailed_trade_log.append({
                                    'date': today.strftime('%Y-%m-%d'), 'action': 'open', 'contract': code, 'direction': direction,
                                    'lots': num_lots, 'entry_price': entry_price, 'exit_price': None, 'pnl': None,
                                    'commission': COMMISSION_PER_CONTRACT * num_lots, 'net_pnl': None, 'margin_used': margin_required, 'margin_released': 0,
                                    'cash_before': cash_before_this_trade, 'cash_after': cash, 'total_assets_after': total_assets, 
                                    'k_param': k, 'N_param': N, 'underlying_price': underlying_close_price_for_margin, 
                                    'strategy_group_id': trade_group_id, 'dte_at_open': trade_info.get('DTE', None)
                                })
        current_equity = total_assets
        if portfolio:
            unrealized_pnl = 0
            today_options = df.loc[df.index == today.strftime('%Y-%m-%d')]
            if not today_options.empty:
                today_options_indexed = today_options.set_index(col2)
                for code, pos in portfolio.items():
                    if code in today_options_indexed.index:
                        current_price = today_options_indexed.loc[code]['settlement_price']
                        pnl_per_lot = (current_price - pos['entry_price']) if pos['direction'] == 'long' else (pos['entry_price'] - current_price)
                        unrealized_pnl += pnl_per_lot * pos['lots']
            current_equity += unrealized_pnl
        equity_curve.append({'date': today, 'equity': current_equity})

    if not equity_curve: return None
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    equity_df.to_csv(os.path.join(RESULTS_DIR, f'daily_equity_log_k{k}_N{N}.csv'))
    
    total_return = (equity_df['equity'].iloc[-1] / INITIAL_CAPITAL) - 1
    num_years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25 if equity_df.index.size > 1 else 0
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    daily_returns = equity_df['equity'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    roll_max = equity_df['equity'].cummax()
    daily_drawdown = equity_df['equity'] / roll_max - 1.0
    max_drawdown = daily_drawdown.min()
    wins = sum(1 for trade in trade_log if trade['pnl'] > 0)
    win_rate = wins / len(trade_log) if trade_log else 0
    
    if detailed_trade_log:
        trade_df = pd.DataFrame(detailed_trade_log)
        trade_df.to_csv(os.path.join(RESULTS_DIR, f'trade_log_k{k}_N{N}.csv'), index=False, encoding='utf-8-sig')
        
    return {
        'k': k, 'N': N, 'CAGR': cagr, 'Sharpe Ratio': sharpe_ratio, 
        'Max Drawdown': max_drawdown, 'Win Rate': win_rate, 
        'Final Equity': equity_df['equity'].iloc[-1]
    }

def plot_and_save_heatmap(df, value_col, title, filename):
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot(index='k', columns='N', values=value_col)
    sorted_columns = sorted(pivot_table.columns, reverse=True)
    pivot_table = pivot_table.reindex(columns=sorted_columns)
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title, fontsize=16)
    plt.xlabel("N (DTE target)")
    plt.ylabel("k (OTM level)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"热力图已保存至: {filename}")

if __name__ == "__main__":
    print("开始执行最终版回测...")
    master_df = prepare_data(DATA_PATH)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    
    param_combinations = list(itertools.product(K_VALUES, N_RANGE))
    print(f"开始并行回测 {len(param_combinations)} 组参数 (使用 {N_JOBS if N_JOBS > 0 else 'all'} 个核心)...")

    results_list = Parallel(n_jobs=N_JOBS)(
        delayed(run_backtest_for_params)(params, master_df)
        for params in tqdm(param_combinations, desc="回测进度")
    )
    
    all_results = [res for res in results_list if res is not None]
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(RESULTS_DIR, 'all_results.csv'), index=False, encoding='utf-8-sig')
        print(f"所有回测结果已保存。")

        print("正在生成性能指标热力图...")
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"无法设置中文字体'SimHei'，热力图可能无法正确显示中文: {e}")

        plot_and_save_heatmap(results_df, 'CAGR', '年化收益率 (CAGR)', os.path.join(RESULTS_DIR, 'cagr_heatmap.png'))
        plot_and_save_heatmap(results_df, 'Sharpe Ratio', '夏普比率 (Sharpe Ratio)', os.path.join(RESULTS_DIR, 'sharpe_heatmap.png'))
        plot_and_save_heatmap(results_df, 'Max Drawdown', '最大回撤 (Max Drawdown)', os.path.join(RESULTS_DIR, 'max_drawdown_heatmap.png'))
        plot_and_save_heatmap(results_df, 'Win Rate', '胜率 (Win Rate)', os.path.join(RESULTS_DIR, 'win_rate_heatmap.png'))
    else:
        print("回测未产生任何有效结果。")
    print("回测执行完毕。")
