import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --- 1. 策略核心参数 ---
# 入场相关
START_DATE_STR = '2024-01-15'
INITIAL_CAPITAL = 2_000_000  # 初始资金
# 开仓价格模式: 'open' 或 'close'。为提升首日收益与K线的一致性，默认使用收盘价执行开仓
START_PRICE_MODE = 'close'
# 期权行权价选择方式:
# - 'atm': 以当日 options 数据中的 underlying_price 选择最接近的平值行权价(忽略 STRIKE_PRICE_OFFSET)
# - 'nearest': 以 underlying_price + STRIKE_PRICE_OFFSET 为目标，选择最近行权价
# - 'floor': 以 underlying_price + STRIKE_PRICE_OFFSET 为目标，选择不高于目标价的最高行权价
# - 'ceiling': 以 underlying_price + STRIKE_PRICE_OFFSET 为目标，选择不低于目标价的最低行权价
SELECT_STRIKE_METHOD = 'nearest'

# 头寸相关（全局设置）
FUTURES_INITIAL_LOTS = 100
# 保护看跌期权持仓手数（全程不变，直到到期平仓）
PUT_OPTION_LOTS = 300

# 期权选择相关
STRIKE_PRICE_OFFSET = -200

# 网格交易相关
PRICE_GRID_INTERVAL = 50
LOTS_ADJUSTMENT_STEP = 10

# 仓位限制
FUTURES_MAX_LOTS = 300
FUTURES_MIN_LOTS = 0

# 交易成本
COMMISSION_FUTURES = 3.0  # 每手期货单边手续费
COMMISSION_OPTIONS = 0.5  # 每手期权单边手续费
SLIPPAGE = 1  # 滑点, 按价格点计算 (暂未在逻辑中实现)

# --- 2. 文件路径 ---
FUTURES_DATA_PATH = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2405/futures_data.csv'
OPTIONS_DATA_PATH = 'Protected_Futures_Reverse_Grid_Strategy/data/PTA2405/options_data.csv'
RESULTS_DIR = 'Protected_Futures_Reverse_Grid_Strategy/backtest_results/PTA2405'

# --- 3. 数据准备 ---
def prepare_data(futures_path, options_path):
    """加载并预处理期货和期权数据"""
    if not os.path.exists(futures_path):
        raise FileNotFoundError(f"期货数据文件未找到: {futures_path}")
    if not os.path.exists(options_path):
        raise FileNotFoundError(f"期权数据文件未找到: {options_path}")

    futures_df = pd.read_csv(futures_path)
    options_df = pd.read_csv(options_path)

    futures_df['datetime'] = pd.to_datetime(futures_df['datetime'])
    options_df['datetime'] = pd.to_datetime(options_df['datetime'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        futures_df[col] = pd.to_numeric(futures_df[col], errors='coerce')
    for col in ['underlying_price', 'strike_price', 'close']:
        options_df[col] = pd.to_numeric(options_df[col], errors='coerce')
        
    futures_df.dropna(inplace=True)
    options_df.dropna(inplace=True)

    print(f"期货数据加载完成, {len(futures_df)} 条记录。")
    print(f"期权数据加载完成, {len(options_df)} 条记录。")
    
    return futures_df.set_index('datetime'), options_df.set_index('datetime')

# --- 4. 可视化模块 ---
def plot_results(daily_log_df, trade_log_df):
    """使用Plotly进行专业的可视化分析仪表盘"""
    if daily_log_df.empty:
        print("没有数据可供可视化。")
        return

    # 关键指标计算
    equity = daily_log_df['total_value']
    returns = equity.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    ann_factor = 252
    vol_annual = float(returns.std(ddof=0) * np.sqrt(ann_factor)) if len(returns) > 1 else 0.0
    ann_return = float(returns.mean() * ann_factor) if len(returns) > 1 else 0.0
    sharpe = float(ann_return / vol_annual) if vol_annual > 0 else 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    # CAGR 估算
    num_days = len(equity)
    years = num_days / ann_factor if num_days > 1 else 0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    # Sortino（使用下行波动）
    downside = returns.copy()
    downside[downside > 0] = 0.0
    downside_std_ann = float(downside.std(ddof=0) * np.sqrt(ann_factor)) if len(downside) > 1 else 0.0
    sortino = float(ann_return / downside_std_ann) if downside_std_ann > 0 else 0.0
    # Calmar
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    # 日度胜率与利润因子（以每日PnL近似）
    daily_pnl = daily_log_df['daily_pnl'] if 'daily_pnl' in daily_log_df.columns else pd.Series(index=daily_log_df.index, dtype=float).fillna(0)
    win_rate = float((daily_pnl > 0).mean()) if len(daily_pnl) > 0 else 0.0
    gross_profit = float(daily_pnl[daily_pnl > 0].sum())
    gross_loss = float(daily_pnl[daily_pnl < 0].sum())
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else np.nan
    # 其余信息
    num_trades = int(len(trade_log_df)) if trade_log_df is not None else 0
    start_dt = daily_log_df.index.min()
    end_dt = daily_log_df.index.max()

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            '期货价格与网格交易点',
            '账户净值曲线 (Equity Curve)',
            '期货持仓手数',
            '每日盈亏 (Daily PnL)',
            '期权价格 (受保护看跌)',
            '关键指标与说明'
        ),
        row_heights=[0.32, 0.22, 0.16, 0.14, 0.10, 0.10],
        specs=[[{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "table"}]]
    )

    # --- 子图1: 期货价格与交易信号 ---
    # K线图（移除不兼容的 hovertemplate，使用默认悬浮信息）
    fig.add_trace(
        go.Candlestick(
            x=daily_log_df.index,
            open=daily_log_df['futures_open'],
            high=daily_log_df['futures_high'],
            low=daily_log_df['futures_low'],
            close=daily_log_df['futures_close'],
            name='期货K线'
        ),
        row=1, col=1
    )

    # 交易信号
    buy_signals = daily_log_df[daily_log_df['trade_action'].str.contains("buy", na=False)]
    sell_signals = daily_log_df[daily_log_df['trade_action'].str.contains("sell", na=False)]
    
    # 用于自定义提示的数据列（兼容无 trade_price/commission 的情况）
    trade_price_series = daily_log_df['trade_price'] if 'trade_price' in daily_log_df.columns else daily_log_df['futures_close']

    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['futures_close'],
            mode='markers',
            marker=dict(color='red', symbol='triangle-up', size=10, line=dict(width=1, color='DarkSlateGrey')),
            name='买入/加仓',
            text=[
                f"日期: {idx:%Y-%m-%d}<br>动作: {act}<br>手数: {abs(int(lots))}<br>成交价: {px:.2f}"
                for idx, act, lots, px in zip(
                    buy_signals.index,
                    buy_signals['trade_action'],
                    buy_signals['trade_lots'],
                    trade_price_series.loc[buy_signals.index]
                )
            ],
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['futures_close'],
            mode='markers',
            marker=dict(color='limegreen', symbol='triangle-down', size=10, line=dict(width=1, color='DarkSlateGrey')),
            name='卖出/减仓',
            text=[
                f"日期: {idx:%Y-%m-%d}<br>动作: {act}<br>手数: {abs(int(lots))}<br>成交价: {px:.2f}"
                for idx, act, lots, px in zip(
                    sell_signals.index,
                    sell_signals['trade_action'],
                    sell_signals['trade_lots'],
                    trade_price_series.loc[sell_signals.index]
                )
            ],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    fig.update_yaxes(title_text="价格", row=1, col=1, tickformat=',.2f', showgrid=True, gridcolor='LightGray')

    # --- 子图2: 账户净值曲线 ---
    fig.add_trace(
        go.Scatter(
            x=daily_log_df.index,
            y=daily_log_df['total_value'],
            name='账户总净值',
            line=dict(color='orange', width=2),
            text=[f"日期: {idx:%Y-%m-%d}<br>净值: {val:,.2f}" for idx, val in zip(daily_log_df.index, daily_log_df['total_value'])],
            hoverinfo='text'
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="净值 ($)", row=2, col=1, tickformat=',.2f', showgrid=True, gridcolor='LightGray')

    # 增加基准（期货价格标准化）以便对比走势
    if 'futures_close' in daily_log_df.columns and len(daily_log_df) > 0:
        base_price = float(daily_log_df['futures_close'].iloc[0])
        if base_price > 0:
            benchmark_equity = daily_log_df['futures_close'] / base_price * INITIAL_CAPITAL
            fig.add_trace(
                go.Scatter(
                    x=daily_log_df.index,
                    y=benchmark_equity,
                    name='基准(期货标准化)',
                    line=dict(color='steelblue', width=1, dash='dot'),
                    text=[f"日期: {idx:%Y-%m-%d}<br>基准净值: {val:,.2f}" for idx, val in zip(daily_log_df.index, benchmark_equity)],
                    hoverinfo='text'
                ),
                row=2, col=1
            )

    # --- 子图3: 期货持仓手数 ---
    fig.add_trace(
        go.Scatter(
            x=daily_log_df.index,
            y=daily_log_df['futures_lots'],
            name='期货持仓手数',
            fill='tozeroy',
            mode='lines',
            line=dict(color='grey', width=1),
            text=[f"日期: {idx:%Y-%m-%d}<br>持仓手数: {val:,.0f}" for idx, val in zip(daily_log_df.index, daily_log_df['futures_lots'])],
            hoverinfo='text'
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="手数", row=3, col=1, tickformat=',.0f', showgrid=True, gridcolor='LightGray')

    # --- 子图4: 每日盈亏 ---
    colors = ['limegreen' if pnl >= 0 else 'red' for pnl in daily_log_df['daily_pnl']]
    fig.add_trace(
        go.Bar(
            x=daily_log_df.index,
            y=daily_log_df['daily_pnl'],
            name='每日盈亏',
            marker_color=colors,
            text=[f"日期: {idx:%Y-%m-%d}<br>每日盈亏: {val:,.2f}" for idx, val in zip(daily_log_df.index, daily_log_df['daily_pnl'])],
            hoverinfo='text'
        ),
        row=4, col=1
    )
    fig.update_yaxes(title_text="盈亏 ($)", row=4, col=1, tickformat=',.2f', showgrid=True, gridcolor='LightGray')
    
    # --- 整体布局设置 ---
    fig.update_layout(
        title_text='<b>受保护的期货逆势网格策略 - 回测分析报告</b>',
        height=2200,
        legend_title='图例',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='closest'
    )
    fig.update_xaxes(title_text='日期', row=4, col=1, showgrid=True, gridcolor='LightGray')

    # --- 子图5: 关键指标与说明 ---
    # --- 子图5: 期权价格与总价值（受保护看跌）---
    if 'put_option_price' in daily_log_df.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_log_df.index,
                y=daily_log_df['put_option_price'],
                name='看跌期权收盘价',
                line=dict(color='purple', width=1.5),
                mode='lines+markers'
            ),
            row=5, col=1, secondary_y=False
        )
    if 'options_total_value' in daily_log_df.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_log_df.index,
                y=daily_log_df['options_total_value'],
                name='期权总市值',
                line=dict(color='darkmagenta', width=1, dash='dot')
            ),
            row=5, col=1, secondary_y=False
        )
    fig.update_yaxes(title_text="期权价格/市值", row=5, col=1, tickformat=',.2f', showgrid=True, gridcolor='LightGray')
    fig.update_xaxes(title_text='日期', row=5, col=1, showgrid=True, gridcolor='LightGray')

    metrics_header = ['指标', '取值']
    metrics_rows = [
        ['开始日期', start_dt.strftime('%Y-%m-%d') if not pd.isna(start_dt) else '-'],
        ['结束日期', end_dt.strftime('%Y-%m-%d') if not pd.isna(end_dt) else '-'],
        ['交易日数', f"{num_days}"],
        ['总回报率', f"{total_return:.2%}"],
        ['年化收益率', f"{ann_return:.2%}"],
        ['CAGR', f"{cagr:.2%}"],
        ['最大回撤', f"{max_drawdown:.2%}"],
        ['年化波动率', f"{vol_annual:.2%}"],
        ['夏普比率(0RF)', f"{sharpe:.2f}"],
        ['Sortino', f"{sortino:.2f}"],
        ['Calmar', f"{calmar:.2f}"],
        ['日度胜率', f"{win_rate:.2%}"],
        ['Profit Factor(按日)', f"{profit_factor:.2f}" if not np.isnan(profit_factor) else 'NA'],
        ['交易次数', f"{num_trades}"],
    ]

    table_cells = list(map(list, zip(*metrics_rows)))
    fig.add_trace(
        go.Table(
            header=dict(values=metrics_header, fill_color='lightgrey', align='left'),
            cells=dict(values=table_cells, align='left')
        ),
        row=6, col=1
    )

    # --- 保存图表 ---
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plot_path = os.path.join(RESULTS_DIR, 'interactive_backtest_dashboard.html')
    fig.write_html(plot_path)
    print(f"专业版交互式图表已保存至: {plot_path}")

# --- 5. 核心回测逻辑 (事件驱动) ---
def run_backtest_event_driven(futures_df, options_df):
    """执行事件驱动的策略回测（修正版）"""
    
    # --- 初始化 ---
    cash = INITIAL_CAPITAL
    futures_lots = 0
    futures_avg_price = 0.0
    options_position = {} # key: strike, value: {'lots', 'entry_price'}
    
    daily_log = []
    trade_log = []

    # 确定交易区间
    all_trading_days = futures_df.index.unique().sort_values()
    start_date = all_trading_days[all_trading_days >= pd.to_datetime(START_DATE_STR)][0]
    option_expiry_date = options_df.index.max()
    
    # 基准价格（确保为标量；同日多行取第一行）
    entry_row = futures_df.loc[start_date]
    if isinstance(entry_row, pd.DataFrame):
        start_open = float(entry_row['open'].iloc[0])
        start_close = float(entry_row['close'].iloc[0])
    else:
        start_open = float(entry_row['open'])
        start_close = float(entry_row['close'])

    # 作为网格基准与首日执行价参考
    if START_PRICE_MODE == 'close':
        entry_price = start_close
    else:
        entry_price = start_open
    
    # 净值口径：从策略启动前一天的权益=初始资金开始，首日盈亏仅反映从建仓价到首日收盘的变化
    last_total_value = INITIAL_CAPITAL

    # --- 开始每日循环 ---
    for today in all_trading_days:
        if today < start_date:
            continue
        if today > option_expiry_date:
            break

        futures_data_today = futures_df.loc[today]
        if isinstance(futures_data_today, pd.DataFrame) and not futures_data_today.empty:
            futures_data_today = futures_data_today.iloc[0] # 处理同一天有多行数据的情况
            
        futures_open = futures_data_today['open']
        futures_high = futures_data_today['high']
        futures_low = futures_data_today['low']
        futures_close = futures_data_today['close']
        
        # --- 初始化每日状态 ---
        daily_snapshot = {
            'date': today,
            'futures_open': futures_open,
            'futures_high': futures_high,
            'futures_low': futures_low,
            'futures_close': futures_close,
            'futures_lots': futures_lots,
            'trade_action': '',
            'trade_lots': 0,
            'trade_price': np.nan,
            'commission_paid': 0.0,
            'daily_pnl': 0,
            'cash': cash
        }
        
        # --- 开仓日逻辑 ---
        if today == start_date:
            # 1. 建立期货头寸
            trade_price = entry_price
            futures_lots = FUTURES_INITIAL_LOTS
            futures_avg_price = trade_price
            cost = futures_lots * trade_price
            commission = FUTURES_INITIAL_LOTS * COMMISSION_FUTURES
            cash -= (cost + commission)
            trade_log.append({'date': today, 'action': 'buy_futures_initial', 'lots': FUTURES_INITIAL_LOTS, 'price': trade_price, 'commission': commission})
            daily_snapshot.update({'trade_action': 'buy_futures_initial', 'trade_lots': FUTURES_INITIAL_LOTS, 'trade_price': trade_price, 'commission_paid': commission})
            print(f"[期货建仓] 日期: {today.date()} | 买入手数: {FUTURES_INITIAL_LOTS} | 价格: {trade_price:.2f} | 手续费: {commission:.2f} | 现金余额: {cash:,.2f}")

            # 2. 建立期权保护头寸
            # 以当日期权数据中的 underlying_price 为基准，计算目标行权价
            # 安全获取当日期权数据，若无则回退至最近一个不晚于今日的交易日
            options_today = options_df.loc[options_df.index.isin([today])]
            if options_today.empty:
                prev_dates = options_df.index[options_df.index <= today]
                if prev_dates.empty:
                    raise ValueError(f"开仓日之前无可用期权数据: {today.date()}")
                nearest = prev_dates.max()
                options_today = options_df.loc[options_df.index.isin([nearest])]
            # 以当日标的参考价（按当日第一条记录）为准，避免“中位数”等聚合引入的错配
            if 'underlying_price' in options_today.columns and not options_today['underlying_price'].isna().all():
                underlying_ref = float(options_today['underlying_price'].iloc[0])
            else:
                underlying_ref = float(entry_price)
            # 计算目标行权价（平值或带偏移）
            if SELECT_STRIKE_METHOD == 'atm':
                target_strike = underlying_ref
            else:
                target_strike = underlying_ref + STRIKE_PRICE_OFFSET

            put_options = options_today[options_today['option_type'] == 'put']
            if put_options.empty:
                raise ValueError(f"在 {today.date()} 及之前最近可用交易日未找到看跌期权")
            
            # 选择行权价
            method = SELECT_STRIKE_METHOD
            if method == 'floor':
                candidates = put_options[put_options['strike_price'] <= target_strike]
                if not candidates.empty:
                    best_option = candidates.sort_values('strike_price', ascending=False).iloc[0]
                else:
                    best_option = put_options.assign(_diff=(put_options['strike_price'] - target_strike).abs()) \
                                           .sort_values('_diff', ascending=True).iloc[0]
            elif method == 'ceiling':
                candidates = put_options[put_options['strike_price'] >= target_strike]
                if not candidates.empty:
                    best_option = candidates.sort_values('strike_price', ascending=True).iloc[0]
                else:
                    best_option = put_options.assign(_diff=(put_options['strike_price'] - target_strike).abs()) \
                                           .sort_values('_diff', ascending=True).iloc[0]
            else:  # 'atm' 或 'nearest' 都按最近处理
                best_option = put_options.assign(_diff=(put_options['strike_price'] - target_strike).abs()) \
                                       .sort_values('_diff', ascending=True).iloc[0]
            option_price = float(best_option['close'])
            option_strike = float(best_option['strike_price'])

            cost = PUT_OPTION_LOTS * option_price
            commission = PUT_OPTION_LOTS * COMMISSION_OPTIONS
            cash -= (cost + commission)
            # 记录为看跌期权持仓，后续估值严格按 option_type=put 过滤
            options_position[option_strike] = {
                'lots': PUT_OPTION_LOTS,
                'entry_price': option_price,
                'option_type': 'put'
            }
            trade_log.append({'date': today, 'action': 'buy_put_option', 'lots': PUT_OPTION_LOTS, 'price': option_price, 'commission': commission, 'strike': option_strike})
            print(f"[保护期权] 日期: {today.date()} | 买入看跌: {PUT_OPTION_LOTS} 手 | 价格: {option_price:.2f} | 行权价: {option_strike:.2f} | 手续费: {commission:.2f} | 现金余额: {cash:,.2f}")
            print(f"策略于 {start_date.date()} 启动。期货@{trade_price:.2f}, 期权@{option_price:.2f}(行权价{option_strike})")
            # 将期权手续费计入当日快照
            daily_snapshot['commission_paid'] += commission

        # --- 持仓调整逻辑 (非开仓日/平仓日) ---
        elif today < option_expiry_date:
            price_diff = futures_close - entry_price
            
            if price_diff > 0:
                steps = np.floor(price_diff / PRICE_GRID_INTERVAL)
                target_lots = FUTURES_INITIAL_LOTS - steps * LOTS_ADJUSTMENT_STEP
                target_lots = max(target_lots, FUTURES_MIN_LOTS)
            elif price_diff < 0:
                steps = np.floor(abs(price_diff) / PRICE_GRID_INTERVAL)
                target_lots = FUTURES_INITIAL_LOTS + steps * LOTS_ADJUSTMENT_STEP
                target_lots = min(target_lots, FUTURES_MAX_LOTS)
            else:
                target_lots = FUTURES_INITIAL_LOTS
            
            lots_to_trade = int(target_lots - futures_lots)  # 确保为整数
            
            # 仅当买入卖出方向发生切换时，跳过当日信号，避免同日既买又卖的频繁来回（去抖动）
            if lots_to_trade != 0:
                trade_price = futures_close
                commission = abs(lots_to_trade) * COMMISSION_FUTURES
                cash -= commission

                if lots_to_trade > 0: # 买入
                    action = 'buy_futures_grid'
                    cost = lots_to_trade * trade_price
                    cash -= cost
                    # 更新成本价
                    new_total_value = (futures_avg_price * futures_lots) + (trade_price * lots_to_trade)
                    futures_lots += lots_to_trade
                    futures_avg_price = new_total_value / futures_lots
                else: # 卖出 (lots_to_trade < 0)
                    action = 'sell_futures_grid'
                    revenue = abs(lots_to_trade) * trade_price
                    cash += revenue
                    # 平均成本价在部分卖出后通常保持不变
                    futures_lots += lots_to_trade
                
                trade_log.append({'date': today, 'action': action, 'lots': lots_to_trade, 'price': trade_price, 'commission': commission})
                daily_snapshot.update({'trade_action': action, 'trade_lots': lots_to_trade, 'trade_price': trade_price, 'commission_paid': commission})
                side_cn = '买入' if lots_to_trade > 0 else '卖出'
                print(f"[网格调仓] 日期: {today.date()} | {side_cn}手数: {abs(lots_to_trade)} | 价格: {trade_price:.2f} | 手续费: {commission:.2f} | 当前持仓: {futures_lots} | 现金余额: {cash:,.2f}")

        # --- 平仓日逻辑 ---
        if today == option_expiry_date:
            print(f"策略于 {today.date()} 到期平仓。")
            # 1. 平期货
            if futures_lots != 0:
                revenue = futures_lots * futures_close
                cash += revenue
                commission = abs(futures_lots) * COMMISSION_FUTURES
                cash -= commission
                trade_log.append({'date': today, 'action': 'sell_futures_final', 'lots': -futures_lots, 'price': futures_close, 'commission': commission})
                daily_snapshot.update({'trade_action': 'sell_futures_final', 'trade_lots': -futures_lots, 'trade_price': futures_close, 'commission_paid': commission})
                print(f"[期货平仓] 日期: {today.date()} | 卖出手数: {abs(futures_lots)} | 价格: {futures_close:.2f} | 手续费: {commission:.2f} | 现金余额: {cash:,.2f}")
                futures_lots = 0

            # 2. 平期权
            options_today = options_df.loc[options_df.index.isin([today])]
            for strike, pos in options_position.items():
                # 仅按看跌期权估值，防止与同行权价的看涨混淆
                final_opt_price_series = options_today[(options_today['strike_price'] == strike) & (options_today['option_type'] == 'put')]['close']
                final_opt_price = final_opt_price_series.iloc[0] if not final_opt_price_series.empty else 0
                
                revenue = pos['lots'] * final_opt_price
                cash += revenue
                commission = pos['lots'] * COMMISSION_OPTIONS
                cash -= commission
                trade_log.append({'date': today, 'action': 'sell_option_final', 'lots': -pos['lots'], 'price': final_opt_price, 'commission': commission, 'strike': strike})
                print(f"[期权平仓] 日期: {today.date()} | 卖出看跌: {pos['lots']} 手 | 价格: {final_opt_price:.2f} | 行权价: {strike:.2f} | 手续费: {commission:.2f} | 现金余额: {cash:,.2f}")
            options_position.clear()

        # --- 每日市值计算 (Mark-to-Market) ---
        # 1. 期货价值
        futures_market_value = futures_lots * futures_close

        # 2. 期权价值与记录期权价格（仅保护看跌）
        options_market_value = 0
        options_today = options_df.loc[options_df.index.isin([today])]
        # 默认当日期权价格为空，用于 daily_log 补充
        put_option_price_today = np.nan
        options_total_lots_today = 0
        if not options_today.empty:
            for strike, pos in options_position.items():
                # 仅按看跌期权估值，防止与同行权价的看涨混淆
                current_opt_price_series = options_today[(options_today['strike_price'] == strike) & (options_today['option_type'] == 'put')]['close']
                current_opt_price = current_opt_price_series.iloc[0] if not current_opt_price_series.empty else pos['entry_price']
                options_market_value += pos['lots'] * current_opt_price
                options_total_lots_today += pos['lots']
                # 记录一个代表性的保护看跌价格（若持有多个行权价，可取加权平均）
                if np.isnan(put_option_price_today):
                    put_option_price_today = current_opt_price
                else:
                    # 加权平均（按持仓手数）
                    total_lots = 0
                    accum = 0.0
                    for s2, p2 in options_position.items():
                        series2 = options_today[(options_today['strike_price'] == s2) & (options_today['option_type'] == 'put')]['close']
                        price2 = series2.iloc[0] if not series2.empty else p2['entry_price']
                        total_lots += p2['lots']
                        accum += p2['lots'] * price2
                    put_option_price_today = accum / total_lots if total_lots > 0 else current_opt_price
        
        # 总价值 = 现金 + 期货市值 + 期权市值
        total_value = cash + futures_market_value + options_market_value
        
        # 计算并记录每日盈亏
        daily_pnl = total_value - last_total_value
        last_total_value = total_value

        daily_snapshot['futures_lots'] = futures_lots
        daily_snapshot['total_value'] = total_value
        daily_snapshot['daily_pnl'] = daily_pnl
        daily_snapshot['cash'] = cash
        daily_snapshot['futures_market_value'] = futures_market_value
        daily_snapshot['options_market_value'] = options_market_value
        daily_snapshot['put_option_price'] = put_option_price_today
        daily_snapshot['options_total_value'] = options_market_value
        daily_snapshot['options_total_lots'] = options_total_lots_today
        daily_log.append(daily_snapshot)

    # --- 结果整合与输出 ---
    daily_log_df = pd.DataFrame(daily_log).set_index('date')

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    daily_log_df.to_csv(os.path.join(RESULTS_DIR, 'daily_log.csv'))
    
    print("\n--- 回测性能总结 ---")
    final_equity = daily_log_df['total_value'].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    print(f"初始资金: {INITIAL_CAPITAL:,.2f}")
    print(f"最终权益: {final_equity:,.2f}")
    print(f"总回报率: {total_return:.2%}")
    print(f"总交易记录数: {len(trade_log_df)}")
    print("--- 结果已保存 ---")
    
    return daily_log_df, trade_log_df

if __name__ == "__main__":
    print("开始执行'受保护的期货逆势网格策略'回测 (事件驱动版)...")
    try:
        futures_data, options_data = prepare_data(FUTURES_DATA_PATH, OPTIONS_DATA_PATH)
        daily_results, trade_log = run_backtest_event_driven(futures_data, options_data)
        if daily_results is not None:
            plot_results(daily_results, trade_log)
        print("\n回测执行完毕。")
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
