# streamlit run portfolio_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

# Display the disclaimer
disclaimer_text = """
## Disclaimer

**This dashboard is for informational and educational purposes only. It is not intended as investment advice or as a recommendation to buy, sell, or hold any securities.**

The information and analysis provided on this dashboard are based on historical data and assumptions that may not accurately reflect current market conditions. Investing in financial markets carries risks, and past performance is not indicative of future results.

Users are encouraged to conduct their own research and consult with a qualified financial advisor before making investment decisions. Any actions taken based on the information provided here are the sole responsibility of the user.

The creators and contributors of this dashboard are not responsible for any investment decisions, financial losses, or other consequences resulting from the use of this tool.

Please use this dashboard as a starting point for your own research. Your financial situation and risk tolerance should be carefully considered before making investment choices.
"""

# Create a sidebar with two options
with st.sidebar:
    st.image('https://cdn-icons-png.flaticon.com/512/5329/5329260.png')
    st.title('Portfolio Analysis Dashboard')
    selected_tab = st.radio('',["Portfolio Analysis","Portfolio Optimization", "Stock Price Forecasting"])
    st.info(disclaimer_text)

# Depending on the selected tab, show the corresponding content
if selected_tab == "Portfolio Analysis":
    st.title("Portfolio Analysis")
    #st.markdown(disclaimer_text)
    
    # Get portfolio data from the user
    assets = st.text_input("Enter the portfolio stocks (comma-separated)", "AAPL,MSFT")
    asset_list = [asset.strip() for asset in assets.split(',')]

    start_date = st.date_input("Select the starting date for your analysis", value=pd.to_datetime('2020-01-01'))

    # Download data from Yahoo Finance
    data = yf.download(asset_list, start=start_date)['Adj Close']

    # Get user-defined weights
    weights = st.text_input("Enter the portfolio weights (comma-separated)", "0.5,0.5")
    weights = [float(weight.strip()) for weight in weights.split(',')]

    # Normalize the weights to sum to 1
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Calculate daily returns
    daily_ret_df = data.pct_change()[1:]
    daily_ret_df

    # Calculate portfolio daily returns
    pf_daily_ret = (daily_ret_df * weights).sum(axis=1)
   
    # Calculate cumulative daily returns
    cumulative_returns = (pf_daily_ret + 1).cumprod() - 1

    # Calculate Sharpe Ratio
    risk_free_rate = st.number_input("Enter the risk-free rate (e.g., 0.03 for 3%)", min_value=0.0, max_value=1.0, value=0.03)
    excess_returns = pf_daily_ret - risk_free_rate / 252  # Assuming 252 trading days in a year
    sharpe_ratio = (252 ** 0.5) * excess_returns.mean() / excess_returns.std()

    # Calculate Standard Deviation
    portfolio_std_dev = excess_returns.std()

    # Convert the index to datetime
    cumulative_returns.index = pd.to_datetime(cumulative_returns.index)

    # Calculate day change, YTD change, total change or gain, and average daily return
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    yesterday = pd.to_datetime(yesterday)  # Convert to datetime
    ytd_start_date = pd.to_datetime(f'{yesterday.year}-01-01')
    ytd_idx = cumulative_returns.index.get_loc(cumulative_returns[cumulative_returns.index <= yesterday].idxmax())
    ytd_return = cumulative_returns.iloc[ytd_idx]
    total_change = cumulative_returns.iloc[-1]
    day_change = cumulative_returns.iloc[-1] - cumulative_returns.iloc[-2]
    average_daily_return = pf_daily_ret.mean()


    try:
        # Benchmark comparison (S&P 500)
        benchmark = yf.download('^GSPC', start=start_date)['Adj Close']
        benchmark_returns = benchmark.pct_change()
        benchmark_cumulative_returns = (benchmark_returns + 1).cumprod() - 1

        # Calculate benchmark metrics
        benchmark_day_change = benchmark_cumulative_returns.iloc[-1] - benchmark_cumulative_returns.iloc[-2]
        benchmark_ytd_idx = benchmark_cumulative_returns.index.get_loc(benchmark_cumulative_returns[benchmark_cumulative_returns.index <= yesterday].idxmax())
        benchmark_ytd_return = benchmark_cumulative_returns.iloc[benchmark_ytd_idx]
        benchmark_total_change = benchmark_cumulative_returns.iloc[-1]
        benchmark_average_daily_return = benchmark_returns.mean()

    except Exception as e:
        st.warning(f"Failed to download S&P 500 data. Error: {e}")
        # Set default values or handle the situation as needed
        benchmark_day_change = benchmark_ytd_return = benchmark_total_change = benchmark_average_daily_return = None

    # Format the data as percentages
    comparison_data = {
        "Metric": ["Day Change", "YTD Return", "Total Return", "Average Daily Return"],
        "Portfolio": [day_change * 100, ytd_return * 100, total_change * 100, average_daily_return * 100],
        "S&P 500": [benchmark_day_change * 100, benchmark_ytd_return * 100, benchmark_total_change * 100, benchmark_average_daily_return * 100],
    }

    # Display the performance comparison in a table as percentages
    st.subheader("Portfolio Returns vs. S&P 500")
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('Metric', inplace=True)
    comparison_df["Portfolio"] = comparison_df["Portfolio"].map("{:.2f}%".format)
    comparison_df["S&P 500"] = comparison_df["S&P 500"].map("{:.2f}%".format)
    st.table(comparison_df)

    # Plot a histogram of daily returns
    st.subheader("Histogram of Daily Returns")
    fig, ax = plt.subplots()
    pf_daily_ret.hist(bins=20, alpha=0.7, label='Portfolio Daily Returns', color='blue')
    benchmark_returns.hist(bins=20, alpha=0.7, label='S&P 500 Daily Returns', color='orange')
    plt.xlabel('Daily % Return')
    # plt.ylabel('Value')
    plt.legend()
    st.pyplot(fig)


#################
    # Calculate risk statistics
    portfolio_annualized_return = (cumulative_returns[-1] + 1) ** (252 / len(cumulative_returns)) - 1
    benchmark_annualized_return = (benchmark_cumulative_returns[-1] + 1) ** (252 / len(benchmark_cumulative_returns)) - 1

    portfolio_std_dev = pf_daily_ret.std() * np.sqrt(252)
    benchmark_std_dev = benchmark_returns.std() * np.sqrt(252)

    # Calculate alpha and beta using CAPM model
    # cov_matrix = daily_ret_df.cov() # daily_ret_df = data.pct_change()[1:]
    # cov_matrix2 = benchmark_returns.to_frame().cov()  # Convert benchmark_returns to DataFrame

    # portfolio_beta = cov_matrix.loc[asset_list].mean() / cov_matrix2.mean()
    # portfolio_alpha = (portfolio_annualized_return - risk_free_rate) - (portfolio_beta * (benchmark_annualized_return - risk_free_rate))

    # # Create a DataFrame for risk statistics
    # risk_stats_data = {
    #     "Metric": ["Alpha", "Beta", "Standard Deviation"],
    #     "Portfolio": [portfolio_alpha, portfolio_beta, portfolio_std_dev],
    #     "S&P 500": ["N/A", 1.0, benchmark_std_dev],
    # }
    # risk_stats_df = pd.DataFrame(risk_stats_data)

    # # Display the risk statistics in a table
    # st.subheader("Risk Statistics (Portfolio vs. S&P 500)")
    # st.table(risk_stats_df)
    # Calculate alpha and beta using CAPM model
    cov_matrix = daily_ret_df.cov()
    cov_matrix2 = benchmark_returns.to_frame().cov()  # Convert benchmark_returns to DataFrame

    # Calculate beta
    portfolio_beta = (cov_matrix.loc[asset_list].mean() / cov_matrix2.iloc[0].mean()).mean()

    # Calculate alpha
    portfolio_alpha = portfolio_annualized_return - risk_free_rate - portfolio_beta * (benchmark_annualized_return - risk_free_rate)

    # Create a DataFrame for risk statistics
    risk_stats_data = {
        "Metric": ["Alpha", "Beta", "Standard Deviation"],
        "Portfolio": [portfolio_alpha, portfolio_beta, portfolio_std_dev],
        "S&P 500": ["N/A", 1.0, benchmark_std_dev],
    }
    risk_stats_df = pd.DataFrame(risk_stats_data)

    # Display the risk statistics in a table
    st.subheader("Risk Statistics (Portfolio vs. S&P 500)")
    st.table(risk_stats_df)

######
    # Plot portfolio vs. S&P 500
    port_bench_df = pd.concat([cumulative_returns, benchmark_cumulative_returns], axis=1)
    port_bench_df.columns = ['Portfolio', 'S&P 500']

    st.subheader('Portfolio vs. S&P 500 Comparison')
    st.line_chart(data=port_bench_df)

    st.subheader('Portfolio Composition')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(weights, labels=data.columns, autopct='%1.1f%%', textprops={'color':'black'})
    st.pyplot(fig)


elif selected_tab == "Portfolio Optimization":
    st.title("Portfolio Optimization")

elif selected_tab == "Stock Price Forecasting":
    st.title("Stock Price Forecasting")

