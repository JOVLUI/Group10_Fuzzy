import streamlit as st
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import io
import base64

from MamdaniValidate import PortfolioAdjustmentFuzzySystem
from SugenoValidate import PortfolioAdjustmentFuzzySugeno
from MarketCondition import MarketConditionFuzzy
from EconomicIndicator import EconomicIndicatorFuzzy
from FinancialGoal import InvestmentHorizonFuzzy
from PortfolioDiv import determine_diversification_level


def plot_to_base64(plt):
    """将Matplotlib图转换为Base64"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"


def create_fuzzy_visualization(input_value, universe, membership_funcs, title, xlabel):
    """创建模糊集可视化"""
    plt.figure(figsize=(10, 6))

    # 绘制模糊集
    for label, membership_func in membership_funcs.items():
        plt.plot(universe, membership_func, label=label)

    # 标记输入值
    plt.axvline(x=input_value, color='r', linestyle='--',
                label=f'Score: {input_value:.2f}')

    # 计算隶属度
    memberships = {label: fuzz.interp_membership(universe, func, input_value)
                   for label, func in membership_funcs.items()}

    # 添加注释
    annotation_text = f"Score: {input_value:.2f}\n" + \
                      "\n".join([f"{label} Membership: {membership:.2f}"
                                 for label, membership in memberships.items()])

    plt.annotate(annotation_text,
                 xy=(input_value, 0.5),
                 xytext=(10, 30),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.tight_layout()

    return plot_to_base64(plt)

def create_portfolio_adjustment_visualization(input_value):
    plt.figure(figsize=(10, 6))
    universe = np.linspace(0, 100, 200)
    membership_funcs = {
        'Aggressive Diversification': fuzz.trapmf(universe, [0, 0, 20, 40]),
        'Moderate Rebalancing': fuzz.trapmf(universe, [30, 40, 60, 70]),
        'Conservative Rebalancing': fuzz.trapmf(universe, [60, 70, 90, 100]),
    }
    return create_fuzzy_visualization(
        input_value,
        universe,
        membership_funcs,
        "Portfolio Adjustment Fuzzy Sets",
        "Adjustment score"
    )

def create_risk_visualization(input_value):
    universe = np.linspace(0, 100, 200)
    membership_funcs = {
        'Low Risk': fuzz.trapmf(universe, [0, 0, 20, 40]),
        'Medium Risk': fuzz.trapmf(universe, [30, 40, 60, 70]),
        'High Risk': fuzz.trapmf(universe, [60, 70, 100, 100])
    }
    return create_fuzzy_visualization(
        input_value,
        universe,
        membership_funcs,
        "Risk tolerance fuzzy set",
        "Risk tolerance score"
    )


def create_market_condition_visualization(input_value):
    universe = np.linspace(0, 100, 200)
    membership_funcs = {
        'Bearish': fuzz.trapmf(universe, [0, 0, 20, 40]),
        'Neutral': fuzz.trapmf(universe, [30, 40, 60, 70]),
        'Bullish': fuzz.trapmf(universe, [60, 70, 100, 100])
    }
    return create_fuzzy_visualization(
        input_value,
        universe,
        membership_funcs,
        "Fuzzy set of market conditions",
        "Market Conditions"
    )


def create_economic_indicator_visualization(input_value):
    universe = np.linspace(0, 100, 200)
    membership_funcs = {
        'Negative': fuzz.trapmf(universe, [0, 0, 20, 40]),
        'Neutral': fuzz.trapmf(universe, [30, 40, 60, 70]),
        'Positive': fuzz.trapmf(universe, [60, 70, 100, 100])
    }
    return create_fuzzy_visualization(
        input_value,
        universe,
        membership_funcs,
        "Fuzzy set of economic indicators",
        "Economic indicators score"
    )


def create_portfolio_div_visualization(input_value):
    universe = np.linspace(0, 100, 200)
    membership_funcs = {
        'Poor': fuzz.trapmf(universe, [0, 0, 20, 40]),
        'Moderate': fuzz.trapmf(universe, [30, 40, 60, 70]),
        'Good': fuzz.trapmf(universe, [60, 70, 100, 100])
    }
    return create_fuzzy_visualization(
        input_value,
        universe,
        membership_funcs,
        "Portfolio Diversification Fuzzy Set",
        "Portfolio Diversification score"
    )


def create_financial_goal_visualization(input_value):
    from FinancialGoal import InvestmentHorizonFuzzy

    financial_goal_system = InvestmentHorizonFuzzy()
    universe = np.linspace(0, 120, 200)

    short_term_values = [financial_goal_system.short_term_membership(x) for x in universe]
    balanced_values = [financial_goal_system.balanced_membership(x) for x in universe]
    long_term_values = [financial_goal_system.long_term_membership(x) for x in universe]

    plt.figure(figsize=(10, 6))
    plt.plot(universe, short_term_values, label='Short-term', color='blue')
    plt.plot(universe, balanced_values, label='Balanced', color='green')
    plt.plot(universe, long_term_values, label='Long-term', color='red')
    plt.axvline(x=input_value, color='purple', linestyle='--',
                label=f'Investment Horizon: {input_value:.2f}')

    # Calculate membership values
    short_term_membership = financial_goal_system.short_term_membership(input_value)
    balanced_membership = financial_goal_system.balanced_membership(input_value)
    long_term_membership = financial_goal_system.long_term_membership(input_value)

    plt.annotate(
        f'Investment Horizon: {input_value:.2f}\n'
        f'Short-term Membership: {short_term_membership:.2f}\n'
        f'Balanced Membership: {balanced_membership:.2f}\n'
        f'Long-term Membership: {long_term_membership:.2f}',
        xy=(input_value, 0.5),
        xytext=(10, 30),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )

    plt.title('Investment Horizon Membership Functions')
    plt.xlabel('Investment Months')
    plt.ylabel('Membership Degree')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    return plot_to_base64(plt)


def main():
    st.set_page_config(page_title="Portfolio fuzzy inference analysis", layout="wide")

    # 添加自定义CSS使内容居中和减少侧边栏间距
    st.markdown("""
    <style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    h1 {
        font-size: 30px !important;
    }
    
    /* 减少侧边栏控件间距 */
    .stSidebar .stSlider,
    .stSidebar .stSelectbox,
    .stSidebar .stButton {
        margin-bottom: 5px !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }

    /* 减少标题间距 */
    .stSidebar h2 {
        margin-bottom: 10px !important;
        margin-top: 10px !important;
    }

    /* 调整滑块标签样式 */
    .stSidebar .stSlider > label {
        margin-bottom: 2px !important;
        font-size: 0.9em !important;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        margin-top: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 左侧侧边栏
    with st.sidebar:
        st.header("Model Selection")
        model_type = st.selectbox(
            "Selecting a fuzzy inference model",
            ["Mamdani Model", "Sugeno Model"]
        )

        st.header("Investment Parameter Input")

        risk_tolerance = st.slider(
            "Risk Tolerance (0-100)",
            min_value=0,
            max_value=100,
            value=50
        )

        market_condition = st.slider(
            "Market Condition(0-100)",
            min_value=0,
            max_value=100,
            value=50
        )

        economic_indicator = st.slider(
            "Economic Indicator(0-100)",
            min_value=0,
            max_value=100,
            value=50
        )

        portfolio_div = st.slider(
            "Portfolio Diversification (0-100)",
            min_value=0,
            max_value=100,
            value=50
        )

        financial_goal = st.slider(
            "Investment Horizon (months)",
            min_value=0,
            max_value=120,
            value=36
        )

        analyze_button = st.button("Executive Reasoning ")

    # 主内容区域
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        st.title("📊 Intelligent Portfolio Adjustment Fuzzy Inference System ( By Group 10 of APU)")

        if analyze_button:
            # 初始化模糊系统
            fuzzy_system = (PortfolioAdjustmentFuzzySystem() if model_type == "Mamdani Model"
                            else PortfolioAdjustmentFuzzySugeno())

            # 计算调整得分
            adjustment_score = fuzzy_system.compute_portfolio_adjustment(
                risk_tolerance,
                market_condition,
                economic_indicator,
                portfolio_div,
                financial_goal
            )

            # 生成推荐
            if adjustment_score < 40:
                recommendation = "Diversify Portfolio"
            elif adjustment_score < 70:
                recommendation = "Rebalance Portfolio"
            else:
                recommendation = "Hold Current Portfolio"

            # 显示结果
            st.success(f"Adjusted Score: {adjustment_score:.2f}")
            st.info(f"Recommendation:  {recommendation} "
                    f"  -  (Risk Tolerance: {risk_tolerance}  "
                    f"Market Condition: {market_condition}  "
                    f"Economic Indicator: {economic_indicator}  "
                     f"Portfolio Diversification: {portfolio_div}  "
                    f"Investment Horizon: {financial_goal})")

            # 可视化标签页
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "💹Portfolio Adjustment",
                "⚡️Risk Tolerance",
                "🏪Market Condition",
                "💰️Economic Indicators",
                "👨‍👩‍👦Portfolio Diversification",
                "🗓️Investment Horizon"
            ])

            with tab1:
                portfolio_adjustment_viz = create_portfolio_adjustment_visualization(adjustment_score)
                st.image(portfolio_adjustment_viz, width=900)

            with tab2:
                risk_viz = create_risk_visualization(risk_tolerance)
                st.image(risk_viz, width=900)

            with tab3:
                market_viz = create_market_condition_visualization(market_condition)
                st.image(market_viz, width=900)

            with tab4:
                economic_viz = create_economic_indicator_visualization(economic_indicator)
                st.image(economic_viz, width=900)

            with tab5:
                portfolio_div_viz = create_portfolio_div_visualization(portfolio_div)
                st.image(portfolio_div_viz, width=900)

            with tab6:
                financial_goal_viz = create_financial_goal_visualization(financial_goal)
                st.image(financial_goal_viz, width=900)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
