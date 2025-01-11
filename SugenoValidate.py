import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

from FinancialGoal import InvestmentHorizonFuzzy
from MarketCondition import MarketConditionFuzzy
from PortfolioDiv import determine_diversification_level
from EconomicIndicator import EconomicIndicatorFuzzy


class PortfolioAdjustmentFuzzySugeno:
    def __init__(self):
        # Input variables
        self.risk_tolerance = ctrl.Antecedent(np.linspace(0, 100, 200), 'risk_tolerance')
        self.market_conditions = ctrl.Antecedent(np.linspace(0, 100, 200), 'market_conditions')
        self.economic_indicators = ctrl.Antecedent(np.linspace(0, 100, 200), 'economic_indicators')
        self.portfolio_diversification = ctrl.Antecedent(np.linspace(0, 100, 200), 'portfolio_diversification')
        self.financial_goals = ctrl.Antecedent(np.linspace(0, 100, 200), 'financial_goals')

        # Output variable
        self.portfolio_adjustment = ctrl.Consequent(np.linspace(0, 100, 200), 'portfolio_adjustment')

        # Initialize fuzzy sets and rules
        self._setup_fuzzy_sets()
        self._define_sugeno_rules()

        # Create control system
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def _setup_fuzzy_sets(self):
        # Risk Tolerance
        self.risk_tolerance['low'] = fuzz.trapmf(self.risk_tolerance.universe, [0, 0, 20, 40])
        self.risk_tolerance['medium'] = fuzz.trapmf(self.risk_tolerance.universe, [30, 40, 60, 70])
        self.risk_tolerance['high'] = fuzz.trapmf(self.risk_tolerance.universe, [60, 70, 100, 100])

        # Market Conditions
        self.market_conditions['bearish'] = fuzz.trapmf(self.market_conditions.universe, [0, 0, 20, 40])
        self.market_conditions['neutral'] = fuzz.trapmf(self.market_conditions.universe, [30, 40, 60, 70])
        self.market_conditions['bullish'] = fuzz.trapmf(self.market_conditions.universe, [60, 70, 100, 100])

        # Economic Indicators
        self.economic_indicators['negative'] = fuzz.trapmf(self.economic_indicators.universe, [0, 0, 20, 40])
        self.economic_indicators['positive'] = fuzz.trapmf(self.economic_indicators.universe, [60, 70, 100, 100])

        # Portfolio Diversification
        self.portfolio_diversification['poor'] = fuzz.trapmf(self.portfolio_diversification.universe, [0, 0, 20, 40])
        self.portfolio_diversification['moderate'] = fuzz.trapmf(self.portfolio_diversification.universe,
                                                                 [30, 40, 60, 70])
        self.portfolio_diversification['good'] = fuzz.trapmf(self.portfolio_diversification.universe,
                                                             [60, 70, 100, 100])

        # Financial Goals
        self.financial_goals['short_term'] = fuzz.trapmf(self.financial_goals.universe, [0, 0, 20, 40])
        self.financial_goals['balanced'] = fuzz.trapmf(self.financial_goals.universe, [30, 40, 60, 70])
        self.financial_goals['long_term'] = fuzz.trapmf(self.financial_goals.universe, [60, 70, 100, 100])

        # Sugeno output
        universe = np.linspace(0, 100, 200)

        # 创建输出变量的模糊集
        self.portfolio_adjustment['diversify'] = fuzz.trimf(universe, [0, 0, 30])
        self.portfolio_adjustment['rebalance'] = fuzz.trimf(universe, [30, 50, 70])
        self.portfolio_adjustment['hold'] = fuzz.trimf(universe, [70, 100, 100])

    def _define_sugeno_rules(self):
        self.rules = [
            # Rule 1: Low risk, bearish market, poor diversification
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.market_conditions['bearish'] &
                self.portfolio_diversification['poor'],
                self.portfolio_adjustment['diversify']
            ),

            # Rule 2: Medium risk, negative economic indicators, long-term goals
            ctrl.Rule(
                self.risk_tolerance['medium'] &
                self.economic_indicators['negative'] &
                self.financial_goals['long_term'],
                self.portfolio_adjustment['rebalance']
            ),

            # Rule 3: Bullish market, moderate diversification, balanced goals
            ctrl.Rule(
                self.market_conditions['bullish'] &
                self.portfolio_diversification['moderate'] &
                self.financial_goals['balanced'],
                self.portfolio_adjustment['diversify']
            ),

            # Rule 4: High risk, positive economic indicators, short-term goals
            ctrl.Rule(
                self.risk_tolerance['high'] &
                self.economic_indicators['positive'] &
                self.financial_goals['short_term'],
                self.portfolio_adjustment['hold']
            ),

            # Rule 5: Negative economic indicators or poor diversification, long-term goals
            ctrl.Rule(
                (self.economic_indicators['negative'] |
                 self.portfolio_diversification['poor']) &
                self.financial_goals['long_term'],
                self.portfolio_adjustment['rebalance']
            ),

            # Rule 6: Low risk, neutral market, balanced goals
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.market_conditions['neutral'] &
                self.financial_goals['balanced'],
                self.portfolio_adjustment['hold']
            ),

            # Rule 7: High risk or good diversification, positive economic indicators
            ctrl.Rule(
                (self.risk_tolerance['high'] |
                 self.portfolio_diversification['good']) &
                self.economic_indicators['positive'],
                self.portfolio_adjustment['hold']
            ),

            # Rule 8: Bearish market, negative economic indicators, short-term goals
            ctrl.Rule(
                self.market_conditions['bearish'] &
                self.economic_indicators['negative'] &
                self.financial_goals['short_term'],
                self.portfolio_adjustment['rebalance']
            ),

            # Rule 9: Neutral market or positive economic indicators, medium risk
            ctrl.Rule(
                (self.market_conditions['neutral'] |
                 self.economic_indicators['positive']) &
                self.risk_tolerance['medium'],
                self.portfolio_adjustment['hold']
            ),

            # Rule 10: Low risk, negative economic indicators, poor diversification
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.economic_indicators['negative'] &
                self.portfolio_diversification['poor'],
                self.portfolio_adjustment['diversify']
            )
        ]

    def compute_portfolio_adjustment(self, risk_tolerance, market_condition,
                                     economic_indicator, portfolio_div, financial_goal):
        # Ensure input variable names match the system definition
        self.simulator.input['risk_tolerance'] = risk_tolerance
        self.simulator.input['market_conditions'] = market_condition
        self.simulator.input['economic_indicators'] = economic_indicator
        self.simulator.input['portfolio_diversification'] = portfolio_div
        self.simulator.input['financial_goals'] = financial_goal

        # Compute results
        self.simulator.compute()

        # Return output
        return self.simulator.output['portfolio_adjustment']

    def visualize_final_decision(self, adjustment_score, recommendation):
        x = np.linspace(0, 100, 200)

        # 第一个窗口：三角形隶属度
        plt.figure(figsize=(10, 6))
        diversify_tri = fuzz.trimf(x, [0, 0, 30])
        rebalance_tri = fuzz.trimf(x, [30, 50, 70])
        hold_tri = fuzz.trimf(x, [70, 100, 100])

        plt.plot(x, diversify_tri, label='Diversify', color='blue')
        plt.plot(x, rebalance_tri, label='Rebalance', color='green')
        plt.plot(x, hold_tri, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='^', label=f'Decision: {adjustment_score:.2f}')

        # 计算隶属度
        diversify_membership = fuzz.interp_membership(x, diversify_tri, adjustment_score)
        rebalance_membership = fuzz.interp_membership(x, rebalance_tri, adjustment_score)
        hold_membership = fuzz.interp_membership(x, hold_tri, adjustment_score)

        plt.annotate(
            f'Adjustment Score: {adjustment_score:.2f}\n'
            f'Diversify Membership: {diversify_membership:.2f}\n'
            f'Rebalance Membership: {rebalance_membership:.2f}\n'
            f'Hold Membership: {hold_membership:.2f}\n'
            f'Recommendation: {recommendation}',
            xy=(adjustment_score, 0.5),
            xytext=(10, 30),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        plt.title('Triangle Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 第二个窗口：梯形隶属度
        plt.figure(figsize=(10, 6))
        diversify_trap = fuzz.trapmf(x, [0, 0, 20, 40])
        rebalance_trap = fuzz.trapmf(x, [30, 40, 60, 70])
        hold_trap = fuzz.trapmf(x, [60, 70, 100, 100])

        plt.plot(x, diversify_trap, label='Diversify', color='blue')
        plt.plot(x, rebalance_trap, label='Rebalance', color='green')
        plt.plot(x, hold_trap, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='s', label=f'Decision: {adjustment_score:.2f}')

        # 计算隶属度
        diversify_trap_membership = fuzz.interp_membership(x, diversify_trap, adjustment_score)
        rebalance_trap_membership = fuzz.interp_membership(x, rebalance_trap, adjustment_score)
        hold_trap_membership = fuzz.interp_membership(x, hold_trap, adjustment_score)

        plt.annotate(
            f'Adjustment Score: {adjustment_score:.2f}\n'
            f'Diversify Membership: {diversify_trap_membership:.2f}\n'
            f'Rebalance Membership: {rebalance_trap_membership:.2f}\n'
            f'Hold Membership: {hold_trap_membership:.2f}\n'
            f'Recommendation: {recommendation}',
            xy=(adjustment_score, 0.5),
            xytext=(10, 30),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        plt.title('Trapezoid Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 第三个窗口：高斯隶属度
        plt.figure(figsize=(10, 6))
        diversify_gauss = fuzz.gaussmf(x, 15, 10)
        rebalance_gauss = fuzz.gaussmf(x, 50, 15)
        hold_gauss = fuzz.gaussmf(x, 85, 10)

        plt.plot(x, diversify_gauss, label='Diversify', color='blue')
        plt.plot(x, rebalance_gauss, label='Rebalance', color='green')
        plt.plot(x, hold_gauss, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='o', label=f'Decision: {adjustment_score:.2f}')

        # 计算隶属度
        diversify_gauss_membership = fuzz.interp_membership(x, diversify_gauss, adjustment_score)
        rebalance_gauss_membership = fuzz.interp_membership(x, rebalance_gauss, adjustment_score)
        hold_gauss_membership = fuzz.interp_membership(x, hold_gauss, adjustment_score)

        plt.annotate(
            f'Adjustment Score: {adjustment_score:.2f}\n'
            f'Diversify Membership: {diversify_gauss_membership:.2f}\n'
            f'Rebalance Membership: {rebalance_gauss_membership:.2f}\n'
            f'Hold Membership: {hold_gauss_membership:.2f}\n'
            f'Recommendation: {recommendation}',
            xy=(adjustment_score, 0.5),
            xytext=(10, 30),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        plt.title('Gaussian Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()


def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = input(prompt)
            # Handle empty input
            if not value:
                print("Input cannot be empty. Please try again.")
                continue

            # Convert to float
            float_value = float(value)

            # Check range constraints
            if min_val is not None and float_value < min_val:
                print(f"Value must be at least {min_val}")
                continue

            if max_val is not None and float_value > max_val:
                print(f"Value must be at most {max_val}")
                continue

            return float_value

        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main():
    print("********* Sugeno Portfolio Adjustment Model *********")

    fuzzy_system = PortfolioAdjustmentFuzzySugeno()
    market_condition_system = MarketConditionFuzzy()
    economic_indicator_system = EconomicIndicatorFuzzy()
    financial_goal_system = InvestmentHorizonFuzzy()

    while True:
        try:
            # Risk Tolerance Input
            print("=== Enter Risk Tolerance Score ===")
            risk_tolerance = get_float_input("Low: 0-30, Neutral:30-70, Bullish:70-100:  ", min_val=0, max_val=100)

            # Determine Risk Level with explicit assignment
            risk_level = "Low"  # Default value
            if risk_tolerance >= 40:
                risk_level = "Medium"
            if risk_tolerance >= 70:
                risk_level = "High"

            plt.figure(figsize=(10, 6))
            risk_universe = np.linspace(0, 100, 200)
            low_risk = fuzz.trapmf(risk_universe, [0, 0, 20, 40])
            medium_risk = fuzz.trapmf(risk_universe, [30, 40, 60, 70])
            high_risk = fuzz.trapmf(risk_universe, [60, 70, 100, 100])

            plt.plot(risk_universe, low_risk, label='Low Risk')
            plt.plot(risk_universe, medium_risk, label='Medium Risk')
            plt.plot(risk_universe, high_risk, label='High Risk')
            plt.axvline(x=risk_tolerance, color='r', linestyle='--', label=f'Risk Score: {risk_tolerance:.2f}')

            # Calculate membership values
            low_risk_membership = fuzz.interp_membership(risk_universe, low_risk, risk_tolerance)
            medium_risk_membership = fuzz.interp_membership(risk_universe, medium_risk, risk_tolerance)
            high_risk_membership = fuzz.interp_membership(risk_universe, high_risk, risk_tolerance)

            plt.annotate(
                f'Risk Score: {risk_tolerance:.2f}\n'
                f'Low Risk Membership: {low_risk_membership:.2f}\n'
                f'Medium Risk Membership: {medium_risk_membership:.2f}\n'
                f'High Risk Membership: {high_risk_membership:.2f}\n'
                f'Risk Level: {risk_level}',
                xy=(risk_tolerance, 0.5),
                xytext=(10, 30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

            plt.title('Risk Tolerance Fuzzy Sets')
            plt.xlabel('Risk Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Market Condition Input
            print("=============================================================")
            print("Enter Market Condition Score")
            market_condition = get_float_input("(Bearish: 0-30, Neutral: 30-70, High Growth: 70-100): ", min_val=0,
                                               max_val=100)

            market_condition_result = market_condition_system.determine_market_condition(market_condition)

            plt.figure(figsize=(10, 6))
            market_universe = np.linspace(0, 100, 200)
            bearish = fuzz.trapmf(market_universe, [0, 0, 20, 40])
            neutral = fuzz.trapmf(market_universe, [30, 40, 60, 70])
            bullish = fuzz.trapmf(market_universe, [60, 70, 100, 100])

            plt.plot(market_universe, bearish, label='Bearish')
            plt.plot(market_universe, neutral, label='Neutral')
            plt.plot(market_universe, bullish, label='Bullish')
            plt.axvline(x=market_condition, color='r', linestyle='--',
                        label=f'Market Condition: {market_condition:.2f}')

            # Calculate membership values
            bearish_membership = fuzz.interp_membership(market_universe, bearish, market_condition)
            neutral_membership = fuzz.interp_membership(market_universe, neutral, market_condition)
            bullish_membership = fuzz.interp_membership(market_universe, bullish, market_condition)

            plt.annotate(
                f'Market Condition: {market_condition:.2f}\n'
                f'Bearish Membership: {bearish_membership:.2f}\n'
                f'Neutral Membership: {neutral_membership:.2f}\n'
                f'Bullish Membership: {bullish_membership:.2f}\n'
                f'Condition: {market_condition_result[0]}',
                xy=(market_condition, 0.5),
                xytext=(10, 30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

            plt.title('Market Condition Fuzzy Sets')
            plt.xlabel('Market Condition Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Economic Indicator Input
            print("=============================================================")
            print("Enter Economic Indicator Score")
            economic_indicator = get_float_input("Negative: 0-30, Neutral: 30-70, Positive: 70-100: ", min_val=0,
                                                 max_val=100)

            economic_condition = economic_indicator_system.determine_economic_condition(economic_indicator)

            plt.figure(figsize=(10, 6))
            economic_universe = np.linspace(0, 100, 200)
            negative = economic_indicator_system.trapezoidal_membership(economic_universe, 0, 0, 20, 30)
            neutral = economic_indicator_system.trapezoidal_membership(economic_universe, 30, 40, 60, 70)
            positive = economic_indicator_system.trapezoidal_membership(economic_universe, 70, 80, 100, 100)

            plt.plot(economic_universe, negative, label='Negative', color='red')
            plt.plot(economic_universe, neutral, label='Neutral', color='green')
            plt.plot(economic_universe, positive, label='Positive', color='blue')
            plt.axvline(x=economic_indicator, color='purple', linestyle='--',
                        label=f'Economic Indicator: {economic_indicator:.2f}')

            # Calculate membership values
            negative_membership = economic_indicator_system.trapezoidal_membership(economic_indicator, 0, 0, 20, 30)
            neutral_membership = economic_indicator_system.trapezoidal_membership(economic_indicator, 30, 40, 60, 70)
            positive_membership = economic_indicator_system.trapezoidal_membership(economic_indicator, 70, 80, 100, 100)

            plt.annotate(
                f'Economic Indicator: {economic_indicator:.2f}\n'
                f'Negative Membership: {negative_membership:.2f}\n'
                f'Neutral Membership: {neutral_membership:.2f}\n'
                f'Positive Membership: {positive_membership:.2f}\n'
                f'Economic Condition: {economic_condition}',
                xy=(economic_indicator, 0.5),
                xytext=(10, 30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

            plt.title('Economic Indicator Fuzzy Sets')
            plt.xlabel('Economic Indicator Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Portfolio Diversification Input
            print("=============================================================")
            print("Enter Portfolio Diversification Score")
            portfolio_div = get_float_input("Poor: 0-40, Moderate: 40-70, Good: 70-100: ", min_val=0, max_val=100)

            portfolio_div_level, portfolio_div_membership = determine_diversification_level(portfolio_div)

            plt.figure(figsize=(10, 6))
            div_universe = np.linspace(0, 100, 200)
            poor = fuzz.trapmf(div_universe, [0, 0, 20, 40])
            moderate = fuzz.trapmf(div_universe, [30, 40, 60, 70])
            good = fuzz.trapmf(div_universe, [60, 70, 100, 100])

            plt.plot(div_universe, poor, label='Poor')
            plt.plot(div_universe, moderate, label='Moderate')
            plt.plot(div_universe, good, label='Good')
            plt.axvline(x=portfolio_div, color='r', linestyle='--',
                        label=f'Diversification: {portfolio_div:.2f}')

            # Calculate membership values
            poor_membership = fuzz.interp_membership(div_universe, poor, portfolio_div)
            moderate_membership = fuzz.interp_membership(div_universe, moderate, portfolio_div)
            good_membership = fuzz.interp_membership(div_universe, good, portfolio_div)

            plt.annotate(
                f'Diversification Score: {portfolio_div:.2f}\n'
                f'Poor Membership: {poor_membership:.2f}\n'
                f'Moderate Membership: {moderate_membership:.2f}\n'
                f'Good Membership: {good_membership:.2f}\n'
                f'Diversification Level: {portfolio_div_level}',
                xy=(portfolio_div, 0.5),
                xytext=(10, 30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

            plt.title('Portfolio Diversification Fuzzy Sets')
            plt.xlabel('Diversification Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Financial Goal Input
            print("=============================================================")
            print("Enter Investment Horizon (months):")
            financial_goal = get_float_input("Short-term: 0-12, Balanced: 12-36, Long-term: >36: ", min_val=0)

            goal_type, membership_value = financial_goal_system.determine_investment_horizon(financial_goal)

            plt.figure(figsize=(10, 6))
            goal_universe = np.linspace(0, 120, 200)
            short_term_values = [financial_goal_system.short_term_membership(x) for x in goal_universe]
            balanced_values = [financial_goal_system.balanced_membership(x) for x in goal_universe]
            long_term_values = [financial_goal_system.long_term_membership(x) for x in goal_universe]

            plt.plot(goal_universe, short_term_values, label='Short-term', color='blue')
            plt.plot(goal_universe, balanced_values, label='Balanced', color='green')
            plt.plot(goal_universe, long_term_values, label='Long-term', color='red')

            plt.axvline(x=financial_goal, color='purple', linestyle='--',
                        label=f'Investment Horizon: {financial_goal:.2f}')

            # Calculate membership values
            short_term_membership = financial_goal_system.short_term_membership(financial_goal)
            balanced_membership = financial_goal_system.balanced_membership(financial_goal)
            long_term_membership = financial_goal_system.long_term_membership(financial_goal)

            plt.annotate(
                f'Investment Horizon: {financial_goal:.2f}\n'
                f'Short-term Membership: {short_term_membership:.2f}\n'
                f'Balanced Membership: {balanced_membership:.2f}\n'
                f'Long-term Membership: {long_term_membership:.2f}\n'
                f'Goal Type: {goal_type}',
                xy=(financial_goal, 0.5),
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
            plt.show()

            # Compute portfolio adjustment
            adjustment_score = fuzzy_system.compute_portfolio_adjustment(
                risk_tolerance,
                market_condition,
                economic_indicator,
                portfolio_div,
                financial_goal
            )

            # Generate recommendation
            if adjustment_score < 30:
                recommendation = "Aggressive Diversification"
            elif adjustment_score < 50:
                recommendation = "Moderate Rebalancing"
            elif adjustment_score < 70:
                recommendation = "Conservative Rebalancing"
            else:
                recommendation = "Hold Current Portfolio"

            # Display results
            print(f"\n--- Portfolio Adjustment Analysis ---")
            print(f"Adjustment Score: {adjustment_score:.2f}")
            print(f"Recommendation: {recommendation}")

            fuzzy_system.visualize_final_decision(adjustment_score, recommendation)

            # Continue or exit
            if input("\nContinue assessment? (y/n): ").lower() != 'y':
                break

        except Exception as e:
            print(f"Assessment Error: {e}")

    print("Thank you, Bye!")


if __name__ == "__main__":
    main()
