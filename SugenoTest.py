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
            # Low-risk scenario rules
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.market_conditions['bearish'] &
                self.economic_indicators['negative'],
                self.portfolio_adjustment['diversify']
            ),
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.market_conditions['neutral'] &
                self.portfolio_diversification['poor'],
                self.portfolio_adjustment['rebalance']
            ),

            # Medium-risk scenario rules
            ctrl.Rule(
                self.risk_tolerance['medium'] &
                self.market_conditions['bullish'] &
                self.economic_indicators['positive'] &
                self.portfolio_diversification['good'],
                self.portfolio_adjustment['hold']
            ),
            ctrl.Rule(
                self.risk_tolerance['medium'] &
                self.market_conditions['neutral'] &
                self.financial_goals['balanced'],
                self.portfolio_adjustment['rebalance']
            ),

            # High-risk scenario rules
            ctrl.Rule(
                self.risk_tolerance['high'] &
                self.market_conditions['bearish'] &
                self.economic_indicators['negative'],
                self.portfolio_adjustment['rebalance']
            ),

            # Composite condition rules
            ctrl.Rule(
                self.risk_tolerance['low'] &
                self.market_conditions['bullish'] &
                self.financial_goals['short_term'],
                self.portfolio_adjustment['diversify']
            ),
            ctrl.Rule(
                self.risk_tolerance['high'] &
                self.market_conditions['neutral'] &
                self.portfolio_diversification['moderate'],
                self.portfolio_adjustment['hold']
            ),

            # Economic indicator and market condition combination rules
            ctrl.Rule(
                self.economic_indicators['negative'] &
                self.market_conditions['bearish'] &
                self.portfolio_diversification['poor'],
                self.portfolio_adjustment['diversify']
            ),
            ctrl.Rule(
                self.economic_indicators['positive'] &
                self.market_conditions['bullish'] &
                self.financial_goals['long_term'],
                self.portfolio_adjustment['hold']
            ),

            # Comprehensive risk and financial goal rules
            ctrl.Rule(
                self.risk_tolerance['medium'] &
                self.financial_goals['balanced'] &
                self.portfolio_diversification['moderate'],
                self.portfolio_adjustment['rebalance']
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
        plt.figure(figsize=(18, 6))
        x = np.linspace(0, 100, 200)

        # 子图1：三角形
        plt.subplot(1, 3, 1)
        diversify_tri = fuzz.trimf(x, [0, 0, 30])
        rebalance_tri = fuzz.trimf(x, [30, 50, 70])
        hold_tri = fuzz.trimf(x, [70, 100, 100])

        plt.plot(x, diversify_tri, label='Diversify', color='blue')
        plt.plot(x, rebalance_tri, label='Rebalance', color='green')
        plt.plot(x, hold_tri, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='^', label=f'Decision: {adjustment_score:.2f}')

        plt.title('Triangle Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        # 子图2：梯形
        plt.subplot(1, 3, 2)
        diversify_trap = fuzz.trapmf(x, [0, 0, 20, 40])
        rebalance_trap = fuzz.trapmf(x, [30, 40, 60, 70])
        hold_trap = fuzz.trapmf(x, [60, 70, 100, 100])

        plt.plot(x, diversify_trap, label='Diversify', color='blue')
        plt.plot(x, rebalance_trap, label='Rebalance', color='green')
        plt.plot(x, hold_trap, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='s', label=f'Decision: {adjustment_score:.2f}')

        plt.title('Trapezoid Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        # 子图3：高斯曲线
        plt.subplot(1, 3, 3)
        diversify_gauss = fuzz.gaussmf(x, 15, 10)
        rebalance_gauss = fuzz.gaussmf(x, 50, 15)
        hold_gauss = fuzz.gaussmf(x, 85, 10)

        plt.plot(x, diversify_gauss, label='Diversify', color='blue')
        plt.plot(x, rebalance_gauss, label='Rebalance', color='green')
        plt.plot(x, hold_gauss, label='Hold', color='red')

        plt.scatter(adjustment_score, 0.5, color='purple', s=200,
                    marker='o', label=f'Decision: {adjustment_score:.2f}')

        plt.title('Gaussian Membership')
        plt.xlabel('Adjustment Score')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Portfolio Adjustment Sugeno Decision\nRecommendation: {recommendation}', fontsize=16)
        plt.tight_layout()
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

    while True:
        try:
            # Inputs
            risk_tolerance = get_float_input("Enter Risk Tolerance Score (0-100): ", 0, 100)
            market_condition = get_float_input("Enter Market Condition Score (0-100): ", 0, 100)
            economic_indicator = get_float_input("Enter Economic Indicator Score (0-100): ", 0, 100)
            portfolio_div = get_float_input("Enter Portfolio Diversification Score (0-100): ", 0, 100)
            financial_goal = get_float_input("Enter Financial Goal Score (0-100): ", 0, 100)

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

    print("Thank you for using Sugeno Portfolio Adjustment Model!")


if __name__ == "__main__":
    main()
