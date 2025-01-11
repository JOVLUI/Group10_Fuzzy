import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

from FinancialGoal import InvestmentHorizonFuzzy
from MarketCondition import MarketConditionFuzzy
from PortfolioDiv import determine_diversification_level
from EconomicIndicator import EconomicIndicatorFuzzy
from RiskTolerance import RiskToleranceCalculator


class PortfolioAdjustmentFuzzySystem:
    def __init__(self):
        # Input variables
        self.risk_tolerance = ctrl.Antecedent(np.linspace(0, 100, 200), 'risk_tolerance')
        self.market_conditions = ctrl.Antecedent(np.linspace(0, 100, 200), 'market_conditions')
        self.economic_indicators = ctrl.Antecedent(np.linspace(0, 100, 200), 'economic_indicators')
        self.portfolio_diversification = ctrl.Antecedent(np.linspace(0, 100, 200), 'portfolio_diversification')
        self.financial_goals = ctrl.Antecedent(np.linspace(0, 120, 200), 'financial_goals')

        # Output variable
        self.portfolio_adjustment = ctrl.Consequent(np.linspace(0, 100, 200), 'portfolio_adjustment')

        # Initialize fuzzy sets and rules
        self._setup_fuzzy_sets()
        self._define_rules()

        # Create control system
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def _setup_fuzzy_sets(self):
        # Fuzzy sets for each variable
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
        self.portfolio_diversification['moderate'] = fuzz.trapmf(self.portfolio_diversification.universe, [30, 40, 60, 70])
        self.portfolio_diversification['good'] = fuzz.trapmf(self.portfolio_diversification.universe, [60, 70, 100, 100])

        # Financial Goals
        self.financial_goals['short_term'] = fuzz.trapmf(self.financial_goals.universe, [0, 0, 20, 40])
        self.financial_goals['balanced'] = fuzz.trapmf(self.financial_goals.universe, [30, 40, 60, 70])
        self.financial_goals['long_term'] = fuzz.trapmf(self.financial_goals.universe, [60, 70, 120, 120])

        # Portfolio Adjustment
        self.portfolio_adjustment['diversify'] = fuzz.trapmf(self.portfolio_adjustment.universe, [0, 0, 20, 40])
        self.portfolio_adjustment['rebalance'] = fuzz.trapmf(self.portfolio_adjustment.universe, [30, 40, 60, 70])
        self.portfolio_adjustment['hold'] = fuzz.trapmf(self.portfolio_adjustment.universe, [60, 70, 100, 100])

    def _define_rules(self):
        # Define 10 fuzzy rules covering different scenarios
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
        plt.figure(figsize=(10, 6))

        # Create trapezoidal membership functions
        x = np.linspace(0, 100, 200)

        # Diversify: Trapezoidal function, flat at [0, 40]
        diversify = fuzz.trapmf(x, [0, 0, 20, 40])

        # Rebalance: Trapezoidal function, flat at [40, 70]
        rebalance = fuzz.trapmf(x, [30, 40, 60, 70])

        # Hold: Trapezoidal function, flat at [70, 100]
        hold = fuzz.trapmf(x, [60, 70, 100, 100])

        plt.plot(x, diversify, label='Diversify')
        plt.plot(x, rebalance, label='Rebalance')
        plt.plot(x, hold, label='Hold')

        # Calculate current score's membership
        def calculate_membership(score, membership_func):
            if score <= membership_func[1]:
                return (score - membership_func[0]) / (membership_func[1] - membership_func[0]) if membership_func[1] != membership_func[0] else 1
            elif score <= membership_func[2]:
                return 1
            elif score <= membership_func[3]:
                return (membership_func[3] - score) / (membership_func[3] - membership_func[2]) if membership_func[3] != membership_func[2] else 1
            else:
                return 0

        diversify_membership = calculate_membership(adjustment_score, [0, 0, 20, 40])
        rebalance_membership = calculate_membership(adjustment_score, [30, 40, 60, 70])
        hold_membership = calculate_membership(adjustment_score, [60, 70, 100, 100])

        plt.axvline(x=adjustment_score, color='r', linestyle='--',
                    label=f'Decision: {adjustment_score:.2f}')

        # Annotate membership and coordinate points
        plt.annotate(
            f'Score: {adjustment_score:.2f}\n'
            f'Diversify Membership: {diversify_membership:.2f}\n'
            f'Rebalance Membership: {rebalance_membership:.2f}\n'
            f'Hold Membership: {hold_membership:.2f}',
            xy=(adjustment_score, 0.5),
            xytext=(10, 30),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        plt.title(f'Portfolio Adjustment Fuzzy Decision\nRecommendation: {recommendation}')
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
    print("********* This is Mamdani Model *********")

    fuzzy_system = PortfolioAdjustmentFuzzySystem()
    market_condition_system = MarketConditionFuzzy()
    economic_indicator_system = EconomicIndicatorFuzzy()
    financial_goal_system = InvestmentHorizonFuzzy()

    while True:
        try:
            # Direct Risk Tolerance Input
            print("=== Enter Risk Tolerance Score ===")
            risk_tolerance = get_float_input("Low: 0-30, Neutral:30-70, Bullish:70-100:  ", min_val=0, max_val=100)

            plt.figure(figsize=(10, 6))
            risk_universe = np.linspace(0, 100, 200)
            low_risk = fuzz.trapmf(risk_universe, [0, 0, 20, 40])
            medium_risk = fuzz.trapmf(risk_universe, [30, 40, 60, 70])
            high_risk = fuzz.trapmf(risk_universe, [60, 70, 100, 100])

            plt.plot(risk_universe, low_risk, label='Low Risk')
            plt.plot(risk_universe, medium_risk, label='Medium Risk')
            plt.plot(risk_universe, high_risk, label='High Risk')
            plt.axvline(x=risk_tolerance, color='r', linestyle='--', label=f'Risk Score: {risk_tolerance:.2f}')
            plt.title('Risk Tolerance Fuzzy Sets')
            plt.xlabel('Risk Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Determine Risk Level
            risk_level = "Low" if risk_tolerance < 40 else "Medium" if risk_tolerance < 70 else "High"
            print(f"\nRisk Tolerance Score: {risk_tolerance:.2f}")
            print(f"Risk Level: {risk_level}")

            # Market Condition Input
            print("=============================================================")
            print("Enter Market Condition Score")
            market_condition = get_float_input("(Bearish: 0-30, Neutral: 30-70, High Growth: 70-100): ", min_val=0, max_val=100)

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
            plt.title('Market Condition Fuzzy Sets')
            plt.xlabel('Market Condition Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            print(f"Market Condition: {market_condition_result[0]} (Membership: {market_condition_result[1]:.2f})")

            # Economic Indicator Input
            print("=============================================================")
            print("Enter Economic Indicator Score")
            economic_indicator = get_float_input("Negative: 0-30, Neutral: 30-70, Positive: 70-100: ", min_val=0, max_val=100)
            economic_condition_system = EconomicIndicatorFuzzy()
            economic_condition = economic_condition_system.determine_economic_condition(economic_indicator)

            plt.figure(figsize=(10, 6))
            economic_universe = np.linspace(0, 100, 200)
            negative = economic_condition_system.trapezoidal_membership(economic_universe, 0, 0, 20, 30)
            neutral = economic_condition_system.trapezoidal_membership(economic_universe, 30, 40, 60, 70)
            positive = economic_condition_system.trapezoidal_membership(economic_universe, 70, 80, 100, 100)

            plt.plot(economic_universe, negative, label='Negative', color='red')
            plt.plot(economic_universe, neutral, label='Neutral', color='green')
            plt.plot(economic_universe, positive, label='Positive', color='blue')
            plt.axvline(x=economic_indicator, color='purple', linestyle='--',
                        label=f'Economic Indicator: {economic_indicator:.2f}')
            plt.title('Economic Indicator Fuzzy Sets')
            plt.xlabel('Economic Indicator Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            # Calculate membership for each fuzzy set
            negative_membership = economic_condition_system.trapezoidal_membership(economic_indicator, 0, 0, 20, 30)
            neutral_membership = economic_condition_system.trapezoidal_membership(economic_indicator, 30, 40, 60, 70)
            positive_membership = economic_condition_system.trapezoidal_membership(economic_indicator, 70, 80, 100, 100)

            membership_value = (
                negative_membership if economic_condition == 'Negative' else
                neutral_membership if economic_condition == 'Neutral' else
                positive_membership
            )
            print(f"Economic Condition: {economic_condition} (Membership: {membership_value:.2f})")

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
            plt.axvline(x=portfolio_div, color='r', linestyle='--', label=f'Diversification: {portfolio_div:.2f}')
            plt.title('Portfolio Diversification Fuzzy Sets')
            plt.xlabel('Diversification Score')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.show()

            print(f"Portfolio Diversification Level: {portfolio_div_level} (Membership: {portfolio_div_membership:.2f})")

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
            plt.title('Investment Horizon Membership Functions')
            plt.xlabel('Investment Months')
            plt.ylabel('Membership Degree')
            plt.ylim(0, 1)
            plt.legend()
            plt.show()

            print(f"Investment Horizon: {goal_type} (Membership: {membership_value:.2f})")

            # Compute portfolio adjustment
            adjustment_score = fuzzy_system.compute_portfolio_adjustment(
                risk_tolerance, market_condition, economic_indicator,
                portfolio_div, financial_goal
            )

            print(f"\nPortfolio Adjustment Score: {adjustment_score:.2f}")

            if adjustment_score < 40:
                recommendation = "Diversify Portfolio"
            elif adjustment_score < 70:
                recommendation = "Rebalance Portfolio"
            else:
                recommendation = "Hold Current Portfolio"
            print(f"Recommendation: {recommendation}")

            fuzzy_system.visualize_final_decision(adjustment_score, recommendation)

            continue_choice = input("\nContinue assessment? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except Exception as e:
            print(f"An error occurred: {e}")

    print("Thank you, Bye!")


if __name__ == "__main__":
    main()
