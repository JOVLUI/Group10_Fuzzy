import numpy as np
import matplotlib.pyplot as plt


class InvestmentHorizonFuzzy:
    def __init__(self):
        self.months_universe = np.linspace(0, 120, 200)

    def short_term_membership(self, x):
        """Membership function for short-term investment (0-12 months)"""
        return 1 - 1 / (1 + np.exp(-0.5 * (x - 6)))

    def balanced_membership(self, x):
        """Membership function for balanced investment (12-36 months)"""
        if 12 <= x <= 36:
            return 1.0
        elif x < 12:
            return (x - 0) / (12 - 0)  # Linear increase
        else:
            return 1 - (x - 36) / (48 - 36)  # Linear decrease

    def long_term_membership(self, x):
        """Membership function for long-term investment (>36 months)"""
        return 1 / (1 + np.exp(-0.1 * (x - 72)))

    def plot_membership_functions(self, input_value):
        plt.figure(figsize=(15, 8))

        # Calculate membership for each function
        short_term_values = [self.short_term_membership(x) for x in self.months_universe]
        balanced_values = [self.balanced_membership(x) for x in self.months_universe]
        long_term_values = [self.long_term_membership(x) for x in self.months_universe]

        # Plot membership functions
        plt.plot(self.months_universe, short_term_values, label='Short-term', color='blue')
        plt.plot(self.months_universe, balanced_values, label='Balanced', color='green')
        plt.plot(self.months_universe, long_term_values, label='Long-term', color='red')

        # Mark input value
        plt.axvline(x=input_value, color='purple', linestyle='--', label=f'Input: {input_value}')

        # Calculate input value membership
        input_short = self.short_term_membership(input_value)
        input_balanced = self.balanced_membership(input_value)
        input_long = self.long_term_membership(input_value)

        # Annotate input value membership
        plt.scatter([input_value], [input_short], color='blue', zorder=5)
        plt.scatter([input_value], [input_balanced], color='green', zorder=5)
        plt.scatter([input_value], [input_long], color='red', zorder=5)

        # Display membership values on the plot
        plt.text(input_value + 2, input_short, f'Short-term: {input_short:.2f}', color='blue')
        plt.text(input_value + 2, input_balanced, f'Balanced: {input_balanced:.2f}', color='green')
        plt.text(input_value + 2, input_long, f'Long-term: {input_long:.2f}', color='red')

        plt.title('Investment Horizon Membership Functions')
        plt.xlabel('Investment Months')
        plt.ylabel('Membership Degree')
        plt.ylim(0, 1)  # Set Y-axis range to 0-1
        plt.legend()
        plt.grid(True)
        plt.show()

        return input_short, input_balanced, input_long

    def determine_investment_horizon(self, input_value):
        short_membership = self.short_term_membership(input_value)
        balanced_membership = self.balanced_membership(input_value)
        long_membership = self.long_term_membership(input_value)

        memberships = {
            'Short-term': short_membership,
            'Balanced': balanced_membership,
            'Long-term': long_membership
        }

        horizon_type = max(memberships, key=memberships.get)
        return horizon_type, memberships[horizon_type]


def main():
    fuzzy_system = InvestmentHorizonFuzzy()

    while True:
        try:
            input_value = float(input("Enter investment months (enter negative number to exit): "))

            if input_value < 0:
                break

            horizon_type, membership_value = fuzzy_system.determine_investment_horizon(input_value)

            print(f"\nInvestment Months: {input_value}")
            print(f"Investment Horizon Type: {horizon_type}")
            print(f"Membership: {membership_value:.2f}")

            short, balanced, long_term = fuzzy_system.plot_membership_functions(input_value)

            print(f"\nDetailed Membership:")
            print(f"Short-term: {short:.2f}")
            print(f"Balanced: {balanced:.2f}")
            print(f"Long-term: {long_term:.2f}")

        except ValueError:
            print("Invalid input, please enter a number")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
