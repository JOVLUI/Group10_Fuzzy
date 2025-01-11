import numpy as np
import matplotlib.pyplot as plt


class MarketConditionFuzzy:
    def __init__(self):
        # Market condition variable range
        self.market_universe = np.linspace(0, 100, 200)

    def trapezoidal_membership(self, x, a, b, c, d):
        """
        Create trapezoidal membership function
        Parameters:
        x: Input value
        a, b: Left trapezoid base
        c, d: Right trapezoid base
        """
        # Ensure input is numpy array
        x = np.asarray(x)

        # Create result array with same shape as x
        membership = np.zeros_like(x, dtype=float)

        # Safely handle division by zero
        if b == a:
            b = a + 1e-10  # Add a small value to avoid division by zero

        # Calculate membership
        # Left rising interval
        mask1 = (x >= a) & (x <= b)
        membership[mask1] = (x[mask1] - a) / (b - a)

        # Flat top interval
        mask2 = (x > b) & (x < c)
        membership[mask2] = 1.0

        # Safely handle division by zero for right side
        if d == c:
            d = c + 1e-10  # Add a small value to avoid division by zero

        # Right descending interval
        mask3 = (x >= c) & (x <= d)
        membership[mask3] = np.maximum(0, (d - x[mask3]) / (d - c))

        # Outside interval is 0
        membership[(x < a) | (x > d)] = 0.0

        return membership

    def determine_market_condition(self, market_value):
        """
        Determine market condition based on input value
        """
        bearish = self.trapezoidal_membership(market_value, 0, 0, 20, 30)
        neutral = self.trapezoidal_membership(market_value, 30, 40, 60, 70)
        bullish = self.trapezoidal_membership(market_value, 70, 80, 100, 100)

        memberships = {
            'Bearish': bearish,
            'Neutral': neutral,
            'Bullish': bullish
        }

        market_condition = max(memberships, key=memberships.get)
        return market_condition, memberships[market_condition]

    def generate_market_condition_plot(self, market_value):
        # Create figure
        plt.figure(figsize=(14, 8))

        # Bearish market condition (trapezoidal membership)
        bearish = self.trapezoidal_membership(self.market_universe, 0, 0, 20, 30)
        plt.plot(self.market_universe, bearish, label='Bearish')

        # Neutral market condition (trapezoidal membership)
        neutral = self.trapezoidal_membership(self.market_universe, 30, 40, 60, 70)
        plt.plot(self.market_universe, neutral, label='Neutral')

        # Bullish market condition (trapezoidal membership)
        bullish = self.trapezoidal_membership(self.market_universe, 70, 80, 100, 100)
        plt.plot(self.market_universe, bullish, label='Bullish')

        # Calculate membership values for the input
        bearish_membership = self.trapezoidal_membership(market_value, 0, 0, 20, 30)
        neutral_membership = self.trapezoidal_membership(market_value, 30, 40, 60, 70)
        bullish_membership = self.trapezoidal_membership(market_value, 70, 80, 100, 100)

        # Mark user input value
        plt.axvline(x=market_value, color='r', linestyle='--', label=f'Market Value: {market_value}')

        # Mark membership points
        plt.scatter([market_value], [bearish_membership], color='blue', zorder=5)
        plt.scatter([market_value], [neutral_membership], color='green', zorder=5)
        plt.scatter([market_value], [bullish_membership], color='red', zorder=5)

        # Add text annotations for membership values
        plt.text(market_value + 2, bearish_membership, f'Bearish: {bearish_membership:.2f}', color='blue')
        plt.text(market_value + 2, neutral_membership, f'Neutral: {neutral_membership:.2f}', color='green')
        plt.text(market_value + 2, bullish_membership, f'Bullish: {bullish_membership:.2f}', color='red')

        # Determine market condition
        market_condition, max_membership = self.determine_market_condition(market_value)

        plt.title(f'Market Condition Fuzzy Sets\nCurrent Market Condition: {market_condition}')
        plt.xlabel('Market Condition Scale')
        plt.ylabel('Membership Degree')
        plt.ylim(0, 1.1)  # Adjust y-axis to show annotations
        plt.legend()
        plt.grid(True)
        plt.show()

        return bearish_membership, neutral_membership, bullish_membership


def main():
    market_fuzzy = MarketConditionFuzzy()

    while True:
        try:
            # User input market value
            market_value = float(input("Enter market condition value (0-100): "))

            # Check input range
            if market_value < 0 or market_value > 100:
                print("Input value must be between 0-100")
                continue

            # Generate market condition fuzzy plot
            bearish, neutral, bullish = market_fuzzy.generate_market_condition_plot(market_value)

            # Print detailed membership values
            print(f"\nDetailed Membership Values:")
            print(f"Bearish: {bearish:.2f}")
            print(f"Neutral: {neutral:.2f}")
            print(f"Bullish: {bullish:.2f}")

            # Ask to continue
            continue_choice = input("\nContinue assessment? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except ValueError:
            print("Invalid input, please enter a number")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Thank you for using the Market Condition Assessment System")


if __name__ == "__main__":
    main()
