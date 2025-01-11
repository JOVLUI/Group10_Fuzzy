import numpy as np
import matplotlib.pyplot as plt


class EconomicIndicatorFuzzy:
    def __init__(self):
        # Economic indicator variable range
        self.economic_universe = np.linspace(0, 100, 200)

    def trapezoidal_membership(self, x, a, b, c, d):
        """
        Create trapezoidal membership function
        Parameters:
        x: Input value domain
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

        # Safely handle division by zero
        if d == c:
            d = c + 1e-10  # Add a small value to avoid division by zero

        # Right descending interval
        mask3 = (x >= c) & (x <= d)
        membership[mask3] = np.maximum(0, (d - x[mask3]) / (d - c))

        # Outside interval is 0
        membership[(x < a) | (x > d)] = 0.0

        return membership

    def determine_economic_condition(self, economic_value):
        """
        Determine economic condition based on input value
        """
        negative = self.trapezoidal_membership(economic_value, 0, 0, 20, 30)
        neutral = self.trapezoidal_membership(economic_value, 30, 40, 60, 70)
        positive = self.trapezoidal_membership(economic_value, 70, 80, 100, 100)

        if negative > neutral and negative > positive:
            return "Negative"
        elif neutral > negative and neutral > positive:
            return "Neutral"
        else:
            return "Positive"

    def generate_economic_indicator_plot(self, economic_value):
        # Create figure
        plt.figure(figsize=(12, 7))

        # Negative economic indicator (trapezoidal membership)
        negative = self.trapezoidal_membership(self.economic_universe, 0, 0, 20, 30)
        plt.plot(self.economic_universe, negative, label='Negative')

        # Neutral economic indicator (trapezoidal membership)
        neutral = self.trapezoidal_membership(self.economic_universe, 30, 40, 60, 70)
        plt.plot(self.economic_universe, neutral, label='Neutral')

        # Positive economic indicator (trapezoidal membership)
        positive = self.trapezoidal_membership(self.economic_universe, 70, 80, 100, 100)
        plt.plot(self.economic_universe, positive, label='Positive')

        # Mark user input value
        plt.axvline(x=economic_value, color='r', linestyle='--', label=f'Economic Value: {economic_value}')

        # Determine economic condition
        economic_condition = self.determine_economic_condition(economic_value)

        plt.title(f'Economic Indicator Fuzzy Sets\nCurrent Economic Condition: {economic_condition}')
        plt.xlabel('Economic Indicator Scale')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    economic_fuzzy = EconomicIndicatorFuzzy()

    while True:
        try:
            # User input economic indicator value
            economic_value = float(input("Enter economic indicator value (0-100): "))

            # Check input range
            if economic_value < 0 or economic_value > 100:
                print("Input value must be between 0-100")
                continue

            # Generate economic indicator fuzzy plot
            economic_fuzzy.generate_economic_indicator_plot(economic_value)

            # Ask to continue
            continue_choice = input("\nContinue assessment? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except ValueError:
            print("Invalid input, please enter a number")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Thank you for using the Economic Indicator Assessment System")


if __name__ == "__main__":
    main()
