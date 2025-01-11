import numpy as np
import matplotlib.pyplot as plt


def trapezoidal_membership(x, a, b, c, d):
    """
    Define a trapezoidal membership function.
    a, b: left foot and left shoulder of the trapezoid
    c, d: right shoulder and right foot of the trapezoid
    """
    # Prevent division by zero
    left_slope = (b - a) if (b - a) != 0 else 1e-6
    right_slope = (d - c) if (d - c) != 0 else 1e-6

    # Correct the np.minimum usage
    left_condition = (x - a) / left_slope
    right_condition = (d - x) / right_slope

    return np.maximum(np.minimum(left_condition, np.minimum(1, right_condition)), 0)


def determine_diversification_level(input_value):
    """
    Determine the portfolio diversification level based on membership values
    """
    poor_value = trapezoidal_membership(input_value, 0, 0, 20, 40)
    moderate_value = trapezoidal_membership(input_value, 40, 50, 60, 70)
    good_value = trapezoidal_membership(input_value, 70, 80, 100, 100)

    # Find the highest membership
    max_membership = max(poor_value, moderate_value, good_value)

    if max_membership == poor_value:
        return "Poor", poor_value
    elif max_membership == moderate_value:
        return "Moderate", moderate_value
    else:
        return "Good", good_value


def plot_fuzzy_logic(input_value):
    """Plot the fuzzy logic membership functions and highlight the input value."""
    # Define the x-axis range
    x = np.linspace(0, 100, 500)

    # Define the trapezoidal membership functions
    poor = trapezoidal_membership(x, 0, 0, 20, 40)
    moderate = trapezoidal_membership(x, 40, 50, 60, 70)
    good = trapezoidal_membership(x, 70, 80, 100, 100)

    # Calculate the membership values for the input value
    poor_value = trapezoidal_membership(input_value, 0, 0, 20, 40)
    moderate_value = trapezoidal_membership(input_value, 40, 50, 60, 70)
    good_value = trapezoidal_membership(input_value, 70, 80, 100, 100)

    # Plot the membership functions
    plt.figure(figsize=(10, 6))
    plt.plot(x, poor, label="Poor", color="red")
    plt.plot(x, moderate, label="Moderate", color="orange")
    plt.plot(x, good, label="Good", color="green")

    # Highlight the input value on the plot
    plt.scatter([input_value], [poor_value], color="red", zorder=5)
    plt.scatter([input_value], [moderate_value], color="orange", zorder=5)
    plt.scatter([input_value], [good_value], color="green", zorder=5)

    # Annotate the input value
    plt.text(input_value, poor_value + 0.05, f"{poor_value:.2f}", color="red", ha="center")
    plt.text(input_value, moderate_value + 0.05, f"{moderate_value:.2f}", color="orange", ha="center")
    plt.text(input_value, good_value + 0.05, f"{good_value:.2f}", color="green", ha="center")

    # Add labels, legend, and title
    plt.title("Fuzzy Logic Membership Functions for Portfolio Diversification")
    plt.xlabel("Portfolio Diversification Score")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    while True:
        try:
            # User input portfolio diversification score
            input_value = float(input("Enter portfolio diversification score (0-100): "))

            # Check input range
            if input_value < 0 or input_value > 100:
                print("Input value must be between 0-100")
                continue

            # Determine diversification level
            level, membership = determine_diversification_level(input_value)
            print(f"Portfolio Diversification Level: {level}")
            print(f"Membership: {membership:.2f}")

            # Generate fuzzy logic plot
            plot_fuzzy_logic(input_value)

            # Ask to continue
            continue_choice = input("\nContinue assessment? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except ValueError:
            print("Invalid input, please enter a number")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Thank you for using the Portfolio Diversification Assessment System")


if __name__ == "__main__":
    main()
