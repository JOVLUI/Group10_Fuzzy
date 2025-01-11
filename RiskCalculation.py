import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from RiskTolerance import RiskToleranceCalculator


class FuzzyRiskLogicVisualization:
    def __init__(self):
        self.calculator = RiskToleranceCalculator()

    def trapezoidal_membership(self, x, a, b, c, d):
        """
        Create trapezoidal membership function
        Parameters:
        x: Input value domain
        a, b: Left trapezoid base
        c, d: Right trapezoid base
        """
        return np.maximum(
            np.minimum(
                np.minimum((x - a) / (b - a), 1.0),
                (d - x) / (d - c)
            ),
            0.0
        )

    def generate_fuzzy_plot(self, age, income, experience):
        # Calculate risk score
        risk_score = self.calculator.calculate_risk_tolerance(age, income, experience)

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Age trapezoidal membership
        age_universe = np.linspace(20, 60, 200)
        age_low = self.trapezoidal_membership(age_universe, 20, 30, 35, 40)
        age_medium = self.trapezoidal_membership(age_universe, 35, 40, 45, 50)
        age_high = self.trapezoidal_membership(age_universe, 45, 50, 55, 60)

        # Calculate age membership values
        age_low_value = self.trapezoidal_membership(age, 20, 30, 35, 40)
        age_medium_value = self.trapezoidal_membership(age, 35, 40, 45, 50)
        age_high_value = self.trapezoidal_membership(age, 45, 50, 55, 60)

        ax1.plot(age_universe, age_low, label='Young')
        ax1.plot(age_universe, age_medium, label='Middle')
        ax1.plot(age_universe, age_high, label='Old')
        ax1.axvline(x=age, color='r', linestyle='--', label='User Age')

        # Add membership value annotations
        ax1.scatter([age], [age_low_value], color='blue')
        ax1.scatter([age], [age_medium_value], color='green')
        ax1.scatter([age], [age_high_value], color='red')
        ax1.text(age + 1, age_low_value, f'Young: {age_low_value:.2f}', color='blue')
        ax1.text(age + 1, age_medium_value, f'Middle: {age_medium_value:.2f}', color='green')
        ax1.text(age + 1, age_high_value, f'Old: {age_high_value:.2f}', color='red')

        ax1.set_title('Age Membership Functions')
        ax1.set_xlabel('Age')
        ax1.legend()

        # Income trapezoidal membership
        income_universe = np.linspace(0, 15000, 200)
        income_low = self.trapezoidal_membership(income_universe, 0, 2000, 3000, 5000)
        income_medium = self.trapezoidal_membership(income_universe, 3000, 5000, 8000, 10000)
        income_high = self.trapezoidal_membership(income_universe, 8000, 10000, 12000, 15000)

        # Calculate income membership values
        income_low_value = self.trapezoidal_membership(income, 0, 2000, 3000, 5000)
        income_medium_value = self.trapezoidal_membership(income, 3000, 5000, 8000, 10000)
        income_high_value = self.trapezoidal_membership(income, 8000, 10000, 12000, 15000)

        ax2.plot(income_universe, income_low, label='Low')
        ax2.plot(income_universe, income_medium, label='Medium')
        ax2.plot(income_universe, income_high, label='High')
        ax2.axvline(x=income, color='r', linestyle='--', label='User Income')

        # Add membership value annotations
        ax2.scatter([income], [income_low_value], color='blue')
        ax2.scatter([income], [income_medium_value], color='green')
        ax2.scatter([income], [income_high_value], color='red')
        ax2.text(income + 100, income_low_value, f'Low: {income_low_value:.2f}', color='blue')
        ax2.text(income + 100, income_medium_value, f'Medium: {income_medium_value:.2f}', color='green')
        ax2.text(income + 100, income_high_value, f'High: {income_high_value:.2f}', color='red')

        ax2.set_title('Income Membership Functions')
        ax2.set_xlabel('Income')
        ax2.legend()

        # Experience trapezoidal membership
        experience_universe = np.linspace(0, 10, 200)
        experience_low = self.trapezoidal_membership(experience_universe, 0, 1, 2, 4)
        experience_medium = self.trapezoidal_membership(experience_universe, 2, 4, 6, 8)
        experience_high = self.trapezoidal_membership(experience_universe, 6, 8, 9, 10)

        # Calculate experience membership values
        experience_low_value = self.trapezoidal_membership(experience, 0, 1, 2, 4)
        experience_medium_value = self.trapezoidal_membership(experience, 2, 4, 6, 8)
        experience_high_value = self.trapezoidal_membership(experience, 6, 8, 9, 10)

        ax3.plot(experience_universe, experience_low, label='Low')
        ax3.plot(experience_universe, experience_medium, label='Medium')
        ax3.plot(experience_universe, experience_high, label='High')
        ax3.axvline(x=experience, color='r', linestyle='--', label='User Experience')

        # Add membership value annotations
        ax3.scatter([experience], [experience_low_value], color='blue')
        ax3.scatter([experience], [experience_medium_value], color='green')
        ax3.scatter([experience], [experience_high_value], color='red')
        ax3.text(experience + 0.2, experience_low_value, f'Low: {experience_low_value:.2f}', color='blue')
        ax3.text(experience + 0.2, experience_medium_value, f'Medium: {experience_medium_value:.2f}', color='green')
        ax3.text(experience + 0.2, experience_high_value, f'High: {experience_high_value:.2f}', color='red')

        ax3.set_title('Experience Membership Functions')
        ax3.set_xlabel('Experience')
        ax3.legend()

        # Risk trapezoidal membership
        risk_universe = np.linspace(0, 100, 200)
        risk_low = self.trapezoidal_membership(risk_universe, 0, 20, 30, 40)
        risk_medium = self.trapezoidal_membership(risk_universe, 30, 40, 60, 70)
        risk_high = self.trapezoidal_membership(risk_universe, 60, 70, 80, 100)

        # Calculate risk membership values
        risk_low_value = self.trapezoidal_membership(risk_score, 0, 20, 30, 40)
        risk_medium_value = self.trapezoidal_membership(risk_score, 30, 40, 60, 70)
        risk_high_value = self.trapezoidal_membership(risk_score, 60, 70, 80, 100)

        ax4.plot(risk_universe, risk_low, label='Low')
        ax4.plot(risk_universe, risk_medium, label='Medium')
        ax4.plot(risk_universe, risk_high, label='High')
        ax4.axvline(x=risk_score, color='r', linestyle='--', label='Risk Score')

        # Add membership value annotations
        ax4.scatter([risk_score], [risk_low_value], color='blue')
        ax4.scatter([risk_score], [risk_medium_value], color='green')
        ax4.scatter([risk_score], [risk_high_value], color='red')
        ax4.text(risk_score + 2, risk_low_value, f'Low: {risk_low_value:.2f}', color='blue')
        ax4.text(risk_score + 2, risk_medium_value, f'Medium: {risk_medium_value:.2f}', color='green')
        ax4.text(risk_score + 2, risk_high_value, f'High: {risk_high_value:.2f}', color='red')

        ax4.set_title('Risk Membership Functions')
        ax4.set_xlabel('Risk Score')
        ax4.legend()

        plt.suptitle(f'Fuzzy Logic Risk Assessment\nRisk Score: {risk_score:.2f}')
        plt.tight_layout()
        plt.show()


def main():
    visualizer = FuzzyRiskLogicVisualization()

    while True:
        try:
            # User input
            print("\nRisk Tolerance Assessment")
            age = float(input("Enter age: "))
            income = float(input("Enter monthly income: "))
            experience = float(input("Enter investment experience (years): "))

            # Generate fuzzy logic plot
            visualizer.generate_fuzzy_plot(age, income, experience)

            # Ask to continue
            continue_choice = input("\nContinue assessment? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except ValueError:
            print("Invalid input, please enter a number")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Thank you for using the Risk Assessment System")


if __name__ == "__main__":
    main()
