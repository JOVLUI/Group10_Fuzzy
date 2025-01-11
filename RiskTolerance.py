class RiskToleranceCalculator:
    def __init__(self):
        # Define fuzzy rules
        self.rules = [
            # Young, high income, experienced -> high risk
            {
                'age_range': (20, 35),
                'income_range': (6000, 15000),
                'experience_range': (5, 10),
                'risk_level': 'high',
                'risk_score': (60, 100)
            },
            # Middle-aged, medium income, intermediate experience -> medium risk
            {
                'age_range': (30, 50),
                'income_range': (2000, 8000),
                'experience_range': (3, 6),
                'risk_level': 'medium',
                'risk_score': (40, 70)
            },
            # Elderly, low income, no experience -> low risk
            {
                'age_range': (45, 60),
                'income_range': (0, 3000),
                'experience_range': (0, 2),
                'risk_level': 'low',
                'risk_score': (0, 40)
            },
            # Young, medium income, intermediate experience -> medium risk
            {
                'age_range': (20, 35),
                'income_range': (2000, 8000),
                'experience_range': (3, 6),
                'risk_level': 'medium',
                'risk_score': (40, 70)
            },
            # Middle-aged, high income, experienced -> high risk
            {
                'age_range': (30, 50),
                'income_range': (6000, 15000),
                'experience_range': (5, 10),
                'risk_level': 'high',
                'risk_score': (60, 100)
            },
            # Elderly, medium income, intermediate experience -> medium risk
            {
                'age_range': (45, 60),
                'income_range': (2000, 8000),
                'experience_range': (3, 6),
                'risk_level': 'medium',
                'risk_score': (40, 70)
            }
        ]

    def _is_in_range(self, value, range_tuple):
        """Check if value is within the specified range"""
        return range_tuple[0] <= value <= range_tuple[1]

    def calculate_risk_tolerance(self, age, income, experience):
        """Calculate risk tolerance"""
        # Find the best matching rule
        best_match = None
        min_distance = float('inf')

        for rule in self.rules:
            # Calculate match for each dimension
            age_match = self._is_in_range(age, rule['age_range'])
            income_match = self._is_in_range(income, rule['income_range'])
            experience_match = self._is_in_range(experience, rule['experience_range'])

            # If completely matched
            if age_match and income_match and experience_match:
                return sum(rule['risk_score']) / 2

            # Calculate distance
            age_distance = self._calculate_distance(age, rule['age_range'])
            income_distance = self._calculate_distance(income, rule['income_range'])
            exp_distance = self._calculate_distance(experience, rule['experience_range'])

            total_distance = age_distance + income_distance + exp_distance

            # Update best match
            if total_distance < min_distance:
                min_distance = total_distance
                best_match = rule

        # If best match found, return middle of risk score
        if best_match:
            return sum(best_match['risk_score']) / 2

        # Default to medium risk
        return 50

    def _calculate_distance(self, value, range_tuple):
        """Calculate distance to range"""
        min_val, max_val = range_tuple
        if value < min_val:
            return min_val - value
        elif value > max_val:
            return value - max_val
        return 0
