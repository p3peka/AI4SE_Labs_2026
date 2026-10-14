"""Name-multiset F1, calculated offline from executed SQL results."""

from collections import Counter


def name_multiset_f1(predicted_rows, expected_rows, column_count=None):
    """Partial name overlap; preserve duplicates and ignore house numbers.

    Failure or wrong projection scores zero. Two empty results score one.
    Names are compared exactly in the second column of HOUSE_NUMBER/NAME.
    """
    if predicted_rows is None or (column_count is not None and column_count != 2):
        return 0.0
    if any(len(row) != 2 for row in predicted_rows):
        return 0.0
    if any(len(row) != 2 for row in expected_rows):
        raise ValueError("Expected HOUSE_NUMBER/NAME reference rows")
    predicted = Counter(row[1] for row in predicted_rows)
    expected = Counter(row[1] for row in expected_rows)
    total = sum(predicted.values()) + sum(expected.values())
    return 1.0 if total == 0 else 2.0 * sum((predicted & expected).values()) / total
