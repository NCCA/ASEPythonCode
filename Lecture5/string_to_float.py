#!/usr/bin/env -S uv run --script
def string_to_float(value: str) -> float:
    """
    Convert a string to a float, handling commas and periods appropriately.

    Args:
        value (str): The string to convert.

    Returns:
        float: The converted float value.

    Raises:
        ValueError: If the string cannot be converted to a float.
    """
    # Remove any commas from the string
    value = value.replace(",", "")

    # Convert the string to a float
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Cannot convert '{value}' to float.") from e


assert string_to_float("1.23") == 1.23
assert string_to_float("-1.23") == -1.23
assert string_to_float("1,234.56") == 1234.56
assert string_to_float("1,234") == 1234.0
assert string_to_float("0.001") == 0.001
assert string_to_float("1000") == 1000.0
assert string_to_float("1,000.00") == 1000.0
assert string_to_float("1,000.123") == 1000.123
assert string_to_float("1,234,567.89") == 1234567.89
assert string_to_float("1,234,567") == 1234567.0
assert string_to_float("0") == 0.0
try:
    assert string_to_float("text")
except ValueError:
    pass
