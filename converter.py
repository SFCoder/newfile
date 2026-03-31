def convert_temperature(value, from_unit, to_unit):
    """Convert temperature between Fahrenheit, Celsius, and Kelvin.

    Args:
        value: The temperature value to convert.
        from_unit: Source unit ('C', 'F', or 'K').
        to_unit: Target unit ('C', 'F', or 'K').

    Returns:
        The converted temperature as a float.
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    if from_unit == to_unit:
        return float(value)

    # Convert to Celsius first
    if from_unit == 'F':
        celsius = (value - 32) * 5 / 9
    elif from_unit == 'K':
        celsius = value - 273.15
    elif from_unit == 'C':
        celsius = value
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    # Convert from Celsius to target
    if to_unit == 'C':
        return celsius
    elif to_unit == 'F':
        return celsius * 9 / 5 + 32
    elif to_unit == 'K':
        return celsius + 273.15
    else:
        raise ValueError(f"Unknown unit: {to_unit}")
