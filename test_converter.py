import pytest
from converter import convert_temperature


# Celsius to others
def test_c_to_f():
    assert convert_temperature(0, 'C', 'F') == pytest.approx(32.0)
    assert convert_temperature(100, 'C', 'F') == pytest.approx(212.0)
    assert convert_temperature(-40, 'C', 'F') == pytest.approx(-40.0)

def test_c_to_k():
    assert convert_temperature(0, 'C', 'K') == pytest.approx(273.15)
    assert convert_temperature(100, 'C', 'K') == pytest.approx(373.15)
    assert convert_temperature(-273.15, 'C', 'K') == pytest.approx(0.0)

# Fahrenheit to others
def test_f_to_c():
    assert convert_temperature(32, 'F', 'C') == pytest.approx(0.0)
    assert convert_temperature(212, 'F', 'C') == pytest.approx(100.0)
    assert convert_temperature(-40, 'F', 'C') == pytest.approx(-40.0)

def test_f_to_k():
    assert convert_temperature(32, 'F', 'K') == pytest.approx(273.15)
    assert convert_temperature(212, 'F', 'K') == pytest.approx(373.15)

# Kelvin to others
def test_k_to_c():
    assert convert_temperature(273.15, 'K', 'C') == pytest.approx(0.0)
    assert convert_temperature(373.15, 'K', 'C') == pytest.approx(100.0)
    assert convert_temperature(0, 'K', 'C') == pytest.approx(-273.15)

def test_k_to_f():
    assert convert_temperature(273.15, 'K', 'F') == pytest.approx(32.0)
    assert convert_temperature(373.15, 'K', 'F') == pytest.approx(212.0)

# Same unit
def test_same_unit():
    assert convert_temperature(42, 'C', 'C') == pytest.approx(42.0)
    assert convert_temperature(42, 'F', 'F') == pytest.approx(42.0)
    assert convert_temperature(42, 'K', 'K') == pytest.approx(42.0)

# Invalid unit
def test_invalid_unit():
    with pytest.raises(ValueError):
        convert_temperature(100, 'X', 'C')
    with pytest.raises(ValueError):
        convert_temperature(100, 'C', 'X')
