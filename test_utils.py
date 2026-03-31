import pytest
from utils import summarize_numbers


def test_basic():
    result = summarize_numbers([1, 2, 3, 4, 5])
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["mean"] == 3.0
    assert result["median"] == 3.0


def test_even_count():
    result = summarize_numbers([1, 2, 3, 4])
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["mean"] == 2.5
    assert result["median"] == 2.5


def test_single_element():
    result = summarize_numbers([7])
    assert result["min"] == 7.0
    assert result["max"] == 7.0
    assert result["mean"] == 7.0
    assert result["median"] == 7.0


def test_negatives_and_floats():
    result = summarize_numbers([-3, -1, 0, 2.5, 4])
    assert result["min"] == -3.0
    assert result["max"] == 4.0
    assert pytest.approx(result["mean"]) == 0.5
    assert result["median"] == 0.0
