import pathlib
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from pai.models import ModelPricing, TieredCost, TimeWindowPricing
from pai.pricing import PricingService


@pytest.fixture
def pricing_service():
    """Provides a fresh PricingService instance for each test."""
    return PricingService()


@pytest.fixture
def custom_pricing_filepath(tmp_path: pathlib.Path) -> str:
    """Creates a temporary custom pricing YAML file and returns its path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yaml_file = data_dir / "custom_prices.yaml"
    # This content is a simplified version of the main test data file.
    yaml_file.write_text(
        """
pricing:
  openai:
    "gpt-4o":
      input_cost: 100.0
      output_cost: 200.0
  test-provider:
    "flat-model":
      input_cost: 1.0
      output_cost: 2.0
""",
        encoding="utf-8",
    )
    return str(yaml_file)


@pytest.mark.asyncio
async def test_load_and_merge_custom_pricing(
    pricing_service: PricingService, custom_pricing_filepath: str
):
    """Verify that a custom pricing file is loaded and correctly overrides defaults."""
    # Mock the base pricing data to avoid network calls.
    pricing_service._pricing_data = {
        "openai/gpt-4o": {
            "input_cost_per_token": 0.000005,  # $5/M
            "output_cost_per_token": 0.000015,  # $15/M
        }
    }
    await pricing_service.load_pricing_data(custom_file_path=custom_pricing_filepath)

    assert pricing_service._custom_pricing_data is not None

    # Test that the custom price for "gpt-4o" overrides the base price.
    # Note: get_model_pricing returns costs per million tokens.
    pricing = pricing_service.get_model_pricing("openai", "gpt-4o")
    assert pricing.input_cost_per_token == 100.0
    assert pricing.output_cost_per_token == 200.0

    # Test that a new model defined only in the custom file is available.
    new_model_pricing = pricing_service.get_model_pricing("test-provider", "flat-model")
    assert new_model_pricing.input_cost_per_token == 1.0
    assert new_model_pricing.output_cost_per_token == 2.0


def test_calculate_flat_cost(pricing_service: PricingService):
    """Test cost calculation with simple flat rates."""
    pricing = ModelPricing(input_cost_per_token=1.0, output_cost_per_token=2.0)
    input_cost, output_cost = pricing_service.calculate_cost(
        pricing, 1_000_000, 2_000_000
    )
    assert input_cost == pytest.approx(1.0)
    assert output_cost == pytest.approx(4.0)


def test_calculate_tiered_cost(pricing_service: PricingService):
    """Test cost calculation with a tiered pricing model."""
    pricing = ModelPricing(
        tiered_input_costs=[
            TieredCost(up_to=1000, cost=10.0),  # $10/M for the first 1k tokens
            TieredCost(up_to=-1, cost=5.0),  # $5/M for tokens after that
        ],
        output_cost_per_token=20.0,  # Flat $20/M for output
    )

    # Calculation for 1500 input tokens:
    # (1000 tokens * $10/M) + (500 tokens * $5/M) = $0.01 + $0.0025 = $0.0125
    expected_input_cost = (1000 / 1_000_000 * 10.0) + (500 / 1_000_000 * 5.0)
    # Calculation for 1000 output tokens:
    # (1000 tokens * $20/M) = $0.02
    expected_output_cost = 1000 / 1_000_000 * 20.0

    input_cost, output_cost = pricing_service.calculate_cost(pricing, 1500, 1000)

    assert input_cost == pytest.approx(expected_input_cost)
    assert output_cost == pytest.approx(expected_output_cost)


def test_calculate_time_based_cost(
    pricing_service: PricingService, monkeypatch: pytest.MonkeyPatch
):
    """Test that the correct time-based pricing window is selected."""
    pricing = ModelPricing(
        input_cost_per_token=1.0,  # Fallback "anytime" rate
        output_cost_per_token=2.0,  # Fallback "anytime" rate
        time_windows=[
            # Peak hours: 9 AM to 5 PM UTC
            TimeWindowPricing(start_hour=9, end_hour=17, input_cost=50.0, output_cost=60.0),
            # Off-peak (overnight): 10 PM to 6 AM UTC
            TimeWindowPricing(start_hour=22, end_hour=6, input_cost=5.0, output_cost=10.0),
        ],
    )
    mock_dt = MagicMock()
    monkeypatch.setattr("pai.pricing.datetime", mock_dt)

    # --- Test Peak Hours (e.g., 10:00 UTC) ---
    mock_dt.utcnow.return_value = datetime(2025, 1, 1, 10, 0, 0)
    input_cost, output_cost = pricing_service.calculate_cost(pricing, 1_000_000, 1_000_000)
    assert input_cost == pytest.approx(50.0)
    assert output_cost == pytest.approx(60.0)

    # --- Test Off-Peak Hours (e.g., 23:00 UTC) ---
    mock_dt.utcnow.return_value = datetime(2025, 1, 1, 23, 0, 0)
    input_cost, output_cost = pricing_service.calculate_cost(pricing, 1_000_000, 1_000_000)
    assert input_cost == pytest.approx(5.0)
    assert output_cost == pytest.approx(10.0)

    # --- Test Fallback Hours (e.g., 08:00 UTC) ---
    mock_dt.utcnow.return_value = datetime(2025, 1, 1, 8, 0, 0)
    input_cost, output_cost = pricing_service.calculate_cost(pricing, 1_000_000, 1_000_000)
    assert input_cost == pytest.approx(1.0)
    assert output_cost == pytest.approx(2.0)


def test_calculate_complex_time_and_tiered_cost(
    pricing_service: PricingService, monkeypatch: pytest.MonkeyPatch
):
    """Test a combination of time-based windows and tiered pricing within a window."""
    pricing = ModelPricing(
        input_cost_per_token=1.0,  # Fallback
        time_windows=[
            TimeWindowPricing(
                start_hour=8,
                end_hour=18,
                input_cost=7.0,  # Flat input cost during this window
                output_tiers=[  # Tiered output cost during this window
                    TieredCost(up_to=2048, cost=18.0),
                    TieredCost(up_to=-1, cost=30.0),
                ],
            )
        ],
    )
    mock_dt = MagicMock()
    monkeypatch.setattr("pai.pricing.datetime", mock_dt)

    # Mock time to be inside the window
    mock_dt.utcnow.return_value = datetime(2025, 1, 1, 12, 0, 0)

    # 1M input tokens at flat $7/M rate = $7
    # 3000 output tokens = (2048 @ $18/M) + (952 @ $30/M)
    expected_output_cost = (2048 / 1_000_000 * 18.0) + (952 / 1_000_000 * 30.0)

    input_cost, output_cost = pricing_service.calculate_cost(pricing, 1_000_000, 3000)

    assert input_cost == pytest.approx(7.0)
    assert output_cost == pytest.approx(expected_output_cost)
