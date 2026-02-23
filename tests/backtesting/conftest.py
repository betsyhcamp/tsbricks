import pytest
import pandas as pd
from tsbricks.backtesting.schema import DataConfig


@pytest.fixture
def monthly_panel() -> pd.DataFrame:
    """Two-series monthly panel from 2022-01 to 2023-12 (24 months each)."""
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    rows = []
    for uid in ["A", "B"]:
        for ds in dates:
            rows.append({"unique_id": uid, "ds": ds, "y": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(freq="MS")
