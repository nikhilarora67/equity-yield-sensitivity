from src.data_loader import load_universe


def test_supplied_universe_is_valid_and_unique():
    sectors = load_universe()
    assert len(sectors) == 218
    assert sectors.index.is_unique
    assert sectors.notna().all()
    assert sectors.nunique() == 11
