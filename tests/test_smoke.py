"""Smoke test: the src-layout package installs and imports cleanly."""


def test_package_imports():
    import data_scout

    assert data_scout.__version__
