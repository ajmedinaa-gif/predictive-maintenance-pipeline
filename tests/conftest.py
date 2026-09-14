"""Fixtures compartidas entre módulos de test."""

import pytest

from predictive_maintenance import datasets


@pytest.fixture(scope="session")
def lab180_df():
    return datasets.load("lab180")


@pytest.fixture(scope="session")
def lab180_spec():
    return datasets.get_spec("lab180")
