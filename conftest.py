import pytest

# content of conftest.py

# Pytest will rewrite assertions in test modules, but not elsewhere.
# This tells pytest to also rewrite assertions in utils/helpers.py.
#
pytest.register_assert_rewrite("tests.helpers")
