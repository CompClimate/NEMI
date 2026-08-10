'''
Shared fixtures for the NEMI tests.
'''
import os

import pytest

TEST_EMBEDDINGS_PATH = 'test_embed.npz'


@pytest.fixture
def embeddings_path():
    """ Embeddings path for tests that would otherwise write the default file
    into the working directory.  Removed once the test completes. """
    yield TEST_EMBEDDINGS_PATH
    if os.path.exists(TEST_EMBEDDINGS_PATH):
        os.remove(TEST_EMBEDDINGS_PATH)
