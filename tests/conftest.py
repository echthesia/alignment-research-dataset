from unittest.mock import patch, Mock
import pytest
from align_data.common.alignment_dataset import make_session


@pytest.fixture(autouse=True, scope="session")
def mock_db():
    # This just mocks out all db calls, nothing more
    with patch("align_data.common.alignment_dataset.make_session"), \
         patch("align_data.db.session.make_session"), \
         patch("align_data.sources.greaterwrong.greaterwrong.make_session"):
        yield
