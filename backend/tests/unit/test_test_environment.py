from app.common.config import get_settings
from app.rag.dense_contract import dense_mode_active


def test_default_test_environment_disables_dense_mode() -> None:
    get_settings.cache_clear()
    try:
        assert dense_mode_active(get_settings()) is False
    finally:
        get_settings.cache_clear()
