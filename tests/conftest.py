from io import BytesIO

import pytest
from PIL import Image

from app.config import Settings
from app.db import Store
from app.verdict import decide


class FakePredictor:
    name = "fake-model"
    threshold = 0.8

    def predict(self, raw, filename=None):
        return decide(0.9, self.threshold)


@pytest.fixture
def png_bytes():
    buf = BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", max_upload_mb=1,
                    api_key="test-key")


@pytest.fixture
def client(settings):
    from app.web import create_app
    app = create_app(settings, predictor=FakePredictor(), store=Store(settings.database_url))
    return app.test_client()


@pytest.fixture
def api_client(settings):
    from app.api import create_api
    return create_api(settings, predictor=FakePredictor()).test_client()


def upload(client, data, filename, **kwargs):
    return client.post("/predict", data={"image": (BytesIO(data), filename)},
                       content_type="multipart/form-data", **kwargs)
