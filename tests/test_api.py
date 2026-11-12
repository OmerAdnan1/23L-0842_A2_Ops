from conftest import upload


def test_api_health(api_client):
    assert api_client.get("/health").json["status"] == "ok"


def test_api_rejects_missing_key(api_client, png_bytes):
    assert upload(api_client, png_bytes, "a.png").status_code == 401


def test_api_predicts_with_key(api_client, png_bytes):
    r = upload(api_client, png_bytes, "a.png", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert r.json["label"] == "AI-generated"
    assert r.json["threshold"] == 0.8


def test_api_rejects_bad_image(api_client):
    r = upload(api_client, b"nope", "a.jpg", headers={"X-API-Key": "test-key"})
    assert r.status_code == 400
    assert "error" in r.json
