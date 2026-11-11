from conftest import upload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["database"] == "ok"


def test_prediction_is_shown_and_logged(client, png_bytes):
    r = upload(client, png_bytes, "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    assert b"cat.png" in client.get("/history").data


def test_rejects_wrong_file_type(client):
    r = upload(client, b"just some text", "notes.txt")
    assert r.status_code == 400
    assert b"Unsupported file type" in r.data


def test_rejects_file_over_size_limit(client):
    r = upload(client, b"\0" * (2 * 1024 * 1024), "big.png")
    assert r.status_code == 413


def test_rejects_undecodable_image(client):
    r = upload(client, b"this is not really a jpeg", "fake.jpg")
    assert r.status_code == 400
    assert b"could not be decoded" in r.data


def test_filename_is_sanitized(client, png_bytes):
    upload(client, png_bytes, "../../etc/passwd.png")
    history = client.get("/history").data
    assert b"etc_passwd.png" in history
    assert b"../" not in history
