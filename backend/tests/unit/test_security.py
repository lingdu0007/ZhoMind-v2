from app.common.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_and_verify() -> None:
    hashed = hash_password("secret-123")
    assert hashed != "secret-123"
    assert verify_password("secret-123", hashed)


def test_jwt_roundtrip() -> None:
    token = create_access_token(subject="alice", role="admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"
