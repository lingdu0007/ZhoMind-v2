import re
from ipaddress import ip_address
from socket import inet_aton
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from app.documents.content_admission import recognizable_private_material

_PUBLIC_QUERY_PARAMETERS = {
    "version": r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}",
    "v": r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}",
    "revision": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "rev": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "page": r"[1-9][0-9]{0,5}",
    "lang": r"[a-z]{2,3}(?:-[A-Za-z]{2,4})?",
}
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home")


def _public_query_parameter(key: str, value: str) -> bool:
    pattern = _PUBLIC_QUERY_PARAMETERS.get(key)
    address_value = value.lower().rstrip(".")
    if (
        pattern is None or not re.fullmatch(pattern, value) or recognizable_private_material(value)
        or address_value == "localhost" or address_value.endswith(_LOCAL_SUFFIXES)
    ):
        return False
    try:
        address = ip_address(address_value)
    except ValueError:
        if "." not in value and not value.lower().startswith("0x") and not (value.isdigit() and len(value) >= 7):
            return True
        try:
            address = ip_address(inet_aton(address_value))
        except (OSError, ValueError):
            return True
    return address.is_global and not address.is_multicast


def sanitize_public_source_url(value: str) -> str:
    """Prepare a public locator; authority accepts only the already-clean form."""
    if not isinstance(value, str):
        raise ValueError("public source URL is invalid")
    decoded = value
    for _ in range(3):
        decoded = unquote(decoded)
        if re.search(r"[\x00-\x20\x7f\\]", decoded):
            raise ValueError("public source URL is invalid")
    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower().rstrip(".").encode("idna").decode("ascii")
        port = parsed.port
    except (ValueError, UnicodeError) as exc:
        raise ValueError("public source URL is invalid") from exc
    if (
        parsed.scheme != "https" or not hostname or "%" in hostname or port not in (None, 443)
        or hostname == "localhost" or hostname.endswith(_LOCAL_SUFFIXES)
        or recognizable_private_material(parsed.path)
    ):
        raise ValueError("public source URL is invalid")
    try:
        address = ip_address(hostname)
    except ValueError:
        labels = hostname.split(".")
        if (
            len(labels) < 2 or len(hostname) > 253
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
            or re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", labels[-1])
        ):
            raise ValueError("public source URL is invalid") from None
    else:
        if not address.is_global or address.is_multicast:
            raise ValueError("public source URL is invalid")
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True) if _public_query_parameter(key, val)]
    host = f"[{hostname}]" if ":" in hostname else hostname
    return urlunsplit(("https", host, parsed.path or "/", urlencode(query), ""))


def is_canonical_public_source_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return value == sanitize_public_source_url(value)
    except ValueError:
        return False
