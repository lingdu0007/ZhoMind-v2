MAX_ACTIVE_MEMBERS = 25
MAX_CONCURRENT_CHATS = 2
MAX_PUBLISHED_SOURCES = 500
MAX_DOCUMENT_BUILD_WORKERS = 1
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def first_release_limits() -> dict[str, int]:
    return {
        "active_members": MAX_ACTIVE_MEMBERS,
        "concurrent_chats": MAX_CONCURRENT_CHATS,
        "published_sources": MAX_PUBLISHED_SOURCES,
        "document_build_workers": MAX_DOCUMENT_BUILD_WORKERS,
        "upload_bytes": MAX_UPLOAD_BYTES,
    }
