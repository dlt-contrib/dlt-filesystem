from typing import Any, Dict
from urllib.parse import quote


def shrink_qs_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Let's only use the first element when decoding URL query parameters."""
    return {key: value[0] for key, value in data.items()}


#: The characters RFC 3986 calls unreserved: an escape for one of these means the
#: same thing as the character itself, so unescaping it cannot change a URL.
UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)

#: Everything `requote_uri` leaves alone, `%` included, so an escape already in
#: the URL survives instead of being escaped a second time (`%2F` -> `%252F`).
_SAFE_WITH_PERCENT = "!#$%&'()*+,/:;=?@[]~"
_SAFE_WITHOUT_PERCENT = "!#$&'()*+,/:;=?@[]~"


def _unquote_unreserved(uri: str) -> str:
    """Unescape only the escapes that cannot carry meaning, leaving the rest alone.

    Anything that is not an unreserved character keeps its escape verbatim,
    including a sequence that is not an escape at all (`100%`) and one that decodes
    to a non-ASCII byte (`%C3` in a percent-encoded UTF-8 filename). Treating
    either as an error would send the whole URL down the escape-the-percent path
    and double-escape a URL that was already correct.

    Raises:
        ValueError: only for two alphanumerics that are not hex (`%zz`), the one
            case that cannot be interpreted either way. The caller answers it by
            escaping the stray `%`.
    """
    parts = uri.split("%")
    for index in range(1, len(parts)):
        escape = parts[index][0:2]
        if len(escape) == 2 and escape.isalnum():
            try:
                character = chr(int(escape, 16))
            except ValueError:
                raise ValueError(
                    f"Invalid percent-escape sequence: '{escape}'"
                ) from None
            if character in UNRESERVED:
                parts[index] = character + parts[index][2:]
            else:
                parts[index] = f"%{parts[index]}"
        else:
            parts[index] = f"%{parts[index]}"
    return "".join(parts)


def requote_uri(uri: str) -> str:
    """Escape what a URL cannot carry literally, and nothing that is already escaped.

    This is `requests.utils.requote_uri`, reproduced so an HTTP URL keeps meaning
    exactly what it meant before: a signature in a query string is computed over
    the *encoded* form, so re-encoding `%2F` as `/` (which is what `yarl` does when
    it normalizes a URL) invalidates it, while leaving a literal space unescaped
    produces a request line no server will accept.

    An escape of an unreserved character is decoded (`%7E` -> `~`), because the two
    forms are defined to be equivalent and every canonical signing scheme treats
    them as such.
    """
    try:
        return quote(_unquote_unreserved(uri), safe=_SAFE_WITH_PERCENT)
    except ValueError:
        # A stray `%` that is not an escape at all: escape it as one.
        return quote(uri, safe=_SAFE_WITHOUT_PERCENT)
