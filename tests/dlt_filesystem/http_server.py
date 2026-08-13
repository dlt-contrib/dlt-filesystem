"""Local HTTP servers for exercising the HTTP filesystem source.

Docker-free and credential-free: a document root is generated into a temporary
directory and served from a background thread on an ephemeral port, so the whole
HTTP matrix runs in the unit lane. This is a different pattern from `gcs.py` in
the same directory, which starts a testcontainer.

Three server behaviours are modelled, because the reader stack takes a different
code path through each:

- `ServerMode.RANGE` answers a `Range` request with `206 Partial Content` plus a
  `Content-Range` header, and reports `Content-Length` on `HEAD`. This is the CDN
  and object-store shape, and the only one a file can be read from as a stream.
- `ServerMode.NO_RANGE` ignores `Range` and answers every request with the whole
  body, which is what `http.server` itself does. fsspec raises on a nonzero-range
  request answered this way.
- `ServerMode.CHUNKED` answers with `Transfer-Encoding: chunked` and no
  `Content-Length`, so the client reports no size at all.

Every request is recorded with its **raw** query string and `Range` header, so a
test can assert on what reached the wire rather than only on the rows that came
back. That is what makes signature preservation and partial reads observable.
"""

import base64
import binascii
import gzip
import json
import mimetypes
import socket
import ssl
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import unquote

#: The rows every text-shaped document in the root carries.
PEOPLE: Tuple[dict, ...] = (
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 41},
    {"name": "Charlie", "age": 25},
)

#: Enough records that reading the first batch of one is visibly not reading all
#: of it. The parquet form is written in many row groups; the line-delimited form
#: is what a reader can genuinely stream, since pyarrow asks for a file's whole
#: data section in one read whatever the transport offers.
EVENT_COUNT = 20_000
EVENT_ROW_GROUP_SIZE = 1_000

#: Credentials the basic-auth server expects, with a password that has to be
#: percent-encoded to survive a URI (`@` would end the userinfo, `/` the netloc).
AUTH_USERNAME = "reader"
AUTH_PASSWORD = "p@ss/word"  # noqa: S105 - test fixture credential, not a secret
AUTH_PASSWORD_ENCODED = "p%40ss%2Fword"  # noqa: S105 - the same fixture credential, encoded

#: Body bytes per chunk in `ServerMode.CHUNKED`, small enough to produce several.
CHUNK_SIZE = 4096

#: Any request under this path is answered `403`, whatever the mode. Models the
#: failure a signed URL actually has: a signature that expired or does not match.
FORBIDDEN_PREFIX = "/forbidden/"


class ServerMode(str, Enum):
    """How a fixture server answers a body request."""

    RANGE = "range"
    NO_RANGE = "no_range"
    CHUNKED = "chunked"


@dataclass
class Request:
    """One request as the server saw it, before any client-side normalization."""

    method: str
    path: str
    query: str
    range_header: Optional[str]
    status: int = 0
    bytes_served: int = 0

    @property
    def is_ranged(self) -> bool:
        return self.range_header is not None


@dataclass
class HttpFixture:
    """A running server, its document root, and the requests it has answered.

    The request log is read and written from different threads, so every access
    to it goes through the lock: handler threads record, the test thread reads.
    `requests` hands back a copy for that reason, which also means a test cannot
    accidentally mutate the log it is asserting on.
    """

    base_url: str
    document_root: Path
    mode: ServerMode

    def __post_init__(self) -> None:
        self._requests: List[Request] = []
        self._lock = threading.Lock()

    @property
    def requests(self) -> List[Request]:
        """Return a snapshot of the requests answered so far, in order."""
        with self._lock:
            return list(self._requests)

    def record(self, request: Request) -> None:
        """Add a request to the log, called from a handler thread."""
        with self._lock:
            self._requests.append(request)

    def url(self, name: str, query: str = "", fragment: str = "") -> str:
        """Return the URL for a document, with an optional raw query and fragment."""
        url = f"{self.base_url}/{name}"
        if query:
            url = f"{url}?{query}"
        if fragment:
            url = f"{url}#{fragment}"
        return url

    def body(self, name: str) -> bytes:
        """Return the bytes the server would serve for a document."""
        return (self.document_root / name).read_bytes()

    def clear(self) -> None:
        """Forget every recorded request, so one test does not read another's."""
        with self._lock:
            self._requests.clear()

    def ranged(self) -> List[Request]:
        """Return only the requests that carried a `Range` header.

        Listing probes a concrete URL with an unranged `GET` (fsspec's `_exists`)
        before any reader opens it, so a test about how much the *reader* pulled
        has to leave that probe out.
        """
        return [request for request in self.requests if request.is_ranged]

    def bytes_served(self, ranged_only: bool = False) -> int:
        """Return how many body bytes the server has written."""
        pool = self.ranged() if ranged_only else self.requests
        return sum(request.bytes_served for request in pool)

    def queries(self) -> List[str]:
        """Return the raw query string of every request, in order."""
        return [request.query for request in self.requests]


class _Server(ThreadingHTTPServer):
    """A threading server carrying the fixture state its handler reads."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address,
        handler,
        *,
        fixture: HttpFixture,
        auth: Optional[Tuple[str, str]],
    ):
        super().__init__(address, handler)
        self.fixture = fixture
        self.auth = auth


def _content_type(name: str) -> str:
    """Guess a document's media type, never claiming a content encoding.

    A `.gz` document is served as opaque bytes on purpose: a `Content-Encoding:
    gzip` response would be decompressed by the client, and fsspec then discards
    `Content-Length`, so the file would appear to have no size. Real object
    stores serve `.gz` objects the same way.
    """
    media_type, _ = mimetypes.guess_type(name)
    if name.endswith(".gz") or media_type is None:
        return "application/octet-stream"
    return media_type


def _requested_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Return the inclusive byte offsets a `Range` header selects, or `None`.

    A header that cannot be parsed yields `None`, which the handler answers with
    the whole body: RFC 9110 has an origin server ignore a `Range` it does not
    understand rather than fail the request. A parsed but *unsatisfiable* range
    is returned as-is, because that one owes a `416`, and fsspec depends on it
    (it reads a `416` as "past the end of the file", not as an error).
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].split(",")[0].strip()
    start_text, separator, end_text = spec.partition("-")
    if not separator:
        return None
    try:
        if not start_text:
            # Suffix range, the last N bytes. Measured: fsspec and pyarrow both
            # request explicit offsets, so nothing under test takes this branch.
            # It is here because ordinary clients do send it (curl `--range -N`,
            # media players), and a fixture that broke on it would be lying.
            return max(size - int(end_text), 0), size - 1
        start = int(start_text)
        end = min(int(end_text) if end_text else size - 1, size - 1)
    except ValueError:
        return None
    return start, end


class _Handler(BaseHTTPRequestHandler):
    """Serve the document root, recording what each request asked for."""

    protocol_version = "HTTP/1.1"
    server_version = "omniload-test"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the per-request line `http.server` writes to stderr."""

    def do_HEAD(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        self._serve("HEAD")

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        self._serve("GET")

    # -- internals ---------------------------------------------------------

    def _serve(self, method: str) -> None:
        server: Any = self.server
        fixture: HttpFixture = server.fixture

        raw_path, separator, raw_query = self.path.partition("?")
        record = Request(
            method=method,
            path=raw_path,
            query=raw_query if separator else "",
            range_header=self.headers.get("Range"),
        )
        fixture.record(record)

        if not self._authorized(server.auth):
            self._respond(
                record,
                401,
                b"",
                method=method,
                extra=(("WWW-Authenticate", 'Basic realm="omniload"'),),
            )
            return

        if raw_path.startswith(FORBIDDEN_PREFIX):
            self._respond(record, 403, b"<Error>AccessDenied</Error>", method=method)
            return

        target = self._target(fixture.document_root, raw_path)
        if target is None:
            self._respond(record, 404, b"file not found", method=method)
            return

        body = target.read_bytes()
        content_type = _content_type(target.name)

        if fixture.mode is ServerMode.CHUNKED:
            self._respond_chunked(record, body, content_type, method=method)
            return

        selection = (
            _requested_range(record.range_header, len(body))
            if fixture.mode is ServerMode.RANGE
            else None
        )
        if selection is not None:
            start, end = selection
            # `end` is already clamped to the last byte, so this covers both an
            # offset past the end of the file and a reversed range.
            if start > end:
                self._respond(
                    record,
                    416,
                    b"",
                    method=method,
                    extra=(("Content-Range", f"bytes */{len(body)}"),),
                )
                return
            self._respond(
                record,
                206,
                body[start : end + 1],
                method=method,
                extra=(
                    ("Content-Type", content_type),
                    ("Accept-Ranges", "bytes"),
                    ("Content-Range", f"bytes {start}-{end}/{len(body)}"),
                ),
            )
            return

        extra = [("Content-Type", content_type)]
        if fixture.mode is ServerMode.RANGE:
            extra.append(("Accept-Ranges", "bytes"))
        self._respond(record, 200, body, method=method, extra=tuple(extra))

    def _authorized(self, auth: Optional[Tuple[str, str]]) -> bool:
        if auth is None:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, payload = header.partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            decoded = base64.b64decode(payload).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return False
        username, _, password = decoded.partition(":")
        return (username, password) == auth

    def _target(self, document_root: Path, raw_path: str) -> Optional[Path]:
        """Resolve a request path inside the document root, or `None`."""
        relative = unquote(raw_path).lstrip("/")
        if not relative:
            return None
        candidate = (document_root / relative).resolve()
        if not candidate.is_file():
            return None
        if document_root.resolve() not in candidate.parents:
            return None
        return candidate

    def _respond(
        self,
        record: Request,
        status: int,
        body: bytes,
        *,
        method: str,
        extra: Sequence[Tuple[str, str]] = (),
    ) -> None:
        self.send_response(status)
        for key, value in extra:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        record.status = status
        if method == "GET" and body:
            self.wfile.write(body)
            record.bytes_served += len(body)

    def _respond_chunked(
        self, record: Request, body: bytes, content_type: str, *, method: str
    ) -> None:
        """Answer without a `Content-Length`, so the client learns no size.

        `HEAD` advertises the same chunked framing as `GET`, which is what nginx
        and object-store gateways do, and is measured to be read correctly (a
        client never looks for a body on a HEAD response).
        """
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        record.status = 200
        if method != "GET":
            return
        for start in range(0, len(body), CHUNK_SIZE):
            piece = body[start : start + CHUNK_SIZE]
            self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
            record.bytes_served += len(piece)
        self.wfile.write(b"0\r\n\r\n")


def self_signed_certificate(directory: Path) -> Tuple[Path, Path]:
    """Write a certificate and key for `localhost`, returning both paths."""
    # Fixed timestamps: a generated certificate must not make the fixture depend
    # on the clock, and validity is only ever checked against "now".
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    not_before = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    not_after = datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "localhost")],
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    certificate_file = directory / "server.pem"
    key_file = directory / "server.key"
    certificate_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return certificate_file, key_file


@contextmanager
def serve(
    document_root: Path,
    mode: ServerMode = ServerMode.RANGE,
    auth: Optional[Tuple[str, str]] = None,
    certificate: Optional[Tuple[Path, Path]] = None,
) -> Iterator[HttpFixture]:
    """Run one fixture server for the duration of the context.

    The socket is bound on an ephemeral port and the server is shut down and
    closed on exit, so nothing is left listening and no port is assumed free.
    """
    scheme = "https" if certificate else "http"
    fixture = HttpFixture(base_url="", document_root=document_root, mode=mode)
    server = _Server(("127.0.0.1", 0), _Handler, fixture=fixture, auth=auth)
    if certificate:
        certificate_file, key_file = certificate
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate_file, key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)

    host, port = server.socket.getsockname()[:2]
    # `localhost` rather than the numeric address, so the certificate's DNS name
    # is the one being verified in the TLS case.
    hostname = "localhost" if certificate else host
    fixture.base_url = f"{scheme}://{hostname}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def closed_port() -> int:
    """Return a port nothing is listening on, for the connection-failure path."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_document_root(directory: Path) -> Path:
    """Generate every document the HTTP tests read, returning the root.

    Nothing here is committed: the root is written per session so a format's
    fixture cannot drift from the rows the assertions expect.
    """
    root = directory / "documents"
    root.mkdir(parents=True, exist_ok=True)

    header = ",".join(PEOPLE[0])
    rows = [",".join(str(person[column]) for column in PEOPLE[0]) for person in PEOPLE]
    csv_text = "\n".join([header, *rows]) + "\n"
    headless_text = "\n".join(rows) + "\n"

    (root / "people.csv").write_text(csv_text)
    (root / "people-no-header.csv").write_text(headless_text)
    (root / "people.csv.gz").write_bytes(gzip.compress(csv_text.encode()))
    # Semicolon-separated, readable only with a `#separator=;` reader hint.
    (root / "people-semicolon.csv").write_text(csv_text.replace(",", ";"))
    # No extension a format can be inferred from, so the format has to be named.
    (root / "people.dat").write_text(csv_text)
    # A name that can only be addressed percent-encoded (`caf%C3%A9.csv`).
    (root / "café.csv").write_text(csv_text)

    array_text = json.dumps(list(PEOPLE))
    (root / "people.json").write_text(array_text)
    (root / "people-pretty.json").write_text(json.dumps(list(PEOPLE), indent=2))
    (root / "people-object.json").write_text(json.dumps(PEOPLE[0]))
    (root / "people.json.gz").write_bytes(gzip.compress(array_text.encode()))
    (root / "people.jsonl").write_text(
        "".join(f"{json.dumps(person)}\n" for person in PEOPLE)
    )

    nested = root / "feeds"
    nested.mkdir(exist_ok=True)
    (nested / "events.csv").write_text(csv_text)

    _write_parquet(root / "people.parquet", PEOPLE)
    _write_events_parquet(root / "events.parquet")
    (root / "events.jsonl").write_text(
        "".join(
            json.dumps({"id": index, "payload": f"event-{index:07d}"}) + "\n"
            for index in range(EVENT_COUNT)
        )
    )

    return root


def _write_parquet(path: Path, records: Sequence[dict]) -> None:
    import pyarrow as pa
    from pyarrow import parquet as pq

    pq.write_table(pa.Table.from_pylist(list(records)), path)


def _write_events_parquet(path: Path) -> None:
    """Write a multi-row-group parquet document, uncompressed.

    Many row groups are what let a reader take one batch without touching the
    whole file, and no compression keeps the body big enough that a partial read
    is unambiguous rather than a rounding difference.
    """
    import pyarrow as pa
    from pyarrow import parquet as pq

    table = pa.table(
        {
            "id": pa.array(range(EVENT_COUNT), pa.int64()),
            "payload": pa.array([f"event-{index:07d}" for index in range(EVENT_COUNT)]),
        }
    )
    pq.write_table(table, path, row_group_size=EVENT_ROW_GROUP_SIZE, compression="none")
