def write_bson(path, docs):
    """Write BSON documents concatenated into a single file (on-disk mongodump form)."""
    import bson

    with open(path, "wb") as f:
        for doc in docs:
            f.write(bson.encode(doc))
    return path


def write_cbor(path, value):
    """Write a single top-level CBOR value (an array of records, or one record)."""
    import cbor2

    with open(path, "wb") as f:
        f.write(cbor2.dumps(value))
    return path


def write_feather(path, records, batch_size=None):
    """Write record dictionaries to an Arrow IPC file (Feather V2).

    ``batch_size`` splits the rows into that many-row record batches, which is the only
    way to produce a file with more than one batch: a single ``write_table`` of a
    contiguous table emits exactly one, so a chunking test written against the default
    would pass whatever the reader did with batch boundaries.
    """
    import pyarrow as pa

    table = pa.Table.from_pylist(records)
    with pa.ipc.new_file(path, table.schema) as writer:
        if batch_size is None:
            writer.write_table(table)
        else:
            for batch in table.to_batches(max_chunksize=batch_size):
                writer.write_batch(batch)
    return path


def write_msgpack(path, rows, **packb_kwargs):
    """Write records as a stream of concatenated MessagePack maps (the on-disk form)."""
    import msgpack

    with open(path, "wb") as f:
        for row in rows:
            f.write(msgpack.packb(row, use_bin_type=True, **packb_kwargs))
    return path


def write_orc(path, records):
    """Write record dictionaries to an ORC file."""
    import pyarrow as pa
    from pyarrow import orc

    orc.write_table(pa.Table.from_pylist(records), path)
    return path


def write_xml(path, text):
    """Write raw XML ``text`` to ``path`` as UTF-8 bytes."""
    with open(path, "wb") as f:
        f.write(text.encode("utf-8") if isinstance(text, str) else text)
    return path


def write_yaml(path, text):
    """Write raw YAML ``text`` to ``path``."""
    with open(path, "w") as f:
        f.write(text)
    return path


# --- Apache Avro --------------------------------------------------------------------
#
# Hand-built rather than written with `polars.DataFrame.write_avro`, which is what the
# reader under test uses. A fixture written by that library can only contain what that
# library can *write*, so a suite built from round trips is structurally blind to the
# reader's worst input: a `map` field is an ordinary Avro type that `polars.read_avro`
# panics on and `write_avro` cannot produce. These primitives are also how the schemas
# no writer emits (a `null` field, a multi-branch union) reach a test at all.

_AVRO_SYNC = b"0123456789abcdef"


def encode_avro_long(value):
    """Encode an Avro ``long``: zig-zag, then seven bits a byte, low group first."""
    value = (value << 1) ^ (value >> 63)
    encoded = bytearray()
    while True:
        group = value & 0x7F
        value >>= 7
        if not value:
            encoded.append(group)
            return bytes(encoded)
        encoded.append(group | 0x80)


def encode_avro_bytes(value):
    return encode_avro_long(len(value)) + value


def write_avro_container(path, schema, body=b"", record_count=0, codec=b"null"):
    """Write an object container around an already-encoded block ``body``.

    The escape hatch for schemas no writer produces. ``schema`` is the Avro schema as a
    Python object; ``body`` is the encoded records and ``record_count`` how many of them
    it holds. A container with no records carries no block at all, which is what the
    file destination writes for an empty load.
    """
    import json as _json

    metadata = _json.dumps(schema).encode("utf-8")
    out = bytearray(b"Obj\x01")
    out += encode_avro_long(2)
    out += encode_avro_bytes(b"avro.schema") + encode_avro_bytes(metadata)
    out += encode_avro_bytes(b"avro.codec") + encode_avro_bytes(codec)
    out += encode_avro_long(0)
    out += _AVRO_SYNC
    if record_count:
        out += (
            encode_avro_long(record_count)
            + encode_avro_long(len(body))
            + body
            + _AVRO_SYNC
        )
    with open(path, "wb") as f:
        f.write(bytes(out))
    return path


#: The Avro type each Python value is encoded as. Deliberately small: enough for record
#: fixtures, and no inference cleverness that could quietly disagree with the encoder.
_AVRO_TYPES = ((bool, "boolean"), (int, "long"), (float, "double"), (str, "string"))


def _avro_type_of(value):
    for python_type, avro_type in _AVRO_TYPES:
        if isinstance(value, python_type):
            return avro_type
    raise TypeError(f"No Avro spelling for {type(value).__name__}: {value!r}")


def _avro_encode(value, avro_type):
    import struct

    if avro_type == "boolean":
        return b"\x01" if value else b"\x00"
    if avro_type == "long":
        return encode_avro_long(value)
    if avro_type == "double":
        return struct.pack("<d", value)
    return encode_avro_bytes(value.encode("utf-8"))


def write_avro(path, records, name="root"):
    """Write record dictionaries as an Avro object container, one block.

    The field order and types come from the records: a column carrying a ``None`` in any
    row becomes a ``["null", T]`` union, which is how a sparse fixture is expressed.
    """
    fields, types = [], {}
    for record in records:
        for key, value in record.items():
            if value is None:
                types.setdefault(key, None)
                if key not in fields:
                    fields.append(key)
                continue
            if key not in fields:
                fields.append(key)
            types[key] = _avro_type_of(value)

    nullable = {
        key: any(record.get(key) is None for record in records) for key in fields
    }
    schema = {
        "type": "record",
        "name": name,
        "fields": [
            {
                "name": key,
                "type": ["null", types[key] or "string"]
                if nullable[key]
                else types[key],
            }
            for key in fields
        ],
    }

    body = bytearray()
    for record in records:
        for key in fields:
            value = record.get(key)
            if nullable[key]:
                if value is None:
                    body += encode_avro_long(0)
                    continue
                body += encode_avro_long(1)
            body += _avro_encode(value, types[key] or "string")

    return write_avro_container(path, schema, bytes(body), len(records))
