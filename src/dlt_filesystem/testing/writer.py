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
