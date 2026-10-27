"""Bounded gzip/CSV prefix extraction from a supplied binary stream.

No file opening, dataframe, network, retry, clock, sleep or background work.
Prefix hashes never claim that the entire archive or its CRC was verified.
"""
from __future__ import annotations

import csv
import hashlib
import zlib

from .unified_measurement_v1 import identity


def limit(value, cap):
    if type(value) is not int or not 1 <= value <= cap:
        raise ValueError("positive integer within hard bound required")
    return value


def read_prefix(raw, *, max_rows=128, max_compressed_bytes=65536,
                max_uncompressed_bytes=32768, max_line_bytes=2048):
    """Parse at most max_rows; a <=4096-byte inflate chunk can contain unused rows."""
    limit(max_rows, 4096)
    limit(max_compressed_bytes, 262144)
    limit(max_uncompressed_bytes, 262144)
    limit(max_line_bytes, 4096)
    if raw.tell() != 0:
        raise ValueError("archive prefix must start at byte zero")
    decoder = zlib.decompressobj(31)
    compressed_sha, output, pending = hashlib.sha256(), bytearray(), b""
    compressed, newlines, members, complete = 0, 0, 0, False
    stop = None
    for _ in range(1024):  # hard termination even for malformed concatenated members
        if newlines >= max_rows + 1:
            stop = "row_budget"
            break
        if len(output) >= max_uncompressed_bytes:
            stop = "uncompressed_byte_budget"
            break
        if decoder.eof:
            members += 1
            if members > 32:
                raise ValueError("gzip member count bound exceeded")
            pending = decoder.unused_data
            decoder = zlib.decompressobj(31)
            next_member = True
        else:
            next_member = False
        if not pending:
            if compressed == max_compressed_bytes:
                stop = "compressed_byte_budget"
                break
            chunk = raw.read(min(4096, max_compressed_bytes - compressed))
            if not chunk:
                if next_member:
                    complete, stop = True, "archive_eof_verified"
                    break
                raise ValueError("truncated gzip before verified member trailer")
            compressed += len(chunk)
            compressed_sha.update(chunk)
            pending = chunk
        try:
            inflated = decoder.decompress(pending, min(4096, max_uncompressed_bytes - len(output)))
        except zlib.error as exc:
            raise ValueError("invalid gzip header/deflate/CRC") from exc
        pending = decoder.unconsumed_tail
        output.extend(inflated)
        newlines += inflated.count(b"\n")
    else:
        raise ValueError("bounded decompression work counter exceeded")
    data = bytes(output)
    lines = data.split(b"\n")
    tail = lines.pop()
    if complete and tail:
        lines.append(tail)  # an entire CRC-verified archive may omit its final LF
    if not lines:
        raise ValueError("complete CSV header missing before prefix bound")
    rows, header, consumed = [], None, 0
    for index, line in enumerate(lines[:max_rows + 1]):
        if len(line) > max_line_bytes:
            raise ValueError("CSV line bound exceeded")
        try:
            fields = next(csv.reader([line.decode("utf-8").rstrip("\r")], strict=True))
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("invalid UTF-8 or multiline/unterminated CSV record") from exc
        if index == 0:
            header = fields
            if not 1 <= len(header) <= 32 or len(set(header)) != len(header) or not all(header):
                raise ValueError("bounded unique CSV header required")
        else:
            if len(fields) != len(header):
                raise ValueError("CSV field count mismatch")
            rows.append(dict(zip(header, fields)))
        consumed += len(line) + 1
    return {"schema": "bounded_gzip_csv_prefix/1", "header": header, "rows": rows,
            "row_count": len(rows), "stop_reason": stop,
            "compressed_bytes_read": compressed, "uncompressed_bytes_inflated": len(data),
            "unparsed_inflated_bytes": max(0, len(data) - consumed),
            "compressed_prefix_sha256": compressed_sha.hexdigest(),
            "uncompressed_prefix_sha256": hashlib.sha256(data).hexdigest(),
            "row_bank_sha256": identity(rows), "verified_gzip_members": members,
            "entire_archive_read_and_crc_verified": complete, "full_archive_hash_verified": False,
            "limits": {"max_rows": max_rows, "max_compressed_bytes": max_compressed_bytes,
                       "max_uncompressed_bytes": max_uncompressed_bytes, "max_line_bytes": max_line_bytes}}
