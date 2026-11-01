"""Reading a heart-rate .xlsx file: the first sheet, three columns, nothing else.

The sheet is converted to the same CSV text the Fitbit reader already checks, so a
workbook goes through exactly the same header rule and the same parsing as a CSV:
the header must be exactly timestamp, beats per minute, data source; any other
column refuses the file before a single data row is read; and nothing beyond those
three columns is ever read, stored or shown. No other layout is guessed.

Excel stores a date as a number or a date-time cell, not as text. A cell with a
timezone is written out in ISO form; a cell WITHOUT one is written out without it,
and the Fitbit reader then refuses the file ("timestamps without a timezone")
rather than guessing a zone.

A workbook is itself a zip, so its declared uncompressed size is checked before it
is opened, and the file is read in streaming mode and never written to disk.
"""

import csv
import io
import zipfile
from datetime import date, datetime

from .sources import SourceError

MAX_XLSX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024
MAX_XLSX_ROWS = 1_100_000
HEADER = ["timestamp", "beats per minute", "data source"]


def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def xlsx_to_csv_bytes(name, data):
    """CSV bytes (header + the three columns) from the first sheet of a workbook."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        declared = sum(info.file_size for info in archive.infolist())
        archive.close()
    except zipfile.BadZipFile:
        raise SourceError("{} is not a valid Excel (.xlsx) file.".format(name))
    if declared > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise SourceError("{} is too large to read as a workbook. Export it as CSV "
                          "instead.".format(name))

    try:
        from openpyxl import load_workbook
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception:
        raise SourceError("{} could not be read as an Excel (.xlsx) file.".format(name))

    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            raise SourceError("{} has an empty first sheet.".format(name))
        header = [("" if c is None else str(c).strip()) for c in header]
        while header and header[-1] == "":
            header.pop()

        missing = [c for c in HEADER if c not in header]
        if missing or not header:
            raise SourceError(
                "{} is not a Fitbit heart-rate sheet: the first row must have the "
                "columns {}.".format(name, ", ".join(HEADER)))
        if len(header) != len(HEADER):
            raise SourceError(
                "{} has {} column(s) beyond the three heart-rate columns ({}). Only that "
                "layout is accepted, and the other columns are never read. No other "
                "layout is guessed.".format(name, len(header) - len(HEADER),
                                            ", ".join(HEADER)))

        order = [header.index(c) for c in HEADER]
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(HEADER)
        count = 0
        for row in rows:
            count += 1
            if count > MAX_XLSX_ROWS:
                raise SourceError("{} has too many rows to read as a workbook. Export "
                                  "it as CSV instead.".format(name))
            if row is None or all(c is None for c in row):
                continue
            writer.writerow([_cell_text(row[i]) if i < len(row) else "" for i in order])
    finally:
        workbook.close()

    return out.getvalue().encode("utf-8")
