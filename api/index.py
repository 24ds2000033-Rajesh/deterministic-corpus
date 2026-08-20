import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# Constants
# ============================================================

SAFE_INTEGER_MAX = 9007199254740991

URI_RE = re.compile(r"^gs://[^/]+/.+$")
DECIMAL_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

TIME_RE = re.compile(
    r"^"
    r"(\d{4}-\d{2}-\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)

ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}


# ============================================================
# Deterministic JSON
# ============================================================

def cj(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def bkey(value):
    return value.encode("utf-8")


# ============================================================
# CRC32C Castagnoli
# ============================================================

CRC32C_TABLE = []

for i in range(256):
    c = i

    for _ in range(8):
        if c & 1:
            c = (c >> 1) ^ 0x82F63B78
        else:
            c >>= 1

    CRC32C_TABLE.append(c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF

    for byte in data:
        crc = (
            CRC32C_TABLE[(crc ^ byte) & 0xFF]
            ^ (crc >> 8)
        )

    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


# ============================================================
# Timestamp
# ============================================================

def parse_time(value):
    if not isinstance(value, str):
        return None

    m = TIME_RE.fullmatch(value)

    if not m:
        return None

    date_part, hh, mm, ss, fraction, offset = m.groups()

    try:
        year, month, day = map(
            int,
            date_part.split("-"),
        )

        hour = int(hh)
        minute = int(mm)
        second = int(ss)

        if hour > 23 or minute > 59 or second > 59:
            return None

        ms = int(
            (fraction or "").ljust(3, "0")
        )

        if offset == "Z":
            tz = timezone.utc

        else:
            sign = 1 if offset[0] == "+" else -1

            oh = int(offset[1:3])
            om = int(offset[4:6])

            if oh > 14:
                return None

            if om > 59:
                return None

            if oh == 14 and om != 0:
                return None

            delta = timedelta(
                hours=oh,
                minutes=om,
            )

            if sign < 0:
                delta = -delta

            tz = timezone(delta)

        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            ms * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def canonical_time(dt):
    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


# ============================================================
# Unicode canonicalization
# ============================================================

def canonicalize(value):
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.lower()

    result = []
    whitespace = False

    for ch in value:
        if ch.isspace():
            if not whitespace:
                result.append(" ")
            whitespace = True
        else:
            result.append(ch)
            whitespace = False

    return "".join(result).strip()


# ============================================================
# Word sets for contamination
# ============================================================

def word_set(value):
    value = value.lower()

    result = set()
    current = []

    for ch in value:
        category = unicodedata.category(ch)

        if (
            category.startswith("L")
            or category.startswith("N")
        ):
            current.append(ch)
        else:
            if current:
                result.add("".join(current))
                current = []

    if current:
        result.add("".join(current))

    return result


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# Validation helpers
# ============================================================

def valid_uri(value):
    return (
        isinstance(value, str)
        and URI_RE.fullmatch(value) is not None
    )


def valid_generation(value):
    return (
        isinstance(value, str)
        and DECIMAL_RE.fullmatch(value) is not None
    )


def valid_crc_syntax(value):
    return (
        isinstance(value, str)
        and CRC_RE.fullmatch(value) is not None
    )


def valid_revision(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INTEGER_MAX
    )


def valid_row_shape(row):
    if not isinstance(row, dict):
        return False

    if set(row.keys()) != ROW_KEYS:
        return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not isinstance(row["eventTime"], str):
        return False

    if not isinstance(row["text"], str):
        return False

    if not valid_revision(row["revision"]):
        return False

    return True


# ============================================================
# JSONL
# ============================================================

def parse_jsonl(content):
    """
    Returns:
        ("ok", rows)
        ("jsonl", None)
        ("schema", None)
    """

    if content == "":
        return "schema", None

    rows = []

    # splitlines() handles CRLF, LF, etc.
    for line in content.splitlines():

        if line.strip() == "":
            continue

        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return "jsonl", None

        if not valid_row_shape(parsed):
            return "schema", None

        # eventTime is part of the row's required validity.
        if parse_time(parsed["eventTime"]) is None:
            return "schema", None

        rows.append(parsed)

    if not rows:
        return "schema", None

    return "ok", rows


# ============================================================
# Reason sorting
# ============================================================

def reason_list(values):
    return sorted(
        set(values),
        key=bkey,
    )


# ============================================================
# Build service
# ============================================================

def process(body):

    # --------------------------------------------------------
    # Top-level request parsing
    # --------------------------------------------------------

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # The specification explicitly makes these request errors.
    if "policy" not in body:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if (
        "objects" not in body
        or not isinstance(body["objects"], list)
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    policy = body["policy"]
    objects = body["objects"]

    # --------------------------------------------------------
    # Policy validity
    # --------------------------------------------------------

    policy_valid = False
    min_dt = None
    max_dt = None
    threshold = None

    if isinstance(policy, dict):

        min_dt = parse_time(
            policy.get("minTime")
        )

        max_dt = parse_time(
            policy.get("maxTime")
        )

        threshold = policy.get(
            "contaminationThreshold"
        )

        threshold_valid = (
            isinstance(
                threshold,
                (int, float),
            )
            and not isinstance(
                threshold,
                bool,
            )
            and math.isfinite(threshold)
            and 0 <= threshold <= 1
        )

        if (
            min_dt is not None
            and max_dt is not None
            and min_dt <= max_dt
            and threshold_valid
        ):
            policy_valid = True

    # --------------------------------------------------------
    # Object identity/integrity
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for supplied in objects:

        # A non-object is still an object supplied by the
        # caller. All independently applicable object checks
        # are performed.
        if isinstance(supplied, dict):
            uri = supplied.get("uri")
            generation = supplied.get(
                "generation"
            )
            fetched_generation = supplied.get(
                "fetchedGeneration"
            )
            supplied_crc = supplied.get(
                "crc32c"
            )
            schema_id = supplied.get(
                "schemaId"
            )
            content = supplied.get(
                "content"
            )
        else:
            uri = None
            generation = None
            fetched_generation = None
            supplied_crc = None
            schema_id = None
            content = None

        codes = []

        # ----------------------------------------------------
        # URI
        # ----------------------------------------------------

        if not valid_uri(uri):
            codes.append("URI_INVALID")

        # ----------------------------------------------------
        # Generations
        #
        # INVALID is based on syntax/type.
        # MISMATCH is based solely on supplied values.
        # ----------------------------------------------------

        gen_ok = valid_generation(
            generation
        )

        fetched_ok = valid_generation(
            fetched_generation
        )

        if not gen_ok or not fetched_ok:
            codes.append("GENERATION_INVALID")

        if generation != fetched_generation:
            codes.append("GENERATION_MISMATCH")

        # ----------------------------------------------------
        # CRC
        # ----------------------------------------------------

        crc_ok = valid_crc_syntax(
            supplied_crc
        )

        if not crc_ok:
            codes.append("CRC32C_INVALID")

        # Only check mismatch for:
        #   string content
        #   syntactically valid CRC
        if (
            isinstance(content, str)
            and crc_ok
        ):
            actual_crc = crc32c_hex(
                content.encode("utf-8")
            )

            if actual_crc != supplied_crc:
                codes.append(
                    "CRC32C_MISMATCH"
                )

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        if not isinstance(content, str):
            codes.append("SCHEMA_INVALID")

        if schema_id != "training-v1":
            codes.append("SCHEMA_INVALID")

        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        rows = None

        if isinstance(content, str):

            status, rows = parse_jsonl(
                content
            )

            if status == "jsonl":
                codes.append(
                    "JSONL_INVALID"
                )

            elif status == "schema":
                codes.append(
                    "SCHEMA_INVALID"
                )

        codes = reason_list(codes)

        # ----------------------------------------------------
        # Reject whole object if ANY object-level error.
        # ----------------------------------------------------

        if codes:

            rejected_objects.append(
                {
                    "uri": (
                        uri
                        if isinstance(uri, str)
                        else None
                    ),
                    "reasonCodes": codes,
                }
            )

            continue

        # ----------------------------------------------------
        # Object accepted.
        # ----------------------------------------------------

        accepted_objects.append(
            {
                "uri": uri,
                "generation": generation,
                "crc32c": supplied_crc,
                "schemaId": schema_id,
                "rows": rows,
            }
        )

    # --------------------------------------------------------
    # Lineage
    #
    # Only integrity/schema-valid accepted objects enter
    # lineage.
    # --------------------------------------------------------

    lineage = []

    for obj in accepted_objects:
        lineage.append(
            {
                "uri": obj["uri"],
                "generation": obj["generation"],
                "crc32c": obj["crc32c"],
                "schemaId": obj["schemaId"],
            }
        )

    # --------------------------------------------------------
    # Candidate rows
    # --------------------------------------------------------

    candidates = []

    for obj in accepted_objects:
        for raw in obj["rows"]:

            event_dt = parse_time(
                raw["eventTime"]
            )

            normalized = {
                "id": raw["id"],
                "entity": canonicalize(
                    raw["entity"]
                ),
                "eventTime": canonical_time(
                    event_dt
                ),
                "revision": raw["revision"],
                "text": canonicalize(
                    raw["text"]
                ),
            }

            candidates.append(normalized)

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    groups = {}

    for row in candidates:

        key = (
            row["entity"],
            row["eventTime"],
            row["text"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    retained = []
    rejected_rows_raw = []

    for group in groups.values():

        winner = min(
            group,
            key=lambda row: (
                -row["revision"],
                bkey(row["id"]),
            ),
        )

        retained.append(winner)

        for row in group:
            if row is not winner:
                rejected_rows_raw.append(
                    {
                        "id": row["id"],
                        "reasonCodes": [
                            "DUPLICATE"
                        ],
                    }
                )

    # --------------------------------------------------------
    # Policy + window
    # --------------------------------------------------------

    policy_rows = []

    for row in retained:

        codes = []

        if not policy_valid:
            codes.append(
                "POLICY_INVALID"
            )

        else:

            dt = parse_time(
                row["eventTime"]
            )

            if (
                dt < min_dt
                or dt > max_dt
            ):
                codes.append(
                    "OUT_OF_WINDOW"
                )

        if codes:

            rejected_rows_raw.append(
                {
                    "id": row["id"],
                    "reasonCodes": reason_list(
                        codes
                    ),
                }
            )
        else:
            policy_rows.append(row)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in policy_rows:

        first_byte = hashlib.sha256(
            row["entity"].encode("utf-8")
        ).digest()[0]

        bucket = first_byte % 10

        if bucket <= 5:
            splits["train"].append(row)

        elif bucket <= 7:
            splits["validation"].append(row)

        else:
            splits["test"].append(row)

    # --------------------------------------------------------
    # Train contamination
    # --------------------------------------------------------

    train_sets = [
        word_set(row["text"])
        for row in splits["train"]
    ]

    final_splits = {
        "train": list(splits["train"]),
        "validation": [],
        "test": [],
    }

    for split in (
        "validation",
        "test",
    ):

        for row in splits[split]:

            current = word_set(
                row["text"]
            )

            contaminated = False

            for train_set in train_sets:

                if (
                    jaccard(
                        current,
                        train_set,
                    )
                    >= threshold
                ):
                    contaminated = True
                    break

            if contaminated:

                rejected_rows_raw.append(
                    {
                        "id": row["id"],
                        "reasonCodes": [
                            "TRAIN_CONTAMINATION"
                        ],
                    }
                )

            else:
                final_splits[split].append(
                    row
                )

    # --------------------------------------------------------
    # Sort final split rows
    # --------------------------------------------------------

    for split in final_splits:

        final_splits[split].sort(
            key=lambda row: (
                bkey(row["id"]),
                bkey(cj(row)),
            )
        )

    # --------------------------------------------------------
    # JSONL digests
    # --------------------------------------------------------

    digests = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        data = "".join(
            cj(row) + "\n"
            for row in final_splits[split]
        ).encode("utf-8")

        digests[split] = hashlib.sha256(
            data
        ).hexdigest()

    # --------------------------------------------------------
    # Merge row rejection codes by ID
    # --------------------------------------------------------

    row_map = {}

    for item in rejected_rows_raw:

        row_id = item["id"]

        if row_id not in row_map:
            row_map[row_id] = set()

        row_map[row_id].update(
            item["reasonCodes"]
        )

    rejected_rows = [
        {
            "id": row_id,
            "reasonCodes": reason_list(codes),
        }
        for row_id, codes in row_map.items()
    ]

    rejected_rows.sort(
        key=lambda item: (
            bkey(item["id"]),
            bkey(cj(item)),
        )
    )

    # --------------------------------------------------------
    # Object / lineage deterministic ordering
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            bkey(
                item["uri"]
                if isinstance(item["uri"], str)
                else ""
            ),
            bkey(cj(item)),
        )
    )

    lineage.sort(
        key=lambda item: (
            bkey(item["uri"]),
            bkey(cj(item)),
        )
    )

    # --------------------------------------------------------
    # Exact response shape
    # --------------------------------------------------------

    return JSONResponse(
        content={
            "splits": final_splits,
            "rejectedObjects": rejected_objects,
            "rejectedRows": rejected_rows,
            "digests": digests,
            "lineage": lineage,
        }
    )


# ============================================================
# HTTP endpoint
# ============================================================

@app.post("/build-corpus")
async def build_corpus(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    return process(body)
