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

SAFE_INT_MAX = 9007199254740991

OBJECT_CODES = (
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
)

ROW_CODES = (
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
)

ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}

# gs://bucket/object
#
# bucket = non-empty segment
# object = non-empty remainder
URI_RE = re.compile(r"^gs://[^/]+/.+$")

GENERATION_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)


# ============================================================
# Deterministic JSON
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def utf8(value):
    return value.encode("utf-8")


# ============================================================
# CRC32C / Castagnoli
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


def crc32c(data):
    crc = 0xFFFFFFFF

    for byte in data:
        crc = (
            CRC32C_TABLE[(crc ^ byte) & 0xFF]
            ^ (crc >> 8)
        )

    return crc ^ 0xFFFFFFFF


def crc32c_hex(data):
    return f"{crc32c(data):08x}"


# ============================================================
# Timestamp
# ============================================================

def parse_timestamp(value):
    """
    Strictly accepts:

      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sZ
      YYYY-MM-DDTHH:mm:ss.ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ

    or numeric offsets with the same format.

    Offset must be <= 14:00.
    +14:01 and -14:01 are invalid.
    """

    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)

    if match is None:
        return None

    (
        year_s,
        month_s,
        day_s,
        hour_s,
        minute_s,
        second_s,
        fraction,
        offset,
    ) = match.groups()

    try:
        year = int(year_s)
        month = int(month_s)
        day = int(day_s)

        hour = int(hour_s)
        minute = int(minute_s)
        second = int(second_s)

        if hour > 23:
            return None

        if minute > 59:
            return None

        if second > 59:
            return None

        milliseconds = int(
            (fraction or "").ljust(3, "0")
        )

        if offset == "Z":
            tz = timezone.utc

        else:
            sign = 1 if offset[0] == "+" else -1

            offset_hours = int(offset[1:3])
            offset_minutes = int(offset[4:6])

            if offset_hours > 14:
                return None

            if offset_minutes > 59:
                return None

            if (
                offset_hours == 14
                and offset_minutes != 0
            ):
                return None

            delta = timedelta(
                hours=offset_hours,
                minutes=offset_minutes,
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
            milliseconds * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def canonical_timestamp(dt):
    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


# ============================================================
# Canonicalization
# ============================================================

def canonicalize(value):
    # Unicode NFKC
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    # lowercase
    value = value.lower()

    # Unicode whitespace -> one ASCII space
    output = []
    pending_space = False

    for ch in value:
        if ch.isspace():
            pending_space = True
            continue

        if pending_space and output:
            output.append(" ")

        pending_space = False
        output.append(ch)

    return "".join(output)


# ============================================================
# Word sets
# ============================================================

def unicode_word_set(value):
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
# Primitive validation
# ============================================================

def valid_uri(value):
    return (
        isinstance(value, str)
        and URI_RE.fullmatch(value) is not None
    )


def valid_generation(value):
    return (
        isinstance(value, str)
        and GENERATION_RE.fullmatch(value)
        is not None
    )


def valid_crc(value):
    return (
        isinstance(value, str)
        and CRC_RE.fullmatch(value)
        is not None
    )


def valid_revision(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= SAFE_INT_MAX
    )


# ============================================================
# JSONL row validation
# ============================================================

def valid_row(row):
    if not isinstance(row, dict):
        return False

    # EXACTLY the five keys.
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

    # eventTime is also part of row validity.
    if parse_timestamp(row["eventTime"]) is None:
        return False

    return True


def parse_jsonl(content):
    """
    Return:
        ("OK", rows)
        ("JSONL_INVALID", None)
        ("SCHEMA_INVALID", None)
    """

    if content == "":
        return "SCHEMA_INVALID", None

    rows = []

    # Keep JSONL semantics: every non-blank line is one JSON
    # value. Blank lines are ignored.
    for line in content.splitlines():

        if line.strip() == "":
            continue

        try:
            value = json.loads(line)
        except Exception:
            return "JSONL_INVALID", None

        if not valid_row(value):
            return "SCHEMA_INVALID", None

        rows.append(value)

    if len(rows) == 0:
        return "SCHEMA_INVALID", None

    return "OK", rows


# ============================================================
# Reasons
# ============================================================

def sort_reasons(reasons):
    return sorted(
        set(reasons),
        key=utf8,
    )


# ============================================================
# Object validation
# ============================================================

def validate_object(obj):
    """
    Returns either:

        {
          "accepted": True,
          "uri": ...,
          "generation": ...,
          "crc32c": ...,
          "schemaId": ...,
          "rows": [...]
        }

    or:

        {
          "accepted": False,
          "rejected": {...}
        }
    """

    # A malformed/non-object item cannot provide any valid
    # object fields.
    if not isinstance(obj, dict):
        uri = None
        generation = None
        fetched_generation = None
        supplied_crc = None
        schema_id = None
        content = None
    else:
        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_generation = obj.get(
            "fetchedGeneration"
        )
        supplied_crc = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    if not valid_uri(uri):
        reasons.append("URI_INVALID")

    # --------------------------------------------------------
    # Generations
    # --------------------------------------------------------

    generation_ok = valid_generation(
        generation
    )

    fetched_generation_ok = valid_generation(
        fetched_generation
    )

    if not generation_ok:
        reasons.append("GENERATION_INVALID")

    if not fetched_generation_ok:
        reasons.append("GENERATION_INVALID")

    # Mismatch is independently applicable when the two
    # supplied generation values differ.
    if generation != fetched_generation:
        reasons.append("GENERATION_MISMATCH")

    # --------------------------------------------------------
    # CRC syntax
    # --------------------------------------------------------

    crc_ok = valid_crc(supplied_crc)

    if not crc_ok:
        reasons.append("CRC32C_INVALID")

    # --------------------------------------------------------
    # CRC value
    #
    # Only calculate when content is a string and CRC syntax
    # itself is valid.
    # --------------------------------------------------------

    if (
        isinstance(content, str)
        and crc_ok
    ):
        calculated = crc32c_hex(
            content.encode("utf-8")
        )

        if calculated != supplied_crc:
            reasons.append(
                "CRC32C_MISMATCH"
            )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    if not isinstance(content, str):
        reasons.append("SCHEMA_INVALID")

    if schema_id != "training-v1":
        reasons.append("SCHEMA_INVALID")

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    rows = None

    if isinstance(content, str):

        status, rows = parse_jsonl(content)

        if status == "JSONL_INVALID":
            reasons.append("JSONL_INVALID")

        elif status == "SCHEMA_INVALID":
            reasons.append("SCHEMA_INVALID")

    reasons = sort_reasons(reasons)

    if reasons:

        return {
            "accepted": False,
            "rejected": {
                "uri": (
                    uri
                    if isinstance(uri, str)
                    else None
                ),
                "reasonCodes": reasons,
            },
        }

    return {
        "accepted": True,
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": schema_id,
        "rows": rows,
    }


# ============================================================
# Main processing
# ============================================================

def build(body):

    # --------------------------------------------------------
    # Request parsing
    # --------------------------------------------------------

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

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
    # Policy
    # --------------------------------------------------------

    policy_valid = False
    min_time = None
    max_time = None
    threshold = None

    if isinstance(policy, dict):

        min_time = parse_timestamp(
            policy.get("minTime")
        )

        max_time = parse_timestamp(
            policy.get("maxTime")
        )

        threshold = policy.get(
            "contaminationThreshold"
        )

        threshold_ok = (
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
            min_time is not None
            and max_time is not None
            and min_time <= max_time
            and threshold_ok
        ):
            policy_valid = True

    # --------------------------------------------------------
    # Validate every object independently
    # --------------------------------------------------------

    accepted = []
    rejected_objects = []

    for obj in objects:

        result = validate_object(obj)

        if result["accepted"]:
            accepted.append(result)
        else:
            rejected_objects.append(
                result["rejected"]
            )

    # --------------------------------------------------------
    # Lineage
    #
    # ONLY accepted objects.
    # --------------------------------------------------------

    lineage = [
        {
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        }
        for obj in accepted
    ]

    # --------------------------------------------------------
    # Canonicalize rows
    # --------------------------------------------------------

    rows = []

    for obj in accepted:

        for raw in obj["rows"]:

            dt = parse_timestamp(
                raw["eventTime"]
            )

            row = {
                "id": raw["id"],
                "entity": canonicalize(
                    raw["entity"]
                ),
                "eventTime": canonical_timestamp(
                    dt
                ),
                "revision": raw["revision"],
                "text": canonicalize(
                    raw["text"]
                ),
            }

            rows.append(row)

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        dedup_key = (
            row["entity"],
            row["eventTime"],
            row["text"],
        )

        groups.setdefault(
            dedup_key,
            [],
        ).append(row)

    retained = []
    rejected_row_entries = []

    for group in groups.values():

        # Highest revision.
        #
        # For ties, UTF-8-byte-smallest ID.
        winner = min(
            group,
            key=lambda r: (
                -r["revision"],
                utf8(r["id"]),
            ),
        )

        retained.append(winner)

        for row in group:

            if row is winner:
                continue

            rejected_row_entries.append(
                {
                    "id": row["id"],
                    "reasonCodes": [
                        "DUPLICATE"
                    ],
                }
            )

    # --------------------------------------------------------
    # Policy / window
    # --------------------------------------------------------

    eligible = []

    for row in retained:

        reasons = []

        if not policy_valid:
            reasons.append(
                "POLICY_INVALID"
            )
        else:

            dt = parse_timestamp(
                row["eventTime"]
            )

            if (
                dt < min_time
                or dt > max_time
            ):
                reasons.append(
                    "OUT_OF_WINDOW"
                )

        if reasons:
            rejected_row_entries.append(
                {
                    "id": row["id"],
                    "reasonCodes": sort_reasons(
                        reasons
                    ),
                }
            )
        else:
            eligible.append(row)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    split_candidates = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in eligible:

        digest = hashlib.sha256(
            row["entity"].encode("utf-8")
        ).digest()

        bucket = digest[0] % 10

        if bucket <= 5:
            split_candidates["train"].append(
                row
            )
        elif bucket <= 7:
            split_candidates["validation"].append(
                row
            )
        else:
            split_candidates["test"].append(
                row
            )

    # --------------------------------------------------------
    # Contamination
    # --------------------------------------------------------

    train_sets = [
        unicode_word_set(row["text"])
        for row in split_candidates["train"]
    ]

    final_splits = {
        "train": list(
            split_candidates["train"]
        ),
        "validation": [],
        "test": [],
    }

    for split in (
        "validation",
        "test",
    ):

        for row in split_candidates[split]:

            current_words = unicode_word_set(
                row["text"]
            )

            contaminated = False

            for train_words in train_sets:

                if (
                    jaccard(
                        current_words,
                        train_words,
                    )
                    >= threshold
                ):
                    contaminated = True
                    break

            if contaminated:

                rejected_row_entries.append(
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
    # Sort rows
    # --------------------------------------------------------

    for split in final_splits:

        final_splits[split].sort(
            key=lambda row: (
                utf8(row["id"]),
                utf8(compact_json(row)),
            )
        )

    # --------------------------------------------------------
    # Digests
    # --------------------------------------------------------

    digests = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        jsonl = "".join(
            compact_json(row) + "\n"
            for row in final_splits[split]
        )

        digests[split] = hashlib.sha256(
            jsonl.encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------------
    # Aggregate rejected rows
    # --------------------------------------------------------

    row_reasons = {}

    for entry in rejected_row_entries:

        row_id = entry["id"]

        row_reasons.setdefault(
            row_id,
            set(),
        ).update(
            entry["reasonCodes"]
        )

    rejected_rows = [
        {
            "id": row_id,
            "reasonCodes": sort_reasons(
                reasons
            ),
        }
        for row_id, reasons
        in row_reasons.items()
    ]

    rejected_rows.sort(
        key=lambda x: (
            utf8(x["id"]),
            utf8(compact_json(x)),
        )
    )

    # --------------------------------------------------------
    # Deterministic object ordering
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda x: (
            utf8(
                x["uri"]
                if isinstance(x["uri"], str)
                else ""
            ),
            utf8(compact_json(x)),
        )
    )

    lineage.sort(
        key=lambda x: (
            utf8(x["uri"]),
            utf8(compact_json(x)),
        )
    )

    # --------------------------------------------------------
    # EXACT response shape
    # --------------------------------------------------------

    return JSONResponse(
        {
            "splits": final_splits,
            "rejectedObjects": rejected_objects,
            "rejectedRows": rejected_rows,
            "digests": digests,
            "lineage": lineage,
        }
    )


# ============================================================
# Endpoint
# ============================================================

@app.post("/api/build-corpus")
async def build_corpus(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    return build(body)
