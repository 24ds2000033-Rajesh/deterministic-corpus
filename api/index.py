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


# ============================================================
# URI
# ============================================================

# gs://bucket/object
#
# Bucket is everything between gs:// and the first slash.
# Object is everything after that slash and must be non-empty.
#
# Object names may themselves contain additional '/' characters.
URI_RE = re.compile(
    r"^gs://[^/]+/.+$"
)


# ============================================================
# Generation / CRC / timestamp syntax
# ============================================================

GENERATION_RE = re.compile(
    r"^[0-9]+$"
)

CRC_RE = re.compile(
    r"^[0-9a-f]{8}$"
)

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
# Timestamp parsing
# ============================================================

def parse_timestamp(value):
    """
    Accept:

        YYYY-MM-DDTHH:mm:ssZ
        YYYY-MM-DDTHH:mm:ss.sZ
        YYYY-MM-DDTHH:mm:ss.ssZ
        YYYY-MM-DDTHH:mm:ss.sssZ

    and the equivalent numeric offsets.

    Offset magnitude is at most 14:00.
    Offset 14 requires minute 00.
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

            offset_hours = int(
                offset[1:3]
            )

            offset_minutes = int(
                offset[4:6]
            )

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
        dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + "."
        + f"{dt.microsecond // 1000:03d}"
        + "Z"
    )


# ============================================================
# Canonicalization
# ============================================================

def canonicalize(value):
    # Unicode NFKC.
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    # Lowercase.
    value = value.lower()

    # Unicode whitespace -> one ASCII space.
    # Leading/trailing whitespace is removed.
    result = []
    pending_space = False

    for ch in value:

        if ch.isspace():
            pending_space = True
            continue

        if pending_space and result:
            result.append(" ")

        pending_space = False
        result.append(ch)

    return "".join(result)


# ============================================================
# Contamination word sets
# ============================================================

def unicode_word_set(value):
    value = value.lower()

    words = set()
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
                words.add(
                    "".join(current)
                )
                current = []

    if current:
        words.add(
            "".join(current)
        )

    return words


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# Primitive validators
# ============================================================

def valid_uri(value):
    if not isinstance(value, str):
        return False

    return (
        URI_RE.fullmatch(value)
        is not None
    )


def valid_generation(value):
    return (
        isinstance(value, str)
        and GENERATION_RE.fullmatch(
            value
        ) is not None
    )


def valid_crc(value):
    return (
        isinstance(value, str)
        and CRC_RE.fullmatch(
            value
        ) is not None
    )


def valid_revision(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


# ============================================================
# JSON parser
# ============================================================

def reject_nonstandard_json_constant(value):
    """
    Python's json module accepts NaN/Infinity by default,
    although those are not JSON values.

    Treat those as JSON parsing failures.
    """
    raise ValueError(
        "non-standard JSON constant"
    )


# ============================================================
# Row schema
# ============================================================

def valid_row(row):

    if not isinstance(row, dict):
        return False

    # Exactly five keys.
    if set(row.keys()) != ROW_KEYS:
        return False

    # Four string fields.
    if not isinstance(
        row["id"],
        str,
    ):
        return False

    if not isinstance(
        row["entity"],
        str,
    ):
        return False

    if not isinstance(
        row["eventTime"],
        str,
    ):
        return False

    if not isinstance(
        row["text"],
        str,
    ):
        return False

    # Non-negative safe integer.
    if not valid_revision(
        row["revision"]
    ):
        return False

    # eventTime must be valid.
    if parse_timestamp(
        row["eventTime"]
    ) is None:
        return False

    return True


# ============================================================
# JSONL
# ============================================================

def parse_jsonl(content):
    """
    Return:

        ("OK", rows)

        ("JSONL_INVALID", None)

        ("SCHEMA_INVALID", None)
    """

    # Empty file.
    if content == "":
        return (
            "SCHEMA_INVALID",
            None,
        )

    rows = []

    # JSONL records are LF-delimited.
    #
    # We intentionally do not use splitlines(), because it treats
    # Unicode line separators such as U+2028/U+2029 as record
    # separators even though JSONL uses LF.
    for line in content.split("\n"):

        # Accept CRLF.
        if line.endswith("\r"):
            line = line[:-1]

        # Blank lines are ignored.
        if line.strip() == "":
            continue

        try:
            value = json.loads(
                line,
                parse_constant=(
                    reject_nonstandard_json_constant
                ),
            )

        except Exception:
            return (
                "JSONL_INVALID",
                None,
            )

        # Valid JSON but wrong row structure.
        if not valid_row(value):
            return (
                "SCHEMA_INVALID",
                None,
            )

        rows.append(value)

    # File containing only blank lines.
    if not rows:
        return (
            "SCHEMA_INVALID",
            None,
        )

    return (
        "OK",
        rows,
    )


# ============================================================
# Reason sorting
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

    # --------------------------------------------------------
    # Extract values exactly as supplied.
    # --------------------------------------------------------

    if isinstance(obj, dict):

        uri = obj.get("uri")

        generation = obj.get(
            "generation"
        )

        fetched_generation = obj.get(
            "fetchedGeneration"
        )

        supplied_crc = obj.get(
            "crc32c"
        )

        schema_id = obj.get(
            "schemaId"
        )

        content = obj.get(
            "content"
        )

    else:

        uri = None
        generation = None
        fetched_generation = None
        supplied_crc = None
        schema_id = None
        content = None

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    if not valid_uri(uri):
        reasons.append(
            "URI_INVALID"
        )

    # --------------------------------------------------------
    # Generation syntax
    # --------------------------------------------------------

    generation_valid = valid_generation(
        generation
    )

    fetched_generation_valid = (
        valid_generation(
            fetched_generation
        )
    )

    if not generation_valid:
        reasons.append(
            "GENERATION_INVALID"
        )

    if not fetched_generation_valid:
        reasons.append(
            "GENERATION_INVALID"
        )

    # --------------------------------------------------------
    # Generation mismatch
    #
    # This is independently applicable whenever the supplied
    # values differ.
    # --------------------------------------------------------

    if (
        generation
        != fetched_generation
    ):
        reasons.append(
            "GENERATION_MISMATCH"
        )

    # --------------------------------------------------------
    # CRC syntax
    # --------------------------------------------------------

    crc_valid = valid_crc(
        supplied_crc
    )

    if not crc_valid:
        reasons.append(
            "CRC32C_INVALID"
        )

    # --------------------------------------------------------
    # CRC value
    #
    # Only applicable when:
    #   * content is a string
    #   * supplied CRC has valid syntax
    # --------------------------------------------------------

    if (
        isinstance(content, str)
        and crc_valid
    ):

        expected_crc = crc32c_hex(
            content.encode("utf-8")
        )

        if expected_crc != supplied_crc:
            reasons.append(
                "CRC32C_MISMATCH"
            )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    if not isinstance(
        content,
        str,
    ):
        reasons.append(
            "SCHEMA_INVALID"
        )

    if schema_id != "training-v1":
        reasons.append(
            "SCHEMA_INVALID"
        )

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    rows = None

    if isinstance(
        content,
        str,
    ):

        status, rows = parse_jsonl(
            content
        )

        if status == "JSONL_INVALID":

            reasons.append(
                "JSONL_INVALID"
            )

        elif status == "SCHEMA_INVALID":

            reasons.append(
                "SCHEMA_INVALID"
            )

    # --------------------------------------------------------
    # Deterministic reason array.
    # --------------------------------------------------------

    reasons = sort_reasons(
        reasons
    )

    # --------------------------------------------------------
    # Reject object.
    # --------------------------------------------------------

    if reasons:

        return {
            "accepted": False,
            "rejected": {
                "uri": (
                    uri
                    if isinstance(
                        uri,
                        str,
                    )
                    else None
                ),
                "reasonCodes": reasons,
            },
        }

    # --------------------------------------------------------
    # Accepted object.
    # --------------------------------------------------------

    return {
        "accepted": True,
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": schema_id,
        "rows": rows,
    }


# ============================================================
# Main corpus builder
# ============================================================

def build(body):

    # --------------------------------------------------------
    # Request parsing
    # --------------------------------------------------------

    if not isinstance(
        body,
        dict,
    ):
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
        or not isinstance(
            body["objects"],
            list,
        )
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    policy = body["policy"]
    objects = body["objects"]

    # --------------------------------------------------------
    # Policy validation
    # --------------------------------------------------------

    policy_valid = False

    min_time = None
    max_time = None
    threshold = None

    if isinstance(
        policy,
        dict,
    ):

        min_time = parse_timestamp(
            policy.get("minTime")
        )

        max_time = parse_timestamp(
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
            and math.isfinite(
                threshold
            )
            and 0 <= threshold <= 1
        )

        if (
            min_time is not None
            and max_time is not None
            and min_time <= max_time
            and threshold_valid
        ):
            policy_valid = True

    # --------------------------------------------------------
    # Object validation
    # --------------------------------------------------------

    accepted = []
    rejected_objects = []

    for obj in objects:

        result = validate_object(
            obj
        )

        if result["accepted"]:

            accepted.append(
                result
            )

        else:

            rejected_objects.append(
                result["rejected"]
            )

    # --------------------------------------------------------
    # Lineage
    #
    # Only objects that passed ALL object-level checks.
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
    rejected_row_entries = []

    for group in groups.values():

        # Highest revision wins.
        #
        # Equal revision:
        # smallest UTF-8 ID wins.
        winner = min(
            group,
            key=lambda row: (
                -row["revision"],
                utf8(row["id"]),
            ),
        )

        retained.append(
            winner
        )

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

            split_candidates[
                "train"
            ].append(row)

        elif bucket <= 7:

            split_candidates[
                "validation"
            ].append(row)

        else:

            split_candidates[
                "test"
            ].append(row)

    # --------------------------------------------------------
    # Contamination
    # --------------------------------------------------------

    train_sets = [
        unicode_word_set(
            row["text"]
        )
        for row in split_candidates[
            "train"
        ]
    ]

    final_splits = {
        "train": list(
            split_candidates[
                "train"
            ]
        ),
        "validation": [],
        "test": [],
    }

    for split in (
        "validation",
        "test",
    ):

        for row in split_candidates[
            split
        ]:

            current_words = (
                unicode_word_set(
                    row["text"]
                )
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

                final_splits[
                    split
                ].append(row)

    # --------------------------------------------------------
    # Sort split rows
    # --------------------------------------------------------

    for split in final_splits:

        final_splits[
            split
        ].sort(
            key=lambda row: (
                utf8(row["id"]),
                utf8(
                    compact_json(row)
                ),
            )
        )

    # --------------------------------------------------------
    # Deterministic JSONL digests
    # --------------------------------------------------------

    digests = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        jsonl = "".join(
            compact_json(row)
            + "\n"
            for row in final_splits[
                split
            ]
        )

        digests[
            split
        ] = hashlib.sha256(
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
        key=lambda item: (
            utf8(item["id"]),
            utf8(
                compact_json(item)
            ),
        )
    )

    # --------------------------------------------------------
    # Sort rejected objects
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            utf8(
                item["uri"]
                if isinstance(
                    item["uri"],
                    str,
                )
                else ""
            ),
            utf8(
                compact_json(item)
            ),
        )
    )

    # --------------------------------------------------------
    # Sort lineage
    # --------------------------------------------------------

    lineage.sort(
        key=lambda item: (
            utf8(item["uri"]),
            utf8(
                compact_json(item)
            ),
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
# HTTP endpoint
# ============================================================

async def _build_corpus_endpoint(
    request: Request,
):
    try:
        body = await request.json()

    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    return build(body)


# Keep all aliases because your current Vercel deployment
# is already routing grader requests through these paths.

@app.post("/build-corpus")
async def build_corpus(
    request: Request,
):
    return await _build_corpus_endpoint(
        request
    )


@app.post("/api/build-corpus")
async def api_build_corpus(
    request: Request,
):
    return await _build_corpus_endpoint(
        request
    )


@app.post("/build-corpus/build-corpus")
async def duplicated_build_corpus(
    request: Request,
):
    return await _build_corpus_endpoint(
        request
    )


@app.post("/api/index")
async def api_index(
    request: Request,
):
    return await _build_corpus_endpoint(
        request
    )
