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
# Constants / validation
# ============================================================

TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

DECIMAL_RE = re.compile(r"^[0-9]+$")
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")

SAFE_INTEGER_MAX = 9007199254740991

OBJECT_CODES = {
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
}

ROW_CODES = {
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
}


# ============================================================
# CRC32C / Castagnoli
# ============================================================

CRC32C_TABLE = []

for n in range(256):
    c = n

    for _ in range(8):
        if c & 1:
            c = (c >> 1) ^ 0x82F63B78
        else:
            c >>= 1

    CRC32C_TABLE.append(c)


def crc32c(data: bytes) -> int:
    value = 0xFFFFFFFF

    for byte in data:
        value = CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)

    return value ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return f"{crc32c(data):08x}"


# ============================================================
# UTF-8 deterministic ordering
# ============================================================

def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


# ============================================================
# Compact JSON
# ============================================================

def compact_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


# ============================================================
# Timestamp validation
# ============================================================

def parse_timestamp(value):
    """
    Accept exactly:

      YYYY-MM-DDTHH:mm:ssZ
      YYYY-MM-DDTHH:mm:ss.sssZ

    or the equivalent numeric offset.

    Fraction: 1-3 digits.
    Offset magnitude <= 14:00.
    Offset hour 14 requires minutes 00.
    """

    if not isinstance(value, str):
        return None

    match = TIMESTAMP_RE.fullmatch(value)

    if not match:
        return None

    base, fraction, offset = match.groups()

    try:
        base_dt = datetime.strptime(
            base,
            "%Y-%m-%dT%H:%M:%S",
        )

        milliseconds = int((fraction or "").ljust(3, "0"))

        if offset == "Z":
            tz = timezone.utc

        else:
            sign = 1 if offset[0] == "+" else -1

            hours = int(offset[1:3])
            minutes = int(offset[4:6])

            if hours > 14:
                return None

            if minutes > 59:
                return None

            if hours == 14 and minutes != 0:
                return None

            delta = timedelta(
                hours=hours,
                minutes=minutes,
            )

            if sign < 0:
                delta = -delta

            tz = timezone(delta)

        dt = base_dt.replace(
            microsecond=milliseconds * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def canonical_timestamp(dt: datetime) -> str:
    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# Unicode canonicalization
# ============================================================

def canonicalize_text(value: str) -> str:
    """
    NFKC
    lowercase
    trim
    collapse Unicode whitespace to one ASCII space
    """

    value = unicodedata.normalize("NFKC", value)

    value = value.lower()

    result = []
    previous_whitespace = False

    for char in value:
        if char.isspace():
            if not previous_whitespace:
                result.append(" ")

            previous_whitespace = True

        else:
            result.append(char)
            previous_whitespace = False

    return "".join(result).strip()


# ============================================================
# URI validation
# ============================================================

def valid_uri(value) -> bool:
    if not isinstance(value, str):
        return False

    # gs://bucket/object
    #
    # Bucket cannot contain "/".
    # Object must be non-empty.
    return re.fullmatch(
        r"gs://[^/]+/.+",
        value,
    ) is not None


# ============================================================
# Generation validation
# ============================================================

def valid_generation(value) -> bool:
    return (
        isinstance(value, str)
        and DECIMAL_RE.fullmatch(value) is not None
    )


# ============================================================
# CRC validation
# ============================================================

def valid_crc(value) -> bool:
    return (
        isinstance(value, str)
        and CRC32C_RE.fullmatch(value) is not None
    )


# ============================================================
# Row schema
# ============================================================

REQUIRED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}


def valid_row_shape(row) -> bool:
    if not isinstance(row, dict):
        return False

    # Exactly these five keys.
    if set(row.keys()) != REQUIRED_ROW_KEYS:
        return False

    # Four text fields.
    for field in (
        "id",
        "entity",
        "eventTime",
        "text",
    ):
        if not isinstance(row[field], str):
            return False

    revision = row["revision"]

    # bool is an int subclass, so explicitly reject it.
    if isinstance(revision, bool):
        return False

    if not isinstance(revision, int):
        return False

    if revision < 0:
        return False

    if revision > SAFE_INTEGER_MAX:
        return False

    return True


# ============================================================
# JSONL parsing
# ============================================================

def parse_jsonl(content: str):
    """
    Returns:

        rows, jsonl_invalid, schema_invalid

    Blank lines are ignored.
    """

    rows = []

    if content == "":
        return [], False, True

    for line in content.splitlines():

        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except Exception:
            return [], True, False

        rows.append(value)

    if not rows:
        return [], False, True

    for row in rows:
        if not valid_row_shape(row):
            return [], False, True

    return rows, False, False


# ============================================================
# Contamination word-set
# ============================================================

def unicode_word_set(value: str):
    """
    Extract runs of Unicode letters/numbers.

    The value is lowercased first.
    """

    value = value.lower()

    words = set()
    current = []

    for char in value:
        category = unicodedata.category(char)

        if category.startswith("L") or category.startswith("N"):
            current.append(char)

        else:
            if current:
                words.add("".join(current))
                current = []

    if current:
        words.add("".join(current))

    return words


def jaccard_similarity(left, right) -> float:
    if not left and not right:
        return 1.0

    union = left | right

    if not union:
        return 1.0

    return len(left & right) / len(union)


# ============================================================
# Deterministic reason-code sorting
# ============================================================

def sorted_reason_codes(codes):
    return sorted(
        set(codes),
        key=utf8_key,
    )


# ============================================================
# Rejected row aggregation
# ============================================================

def aggregate_rejected_rows(records):
    by_id = {}

    for record in records:
        row_id = record["id"]

        if row_id not in by_id:
            by_id[row_id] = set()

        by_id[row_id].update(record["reasonCodes"])

    result = []

    for row_id, codes in by_id.items():
        result.append(
            {
                "id": row_id,
                "reasonCodes": sorted_reason_codes(codes),
            }
        )

    result.sort(
        key=lambda x: (
            utf8_key(x["id"]),
            utf8_key(compact_json(x)),
        )
    )

    return result


# ============================================================
# Object sorting
# ============================================================

def sort_rejected_objects(records):
    records.sort(
        key=lambda x: (
            utf8_key(x["uri"] or ""),
            utf8_key(compact_json(x)),
        )
    )

    return records


def sort_lineage(records):
    records.sort(
        key=lambda x: (
            utf8_key(x["uri"]),
            utf8_key(compact_json(x)),
        )
    )

    return records


# ============================================================
# Main corpus builder
# ============================================================

def build_corpus(request_data):

    # --------------------------------------------------------
    # Required top-level validation
    # --------------------------------------------------------

    if "policy" not in request_data:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(request_data["objects"], list):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    policy = request_data["policy"]
    objects = request_data["objects"]

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

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

    else:
        min_time = None
        max_time = None
        threshold = None

    policy_valid = (
        min_time is not None
        and max_time is not None
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and 0 <= threshold <= 1
        and min_time <= max_time
    )

    rejected_objects = []
    rejected_rows = []
    valid_objects = []

    # --------------------------------------------------------
    # Object validation
    # --------------------------------------------------------

    for obj in objects:

        if isinstance(obj, dict):
            supplied_uri = obj.get("uri")
            generation = obj.get("generation")
            fetched_generation = obj.get(
                "fetchedGeneration"
            )
            supplied_crc = obj.get("crc32c")
            schema_id = obj.get("schemaId")
            content = obj.get("content")

        else:
            supplied_uri = None
            generation = None
            fetched_generation = None
            supplied_crc = None
            schema_id = None
            content = None

        object_codes = []

        # URI
        if not valid_uri(supplied_uri):
            object_codes.append("URI_INVALID")

        # Generations
        generation_valid = valid_generation(
            generation
        )

        fetched_generation_valid = valid_generation(
            fetched_generation
        )

        if not generation_valid:
            object_codes.append("GENERATION_INVALID")

        if not fetched_generation_valid:
            object_codes.append("GENERATION_INVALID")

        # Mismatch means unequal supplied values.
        if generation != fetched_generation:
            object_codes.append("GENERATION_MISMATCH")

        # CRC syntax
        crc_valid = valid_crc(supplied_crc)

        if not crc_valid:
            object_codes.append("CRC32C_INVALID")

        # CRC mismatch is checked only when:
        # content is a string AND CRC syntax is valid.
        if (
            isinstance(content, str)
            and crc_valid
        ):
            calculated_crc = crc32c_hex(
                content.encode("utf-8")
            )

            if calculated_crc != supplied_crc:
                object_codes.append(
                    "CRC32C_MISMATCH"
                )

        # Schema
        if (
            not isinstance(content, str)
            or schema_id != "training-v1"
        ):
            object_codes.append("SCHEMA_INVALID")

        parsed_rows = []

        # JSONL/schema validation
        if isinstance(content, str):

            (
                parsed_rows,
                jsonl_invalid,
                schema_invalid,
            ) = parse_jsonl(content)

            if jsonl_invalid:
                object_codes.append(
                    "JSONL_INVALID"
                )

            if schema_invalid:
                object_codes.append(
                    "SCHEMA_INVALID"
                )

        # ----------------------------------------------------
        # Every parsed row must have a valid eventTime.
        #
        # There is no row-level INVALID_TIME code in the
        # specification, so an invalid eventTime makes the
        # object schema-invalid.
        # ----------------------------------------------------

        normalized_rows = []

        if parsed_rows:

            for row in parsed_rows:

                event_dt = parse_timestamp(
                    row["eventTime"]
                )

                if event_dt is None:
                    object_codes.append(
                        "SCHEMA_INVALID"
                    )
                    normalized_rows = []
                    break

                normalized_rows.append(
                    {
                        "id": row["id"],
                        "entity": canonicalize_text(
                            row["entity"]
                        ),
                        "eventTime": canonical_timestamp(
                            event_dt
                        ),
                        "revision": row["revision"],
                        "text": canonicalize_text(
                            row["text"]
                        ),
                    }
                )

        # ----------------------------------------------------
        # Object accepted/rejected
        # ----------------------------------------------------

        object_codes = sorted_reason_codes(
            object_codes
        )

        if object_codes:

            rejected_objects.append(
                {
                    "uri": (
                        supplied_uri
                        if isinstance(
                            supplied_uri,
                            str
                        )
                        else None
                    ),
                    "reasonCodes": object_codes,
                }
            )

            continue

        valid_objects.append(
            {
                "uri": supplied_uri,
                "generation": generation,
                "crc32c": supplied_crc,
                "schemaId": schema_id,
                "rows": normalized_rows,
            }
        )

    # --------------------------------------------------------
    # Lineage only for accepted objects
    # --------------------------------------------------------

    lineage = [
        {
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        }
        for obj in valid_objects
    ]

    # --------------------------------------------------------
    # Flatten rows
    # --------------------------------------------------------

    candidates = []

    for obj in valid_objects:

        for row in obj["rows"]:

            candidates.append(
                {
                    "uri": obj["uri"],
                    "generation": obj["generation"],
                    "crc32c": obj["crc32c"],
                    "schemaId": obj["schemaId"],
                    "row": row,
                }
            )

    # --------------------------------------------------------
    # Deduplication
    #
    # Key:
    #   entity,eventTime,text
    #
    # Winner:
    #   highest revision
    #   then smallest UTF-8 ID
    # --------------------------------------------------------

    groups = {}

    for candidate in candidates:

        row = candidate["row"]

        key = (
            row["entity"],
            row["eventTime"],
            row["text"],
        )

        groups.setdefault(
            key,
            [],
        ).append(candidate)

    retained = []

    for group in groups.values():

        highest_revision = max(
            item["row"]["revision"]
            for item in group
        )

        tied = [
            item
            for item in group
            if item["row"]["revision"]
            == highest_revision
        ]

        winner = min(
            tied,
            key=lambda item: utf8_key(
                item["row"]["id"]
            ),
        )

        retained.append(winner)

        for item in group:

            if item is not winner:

                rejected_rows.append(
                    {
                        "id": item["row"]["id"],
                        "reasonCodes": [
                            "DUPLICATE"
                        ],
                    }
                )

    # --------------------------------------------------------
    # Policy + window
    # --------------------------------------------------------

    eligible = []

    for candidate in retained:

        row = candidate["row"]
        row_codes = []

        if not policy_valid:

            row_codes.append(
                "POLICY_INVALID"
            )

        else:

            event_dt = parse_timestamp(
                row["eventTime"]
            )

            if (
                event_dt < min_time
                or event_dt > max_time
            ):
                row_codes.append(
                    "OUT_OF_WINDOW"
                )

        if row_codes:

            rejected_rows.append(
                {
                    "id": row["id"],
                    "reasonCodes":
                        sorted_reason_codes(
                            row_codes
                        ),
                }
            )

        else:

            eligible.append(candidate)

    # --------------------------------------------------------
    # Deterministic split
    #
    # SHA256(UTF8(entity))[0] % 10
    # --------------------------------------------------------

    split_candidates = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for candidate in eligible:

        entity = candidate["row"]["entity"]

        digest = hashlib.sha256(
            entity.encode("utf-8")
        ).digest()

        bucket = digest[0] % 10

        if bucket <= 5:
            split = "train"

        elif bucket <= 7:
            split = "validation"

        else:
            split = "test"

        split_candidates[split].append(
            candidate["row"]
        )

    # --------------------------------------------------------
    # Train contamination
    #
    # Word-set is calculated from canonicalized text.
    # --------------------------------------------------------

    train_word_sets = [
        unicode_word_set(row["text"])
        for row in split_candidates["train"]
    ]

    final_splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split in (
        "train",
        "validation",
        "test",
    ):

        for row in split_candidates[split]:

            if split == "train":

                final_splits["train"].append(
                    row
                )

                continue

            current_words = unicode_word_set(
                row["text"]
            )

            contaminated = False

            for train_words in train_word_sets:

                similarity = jaccard_similarity(
                    current_words,
                    train_words,
                )

                if similarity >= threshold:
                    contaminated = True
                    break

            if contaminated:

                rejected_rows.append(
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
    # Sort split rows
    #
    # UTF-8 ID, then compact JSON
    # --------------------------------------------------------

    for split in (
        "train",
        "validation",
        "test",
    ):

        final_splits[split].sort(
            key=lambda row: (
                utf8_key(row["id"]),
                utf8_key(
                    compact_json(row)
                ),
            )
        )

    # --------------------------------------------------------
    # Exact JSONL serialization + SHA256
    # --------------------------------------------------------

    digests = {}

    for split in (
        "train",
        "validation",
        "test",
    ):

        serialized = "".join(
            compact_json(row) + "\n"
            for row in final_splits[split]
        )

        serialized_bytes = serialized.encode(
            "utf-8"
        )

        digests[split] = hashlib.sha256(
            serialized_bytes
        ).hexdigest()

    # --------------------------------------------------------
    # Deterministic rejected rows
    # --------------------------------------------------------

    rejected_rows = aggregate_rejected_rows(
        rejected_rows
    )

    # --------------------------------------------------------
    # Deterministic rejected objects
    # --------------------------------------------------------

    rejected_objects = sort_rejected_objects(
        rejected_objects
    )

    # --------------------------------------------------------
    # Deterministic lineage
    # --------------------------------------------------------

    lineage = sort_lineage(lineage)

    # --------------------------------------------------------
    # Exact response shape
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

@app.post("/build-corpus")
async def build_corpus(request: Request):

    try:
        body = await request.json()

    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # Missing policy is explicitly INVALID_INPUT.
    if "policy" not in body:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # Non-array objects is explicitly INVALID_INPUT.
    if not isinstance(
        body.get("objects"),
        list,
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    return build_corpus(body)
