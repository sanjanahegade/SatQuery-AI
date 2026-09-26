"""Normalize Optical VQA questions before they are sent to GeoChat."""

import re


_TRAILING_IMAGE = re.compile(
    r"[\s,]*\bin (?:the|this) images?\b[.?!]*$",
    re.IGNORECASE,
)


_LEADING_FILLER = re.compile(
    r"^(?:please\s+)?(?:look directly at the image[.!]?\s*)?"
    r"(?:are\s+there|is\s+there|is|are|which(?:\s+one)?|what)\s+",
    re.IGNORECASE,
)


_COMPARISON_PATTERNS = (
    re.compile(
        r"\b(?:is|are)\s+(?P<a>.+?)\s+"
        r"(?:more|greater|larger|bigger|higher)\s+or\s+"
        r"(?:more|greater|larger|bigger|higher)?\s*"
        r"(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<a>.+?)\s+or\s+(?P<b>.+?)\s+which\s+is\s+"
        r"(?:more|greater|larger|bigger|higher)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^which\s+is\s+bigger\s+(?P<a>.+?)\s+or\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^which\s+(?:covers|occupies)\s+more\s+area\s+"
        r"(?P<a>.+?)\s+or\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmore\s+(?P<a>.+?)\s+or\s+(?:more\s+)?(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:more|greater|larger|higher)\s+"
        r"(?P<a>.+?)\s+than\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"which\s+occupies\s+more[^:]*:\s*"
        r"(?P<a>.+?)\s+or\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"which\s+is\s+(?:more|greater|larger|higher)[^:]*:\s*"
        r"(?P<a>.+?)\s+or\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"compare(?:\s+the)?"
        r"(?:\s+visible\s+area(?:\s+covered\s+by)?)?\s+"
        r"(?P<a>.+?)\s+and\s+(?P<b>.+?)"
        r"(?:[.?!].*)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"which\s+is\s+(?:more|greater|larger|higher)\b"
        r".*?\b(?P<a>.+?)\s+or\s+(?P<b>.+?)$",
        re.IGNORECASE,
    ),
)


def _strip_trailing_image_phrase(text):
    return _TRAILING_IMAGE.sub("", text).strip(
        " \t.?!,:;"
    )


def _clean_entity(text):
    text = (text or "").strip(
        " \t.?!,:;\"'"
    )

    text = _strip_trailing_image_phrase(text)

    text = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip(
        " \t.?!,:;\"'"
    )

    return text


def extract_comparison_entities(question):
    """Return (entity_a, entity_b) for supported two-way comparisons."""

    if not question or not str(question).strip():
        return None

    cleaned = _strip_trailing_image_phrase(
        str(question).strip()
    )

    cleaned = cleaned.rstrip(
        " .?!,:;"
    )

    if re.fullmatch(
        r"which\s+side\s+is\s+"
        r"(?:more|greater|larger|bigger|higher)",
        cleaned,
        re.IGNORECASE,
    ):
        return "first side", "second side"

    if re.fullmatch(
        r"are\s+both\s+(?:the\s+)?same",
        cleaned,
        re.IGNORECASE,
    ):
        return "first entity", "second entity"

    match = re.fullmatch(
        r"is\s+(?P<a>.+?)\s+more\s+or\s+(?P<b>.+)",
        cleaned,
        re.IGNORECASE,
    )

    if match:
        entity_a = _clean_entity(
            match.group("a")
        )

        entity_b = _clean_entity(
            match.group("b")
        )

        if entity_a and entity_b:
            return entity_a, entity_b

    first_sentence, _, rest = cleaned.partition(". ")

    if rest.lower().startswith("which "):
        cleaned = first_sentence

    candidates = [cleaned]

    stripped = _LEADING_FILLER.sub(
        "",
        cleaned,
    ).strip(
        " .?!,:;"
    )

    if stripped and stripped.lower() != cleaned.lower():
        candidates.append(stripped)

    for text in candidates:

        for pattern in _COMPARISON_PATTERNS:

            match = pattern.search(text)

            if not match:
                continue

            entity_a = _clean_entity(
                match.group("a")
            )

            entity_b = _clean_entity(
                match.group("b")
            )

            if not entity_a or not entity_b:
                continue

            if entity_a.lower() == entity_b.lower():
                continue

            if (
                " or " in entity_a.lower()
                or " and " in entity_a.lower()
            ):
                continue

            return entity_a, entity_b

    return None


def _is_infrastructure_question(question):
    """Detect questions asking about infrastructure."""

    text = (question or "").lower()

    infrastructure_terms = (
        "infrastructure",
        "infrastructures",
        "road",
        "roads",
        "street",
        "streets",
        "highway",
        "transportation",
        "parking",
        "buildings",
        "building",
        "facilities",
        "structures around",
        "infrastructure around",
    )

    return any(
        term in text
        for term in infrastructure_terms
    )


def normalize_prompt(question):
    """Return an evidence-grounded GeoChat question string."""

    if question is None:
        return question

    original = str(question).strip()

    if not original:
        return original

    # Comparison questions
    entities = extract_comparison_entities(
        original
    )

    if entities is not None:

        entity_a, entity_b = entities

        return (
            "You are an evidence-grounded satellite "
            "image analyst. Look directly at the image "
            "and compare only the visible areas of the "
            "two requested entities. "
            f"Which occupies more visible area: "
            f"{entity_a} or {entity_b}? "
            f"Answer with only one of: "
            f"{entity_a}, {entity_b}, or approximately equal. "
            "Do not invent objects or infer unsupported "
            "details."
        )

    lower_q = original.lower()

    # Challenge questions
    challenge_instruction = ""

    if any(
        phrase in lower_q
        for phrase in (
            "why did you say",
            "that is incorrect",
            "there is no",
            "are you sure",
            "is that correct",
        )
    ):
        challenge_instruction = (
            " The user is challenging an earlier claim. "
            "Do not apologize or agree automatically. "
            "Re-examine the image carefully and state "
            "the strongest visible evidence supporting "
            "or contradicting the claim."
        )

    # Infrastructure questions
    if _is_infrastructure_question(original):

        return (
            "You are an evidence-grounded satellite image "
            "analyst. Carefully inspect the entire image "
            "before answering. "
            "The user is asking about visible infrastructure "
            "or infrastructure-related objects. "
            "Explicitly check for roads, streets, highways, "
            "parking areas, buildings, sports facilities, "
            "large structures, and other clearly visible "
            "constructed features. "
            "A road may be visible even if it is small, "
            "partially obscured, curved, or surrounding "
            "the main feature. "
            "Do not say that a feature is absent unless "
            "you have carefully checked the relevant image "
            "regions. "
            "Do not invent infrastructure that is not "
            "visibly supported. "
            "Distinguish clearly between what is directly "
            "visible and what is inferred. "
            "If a feature is ambiguous, say that it is "
            "uncertain rather than claiming it is absent. "
            f"Question: {original}"
            + challenge_instruction
        )

    # Normal VQA
    return (
        "You are an evidence-grounded satellite image "
        "analyst. Inspect the image carefully and answer "
        "the user's latest question. "
        "Use only details that are visibly supported by "
        "the image, including readable map labels or "
        "other visible text. "
        "Separate direct observations from inferences. "
        "Do not invent landmarks, objects, locations, "
        "scene types, or text. "
        "For location questions, cite visible labels "
        "or say that the image does not provide enough "
        "evidence for a reliable location. "
        "If the question challenges an earlier answer, "
        "reassess the image and explain which visible "
        "evidence supports or contradicts it. "
        "Do not apologize or reverse the answer "
        "automatically. "
        "If evidence is weak or ambiguous, say so clearly. "
        "Be concise and do not repeat these instructions. "
        + challenge_instruction
        + f" Question: {original}"
    )