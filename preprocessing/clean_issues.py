"""
NexResolve — Step 1: Clean Issues
Input : data/raw/issues.csv
Output: data/intermediate/cleaned_issues.csv

Run: python preprocessing/clean_issues.py
"""

import re
import ast
import pandas as pd
from html.parser import HTMLParser

# ─────────────────────────────────────────────
# HTML stripper
# ─────────────────────────────────────────────
class _MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return " ".join(self.fed)

def strip_html(text: str) -> str:
    s = _MLStripper()
    s.feed(str(text))
    return s.get_data()


# ─────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Full text cleaning pipeline for title/body fields."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Strip HTML tags
    text = strip_html(text)

    # 2. Remove markdown-style code blocks (keep surrounding context)
    text = re.sub(r"```[\s\S]*?```", " [CODE_BLOCK] ", text)
    text = re.sub(r"`[^`]+`", " [INLINE_CODE] ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+", " [URL] ", text)

    # 4. Remove GitHub @mentions, issue refs (#123)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\d+", "", text)

    # 5. Remove markdown table rows  (|---|---|)
    text = re.sub(r"\|[-:]+\|[-:| ]+", " ", text)

    # 6. Remove markdown headers/bullets/bold/italic
    text = re.sub(r"[#*_~`>]", " ", text)

    # 7. Remove hex hashes / long commit-like tokens
    text = re.sub(r"\b[0-9a-f]{10,}\b", " [HASH] ", text)

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Lowercase
    text = text.lower()

    return text


# ─────────────────────────────────────────────
# Label normalisation
# ─────────────────────────────────────────────
# Map raw GitHub labels → canonical intent categories
_LABEL_MAP = {
    "bug":            "bug",
    "*bug":           "bug",
    "question":       "question",
    "*question":      "question",
    "duplicate":      "duplicate",
    "*duplicate":     "duplicate",
    "enhancement":    "enhancement",
    "invalid":        "invalid",
    "spam":           "invalid",
    "wontfix":        "wontfix",
    "help wanted":    "help_wanted",
    "triage-needed":  "triage",
    "info-needed":    "info_needed",
    "insiders-released": "released",
    "ai-translated":  "ai_translated",
    "new release":    "released",
    "chat-billing":   "billing",
    "error-list":     "error",
}

def _parse_labels(raw) -> list:
    """Parse comma-string or stringified-list → Python list."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    raw = str(raw).strip()
    # handle stringified list: "['bug', 'duplicate']"
    if raw.startswith("["):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return [l.strip() for l in raw.split(",") if l.strip()]

def normalise_labels(raw) -> str:
    """Return pipe-separated canonical labels, drop stars/noise."""
    labels = _parse_labels(raw)
    normalised = []
    for lbl in labels:
        lbl_clean = lbl.lstrip("*").strip().lower()
        mapped = _LABEL_MAP.get(lbl_clean, lbl_clean)  # keep as-is if not in map
        if mapped and mapped not in normalised:
            normalised.append(mapped)
    return "|".join(normalised) if normalised else "unlabelled"

def primary_label(normalised: str) -> str:
    """Return the most informative single label for downstream modelling."""
    priority = ["bug", "question", "enhancement", "error", "billing",
                "help_wanted", "triage", "info_needed", "invalid",
                "duplicate", "wontfix", "released", "ai_translated", "unlabelled"]
    lbls = normalised.split("|")
    for p in priority:
        if p in lbls:
            return p
    return lbls[0] if lbls else "unlabelled"


# ─────────────────────────────────────────────
# NLP-derived flags (improved regex-based)
# ─────────────────────────────────────────────
#
# missing_version_flag = 1 → NO version info found  → RL should ask clarification
# missing_version_flag = 0 → version IS present     → no clarification needed
#
# missing_error_flag   = 1 → NO error/exception found → may need clarification
# missing_error_flag   = 0 → error code/trace IS present
#
# Detection layers for VERSION (applied in order):
#   Layer 1 — explicit version numbers   e.g.  1.2.3  v4.5  0.33.5  10.0.26200
#   Layer 2 — labelled field with value  e.g.  "vs code version: 1.111"
#   Layer 3 — blank version templates    e.g.  "vs code version: -"  → still MISSING
#   Layer 4 — release/update keywords    e.g.  "latest update" "insiders release"
#             (weak signal — treated as version-present, lower confidence)

# Layer 1: version number  e.g. 1.2.3  v4.5  10.0.26200
_VERSION_NUMBER_RE = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)*\b",
    re.IGNORECASE
)

# Layer 2: version field with real value  e.g. "version: 1.2"
_VERSION_FIELD_RE = re.compile(
    r"(?:version|vs code version|extension version|os version)[:\s]+v?[\d]+\.[\d]",
    re.IGNORECASE
)

# Layer 3: blank template — user left version empty  e.g. "vs code version: -"
_VERSION_BLANK_RE = re.compile(
    r"(?:version|vs code version|extension version|os version)[:\s]*(?:-|n/?a|unknown|\?)\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Layer 4: release/update keywords — weak version signal
_VERSION_KEYWORD_RE = re.compile(
    r"\b(?:latest update|insiders release|insiders version|after (?:the )?update|"
    r"recent update|new release|after (?:the )?upgrade)\b",
    re.IGNORECASE
)

# Error patterns — expanded to catch real error formats from GitHub issues
_ERROR_RE = re.compile(
    r"\b(?:"
    r"error[:\s][\w\d]+|"           # error: SomeError  or  error 404
    r"exception[:\s][\w\d]+|"       # exception: NullPointerException
    r"0x[0-9a-f]{4,}|"              # hex codes  0x80070005
    r"errno\s*\d+|"                 # errno 13
    r"traceback|"                   # Python traceback
    r"stack trace|"                 # Java/JS stack trace
    r"exit code \d+|"               # exit code 1
    r"failed with|"                 # failed with error
    r"throws?|"                     # throws / throw
    r"crash(?:ed|es)?|"             # crashed / crashes
    r"segfault|"                    # segmentation fault
    r"core dump"                    # core dump
    r")",
    re.IGNORECASE
)


def has_version(text: str) -> int:
    """
    Returns 1 if version information is genuinely present, 0 if missing.
    Blank templates like 'version: -' are treated as MISSING (returns 0).
    missing_version_flag = 1 - has_version(text)
    """
    if not isinstance(text, str):
        return 0

    # Layer 3 check first: blank template with no version number anywhere
    if _VERSION_BLANK_RE.search(text) and not _VERSION_NUMBER_RE.search(text):
        return 0  # template exists but is empty → still missing

    # Layer 1: explicit version number present
    if _VERSION_NUMBER_RE.search(text):
        return 1

    # Layer 2: labelled field with real value
    if _VERSION_FIELD_RE.search(text):
        return 1

    # Layer 4: weak keyword signal
    if _VERSION_KEYWORD_RE.search(text):
        return 1

    return 0  # nothing found → version is missing


def has_error(text: str) -> int:
    """Returns 1 if any error/exception signal is present, 0 otherwise."""
    if not isinstance(text, str):
        return 0
    return int(bool(_ERROR_RE.search(text)))



# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run(input_path: str = "data/raw/issues.csv",
        output_path: str = "data/intermediate/cleaned_issues.csv"):

    print(f"[clean_issues] Loading {input_path} ...")
    df = pd.read_csv(input_path)
    original_shape = df.shape
    print(f"  Rows: {original_shape[0]}  Cols: {original_shape[1]}")

    # ── 1. Drop exact duplicates ────────────────────────────────────────
    df = df.drop_duplicates(subset=["issue_id"])
    print(f"  After dedup: {len(df)} rows")

    # ── 2. Remove bot-authored issues ──────────────────────────────────
    df = df[df["author_type"].str.lower() != "bot"]
    print(f"  After removing bot issues: {len(df)} rows")

    # ── 3. Keep only closed issues ─────────────────────────────────────
    df = df[df["state"] == "closed"]
    print(f"  Closed issues: {len(df)} rows")

    # ── 4. Fix missing body  ────────────────────────────────────────────
    df["body"] = df["body"].fillna("")

    # ── 5. Clean text fields ────────────────────────────────────────────
    print("  Cleaning title & body text ...")
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_body"]  = df["body"].apply(clean_text)
    df["clean_text"]  = (df["clean_title"] + " " + df["clean_body"]).str.strip()

    # ── 6. Normalise labels ─────────────────────────────────────────────
    df["labels_normalised"] = df["labels"].apply(normalise_labels)
    df["primary_label"]     = df["labels_normalised"].apply(primary_label)

    # ── 7. Fix timestamps ───────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["closed_at"]  = pd.to_datetime(df["closed_at"],  errors="coerce", utc=True)

    # ── 8. Resolution time (re-compute to guard against raw errors) ─────
    df["resolution_time_hours"] = (
        (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600
    ).round(2)
    df["resolution_time_hours"] = df["resolution_time_hours"].clip(lower=0)

    # ── 9. NLP flags (entity hints) ─────────────────────────────────────
    df["missing_version_flag"] = df["clean_text"].apply(lambda t: 1 - has_version(t))
    df["missing_error_flag"]   = df["clean_text"].apply(lambda t: 1 - has_error(t))

    # ── 10. has_solution_comment placeholder (filled in merge step) ─────
    df["has_solution_comment"] = 0

    # ── 11. Assignee fill ───────────────────────────────────────────────
    df["assignee_login"] = df["assignee_login"].fillna("unassigned")

    # ── 12. Select & order output columns ───────────────────────────────
    out_cols = [
        "repo", "issue_id", "issue_number",
        "clean_title", "clean_body", "clean_text",
        "labels_normalised", "primary_label",
        "state", "created_at", "closed_at", "resolution_time_hours",
        "comments_count", "author_login", "author_type", "assignee_login",
        "missing_version_flag", "missing_error_flag", "has_solution_comment",
    ]
    df = df[out_cols]

    df.to_csv(output_path, index=False)
    print(f"\n[clean_issues] Saved {len(df)} rows → {output_path}")
    print(f"  primary_label distribution:\n{df['primary_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/issues.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/intermediate/cleaned_issues.csv"
    run(inp, out)