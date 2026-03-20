"""
NexResolve — Step 2: Clean Comments
Input : data/raw/comments.csv
Output: data/intermediate/cleaned_comments.csv
        data/intermediate/aggregated_comments.csv  (one row per issue)

Run: python preprocessing/clean_comments.py
"""

import re
import pandas as pd
from html.parser import HTMLParser


# ─────────────────────────────────────────────
# Reuse HTML stripper (same as clean_issues)
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


def clean_comment(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = strip_html(text)
    text = re.sub(r"```[\s\S]*?```", " [CODE_BLOCK] ", text)
    text = re.sub(r"`[^`]+`", " [INLINE_CODE] ", text)
    text = re.sub(r"https?://\S+", " [URL] ", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"\|[-:]+\|[-:| ]+", " ", text)
    text = re.sub(r"[#*_~`>]", " ", text)
    text = re.sub(r"\b[0-9a-f]{10,}\b", " [HASH] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ─────────────────────────────────────────────
# Bot detection heuristics
# ─────────────────────────────────────────────
_BOT_KEYWORDS = [
    "bot", "[bot]", "github-actions", "dependabot",
    "codecov", "stale[bot]", "auto-close",
]

def is_bot_author(login: str) -> bool:
    if not isinstance(login, str):
        return False
    login_lower = login.lower()
    return any(kw in login_lower for kw in _BOT_KEYWORDS)


# ─────────────────────────────────────────────
# Solution-comment detection
# ─────────────────────────────────────────────
_SOLUTION_PATTERNS = re.compile(
    r"\b(fix(?:ed|es)?|resolv(?:ed|es)?|solv(?:ed|es)?|workaround|close[sd]?|"
    r"the\s+issue\s+is|should\s+work|try\s+this|solution\s+is|answer\s+is)\b",
    re.IGNORECASE,
)

def is_solution_comment(text: str) -> int:
    return int(bool(_SOLUTION_PATTERNS.search(str(text))))


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run(input_path: str = "data/raw/comments.csv",
        issues_path: str = "data/intermediate/cleaned_issues.csv",
        output_path: str = "data/intermediate/cleaned_comments.csv",
        agg_path: str    = "data/intermediate/aggregated_comments.csv"):

    print(f"[clean_comments] Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(df)}")

    # ── 1. Drop exact duplicates ────────────────────────────────────────
    df = df.drop_duplicates(subset=["comment_id"])

    # ── 2. Remove bot comments ──────────────────────────────────────────
    df["is_bot"] = df["comment_author_login"].apply(is_bot_author)
    df = df[~df["is_bot"]].copy()
    print(f"  After removing bots: {len(df)} rows")

    # ── 3. Remove empty bodies ──────────────────────────────────────────
    df["comment_body"] = df["comment_body"].fillna("").astype(str)
    df = df[df["comment_body"].str.strip() != ""]
    print(f"  After removing empty: {len(df)} rows")

    # ── 4. Clean text ────────────────────────────────────────────────────
    df["clean_comment"] = df["comment_body"].apply(clean_comment)

    # ── 5. Tag solution comments ─────────────────────────────────────────
    df["is_solution_comment"] = df["clean_comment"].apply(is_solution_comment)

    # ── 6. Parse timestamps ──────────────────────────────────────────────
    df["comment_created_at"] = pd.to_datetime(
        df["comment_created_at"], errors="coerce", utc=True
    )

    # ── 7. Select output columns ─────────────────────────────────────────
    out_cols = [
        "repo", "issue_number", "comment_id",
        "clean_comment", "comment_created_at",
        "comment_author_login", "comment_author_type", "author_association",
        "is_solution_comment",
    ]
    df = df[out_cols]
    df.to_csv(output_path, index=False)
    print(f"[clean_comments] Saved cleaned comments → {output_path}")

    # ─────────────────────────────────────────────────────────────────────
    # AGGREGATION: one row per issue
    # ─────────────────────────────────────────────────────────────────────
    agg = df.groupby("issue_number").agg(
        comment_count        = ("comment_id",         "count"),
        has_solution_comment = ("is_solution_comment","max"),
        all_comments_text    = ("clean_comment",      lambda x: " |SEP| ".join(x.dropna())),
        solution_comments    = ("clean_comment",      lambda x: " |SEP| ".join(
                                    x[df.loc[x.index, "is_solution_comment"] == 1].dropna())),
        unique_commenters    = ("comment_author_login","nunique"),
    ).reset_index()

    # ── Back-fill has_solution_comment into cleaned_issues ──────────────
    try:
        ci = pd.read_csv(issues_path)
        sol_map = agg.set_index("issue_number")["has_solution_comment"].to_dict()
        ci["has_solution_comment"] = ci["issue_number"].map(sol_map).fillna(0).astype(int)
        ci.to_csv(issues_path, index=False)
        print(f"  Updated has_solution_comment in {issues_path}")
    except FileNotFoundError:
        print(f"  (Skipped back-fill: {issues_path} not found yet)")

    agg.to_csv(agg_path, index=False)
    print(f"[clean_comments] Saved aggregated comments → {agg_path}  ({len(agg)} issues)")
    return df, agg


if __name__ == "__main__":
    import sys
    run(
        input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/comments.csv",
        issues_path = sys.argv[2] if len(sys.argv) > 2 else "data/intermediate/cleaned_issues.csv",
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/intermediate/cleaned_comments.csv",
        agg_path    = sys.argv[4] if len(sys.argv) > 4 else "data/intermediate/aggregated_comments.csv",
    )