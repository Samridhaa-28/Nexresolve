
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY VOCABULARIES
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = ["version", "error_type", "platform", "hardware"]

# VERSION — regex patterns (ordered most specific first)
VERSION_PATTERNS = [
    r'\b(?:version|v)[:\s]+\d+[\d\.]+\w*',   # version: 1.2.3  or  v1.2.3
    r'\bcode\s+\d+[\d\.]+',                    # code 1.111.0 (VSCode pattern)
    r'\b(?:python|py|torch|cuda|pip)\s*\d+[\d\.]+',  # python 3.10, torch 2.1
    r'\b\d+\.\d+\.\d+\b',                      # strict x.y.z
    r'\b\d+\.\d+\b',                            # x.y (2-part, lower confidence)
]

# PLATFORM — closed vocabulary
PLATFORM_VOCAB = {
    "windows", "linux", "ubuntu", "macos", "mac os", "darwin",
    "wsl", "centos", "debian", "fedora", "android", "ios",
    "win10", "win11", "win7",
}

# HARDWARE — closed vocabulary
HARDWARE_VOCAB = {
    "cuda", "rocm", "mps", "gpu", "cpu", "npu", "tpu",
    "rtx", "gtx", "a100", "v100", "h100", "t4",
    "3090", "3080", "4090", "3050",
    "nvidia", "amd", "intel", "apple silicon", "m1", "m2", "m3",
}

# ERROR_TYPE — patterns + vocabulary
ERROR_PATTERNS = [
    r'\b\w+error\b',         # attributeerror, runtimeerror, valueerror
    r'\b\w+exception\b',     # keyexception, notimplementedexception
    r'\bsegfault\b',
    r'\bsegmentation fault\b',
    r'\boom\b',              # out of memory abbreviation
    r'\bout of memory\b',
    r'\btraceback\b',
    r'\bstack overflow\b',
    r'\bcore dump\b',
    r'\bkilled\b',           # process killed (OOM killer)
    r'\bassert\w*\b',        # assertion error variants
]

ERROR_VOCAB = {
    "attributeerror", "runtimeerror", "valueerror", "typeerror",
    "keyerror", "indexerror", "nameerror", "oserror", "ioerror",
    "memoryerror", "recursionerror", "stopiteration", "syntaxerror",
    "indentationerror", "unicodeerror", "unicodedecodeerror",
    "notimplementederror", "permissionerror", "filenotfounderror",
    "connectionerror", "timeouterror", "overflowerror", "zeroDivisionerror",
    "assertionerror", "importerror", "modulenotfounderror",
    "segfault", "oom", "traceback", "exception", "panic",
}

# GLiNER entity type descriptions (plain English labels)
GLINER_LABELS = [
    "software version number",
    "error or exception type",
    "operating system or platform",
    "GPU or hardware device",
]

GLINER_LABEL_MAP = {
    "software version number":    "version",
    "error or exception type":    "error_type",
    "operating system or platform": "platform",
    "GPU or hardware device":     "hardware",
}


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1 — REGEX EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_regex(text: str) -> dict[str, list[str]]:
    
    text_lower = text.lower()
    results = {e: [] for e in ENTITY_TYPES}

    # VERSION
    found_versions = set()
    for pat in VERSION_PATTERNS:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        for m in matches:
            m_clean = m.strip()
            if m_clean and m_clean not in found_versions:
                found_versions.add(m_clean)
                results["version"].append(m_clean)

    # PLATFORM
    for term in PLATFORM_VOCAB:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            results["platform"].append(term)

    # HARDWARE
    for term in HARDWARE_VOCAB:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            results["hardware"].append(term)

    # ERROR_TYPE — patterns first
    found_errors = set()
    for pat in ERROR_PATTERNS:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        for m in matches:
            m_clean = m.strip().lower()
            if m_clean and m_clean not in found_errors:
                found_errors.add(m_clean)
                results["error_type"].append(m_clean)
    # vocabulary lookup
    for term in ERROR_VOCAB:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower) and term not in found_errors:
            found_errors.add(term)
            results["error_type"].append(term)

    return results


def extract_regex_flags(text: str) -> dict[str, int]:
    """Return binary flags from regex extraction."""
    extracted = extract_regex(text)
    return {
        "has_version":    int(len(extracted["version"])    > 0),
        "has_error_type": int(len(extracted["error_type"]) > 0),
        "has_platform":   int(len(extracted["platform"])   > 0),
        "has_hardware":   int(len(extracted["hardware"])   > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2 — SPACY ENTITY RULER
# ─────────────────────────────────────────────────────────────────────────────

def build_spacy_pipeline():
    """
    Build a spaCy pipeline with a custom EntityRuler.

    Key advantage over regex: token-level matching captures multi-word
    entities like "python 3.10" (two tokens: LOWER=python + LIKE_NUM=true)
    and context-aware matches like "running on ubuntu".

    Returns spaCy nlp pipeline with EntityRuler added.
    """
    import spacy
    from spacy.pipeline import EntityRuler

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

    patterns = []

    # VERSION patterns — token level
    # "python 3.10" — LOWER:python + LIKE_NUM:true
    framework_names = [
        "python", "torch", "pytorch", "tensorflow", "tf",
        "cuda", "numpy", "pip", "conda", "node", "npm",
    ]
    for fw in framework_names:
        patterns.append({
            "label": "VERSION",
            "pattern": [{"LOWER": fw}, {"LIKE_NUM": True}]
        })
        patterns.append({
            "label": "VERSION",
            "pattern": [{"LOWER": fw}, {"TEXT": {"REGEX": r"\d+\.\d+[\d\.]*"}}]
        })

    # "version: 1.2.3" — context pattern
    patterns.append({
        "label": "VERSION",
        "pattern": [
            {"LOWER": {"IN": ["version", "v"]}},
            {"TEXT": {"REGEX": r"\d+[\d\.]+"}}
        ]
    })

    # standalone version numbers x.y.z
    patterns.append({
        "label": "VERSION",
        "pattern": [{"TEXT": {"REGEX": r"\b\d+\.\d+\.\d+\b"}}]
    })

    # PLATFORM — token patterns
    for term in PLATFORM_VOCAB:
        if " " in term:
            words = term.split()
            patterns.append({
                "label": "PLATFORM",
                "pattern": [{"LOWER": w} for w in words]
            })
        else:
            patterns.append({
                "label": "PLATFORM",
                "pattern": [{"LOWER": term}]
            })

    # HARDWARE — token patterns
    for term in HARDWARE_VOCAB:
        if " " in term:
            words = term.split()
            patterns.append({
                "label": "HARDWARE",
                "pattern": [{"LOWER": w} for w in words]
            })
        else:
            patterns.append({
                "label": "HARDWARE",
                "pattern": [{"LOWER": term}]
            })

    # ERROR_TYPE — token patterns for class names
    for term in ERROR_VOCAB:
        patterns.append({
            "label": "ERROR_TYPE",
            "pattern": [{"LOWER": term}]
        })

    # ERROR_TYPE — regex for XError / XException
    patterns.append({
        "label": "ERROR_TYPE",
        "pattern": [{"TEXT": {"REGEX": r"\w+error", "flags": re.IGNORECASE}}]
    })
    patterns.append({
        "label": "ERROR_TYPE",
        "pattern": [{"TEXT": {"REGEX": r"\w+exception", "flags": re.IGNORECASE}}]
    })

    ruler.add_patterns(patterns)
    return nlp


def extract_spacy(text: str, nlp) -> dict[str, list[str]]:
    """Extract entities using spaCy EntityRuler."""
    doc = nlp(text.lower())
    results = {e: [] for e in ENTITY_TYPES}
    label_map = {
        "VERSION":    "version",
        "PLATFORM":   "platform",
        "HARDWARE":   "hardware",
        "ERROR_TYPE": "error_type",
    }
    for ent in doc.ents:
        mapped = label_map.get(ent.label_)
        if mapped:
            results[mapped].append(ent.text.strip())
    return results


def extract_spacy_flags(text: str, nlp) -> dict[str, int]:
    """Return binary flags from spaCy extraction."""
    extracted = extract_spacy(text, nlp)
    return {
        "has_version":    int(len(extracted["version"])    > 0),
        "has_error_type": int(len(extracted["error_type"]) > 0),
        "has_platform":   int(len(extracted["platform"])   > 0),
        "has_hardware":   int(len(extracted["hardware"])   > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 3 — GLINER ZERO-SHOT NER
# ─────────────────────────────────────────────────────────────────────────────

def load_gliner(threshold: float = 0.5):
    
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return model


def extract_gliner(
    text: str,
    model,
    threshold: float = 0.5,
) -> dict[str, list[str]]:
    
    results = {e: [] for e in ENTITY_TYPES}

    # GLiNER works better with shorter texts — truncate at 512 chars
    text_input = text[:512] if len(text) > 512 else text

    try:
        entities = model.predict_entities(
            text_input,
            GLINER_LABELS,
            threshold=threshold,
        )
        for ent in entities:
            mapped = GLINER_LABEL_MAP.get(ent["label"])
            if mapped:
                results[mapped].append(ent["text"].strip())
    except Exception:
        pass  # Return empty if model fails on this text

    return results


def extract_gliner_flags(text: str, model, threshold: float = 0.5) -> dict[str, int]:
   
    extracted = extract_gliner(text, model, threshold)
    return {
        "has_version":    int(len(extracted["version"])    > 0),
        "has_error_type": int(len(extracted["error_type"]) > 0),
        "has_platform":   int(len(extracted["platform"])   > 0),
        "has_hardware":   int(len(extracted["hardware"])   > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID UNION ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_silver_labels(
    df: pd.DataFrame,
    gliner_model=None,
    save_path: str = "models/nlp/entity_silver_labels.pkl",
    gliner_threshold: float = 0.5,
) -> pd.DataFrame:
    
    # Check if already computed
    if os.path.exists(save_path):
        print(f"  Loading cached silver labels from {save_path}")
        with open(save_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  Building silver labels for {len(df)} issues ...")
    texts = df["clean_text"].fillna("").astype(str).tolist()

    # Regex labels
    print("  Running Regex annotation ...")
    regex_flags = []
    for text in texts:
        flags = extract_regex_flags(text)
        regex_flags.append(flags)

    # Second annotator: GLiNER preferred, spaCy as fallback
    # This ensures the union silver standard is always meaningful —
    # never just a copy of regex alone.
    second_flags = []

    if gliner_model is not None:
        print("  Running GLiNER annotation (second annotator) ...")
        for i, text in enumerate(texts):
            flags = extract_gliner_flags(text, gliner_model, gliner_threshold)
            second_flags.append(flags)
            if (i + 1) % 100 == 0:
                print(f"    GLiNER: {i+1}/{len(texts)}")
        print("  GLiNER annotation complete.")

    else:
        # GLiNER not available — use spaCy as second annotator
        # spaCy uses token-level patterns (different from char-level regex)
        # Union of regex + spaCy is a valid silver standard:
        #   regex catches structured template fields
        #   spaCy catches multi-token entities and token-boundary variants
        spacy_nlp_local = None
        try:
            spacy_nlp_local = build_spacy_pipeline()
            print("  Running spaCy annotation (second annotator, GLiNER unavailable) ...")
            for text in texts:
                flags = extract_spacy_flags(text, spacy_nlp_local)
                second_flags.append(flags)
            print("  spaCy annotation complete.")
        except Exception as e:
            print(f"  spaCy also unavailable ({e}) — using Regex-only silver.")
            print("  WARNING: silver = regex → Regex F1 will be 1.0 (not meaningful).")
            print("  Install spaCy or GLiNER for a valid silver standard.")
            second_flags = [{k: 0 for k in ["has_version","has_error_type",
                                              "has_platform","has_hardware"]}
                            for _ in texts]

    gliner_flags = second_flags  # kept as gliner_flags for union logic below

    # Hybrid union
    labels = []
    for r, g in zip(regex_flags, gliner_flags):
        union = {
            "has_version":    int(r["has_version"]    or g["has_version"]),
            "has_error_type": int(r["has_error_type"] or g["has_error_type"]),
            "has_platform":   int(r["has_platform"]   or g["has_platform"]),
            "has_hardware":   int(r["has_hardware"]   or g["has_hardware"]),
        }
        union["entity_count"] = sum(union.values())
        labels.append(union)

    silver_df = pd.DataFrame(labels)
    silver_df["issue_number"] = df["issue_number"].values

    # Save for reuse
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as fh:
        pickle.dump(silver_df, fh)

    # Print distribution
    print(f"\n  Silver label distribution ({len(silver_df)} issues):")
    for col in ["has_version", "has_error_type", "has_platform", "has_hardware"]:
        n = silver_df[col].sum()
        print(f"    {col:20s}: {n:4d} ({n/len(silver_df)*100:.1f}%)")

    print(f"  Saved → {save_path}")
    return silver_df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    entity_type: str,
) -> dict:
    """Compute precision, recall, F1 for a single entity type (binary)."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    if y_true.sum() == 0 and y_pred.sum() == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "accuracy": 1.0, "support": 0, "entity_type": entity_type}

    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tp  = int(((y_true == 1) & (y_pred == 1)).sum())
    fp  = int(((y_true == 0) & (y_pred == 1)).sum())
    fn  = int(((y_true == 1) & (y_pred == 0)).sum())
    tn  = int(((y_true == 0) & (y_pred == 0)).sum())

    return {
        "entity_type": entity_type,
        "precision":   round(pre, 4),
        "recall":      round(rec, 4),
        "f1":          round(f1,  4),
        "accuracy":    round(acc, 4),
        "support":     int(y_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_approach(
    approach_name: str,
    pred_flags: list[dict],
    silver_df: pd.DataFrame,
) -> dict:
    
    flag_cols = ["has_version", "has_error_type", "has_platform", "has_hardware"]
    entity_map = {
        "has_version":    "version",
        "has_error_type": "error_type",
        "has_platform":   "platform",
        "has_hardware":   "hardware",
    }

    per_entity = {}
    f1_scores  = []

    sep = "=" * 57
    print(f"\n{sep}")
    print(f"  {approach_name}")
    print(sep)
    print(f"  {'Entity':15s} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7} {'n_true':>7}")
    print("  " + "─" * 50)

    for col in flag_cols:
        etype  = entity_map[col]
        y_true = silver_df[col].values.astype(int)
        y_pred = np.array([int(f[col]) for f in pred_flags])
        m      = compute_metrics(y_true, y_pred, etype)
        per_entity[etype] = m
        f1_scores.append(m["f1"])

        bar = "█" * int(m["f1"] * 20)
        print(
            f"  {etype:15s} {m['precision']:>7.3f} {m['recall']:>7.3f} "
            f"{m['f1']:>7.3f} {m['accuracy']:>7.3f} {m['support']:>7}  |{bar}"
        )

    macro_f1 = round(float(np.mean(f1_scores)), 4)
    print(f"\n  Macro F1: {macro_f1:.4f}")

    return {
        "approach_name": approach_name,
        "per_entity":    per_entity,
        "macro_f1":      macro_f1,
        "macro_precision": round(float(np.mean([
            per_entity[e]["precision"] for e in entity_map.values()
        ])), 4),
        "macro_recall": round(float(np.mean([
            per_entity[e]["recall"] for e in entity_map.values()
        ])), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL APPROACHES
# ─────────────────────────────────────────────────────────────────────────────

def run_all_approaches(
    df: pd.DataFrame,
    silver_df: pd.DataFrame,
    save_dir: str = "models/nlp",
    gliner_model=None,
    spacy_nlp=None,
    split_name: str = "val",
) -> list[dict]:
    
    texts = df["clean_text"].fillna("").astype(str).tolist()
    results = []

    # ── Regex ─────────────────────────────────────────────────────────────
    print("\n─── Running Regex extraction ───")
    regex_flags = [extract_regex_flags(t) for t in texts]
    m_regex = evaluate_approach("Regex", regex_flags, silver_df)
    m_regex["split"] = split_name
    results.append(m_regex)

    # ── spaCy ─────────────────────────────────────────────────────────────
    if spacy_nlp is not None:
        print("\n─── Running spaCy extraction ───")
        spacy_flags = [extract_spacy_flags(t, spacy_nlp) for t in texts]
        m_spacy = evaluate_approach("spaCy", spacy_flags, silver_df)
        m_spacy["split"] = split_name
        results.append(m_spacy)
    else:
        print("\n  SKIP spaCy — not installed (pip install spacy && python -m spacy download en_core_web_sm)")

    # ── GLiNER ────────────────────────────────────────────────────────────
    if gliner_model is not None:
        print("\n─── Running GLiNER extraction ───")
        gliner_flags = []
        for i, t in enumerate(texts):
            gliner_flags.append(extract_gliner_flags(t, gliner_model))
            if (i + 1) % 50 == 0:
                print(f"  GLiNER: {i+1}/{len(texts)}")
        m_gliner = evaluate_approach("GLiNER", gliner_flags, silver_df)
        m_gliner["split"] = split_name
        results.append(m_gliner)
    else:
        print("\n  SKIP GLiNER — not installed (pip install gliner)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BEST APPROACH SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best(
    results: list[dict],
    save_dir: str = "models/nlp",
) -> dict:
   
    if not results:
        print("  No results to save.")
        return {}

    results_path = os.path.join(save_dir, "entity_results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            existing = json.load(fh)

    existing.setdefault("val_results", {})
    for r in results:
        existing["val_results"][r["approach_name"]] = r

    # Compare all known approaches
    all_results = list(existing["val_results"].values())
    best = max(all_results, key=lambda x: x.get("macro_f1", 0))

    # Print comparison
    print(f"\n{'=' * 57}")
    print("  ENTITY APPROACH COMPARISON  (Val Set)")
    print(f"{'=' * 57}")
    print(f"  {'Approach':<12} {'MacroP':>8} {'MacroR':>8} {'MacroF1':>9}")
    print("  " + "─" * 42)
    for r in sorted(all_results, key=lambda x: -x.get("macro_f1", 0)):
        marker = " ←" if r["approach_name"] == best["approach_name"] else ""
        print(
            f"  {r['approach_name']:<12}"
            f"{r.get('macro_precision',0):>9.4f}"
            f"{r.get('macro_recall',0):>9.4f}"
            f"{r.get('macro_f1',0):>10.4f}{marker}"
        )
    print(f"\n  ✓ Best: {best['approach_name']}  (macro_f1={best['macro_f1']:.4f})")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)

    summary = {
        "best_approach":    best["approach_name"],
        "selection_metric": "macro_f1",
        "selection_reason": (
            "Macro F1 averaged across all 4 entity types. "
            "F1 chosen over precision because missing an entity (FN) is worse "
            "than a false positive — incomplete RL state is harder to recover "
            "from than a slightly noisy state."
        ),
        "comparison": [
            {
                "approach":        r["approach_name"],
                "macro_f1":        r.get("macro_f1"),
                "macro_precision": r.get("macro_precision"),
                "macro_recall":    r.get("macro_recall"),
            }
            for r in all_results
        ],
        "entity_types":    ENTITY_TYPES,
        "annotation_strategy": (
            "Hybrid Union Silver Standard: "
            "has_entity=1 if Regex OR GLiNER detects entity. "
            "Covers structured template issues (Regex) and "
            "informal free-text issues (GLiNER)."
        ),
    }

    best_path = os.path.join(save_dir, "entity_best_model.json")
    with open(best_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"  Saved → {results_path}")
    print(f"  Saved → {best_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    save_dir: str = "models/nlp",
    gliner_model=None,
    spacy_nlp=None,
) -> dict:
    
    best_path = os.path.join(save_dir, "entity_best_model.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError("No entity_best_model.json. Run --mode train first.")

    with open(best_path, encoding="utf-8") as fh:
        best = json.load(fh)
    best_name = best["best_approach"]

    print(f"\n{'=' * 57}")
    print(f"  TEST SET  |  {best_name}")
    print(f"{'=' * 57}")

    test_df = pd.read_csv("data/splits/test.csv")

    # Build test silver labels
    test_silver = build_silver_labels(
        test_df, gliner_model,
        save_path=os.path.join(save_dir, "entity_silver_labels_test.pkl"),
    )

    texts = test_df["clean_text"].fillna("").astype(str).tolist()

    # Evaluate ALL available approaches on test set
    approaches_to_test = [("Regex", [extract_regex_flags(t) for t in texts])]

    if spacy_nlp is not None:
        approaches_to_test.append(
            ("spaCy", [extract_spacy_flags(t, spacy_nlp) for t in texts])
        )

    if gliner_model is not None:
        print("  Running GLiNER on test set ...")
        g_preds = []
        for i, t in enumerate(texts):
            g_preds.append(extract_gliner_flags(t, gliner_model))
            if (i + 1) % 50 == 0:
                print(f"    GLiNER: {i+1}/{len(texts)}")
        approaches_to_test.append(("GLiNER", g_preds))

    all_metrics   = []
    test_results  = {}
    for aname, pred_flags in approaches_to_test:
        m = evaluate_approach(aname + " [TEST]", pred_flags, test_silver)
        m["approach_name_clean"] = aname
        test_results[aname] = m
        all_metrics.append(m)

    best_test = max(all_metrics, key=lambda x: x.get("macro_f1", 0))
    print("\n  Best on test: "
          + best_test["approach_name_clean"]
          + "  (macro_f1=" + str(best_test["macro_f1"]) + ")")

    results_path = os.path.join(save_dir, "entity_results.json")
    with open(results_path, encoding="utf-8") as fh:
        existing = json.load(fh)
    existing["test_results"]     = best_test
    existing["test_results_all"] = test_results
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
    print("  Test results saved -> " + results_path)
    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES — write to CSVs
# ─────────────────────────────────────────────────────────────────────────────

def generate_features(
    save_dir: str    = "models/nlp",
    issues_path: str = "data/final/cleaned_issues.csv",
    rl_path: str     = "data/final/final_rl_dataset.csv",
    gliner_model=None,
    spacy_nlp=None,
) -> pd.DataFrame:
    
    best_path = os.path.join(save_dir, "entity_best_model.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError("No entity_best_model.json. Run --mode train first.")

    with open(best_path, encoding="utf-8") as fh:
        best = json.load(fh)
    best_name = best["best_approach"]

    ci = pd.read_csv(issues_path, encoding="utf-8")
    texts = ci["clean_text"].fillna("").astype(str).tolist()

    print(f"\n[generate_features] Best approach: {best_name}")
    print(f"  Processing {len(ci)} issues ...")

    # Extract entities
    extracted_lists = []
    flag_dicts      = []

    if best_name == "spaCy" and spacy_nlp is not None:
        extractor = lambda t: extract_spacy(t, spacy_nlp)
    elif best_name == "GLiNER" and gliner_model is not None:
        extractor = lambda t: extract_gliner(t, gliner_model)
    else:
        if best_name != "Regex":
            print(f"  {best_name} not available — falling back to Regex")
        extractor = extract_regex

    for i, text in enumerate(texts):
        extracted = extractor(text)
        extracted_lists.append(extracted)
        flag_dicts.append({
            "has_version":    int(len(extracted["version"])    > 0),
            "has_error_type": int(len(extracted["error_type"]) > 0),
            "has_platform":   int(len(extracted["platform"])   > 0),
            "has_hardware":   int(len(extracted["hardware"])   > 0),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(texts)}")

    # Add entity string columns to cleaned_issues.csv (for inspection)
    flag_cols = ["has_version", "has_error_type", "has_platform", "has_hardware"]
    str_cols  = ["extracted_version", "extracted_error", "extracted_platform", "extracted_hardware"]

    for col in flag_cols + str_cols + ["entity_count"]:
        if col in ci.columns:
            ci = ci.drop(columns=[col])

    ci["extracted_version"]  = [json.dumps(e["version"])    for e in extracted_lists]
    ci["extracted_error"]    = [json.dumps(e["error_type"]) for e in extracted_lists]
    ci["extracted_platform"] = [json.dumps(e["platform"])   for e in extracted_lists]
    ci["extracted_hardware"] = [json.dumps(e["hardware"])   for e in extracted_lists]
    ci["has_version"]    = [f["has_version"]    for f in flag_dicts]
    ci["has_error_type"] = [f["has_error_type"] for f in flag_dicts]
    ci["has_platform"]   = [f["has_platform"]   for f in flag_dicts]
    ci["has_hardware"]   = [f["has_hardware"]   for f in flag_dicts]
    ci["entity_count"]   = ci[flag_cols].sum(axis=1)

    ci.to_csv(issues_path, index=False, encoding="utf-8")
    print(f"  Updated → {issues_path}")

    # Add binary flags only to final_rl_dataset.csv
    rl = pd.read_csv(rl_path, encoding="utf-8")
    for col in flag_cols + ["entity_count"]:
        if col in rl.columns:
            rl = rl.drop(columns=[col])

    merge_data = ci[["issue_number"] + flag_cols + ["entity_count"]]
    rl = rl.merge(merge_data, on="issue_number", how="left")
    rl.to_csv(rl_path, index=False, encoding="utf-8")
    print(f"  Updated → {rl_path}")

    # Summary
    print(f"\n  Entity presence across {len(ci)} issues:")
    for col in flag_cols:
        n = ci[col].sum()
        print(f"    {col:20s}: {n:4d} ({n/len(ci)*100:.1f}%)")
    print(f"    entity_count=0: {(ci['entity_count']==0).sum()} issues (no entities found)")
    print(f"    entity_count=4: {(ci['entity_count']==4).sum()} issues (all 4 present)")

    return ci[["issue_number"] + flag_cols + ["entity_count"]]


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — single prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    text: str,
    save_dir: str = "models/nlp",
    spacy_nlp=None,
    gliner_model=None,
) -> dict:
    
    best_path = os.path.join(save_dir, "entity_best_model.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            "No entity_best_model.json. "
            "Run: python nlp/entity_extractor.py --mode train"
        )

    with open(best_path, encoding="utf-8") as fh:
        best = json.load(fh)
    best_name = best["best_approach"]

    if best_name == "spaCy" and spacy_nlp is not None:
        extracted = extract_spacy(text, spacy_nlp)
    elif best_name == "GLiNER" and gliner_model is not None:
        extracted = extract_gliner(text, gliner_model)
    else:
        extracted = extract_regex(text)
        best_name = "Regex (fallback)"

    flags = {
        "has_version":    int(len(extracted["version"])    > 0),
        "has_error_type": int(len(extracted["error_type"]) > 0),
        "has_platform":   int(len(extracted["platform"])   > 0),
        "has_hardware":   int(len(extracted["hardware"])   > 0),
    }
    flags["entity_count"] = sum(flags.values())

    return {
        "entities":    extracted,
        "flags":       flags,
        "approach":    best_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NexResolve Entity Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="train",
                        choices=["annotate", "train", "test", "generate", "predict"])
    parser.add_argument("--text",              default=None)
    parser.add_argument("--save_dir",          default="models/nlp")
    parser.add_argument("--gliner_threshold",  type=float, default=0.5)
    parser.add_argument("--no_gliner",         action="store_true",
                        help="Skip GLiNER (use if not installed yet)")
    parser.add_argument("--no_spacy",          action="store_true",
                        help="Skip spaCy (use if not installed yet)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────
    spacy_nlp    = None
    gliner_model = None

    if not args.no_spacy:
        try:
            spacy_nlp = build_spacy_pipeline()
            print("  spaCy pipeline loaded.")
        except Exception as e:
            print(f"  spaCy not available: {e}")
            print("  Install: pip install spacy && python -m spacy download en_core_web_sm")

    if not args.no_gliner:
        try:
            print("  Loading GLiNER model (downloads ~170MB on first run) ...")
            gliner_model = load_gliner(args.gliner_threshold)
            print("  GLiNER loaded.")
        except Exception as e:
            print(f"  GLiNER not available: {e}")
            print("  Install: pip install gliner")

    # ── Modes ────────────────────────────────────────────────────────────
    if args.mode == "predict":
        text = args.text or input("Issue text: ").strip()
        result = predict(text, args.save_dir, spacy_nlp, gliner_model)
        print(json.dumps(result, indent=2))
        return

    if args.mode == "annotate":
        # Build silver labels on full dataset (runs once, cached)
        ci = pd.read_csv("data/final/cleaned_issues.csv")
        silver = build_silver_labels(
            ci, gliner_model,
            save_path=os.path.join(args.save_dir, "entity_silver_labels_full.pkl"),
            gliner_threshold=args.gliner_threshold,
        )
        print(f"\n✓ Silver labels built and saved.")
        return

    if args.mode == "train":
        print("Loading val set ...")
        val_df = pd.read_csv("data/splits/val.csv")
        print(f"  Val: {len(val_df)} issues")

        # Build silver labels on val set
        print("\nBuilding silver labels ...")
        silver = build_silver_labels(
            val_df, gliner_model,
            save_path=os.path.join(args.save_dir, "entity_silver_labels_val.pkl"),
            gliner_threshold=args.gliner_threshold,
        )

        # Run all approaches
        results = run_all_approaches(
            val_df, silver, args.save_dir,
            gliner_model, spacy_nlp, split_name="val",
        )

        # Select best
        select_best(results, args.save_dir)

        print("\n✓ Training complete.")
        print("  Next steps:")
        print("    python nlp/entity_extractor.py --mode test")
        print("    python evaluation/entity_eval.py --verbose")
        print("    python nlp/entity_extractor.py --mode generate")
        print("    jupyter notebook notebooks/05_entity_extraction.ipynb")
        return

    if args.mode == "test":
        evaluate_on_test(args.save_dir, gliner_model, spacy_nlp)
        return

    if args.mode == "generate":
        generate_features(args.save_dir,
                          gliner_model=gliner_model, spacy_nlp=spacy_nlp)
        print("\n✓ Entity features written to CSVs.")
        return


if __name__ == "__main__":
    main()