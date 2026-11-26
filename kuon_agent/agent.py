"""The Kuon Agent loop: gather evidence -> think -> answer with citations.

Design goals, in order:
1. Grounded: answer ONLY from collected evidence; admit ignorance otherwise.
2. Inspectable: return the evidence, the notes, and the final answer.
3. Swappable: any engine (Kuon, Ollama/Qwen, llama.cpp) works unchanged.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Callable

from kuon.chat import format_prompt
from kuon_agent.rag import BM25Index
from kuon_agent.search import Source, web_search, wikipedia_search
from kuon_agent.tools import (
    FactLedger,
    business_kpi,
    extract_grounded_numbers,
    parse_amount,
    parse_table,
    period_aliases,
    safe_calc,
    summarize_table,
    verify_grounded_answer,
)


_TOOL_CALL_RE = re.compile(
    r"^\s*TOOL:\s*(calc|kpi)\(([^)\n]{1,220})\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ID_WORDS = re.compile(
    r"\b(?:apa|apakah|siapa|kapan|dimana|mengapa|kenapa|bagaimana|berapa|berikut|sekarang|adalah|"
    r"dengan|untuk|tidak|bisa|sudah|belum|saya|anda|kamu|biaya|bulan|dari|harga|hitung|hpp|kami|"
    r"ke|kotor|minggu|omzet|omset|pendapatan|persen|pesanan|turun|yang)\b",
    re.I,
)
_CHANGE_TERMS = re.compile(
    r"\b(?:perubahan|selisih|turun|naik|penurunan|kenaikan|change[ds]?|increase[ds]?|decrease[ds]?|fell|drop(?:ped)?|rise|rose|growth)\b",
    re.I,
)
_SALES_CHANGE_REQUEST = re.compile(
    r"(?:"
    r"(?:berapa|hitung(?:kan)?|cari|tentukan)?\s*(?:persen(?:tase)?\s+)?(?:perubahan|kenaikan|penurunan)\s+(?:omzet|omset|pendapatan|penjualan)"
    r"|(?:omzet|omset|pendapatan|penjualan)\s+.*?(?:perubahan|turun|naik).*?(?:persen|%)"
    r"|(?:berapa\s+persen(?:tase)?)\s+(?:perubahan\s+|kenaikan\s+|penurunan\s+)?(?:omzet|omset|pendapatan|penjualan)"
    r"|(?:what\s+(?:is|was)|calculate|compute|find)?\s*(?:the\s+)?(?:revenue|sales(?:\s+revenue)?)\s+(?:percent(?:age)?\s+)?change"
    r"|(?:percent(?:age)?\s+change\s+(?:in|of)\s+(?:revenue|sales))"
    r"|by\s+what\s+percent(?:age)?\s+did\s+(?:revenue|sales)"
    r"|how\s+much\s+(?:percent(?:age)?\s+)?did\s+(?:revenue|sales)\s+(?:change|drop|fall|rise|grow|increase|decrease)"
    r")",
    re.I,
)
_REVENUE_HEADERS = frozenset({
    "omzet", "omset", "total omzet", "total omset", "pendapatan", "total pendapatan",
    "revenue", "total revenue", "gross revenue", "sales", "sales revenue", "gross sales",
    "penjualan", "total penjualan",
})
_FROM_TO_RE = re.compile(
    r"(?<!\w)(?:dari|from)\s+([A-Za-z0-9][A-Za-z0-9 _\-]{0,24}?)\s+(?:ke|to|hingga|sampai)\s+([A-Za-z0-9][A-Za-z0-9 _\-]{0,24}?)(?=\s*(?:[?.!,;]|dan\b|and\b|dibanding|versus|vs\b|$))",
    re.I,
)
_ARROW_PERIOD_RE = re.compile(
    r"(?<!\w)([A-Za-z0-9][A-Za-z0-9 _\-]{0,18}?)\s*(?:->|→)\s*([A-Za-z0-9][A-Za-z0-9 _\-]{0,18}?)(?=\s*(?:[?.!,;]|dan\b|and\b|$))",
    re.I,
)
_AMOUNT_TOKEN_RE = re.compile(
    r"\(?[+\-−]?(?:(?:Rp\.?|IDR|USD|EUR|GBP|JPY|[$€£¥])\s*)?"
    r"\d+(?:[.,]\d+)*"
    r"(?:\s*(?:miliar|milyar|billion|bn|juta|jt|million|mn|ribu|rb|k|IDR|USD|EUR|GBP|JPY)\b)?\)?",
    re.I,
)
_CALC_CONSTANTS = frozenset({0.0, 1.0, 2.0, 100.0})

_KPI_REQUEST_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("gross_margin", re.compile(r"\b(?:gross\s+(?:profit\s+)?margin|margin\s+(?:kotor|bruto))\b", re.I)),
    ("net_margin", re.compile(r"\b(?:net\s+(?:profit\s+)?margin|margin\s+(?:bersih|neto))\b", re.I)),
    ("conversion_rate", re.compile(r"\b(?:conversion\s+rate|tingkat\s+konversi|rasio\s+konversi)\b", re.I)),
    ("breakeven_units", re.compile(r"\b(?:break[\s\-]?even(?:\s+units?)?|titik\s+impas|balik\s+modal)\b", re.I)),
    ("aov", re.compile(r"\b(?:aov|average\s+order\s+value|rata-rata\s+(?:nilai\s+)?(?:per\s+)?pesanan)\b", re.I)),
    ("roi", re.compile(r"\b(?:roi|return\s+on\s+investment)\b", re.I)),
)

_AMOUNT_CAPTURE = (
    r"((?:(?:Rp\.?|IDR|USD|EUR|GBP|JPY|[$€£¥])\s*)?"
    r"\d+(?:[.,]\d+)*"
    r"(?:\s*(?:miliar|milyar|billion|bn|juta|jt|million|mn|ribu|rb|k|IDR|USD|EUR|GBP|JPY)\b)?)"
)
_FIELD_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "revenue": (
        re.compile(rf"\b(?:omzet|omset|pendapatan|gross\s+revenue|total\s+revenue|revenue|penjualan)\b(?:\s+(?:bulan|minggu|pekan|tahun|kuartal)\s+(?:ini|lalu))?\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "cogs": (
        re.compile(rf"\b(?:hpp|cogs|cost\s+of\s+goods\s+sold|harga\s+pokok(?:\s+penjualan)?|biaya\s+pokok(?:\s+penjualan)?)\b(?:\s+(?:bulan|minggu|pekan)\s+(?:ini|lalu))?\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "total_cost": (
        re.compile(rf"\b(?:total\s+(?:cost|biaya)|biaya\s+total)\b\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "visitors": (
        re.compile(rf"\b(?:pengunjung(?:\s+situs|\s+toko|\s+aplikasi)?|visitors?|kunjungan|sessions?)\b(?:\s+(?:minggu|pekan|bulan)\s+(?:ini|lalu))?\s*(?:is|was|adalah|sebesar|=|:|of|sebanyak)?\s*{_AMOUNT_CAPTURE}", re.I),
        re.compile(rf"{_AMOUNT_CAPTURE}\s*(?:orang\s+)?(?:pengunjung|store\s+visitors?|website\s+visitors?|visitors?|sessions?)\b", re.I),
    ),
    "buyers": (
        re.compile(rf"\b(?:yang\s+membeli|pembeli|buyers?|completed\s+buyers?)\b\s*(?:is|was|adalah|sebesar|=|:|of|sebanyak)?\s*{_AMOUNT_CAPTURE}", re.I),
        re.compile(rf"{_AMOUNT_CAPTURE}\s*(?:orang\s+)?(?:pembeli|buyers?)\b", re.I),
    ),
    "orders": (
        re.compile(rf"\b(?:jumlah\s+pesanan|total\s+pesanan|pesanan|orders?|transaksi)\b\s*(?:is|was|adalah|sebesar|=|:|of|sebanyak)?\s*{_AMOUNT_CAPTURE}", re.I),
        re.compile(rf"{_AMOUNT_CAPTURE}\s*(?:pesanan|completed\s+orders?|customer\s+orders?|orders?|transaksi)\b", re.I),
    ),
    "fixed_cost": (
        re.compile(rf"\b(?:biaya\s+tetap(?:\s+usaha)?|fixed\s+costs?)\b(?:\s+(?:per\s+bulan|bulanan|kami))?\s*(?:is|was|are|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "unit_price": (
        re.compile(rf"\b(?:harga\s+jual(?:\s+per\s+unit)?|unit\s+(?:selling\s+)?price|selling\s+price(?:\s+per\s+unit)?)\b\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "unit_cost": (
        re.compile(rf"\b(?:biaya\s+variabel(?:\s+per\s+unit)?|unit\s+variable\s+cost|variable\s+cost(?:\s+per\s+unit)?|unit\s+cost|biaya\s+produksi(?:\s+per\s+unit)?)\b\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
    ),
    "gain": (
        re.compile(rf"\b(?:gross\s+contribution|gain|keuntungan\s+kotor|hasil\s+penjualan|pendapatan\s+kampanye|omzet\s+kampanye)\b\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
        re.compile(rf"generated\s+{_AMOUNT_CAPTURE}\s+in\s+(?:gross\s+contribution|revenue|gain)", re.I),
    ),
    "cost": (
        re.compile(rf"\b(?:biaya\s+iklan|biaya\s+kampanye|ad\s+spend|campaign\s+spend|spend|modal|cost)\b\s*(?:is|was|adalah|sebesar|=|:|of)?\s*{_AMOUNT_CAPTURE}", re.I),
        re.compile(rf"(?:on|spent)\s+{_AMOUNT_CAPTURE}\s+(?:spend|on\s+ads|cost)\b", re.I),
    ),
}
_KPI_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "gross_margin": ("revenue", "cogs"),
    "net_margin": ("revenue", "total_cost"),
    "conversion_rate": ("buyers", "visitors"),
    "breakeven_units": ("fixed_cost", "unit_price", "unit_cost"),
    "aov": ("revenue", "orders"),
    "roi": ("gain", "cost"),
}
_MONETARY_FIELDS = frozenset({"revenue", "cogs", "total_cost", "fixed_cost", "unit_price", "unit_cost", "gain", "cost"})


def _extract_embedded_table(text: str) -> str | None:
    """Detect a Markdown, tab-separated (Excel/Sheets), or multi-row CSV table inside a client message."""
    lines = text.splitlines()
    md_block = [ln.strip() for ln in lines if ln.strip().startswith("|") and ln.strip().endswith("|")]
    if len(md_block) >= 3:
        return "\n".join(md_block)
    tsv_block: list[str] = []
    for ln in lines:
        s = ln.strip()
        if "\t" in s and len(s.split("\t")) >= 2 and not s.endswith("?"):
            tsv_block.append(s)
        elif len(tsv_block) >= 3:
            break
        else:
            tsv_block = []
    if len(tsv_block) >= 3:
        return "\n".join(tsv_block)
    csv_block: list[str] = []
    for ln in lines:
        s = ln.strip()
        if "," in s and len(s.split(",")) >= 2 and not s.endswith("?"):
            csv_block.append(s)
        elif len(csv_block) >= 3:
            break
        else:
            csv_block = []
    if len(csv_block) >= 3:
        return "\n".join(csv_block)
    return None


def _extract_labeled_fields(texts: list[str]) -> dict[str, tuple[float, str | None]]:
    """Extract explicitly labeled numeric business fields across user turns (latest turn wins)."""
    bound: dict[str, tuple[float, str | None]] = {}
    for text in texts:
        non_table = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("|"))
        for field_name, patterns in _FIELD_PATTERNS.items():
            for pat in patterns:
                for m in pat.finditer(non_table):
                    val, unit = parse_amount(m.group(1))
                    if val is not None:
                        bound[field_name] = (val, unit)
    return bound


def _number_in_set(val: float, allowed: set[float]) -> bool:
    return any(abs(val - a) <= max(1e-6, 1e-5 * abs(a)) for a in allowed)


def _calc_literals(expr: str) -> list[float]:
    tree = ast.parse(expr.strip().replace(",", ""), mode="eval")
    vals: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            vals.append(float(node.value))
    return vals


def _execute_tool_call(
    kind: str,
    arg_str: str,
    *,
    allowed_numbers: set[float] | None = None,
    bound_fields: dict[str, tuple[float, str | None]] | None = None,
) -> tuple[str | None, str | None]:
    """Deterministically execute a `calc(...)` or `kpi(...)` call emitted by Kuon.

    Returns `(tool_output, error_reason)`. When `allowed_numbers` is provided, every
    numeric argument must be grounded in the user's conversation or verified evidence.
    """
    kind_l = kind.strip().lower()
    try:
        if kind_l == "calc":
            raw_expr = arg_str.strip().strip("'\"")
            if allowed_numbers is not None:
                for lit in _calc_literals(raw_expr):
                    if lit not in _CALC_CONSTANTS and not _number_in_set(lit, allowed_numbers):
                        return None, f"tool_error(ungrounded_args): calc literal {lit} not found in user data"
            val = safe_calc(raw_expr)
            return f"calc({arg_str.strip()}) = {val:.4f}".rstrip("0").rstrip("."), None
        if kind_l == "kpi":
            parts = [p.strip() for p in arg_str.split(",") if p.strip()]
            if not parts:
                return None, "tool_error(invalid_args)"
            metric = parts[0].strip("'\"")
            kwargs: dict[str, float] = {}
            for p in parts[1:]:
                if "=" not in p:
                    return None, "tool_error(invalid_args)"
                key, raw = (s.strip() for s in p.split("=", 1))
                value = float(raw)
                if key in kwargs or not math.isfinite(value):
                    return None, "tool_error(invalid_args)"
                kwargs[key] = value
            if allowed_numbers is not None:
                for key, val in kwargs.items():
                    if not _number_in_set(val, allowed_numbers):
                        return None, f"tool_error(ungrounded_args): {key}={val} not found in user data"
                    if bound_fields and key in bound_fields:
                        expected_val = bound_fields[key][0]
                        if abs(val - expected_val) > max(1e-6, 1e-5 * abs(expected_val)):
                            return None, f"tool_error(ungrounded_args): {key}={val} contradicts stated {expected_val}"
            res = business_kpi(metric, **kwargs)
            if any(isinstance(v, float) and not math.isfinite(v) for v in res.values()):
                return None, "tool_error(non_finite)"
            summary_bits = ", ".join(f"{k}={v}" for k, v in res.items() if k != "metric")
            return f"kpi({metric}): {summary_bits}", None
    except (ValueError, ArithmeticError, SyntaxError, TypeError) as err:
        return None, f"tool_error({type(err).__name__})"
    return None, "tool_error(unknown)"


@dataclass
class AgentConfig:
    use_rag: bool = True
    use_wiki: bool = True
    use_web: bool = False          # opt-in: DuckDuckGo scrape
    fallback_web: bool = False     # if no evidence, try the web
    think: bool = True             # two-stage: notes -> answer
    top_k_rag: int = 3
    max_sources: int = 4
    max_source_chars: int = 1200
    max_answer_tokens: int = 300


NOTES_PROMPT = """Read the evidence and write 1-3 short bullet points that help answer the question.
Only state what the evidence says. If nothing is relevant, write: no relevant evidence.

Evidence:
{evidence}

Question: {question}

Bullet points:"""


SMALLTALK_PATTERNS = [
    re.compile(r"^(halo+|hallo+|hai+|hi+|hello+|hey+|selamat\s+(pagi|siang|sore|malam)|good\s+(morning|afternoon|evening)|assalamualaikum)\b", re.I),
    re.compile(r"\b(thank you|thanks|terima kasih|makasih)\b", re.I),
    re.compile(r"\b(who are you|what is kuon|who is kuon|tell me about yourself|apa itu kuon|siapa kamu|siapa yang membuatmu|who made you|who built you)\b", re.I),
    re.compile(r"\b(dasarnya model apa|pakai model apa|berbasis (?:model )?apa|model dasarmu|what (?:base )?model are you|built on|based on (?:gpt|claude|gemma|llama|qwen))\b", re.I),
    re.compile(r"\b(tell me|write|tulis|ceritakan)\b.{0,40}\b(story|cerita)\b", re.I),
]


_GREETING_PREFIX_RE = re.compile(
    r"^(?:halo+|hallo+|hai+|hi+|hello+|hey+|selamat\s+(?:pagi|siang|sore|malam)|good\s+(?:morning|afternoon|evening)|assalamualaikum)[,!.\s]+",
    re.I,
)


def is_smalltalk(question: str) -> bool:
    """Greetings, thanks, and identity questions are answered directly
    (no search) using the chat template the model was fine-tuned on."""
    return any(pattern.search(question.strip()) for pattern in SMALLTALK_PATTERNS)


def _normalize_smalltalk_prompt(question: str) -> str:
    """Strip leading greeting filler when followed by a direct identity question."""
    q = question.strip()
    stripped = _GREETING_PREFIX_RE.sub("", q).strip()
    if stripped and is_smalltalk(stripped):
        return stripped
    return q


@dataclass
class AgentResult:
    answer: str
    notes: str = ""
    sources: list[dict] = field(default_factory=list)


_FOLLOWUP_RE = re.compile(
    r"^(?:kalau|kalo|terus|lalu|nah|dan|bagaimana\s+dengan|gimana\s+dengan|what\s+about|how\s+about|and\b|what\s+else)\b"
    r"|\b(?:dia|beliau|mereka|itu|tersebut|satunya|lainnya|sebelumnya|tadi|he|she|they|it|that|those|their|the\s+other)\b",
    re.I,
)


def _expand_followup_query(question: str, prior_user_msgs: list[str]) -> str:
    """Attach prior user topic to short/anaphoric follow-ups for better RAG & web search."""
    q = question.strip()
    if not q or not prior_user_msgs:
        return q
    prev = next((m.strip() for m in reversed(prior_user_msgs) if m.strip() and not is_smalltalk(m)), "")
    if not prev:
        return q
    # Skip if previous message was a giant pasted table
    if _extract_embedded_table(prev) is not None:
        prev_prose = " ".join(
            ln.strip() for ln in prev.splitlines()
            if ln.strip() and not ln.strip().startswith("|") and "\t" not in ln and ln.count(",") < 2
        ).strip()
        prev = prev_prose
    if not prev:
        return q
    words = q.split()
    if len(words) <= 10 and (_FOLLOWUP_RE.search(q) or len(words) <= 5):
        return f"{prev[:140]} {q}"
    return q


def _match_period_row(mention: str, rows: list[list[str]], period_header: str) -> int | None:
    """Resolve a period mention (e.g. 'Week 2', 'W2', 'Minggu 1') to a unique row index."""
    clean = " ".join(mention.strip().casefold().split())
    if not clean:
        return None
    exact = [i for i, r in enumerate(rows) if r and " ".join(r[0].strip().casefold().split()) == clean]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    mention_keys = period_aliases(mention, period_header)
    if not mention_keys:
        return None
    hits = [
        i for i, r in enumerate(rows)
        if r and (mention_keys & period_aliases(r[0], period_header))
    ]
    return hits[0] if len(hits) == 1 else None


def _apply_table_followups(
    headers: list[str],
    rows: list[list[str]],
    rev_col: int,
    followups: list[str],
) -> tuple[list[list[str]], list[str]]:
    """Patch missing or corrected revenue cells from follow-up user messages."""
    patched = [list(r) for r in rows]
    notes: list[str] = []
    period_header = headers[0] if headers else ""

    for msg in followups:
        # 1. Explicit period-to-value mentions (e.g. "W2 revenue was 8000", "Omzet M2 = Rp8.250.000")
        matched_any = False
        for i, row in enumerate(patched):
            if not row:
                continue
            label = row[0].strip()
            aliases = period_aliases(label, period_header)
            # Build readable candidate prefixes for this row (e.g., W2, Week 2, Minggu 2, M2)
            cand_patterns: list[str] = [re.escape(label)]
            for key in aliases:
                if key.startswith("week:"):
                    n = key.split(":", 1)[1]
                    cand_patterns.extend([rf"w[\s\-]*{n}", rf"week[\s\-]*{n}", rf"minggu[\s\-]*{n}", rf"m[\s\-]*{n}"])
                elif key.startswith("month:"):
                    n = key.split(":", 1)[1]
                    cand_patterns.extend([rf"month[\s\-]*{n}", rf"bulan[\s\-]*{n}", rf"m[\s\-]*{n}"])
                elif key.startswith("quarter:"):
                    n = key.split(":", 1)[1]
                    cand_patterns.extend([rf"q[\s\-]*{n}", rf"quarter[\s\-]*{n}", rf"kuartal[\s\-]*{n}"])
                elif key.startswith("period:"):
                    n = key.split(":", 1)[1]
                    cand_patterns.extend([rf"p[\s\-]*{n}", rf"period(?:e)?[\s\-]*{n}"])
            alias_re = "(?:" + "|".join(dict.fromkeys(cand_patterns)) + ")"
            pat_forward = re.compile(
                rf"(?<!\w){alias_re}(?!\w)\s*(?:revenue|omzet|pendapatan|sales|penjualan)?\s*(?:was|is|adalah|sebesar|menjadi|to|ke|=|:)?\s*{_AMOUNT_CAPTURE}",
                re.I,
            )
            pat_reverse = re.compile(
                rf"(?:revenue|omzet|pendapatan|sales|penjualan)\s+(?:for|untuk|di|pada)?\s*(?<!\w){alias_re}(?!\w)\s*(?:was|is|adalah|sebesar|menjadi|to|ke|=|:)?\s*{_AMOUNT_CAPTURE}",
                re.I,
            )
            val_str = None
            for p in (pat_reverse, pat_forward):
                m = p.search(msg)
                if m:
                    val_str = m.group(1).strip()
                    break
            if val_str:
                val, _ = parse_amount(val_str)
                if val is not None:
                    while len(row) <= rev_col:
                        row.append("")
                    row[rev_col] = val_str
                    notes.append(f"follow-up {label}={val_str}")
                    matched_any = True

        # 2. Fallback: if user gave a single amount and exactly one row has an empty revenue cell
        if not matched_any:
            empty_idxs = [i for i, r in enumerate(patched) if rev_col >= len(r) or not r[rev_col].strip()]
            amounts = [m.group(0).strip() for m in _AMOUNT_TOKEN_RE.finditer(msg) if parse_amount(m.group(0))[0] is not None]
            if len(empty_idxs) == 1 and len(amounts) == 1:
                idx = empty_idxs[0]
                while len(patched[idx]) <= rev_col:
                    patched[idx].append("")
                patched[idx][rev_col] = amounts[0]
                label = patched[idx][0].strip() if patched[idx] else f"row {idx + 1}"
                notes.append(f"follow-up {label}={amounts[0]}")

    return patched, notes


def _find_sales_request(text: str) -> str:
    return next(
        (
            sentence.strip()
            for line in text.splitlines()
            if not line.strip().startswith("|")
            for sentence in re.split(r"(?<=[.!?])\s+", line)
            if _SALES_CHANGE_REQUEST.search(sentence) and _CHANGE_TERMS.search(sentence)
        ),
        "",
    )


def _answer_sales_change(
    user_msgs: list[str],
    indonesian: bool,
) -> AgentResult | None:
    """Compute a grounded two-period revenue comparison across single-turn or multi-turn chat."""
    table_idx: int | None = None
    table_text: str | None = None
    for idx in range(len(user_msgs) - 1, -1, -1):
        tbl = _extract_embedded_table(user_msgs[idx])
        if tbl is not None:
            table_idx = idx
            table_text = tbl
            break

    # Locate the active sales-change request (latest turn first, else the turn with the table)
    request = ""
    req_idx: int | None = None
    for idx in range(len(user_msgs) - 1, -1, -1):
        cand = _find_sales_request(user_msgs[idx])
        if cand:
            request = cand
            req_idx = idx
            break
    if not request:
        return None
    # If the user moved on to a completely unrelated topic after an earlier sales table, don't hijack it
    if table_idx is not None and req_idx is not None and table_idx < len(user_msgs) - 1:
        latest = user_msgs[-1]
        if _extract_embedded_table(latest) is None and not _AMOUNT_TOKEN_RE.search(latest) and not _SALES_CHANGE_REQUEST.search(latest):
            return None

    def reply(id_text: str, en_text: str) -> AgentResult:
        return AgentResult(id_text if indonesian else en_text)

    try:
        headers, rows = parse_table(table_text or "")
    except ValueError:
        headers, rows = [], []
    columns = [i for i, h in enumerate(headers) if h.casefold() in _REVENUE_HEADERS]
    if len(columns) != 1 or len(rows) < 2:
        return reply(
            "Bisa kirim tabel minimal dua periode dengan satu kolom omzet untuk menghitung persentase perubahannya?",
            "Can you share a table with at least two periods and one revenue column so I can calculate the change?",
        )

    col = columns[0]
    followups = user_msgs[table_idx + 1:] if table_idx is not None else []
    rows, patch_notes = _apply_table_followups(headers, rows, col, followups)
    period_header = headers[0] if headers else ""

    # Resolve explicit period pair ("from A to B" / "dari A ke B" / "A -> B")
    explicit = _FROM_TO_RE.search(request) or _ARROW_PERIOD_RE.search(request)
    if explicit:
        start_mention, end_mention = explicit.group(1).strip(), explicit.group(2).strip()
        old_idx = _match_period_row(start_mention, rows, period_header)
        new_idx = _match_period_row(end_mention, rows, period_header)
        if old_idx is None or new_idx is None or old_idx == new_idx:
            avail = ", ".join(r[0].strip() for r in rows if r and r[0].strip())
            return reply(
                f"Saya belum bisa mencocokkan periode '{start_mention}' ke '{end_mention}' pada tabel ({avail}). Periode mana yang ingin dibandingkan?",
                f"I couldn't uniquely match '{start_mention}' to '{end_mention}' in the table ({avail}). Which two periods should I compare?",
            )
    else:
        if len(rows) != 2:
            avail = ", ".join(r[0].strip() for r in rows if r and r[0].strip())
            return reply(
                f"Tabel memuat {len(rows)} periode ({avail}). Dua periode mana yang ingin dibandingkan?",
                f"The table has {len(rows)} periods ({avail}). Which two periods should I compare?",
            )
        old_idx, new_idx = 0, 1

    old_row, new_row = rows[old_idx], rows[new_idx]
    old_label = old_row[0].strip() if old_row else "period 1"
    new_label = new_row[0].strip() if new_row else "period 2"
    old_raw = old_row[col].strip() if col < len(old_row) else ""
    new_raw = new_row[col].strip() if col < len(new_row) else ""

    missing_labels = [lbl for lbl, raw in ((old_label, old_raw), (new_label, new_raw)) if not raw]
    if missing_labels:
        missing_str = ", ".join(missing_labels)
        return reply(
            f"Bisa isi omzet {missing_str} yang masih kosong? Saya perlu satu angka omzet untuk masing-masing periode.",
            f"Can you fill in the missing revenue for {missing_str}? I need one revenue figure for each period.",
        )

    old_val, old_unit = parse_amount(old_raw)
    new_val, new_unit = parse_amount(new_raw)
    if old_val is None or new_val is None:
        return reply(
            "Ada angka omzet yang formatnya belum terbaca jelas. Bisa tulis angka omzet yang benar untuk kedua periode?",
            "One of the revenue values could not be parsed cleanly. Can you confirm the exact revenue figures for both periods?",
        )
    if old_unit and new_unit and old_unit != new_unit:
        return reply(
            f"Mata uang kedua periode berbeda ({old_label}: {old_unit}, {new_label}: {new_unit}). Bisa samakan satuannya dulu sebelum dihitung?",
            f"The two periods use different currencies ({old_label}: {old_unit}, {new_label}: {new_unit}). Can you convert them to one currency first?",
        )
    if old_val == 0:
        return reply(
            f"Omzet awal ({old_label}) nol, jadi persentase perubahan tidak bisa dihitung. Bandingkan selisih omzet kedua periode dalam satuan uang.",
            f"Starting revenue ({old_label}) is zero, so a percentage change is undefined. Compare the revenue difference in currency instead.",
        )

    result = business_kpi("pct_change", old=old_val, new=new_val)
    pct = f"{result['pct_change'] or 0:.2f}".rstrip("0").rstrip(".")
    if indonesian:
        answer = (
            f"Perubahan omzet dari {old_label} ke {new_label}: {pct.replace('.', ',')}%.\n\n"
            "Bandingkan jumlah pesanan dan rata-rata omzet per pesanan sebelum memilih tindakan yang akan diuji."
        )
    else:
        answer = (
            f"Revenue change from {old_label} to {new_label}: {pct}%.\n\n"
            "Compare order counts and average revenue per order before choosing what to test."
        )
    provenance = f"{old_label}={old_val}, {new_label}={new_val}"
    if patch_notes:
        provenance += " (" + "; ".join(patch_notes) + ")"
    return AgentResult(
        answer=answer,
        notes=f"kpi(pct_change): {provenance}; {result['formula']} = {pct}%",
        sources=[{"title": "tool:kpi", "url": ""}],
    )


def _format_num(val: float, indonesian: bool) -> str:
    txt = f"{val:.2f}".rstrip("0").rstrip(".")
    return txt.replace(".", ",") if indonesian else txt


_TABLE_OP_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("share", re.compile(r"\b(?:kontribusi|pangsa|porsi|persentase\s+kontribusi|share(?:\s+of)?|contribution)\b", re.I)),
    ("max", re.compile(r"\b(?:tertinggi|terbesar|paling\s+(?:tinggi|besar|banyak)|terlaris|highest|largest|biggest|maximum|top\s+\d*|most)\b", re.I)),
    ("min", re.compile(r"\b(?:terendah|terkecil|paling\s+(?:rendah|kecil|sedikit)|lowest|smallest|minimum|least)\b", re.I)),
    ("avg", re.compile(r"\b(?:rata[\s-]?rata|rerata|average|mean)\b", re.I)),
    ("sum", re.compile(r"\b(?:total|jumlah\s+(?:keseluruhan|semua|total)|sum(?:\s+of)?)\b", re.I)),
)
_COL_SYNONYMS: tuple[frozenset[str], ...] = (
    _REVENUE_HEADERS,
    frozenset({"pesanan", "orders", "order", "transaksi", "transactions", "jumlah pesanan"}),
    frozenset({"biaya", "cost", "costs", "hpp", "cogs", "pengeluaran", "expense", "expenses"}),
    frozenset({"laba", "profit", "net profit", "laba bersih", "gross profit", "laba kotor"}),
    frozenset({"unit", "units", "qty", "quantity", "terjual", "sold", "volume", "paket"}),
    frozenset({"pelanggan", "pembeli", "pengunjung", "customers", "buyers", "visitors"}),
)


def _format_table_val(value: float, unit: str | None, indonesian: bool) -> str:
    is_int = abs(value - round(value)) < 1e-6
    if is_int:
        base = f"{int(round(value)):,}"
        formatted = base.replace(",", ".") if indonesian else base
    else:
        base = f"{value:,.2f}".rstrip("0").rstrip(".")
        if indonesian:
            formatted = base.replace(",", "_").replace(".", ",").replace("_", ".")
        else:
            formatted = base
    if unit == "IDR":
        return f"Rp{formatted}"
    if unit == "USD":
        return f"${formatted}"
    if unit == "EUR":
        return f"€{formatted}"
    if unit == "%":
        return f"{formatted}%"
    if unit:
        return f"{unit} {formatted}"
    return formatted


def _answer_table_analytics(user_msgs: list[str], indonesian: bool) -> AgentResult | None:
    """Deterministic table aggregation (sum, mean, max, min, share %) on pasted tables."""
    if not user_msgs:
        return None
    table_text: str | None = None
    for msg in reversed(user_msgs):
        tbl = _extract_embedded_table(msg)
        if tbl is not None:
            table_text = tbl
            break
    if not table_text:
        return None

    latest = user_msgs[-1]
    prose_lines = [
        ln.strip() for ln in latest.splitlines()
        if ln.strip() and not ln.strip().startswith("|") and "\t" not in ln and ln.count(",") < 2
    ]
    query_text = " ".join(prose_lines).strip()
    if not query_text:
        return None

    matched_ops = [op for op, pat in _TABLE_OP_PATTERNS if pat.search(query_text)]
    if not matched_ops:
        return None

    try:
        headers, rows = parse_table(table_text)
    except ValueError:
        return None
    if len(headers) < 2 or not rows:
        return None

    numeric_cols: dict[int, list[tuple[str, float, str | None]]] = {}
    for ci in range(1, len(headers)):
        entries: list[tuple[str, float, str | None]] = []
        for r in rows:
            if ci >= len(r):
                continue
            label = r[0].strip() if r else ""
            val, unit = parse_amount(r[ci])
            if label and val is not None:
                entries.append((label, val, unit))
        if len(entries) >= 2:
            numeric_cols[ci] = entries
    if not numeric_cols:
        return None

    q_low = query_text.casefold()
    chosen_ci: int | None = None
    for ci in numeric_cols:
        h_low = headers[ci].strip().casefold()
        if h_low and re.search(rf"(?<!\w){re.escape(h_low)}(?!\w)", q_low):
            chosen_ci = ci
            break
    if chosen_ci is None:
        for syn_group in _COL_SYNONYMS:
            if any(re.search(rf"(?<!\w){re.escape(s)}(?!\w)", q_low) for s in syn_group):
                cands = [ci for ci in numeric_cols if headers[ci].strip().casefold() in syn_group]
                if len(cands) == 1:
                    chosen_ci = cands[0]
                    break
    if chosen_ci is None:
        if len(numeric_cols) == 1:
            chosen_ci = next(iter(numeric_cols))
        else:
            rev_cands = [ci for ci in numeric_cols if headers[ci].strip().casefold() in _REVENUE_HEADERS]
            if len(rev_cands) == 1:
                chosen_ci = rev_cands[0]
            else:
                return None

    col_name = headers[chosen_ci].strip()
    entries = numeric_cols[chosen_ci]
    units = {u for _, _, u in entries if u}
    if len(units) > 1:
        u_list = ", ".join(sorted(units))
        msg = (
            f"Kolom {col_name} memakai mata uang berbeda ({u_list}). Bisa samakan satuannya dulu sebelum dihitung?"
            if indonesian else
            f"The {col_name} column mixes currencies ({u_list}). Can you standardize them to one currency first?"
        )
        return AgentResult(answer=msg)
    unit = next(iter(units)) if units else None

    total_val = sum(v for _, v, _ in entries)
    avg_val = total_val / len(entries)
    max_entry = max(entries, key=lambda x: x[1])
    min_entry = min(entries, key=lambda x: x[1])

    primary_op = next((op for op in ("max", "min", "avg", "sum", "share") if op in matched_ops), matched_ops[0])

    if primary_op == "max":
        v_txt = _format_table_val(max_entry[1], unit, indonesian)
        share_pct = (max_entry[1] / total_val * 100.0) if total_val > 0 else None
        share_str = (
            f" ({_format_num(share_pct, indonesian)}% dari total)" if share_pct is not None else ""
        ) if indonesian else (
            f" ({_format_num(share_pct, indonesian)}% of total)" if share_pct is not None else ""
        )
        answer = (
            f"{col_name} tertinggi: {max_entry[0]} sebesar {v_txt}{share_str}."
            if indonesian else
            f"Highest {col_name}: {max_entry[0]} at {v_txt}{share_str}."
        )
        note = f"table_analytics(op=max, col={col_name}, n={len(entries)}): label={max_entry[0]}, value={max_entry[1]}, total={round(total_val, 4)}"
    elif primary_op == "min":
        v_txt = _format_table_val(min_entry[1], unit, indonesian)
        share_pct = (min_entry[1] / total_val * 100.0) if total_val > 0 else None
        share_str = (
            f" ({_format_num(share_pct, indonesian)}% dari total)" if share_pct is not None else ""
        ) if indonesian else (
            f" ({_format_num(share_pct, indonesian)}% of total)" if share_pct is not None else ""
        )
        answer = (
            f"{col_name} terendah: {min_entry[0]} sebesar {v_txt}{share_str}."
            if indonesian else
            f"Lowest {col_name}: {min_entry[0]} at {v_txt}{share_str}."
        )
        note = f"table_analytics(op=min, col={col_name}, n={len(entries)}): label={min_entry[0]}, value={min_entry[1]}, total={round(total_val, 4)}"
    elif primary_op == "avg":
        v_txt = _format_table_val(avg_val, unit, indonesian)
        t_txt = _format_table_val(total_val, unit, indonesian)
        answer = (
            f"Rata-rata {col_name}: {v_txt} (dari {len(entries)} baris, total {t_txt})."
            if indonesian else
            f"Average {col_name}: {v_txt} (across {len(entries)} rows, total {t_txt})."
        )
        note = f"table_analytics(op=avg, col={col_name}, n={len(entries)}): mean={round(avg_val, 4)}, total={round(total_val, 4)}"
    elif primary_op == "sum":
        t_txt = _format_table_val(total_val, unit, indonesian)
        avg_txt = _format_table_val(avg_val, unit, indonesian)
        answer = (
            f"Total {col_name}: {t_txt} (dari {len(entries)} baris, rata-rata {avg_txt})."
            if indonesian else
            f"Total {col_name}: {t_txt} (across {len(entries)} rows, average {avg_txt})."
        )
        note = f"table_analytics(op=sum, col={col_name}, n={len(entries)}): sum={round(total_val, 4)}, mean={round(avg_val, 4)}"
    else:
        if total_val == 0:
            return AgentResult(
                answer=(
                    f"Total {col_name} nol sehingga porsi persentase tidak dapat dihitung."
                    if indonesian else
                    f"Total {col_name} is zero, so percentage shares cannot be computed."
                )
            )
        ranked = sorted(entries, key=lambda x: x[1], reverse=True)
        parts = [
            f"{lbl}: {_format_num((val / total_val) * 100.0, indonesian)}% ({_format_table_val(val, unit, indonesian)})"
            for lbl, val, _ in ranked
        ]
        answer = (
            f"Porsi {col_name} terhadap total ({_format_table_val(total_val, unit, indonesian)}):\n- " + "\n- ".join(parts)
            if indonesian else
            f"{col_name} share of total ({_format_table_val(total_val, unit, indonesian)}):\n- " + "\n- ".join(parts)
        )
        note = f"table_analytics(op=share, col={col_name}, n={len(entries)}): total={round(total_val, 4)}"

    return AgentResult(
        answer=answer,
        notes=note,
        sources=[{"title": "tool:table_analytics", "url": ""}],
    )


def _answer_direct_kpi(user_msgs: list[str], indonesian: bool) -> AgentResult | None:
    """Compute grounded standard KPIs (margin, conversion, break-even, AOV, ROI) with provenance."""
    # Find the latest explicit KPI request across user turns
    metric: str | None = None
    req_idx: int | None = None
    for idx in range(len(user_msgs) - 1, -1, -1):
        non_table = "\n".join(ln for ln in user_msgs[idx].splitlines() if not ln.strip().startswith("|"))
        for name, pat in _KPI_REQUEST_PATTERNS:
            if pat.search(non_table):
                metric = name
                req_idx = idx
                break
        if metric is not None:
            break
    if metric is None or req_idx is None:
        return None
    # Do not hijack non-KPI follow-ups or general questions without any numbers/missing-input context
    if req_idx < len(user_msgs) - 1 and not _AMOUNT_TOKEN_RE.search(user_msgs[-1]):
        return None
    # Only handle explicit calculation asks (question mark or calculate/hitung/berapa/what/how many)
    req_text = user_msgs[req_idx]
    if not re.search(r"\?|berapa|hitung|what|how\s+many|calculate|compute", req_text, re.I):
        return None

    bound = _extract_labeled_fields(user_msgs)
    required = _KPI_REQUIRED_FIELDS[metric]
    missing = [f for f in required if f not in bound]
    if missing:
        # Only ask if the user provided at least one relevant input or explicitly asked to compute
        if not any(f in bound for f in required):
            return None
        missing_labels_id = {
            "revenue": "omzet", "cogs": "HPP", "total_cost": "total biaya",
            "buyers": "jumlah pembeli", "visitors": "jumlah pengunjung",
            "fixed_cost": "biaya tetap", "unit_price": "harga jual per unit",
            "unit_cost": "biaya variabel per unit", "gain": "pendapatan/keuntungan kotor",
            "cost": "biaya/modal", "orders": "jumlah pesanan",
        }
        missing_labels_en = {
            "revenue": "revenue", "cogs": "COGS", "total_cost": "total cost",
            "buyers": "buyers", "visitors": "visitors",
            "fixed_cost": "fixed cost", "unit_price": "unit price",
            "unit_cost": "unit variable cost", "gain": "gross contribution/gain",
            "cost": "cost/spend", "orders": "order count",
        }
        if indonesian:
            need = ", ".join(missing_labels_id[m] for m in missing)
            return AgentResult(f"Data {need} belum tersedia untuk menghitung {metric.replace('_', ' ')}. Berapa angka {need} pada periode yang sama?")
        need = ", ".join(missing_labels_en[m] for m in missing)
        return AgentResult(f"Missing {need} to calculate {metric.replace('_', ' ')}. What is the {need} for the same period?")

    # Check currency consistency across monetary inputs
    units = {bound[f][1] for f in required if f in _MONETARY_FIELDS and bound[f][1]}
    if len(units) > 1:
        u_list = ", ".join(sorted(units))
        if indonesian:
            return AgentResult(f"Satuan mata uang tidak sama ({u_list}). Bisa samakan semua angka ke satu mata uang dulu?")
        return AgentResult(f"The inputs use mixed currencies ({u_list}). Can you convert them to one currency first?")

    kwargs = {f: bound[f][0] for f in required}
    try:
        res = business_kpi(metric, **kwargs)
    except (ValueError, ZeroDivisionError) as err:
        if indonesian:
            return AgentResult(f"Perhitungan belum bisa dijalankan ({err}). Bisa periksa kembali angka masukannya?")
        return AgentResult(f"The calculation could not run ({err}). Can you double-check the input figures?")

    inputs_trace = ", ".join(f"{k}={kwargs[k]}" for k in required)
    if metric == "gross_margin":
        val_txt = _format_num(res["margin_pct"], indonesian)
        ans = (
            f"Margin kotor: {val_txt}%.\n\nPeriksa komponen HPP terbesar dan pantau rasio margin kotor setiap minggu."
            if indonesian else
            f"Gross margin: {val_txt}%.\n\nReview the largest cost items and track weekly margin ratio before changing prices."
        )
        note = f"kpi(gross_margin): {inputs_trace}; gross_profit={res['gross_profit']}, margin_pct={res['margin_pct']}, formula={res['formula']}"
    elif metric == "net_margin":
        val_txt = _format_num(res["margin_pct"], indonesian)
        ans = (
            f"Margin bersih: {val_txt}%.\n\nAudit pos biaya operasional terbesar dan pantau rasio margin bersih bulanan."
            if indonesian else
            f"Net margin: {val_txt}%.\n\nAudit the largest operating expense lines and monitor monthly net margin ratio."
        )
        note = f"kpi(net_margin): {inputs_trace}; net_profit={res['net_profit']}, margin_pct={res['margin_pct']}, formula={res['formula']}"
    elif metric == "conversion_rate":
        val_txt = _format_num(res["rate_pct"], indonesian)
        ans = (
            f"Tingkat konversi: {val_txt}%.\n\nCatat alasan utama pengunjung tidak membeli dan ukur tingkat konversi harian."
            if indonesian else
            f"Conversion rate: {val_txt}%.\n\nLog why non-buying visitors leave and monitor daily conversion rate."
        )
        note = f"kpi(conversion_rate): {inputs_trace}; rate_pct={res['rate_pct']}, formula={res['formula']}"
    elif metric == "breakeven_units":
        units_int = res["units"]
        ans = (
            f"Titik impas: {units_int} unit.\n\nPeriksa selisih penjualan mingguan terhadap target dan pantau rasio margin kontribusi."
            if indonesian else
            f"Break-even volume: {units_int} units.\n\nReview weekly unit sales variance against the threshold and track contribution margin ratio."
        )
        note = f"kpi(breakeven_units): {inputs_trace}; contribution_margin={res['contribution_margin']}, units={units_int}, formula={res['formula']}"
    elif metric == "aov":
        val_txt = _format_num(res["aov"], indonesian)
        ans = (
            f"Rata-rata nilai pesanan (AOV): {val_txt}.\n\nUji paket pendamping pada produk utama dan pantau rata-rata per pesanan mingguan."
            if indonesian else
            f"Average order value (AOV): {val_txt}.\n\nTest a complementary bundle on top items and track average revenue per order."
        )
        note = f"kpi(aov): {inputs_trace}; aov={res['aov']}, formula={res['formula']}"
    else:  # roi
        val_txt = _format_num(res["roi_pct"], indonesian)
        ans = (
            f"ROI: {val_txt}%.\n\nUji ulang kampanye pada segmen berintent tinggi dan pantau rasio ROI bersih."
            if indonesian else
            f"ROI: {val_txt}%.\n\nRe-test the campaign on a high-intent segment and track net ROI ratio."
        )
        note = f"kpi(roi): {inputs_trace}; net_gain={res['net_gain']}, roi_pct={res['roi_pct']}, formula={res['formula']}"

    return AgentResult(answer=ans, notes=note, sources=[{"title": "tool:kpi", "url": ""}])


class ContextLimitError(ValueError):
    """The newest user turn does not fit even without history."""


class KuonAgent:
    def __init__(self, engine, config: AgentConfig | None = None,
                 rag_index: BM25Index | None = None,
                 wiki_fn: Callable[[str, int], list[Source]] = wikipedia_search,
                 web_fn: Callable[[str, int], list[Source]] = web_search):
        self.engine = engine
        self.cfg = config or AgentConfig()
        self.rag = rag_index
        self._wiki = wiki_fn
        self._web = web_fn

    # ------------------------------------------------------------ evidence

    def _gather(self, question: str) -> list[Source]:
        gathered: list[Source] = []
        if self.cfg.use_rag and self.rag is not None:
            for _score, chunk in self.rag.search(question, top_k=self.cfg.top_k_rag):
                gathered.append(Source(title=chunk.source, url="", text=chunk.text))
        if self.cfg.use_web:
            gathered.extend(self._web(question, self.cfg.max_sources))
        if self.cfg.use_wiki:
            is_id = bool(_ID_WORDS.search(question))
            if is_id:
                try:
                    id_hits = self._wiki(question, self.cfg.max_sources, lang="id")
                    gathered.extend(id_hits)
                    if not id_hits:
                        gathered.extend(self._wiki(question, self.cfg.max_sources, lang="en"))
                except TypeError:
                    gathered.extend(self._wiki(question, self.cfg.max_sources))
            else:
                gathered.extend(self._wiki(question, self.cfg.max_sources))
        if self.cfg.fallback_web and not gathered:
            gathered.extend(self._web(question, self.cfg.max_sources))

        # dedupe by url/title prefix, trim, cap
        seen: set[str] = set()
        unique: list[Source] = []
        for source in gathered:
            key = source.url or source.title[:80]
            if key in seen:
                continue
            seen.add(key)
            unique.append(Source(title=source.title, url=source.url,
                                 text=source.text[:self.cfg.max_source_chars]))
        return unique[:self.cfg.max_sources]

    @staticmethod
    def _format_evidence(sources: list[Source]) -> str:
        blocks = []
        for i, source in enumerate(sources, start=1):
            header = f"[{i}] {source.title}"
            if source.url:
                header += f" ({source.url})"
            blocks.append(f"{header}\n{source.text}")
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- loop

    def _complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float | None = None,
        on_token: Callable[[str], None] | None = None,
        guard_tool_prefix: bool = False,
    ) -> tuple[str, bool]:
        """Run engine completion, optionally streaming live token deltas.

        When `guard_tool_prefix=True`, buffers only leading characters that could
        still be a `TOOL:` invocation so raw tool syntax is never leaked to a stream.
        """
        stream_fn = getattr(self.engine, "complete_stream", None)
        if on_token is not None and callable(stream_fn):
            kwargs: dict = {"max_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            try:
                gen = stream_fn(prompt, **kwargs)
            except TypeError:
                gen = stream_fn(prompt, max_tokens=max_tokens)
            chunks: list[str] = []
            buf = ""
            flushed = not guard_tool_prefix
            streamed = False
            for delta in gen:
                if not delta:
                    continue
                chunks.append(delta)
                if not flushed:
                    buf += delta
                    lead = buf.lstrip()
                    low = lead.lower()
                    if not lead:
                        continue
                    if (len(low) < 5 and "tool:".startswith(low)) or low.startswith("tool:"):
                        continue
                    flushed = True
                    on_token(buf)
                    streamed = True
                    buf = ""
                else:
                    on_token(delta)
                    streamed = True
            if not flushed and buf and not buf.lstrip().lower().startswith("tool:"):
                on_token(buf)
                streamed = True
            return "".join(chunks).strip(), streamed

        if temperature is not None:
            try:
                text = self.engine.complete(prompt, max_tokens=max_tokens, temperature=temperature)
            except TypeError:
                text = self.engine.complete(prompt, max_tokens=max_tokens)
        else:
            text = self.engine.complete(prompt, max_tokens=max_tokens)
        return text.strip(), False

    def advise(
        self,
        messages: list[tuple[str, str]],
        *,
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentResult:
        # The user's business facts come from the chat, optional client documents
        # (BM25 RAG), and deterministic analytical tools (never public web guesses).
        fits = getattr(self.engine, "prompt_fits", None)
        if fits is None:
            raise NotImplementedError("Business chat requires an engine with token budgeting.")

        turns: list[list[str]] = []
        user_msgs: list[str] = []
        latest_user_text = ""
        for role, content in messages:
            if role == "user":
                turns.append([f"User: {content}"])
                user_msgs.append(content)
                latest_user_text = content
            elif role == "assistant" and turns:
                turns[-1].append(f"Assistant: {content}")

        ledger = FactLedger()
        sources: list[dict] = []
        tool_notes: list[str] = []

        # Detect language from latest user turn, falling back to prior user turn for numeric follow-ups
        indonesian = bool(_ID_WORDS.search(latest_user_text)) or (
            not re.search(r"[A-Za-z]{3,}", latest_user_text)
            and any(_ID_WORDS.search(m) for m in user_msgs[:-1])
        )

        # 1. Deterministic grounded sales-table comparison, table analytics, & direct KPI calculation
        sales_answer = _answer_sales_change(user_msgs, indonesian)
        if sales_answer is not None:
            if on_token is not None and sales_answer.answer:
                on_token(sales_answer.answer)
            return sales_answer

        table_analytics_answer = _answer_table_analytics(user_msgs, indonesian)
        if table_analytics_answer is not None:
            if on_token is not None and table_analytics_answer.answer:
                on_token(table_analytics_answer.answer)
            return table_analytics_answer

        embedded_tbl = _extract_embedded_table(latest_user_text)
        if not embedded_tbl:
            kpi_answer = _answer_direct_kpi(user_msgs, indonesian)
            if kpi_answer is not None:
                if on_token is not None and kpi_answer.answer:
                    on_token(kpi_answer.answer)
                return kpi_answer

        if embedded_tbl:
            try:
                summary = summarize_table(embedded_tbl)
                if summary["numeric_summary"]:
                    bits = []
                    for col, st in summary["numeric_summary"].items():
                        delta = (f", delta={st['first_to_last_delta']} ({st['first_to_last_pct']}%)"
                                 if "first_to_last_pct" in st else "")
                        bits.append(f"{col}: sum={st['sum']}, mean={st['mean']}, min={st['min']}, max={st['max']}{delta}")
                    note = f"Table ({summary['row_count']} rows) — " + "; ".join(bits)
                    ledger.add(note, source="tool:table_summary")
                    sources.append({"title": "tool:table_summary", "url": ""})
                    tool_notes.append(note)
            except ValueError:
                pass

        # 2. Optional client document retrieval (only when the operator loaded a RAG index)
        if self.cfg.use_rag and self.rag is not None and latest_user_text:
            rag_query = _expand_followup_query(latest_user_text, user_msgs[:-1])
            for _score, chunk in self.rag.search(rag_query, top_k=min(2, self.cfg.top_k_rag)):
                snippet = chunk.text[:350].strip()
                if snippet:
                    ledger.add(snippet, source=chunk.source)
                    sources.append({"title": chunk.source, "url": ""})

        token_budget = min(max_tokens, 512) if max_tokens is not None else min(180, self.cfg.max_answer_tokens)

        def _build_fitting_prompt(evidence_text: str | None) -> str:
            conversation = "\n".join(turns[-1])
            prompt = format_prompt(
                conversation.removeprefix("User: "),
                evidence=evidence_text,
                business=True,
            )
            if not fits(prompt, token_budget):
                # Fall back to dropping evidence if client message alone barely fits
                if evidence_text is not None:
                    prompt = format_prompt(conversation.removeprefix("User: "), business=True)
                if not fits(prompt, token_budget):
                    raise ContextLimitError
                return prompt
            for turn in reversed(turns[:-1]):
                candidate = "\n".join([*turn, conversation])
                expanded = format_prompt(
                    candidate.removeprefix("User: "),
                    evidence=evidence_text,
                    business=True,
                )
                if not fits(expanded, token_budget):
                    break
                conversation, prompt = candidate, expanded
            return prompt

        ev_text = ledger.format_evidence() or None
        prompt = _build_fitting_prompt(ev_text)
        # Only stream live when no evidence post-verification override can replace the answer
        answer, streamed = self._complete(
            prompt,
            max_tokens=token_budget,
            temperature=temperature,
            on_token=(on_token if ev_text is None else None),
            guard_tool_prefix=True,
        )

        # 3. Single bounded deterministic tool-call pass with strict argument provenance check
        match = _TOOL_CALL_RE.search(answer)
        if match:
            grounded_context = "\n".join(user_msgs) + "\n" + ledger.format_evidence()
            allowed_nums = extract_grounded_numbers(grounded_context)
            bound_fields = _extract_labeled_fields(user_msgs)
            tool_out, err_note = _execute_tool_call(
                match.group(1),
                match.group(2),
                allowed_numbers=allowed_nums,
                bound_fields=bound_fields,
            )
            if tool_out is None:
                if err_note:
                    tool_notes.append(err_note)
                answer = (
                    "Saya belum bisa memverifikasi hitungan ini karena angka masukan belum cocok dengan data yang Anda berikan. Bisa konfirmasi ukuran, angka, dan satuannya?"
                    if indonesian else
                    "I couldn't verify that calculation because the figures didn't match your stated data. Which metric, figures, and units should I use?"
                )
            else:
                source = f"tool:{match.group(1).lower()}"
                ledger.add(tool_out, source=source)
                sources.append({"title": source, "url": ""})
                tool_notes.append(tool_out)
                answer, _ = self._complete(
                    _build_fitting_prompt(ledger.format_evidence()),
                    max_tokens=token_budget,
                    temperature=temperature,
                )

        # 4. Post-generation grounding verification whenever external/tool evidence was supplied
        final_ev = ledger.format_evidence()
        if final_ev:
            ok, safe_fallback = verify_grounded_answer(
                latest_user_text, final_ev, answer, indonesian=indonesian
            )
            if not ok and safe_fallback:
                answer = safe_fallback

        if on_token is not None and not streamed and answer:
            on_token(answer)

        return AgentResult(
            answer=answer,
            notes="\n".join(tool_notes),
            sources=sources,
        )

    def answer(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentResult:
        token_budget = min(max_tokens, 512) if max_tokens is not None else self.cfg.max_answer_tokens
        prior_users = [c for r, c in (history or [])[:-1] if r == "user" and c.strip()]
        indonesian = bool(_ID_WORDS.search(question)) or (
            not re.search(r"[A-Za-z]{3,}", question) and any(_ID_WORDS.search(m) for m in prior_users)
        )
        if is_smalltalk(question):
            # answer in the model's own chat format, without search
            text, streamed = self._complete(
                format_prompt(_normalize_smalltalk_prompt(question)),
                max_tokens=token_budget,
                temperature=temperature,
                on_token=on_token,
            )
            if on_token is not None and not streamed and text:
                on_token(text)
            return AgentResult(answer=text, notes="", sources=[])

        search_query = _expand_followup_query(question, prior_users)
        sources = self._gather(search_query)
        if not sources:
            # short, non-question inputs are almost always conversational
            # ("haloooo", "p"); real questions must admit ignorance instead
            looks_factual = bool(re.match(
                r"^(who|what|where|when|why|how|which|siapa|apa|kapan|dimana|di mana|berapa|mengapa|kenapa)\b",
                question.strip(), re.I)) or question.strip().endswith("?")
            if not looks_factual and len(question.split()) <= 3:
                text, streamed = self._complete(
                    format_prompt(question),
                    max_tokens=token_budget,
                    temperature=temperature,
                    on_token=on_token,
                )
                if on_token is not None and not streamed and text:
                    on_token(text)
                return AgentResult(answer=text, notes="", sources=[])
            msg = (
                "Saya belum menemukan bukti dari sumber yang tersedia untuk menjawab pertanyaan ini."
                if indonesian else
                "I don't know from my sources, and no evidence was found."
            )
            if on_token is not None:
                on_token(msg)
            return AgentResult(
                answer=msg,
                notes="no evidence collected",
                sources=[],
            )

        prompt_question = search_query if search_query != question else question
        evidence = self._format_evidence(sources)
        fits = getattr(self.engine, "prompt_fits", None)
        if callable(fits):
            # Trim excess tail evidence so long web/RAG pages never left-truncate the prompt
            kept = list(sources)
            while len(kept) > 1 and not fits(format_prompt(prompt_question, evidence=self._format_evidence(kept)), token_budget):
                kept.pop()
            evidence = self._format_evidence(kept)
            if not fits(format_prompt(prompt_question, evidence=evidence), token_budget):
                lo, hi = 0, len(evidence)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if fits(format_prompt(prompt_question, evidence=evidence[:mid]), token_budget):
                        lo = mid
                    else:
                        hi = mid - 1
                evidence = evidence[:lo].rstrip()
        notes = ""
        if self.cfg.think:
            notes, _ = self._complete(
                NOTES_PROMPT.format(evidence=evidence, question=prompt_question),
                max_tokens=120,
                temperature=temperature,
            )

        answer, _ = self._complete(
            format_prompt(prompt_question, evidence=evidence),
            max_tokens=token_budget,
            temperature=temperature,
        )
        ok, safe_fallback = verify_grounded_answer(prompt_question, evidence, answer, indonesian=indonesian)
        if not ok:
            # One bounded grounding retry keeping exact figures and currency symbols from evidence
            hint = (
                "Gunakan angka dan mata uang persis seperti pada kutipan bukti, tanpa mengganti satuan."
                if indonesian else
                "Use the exact numbers and currency symbols from the evidence without converting currencies."
            )
            retry_prompt = format_prompt(f"{prompt_question}\n({hint})", evidence=evidence)
            if not callable(fits) or fits(retry_prompt, token_budget):
                retry_ans, _ = self._complete(
                    retry_prompt,
                    max_tokens=token_budget,
                    temperature=0.0,
                )
                ok_retry, fb_retry = verify_grounded_answer(question, evidence, retry_ans, indonesian=indonesian)
                if ok_retry:
                    answer = retry_ans
                    ok = True
                elif fb_retry:
                    safe_fallback = fb_retry
        if not ok and safe_fallback:
            answer = safe_fallback
        if on_token is not None and answer:
            on_token(answer)

        return AgentResult(
            answer=answer,
            notes=notes,
            sources=[{"title": s.title, "url": s.url} for s in sources],
        )
