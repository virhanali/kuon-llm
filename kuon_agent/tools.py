"""Deterministic business analytical tools for Kuon Business Agent.

Provides domain-general, zero-hallucination analytical primitives (no hardcoded
per-business-type advice rules):
  1. `safe_calc(expr)` — AST-verified arithmetic evaluator (no `eval()` code execution).
  2. `business_kpi(metric, **kwargs)` — Exact standard financial/ops formulas
     (gross margin, net margin, pct change, conversion rate, AOV, break-even units,
      ROI, CAC payback, stockout rate, EOQ).
  3. `summarize_table(text)` — Parses CSV or Markdown tables attached by clients
     and computes exact column totals, means, min/max, and period-over-period deltas.
  4. `FactLedger` — Tracks client-stated facts vs open hypotheses across turns with
     citation indices (`[1]`, `[2]`, ...) so Kuon stays grounded in confirmed evidence.
"""

from __future__ import annotations

import ast
import csv
import io
import math
import operator
import re
from dataclasses import dataclass, field


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calc(expr: str) -> float:
    """Safely evaluate a plain arithmetic expression using Python's AST."""
    cleaned = expr.strip().replace(",", "")
    if not cleaned or len(cleaned) > 200:
        raise ValueError("expression must be non-empty and <= 200 chars")
    tree = ast.parse(cleaned, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _BIN_OPS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    result = float(_eval(tree))
    if not math.isfinite(result):
        raise ValueError("non-finite calculation result")
    return result


_KPI_ALIASES = {
    "break_even_units": "breakeven_units",
    "break_even": "breakeven_units",
    "breakeven": "breakeven_units",
    "gross_profit_margin": "gross_margin",
    "net_profit_margin": "net_margin",
    "percent_change": "pct_change",
    "percentage_change": "pct_change",
    "average_order_value": "aov",
    "return_on_investment": "roi",
    "economic_order_quantity": "eoq",
    "conversion": "conversion_rate",
}


def business_kpi(metric: str, **kwargs: float) -> dict:
    """Compute exact standard business/financial KPIs with formula trace."""
    m = re.sub(r"[\s\-]+", "_", metric.strip().lower()).strip("_")
    m = _KPI_ALIASES.get(m, m)

    def req(key: str) -> float:
        if key not in kwargs or kwargs[key] is None:
            raise ValueError(f"missing required input '{key}' for metric '{metric}'")
        return float(kwargs[key])

    if m == "gross_margin":
        rev, cogs = req("revenue"), req("cogs")
        if rev == 0:
            raise ZeroDivisionError("revenue cannot be zero")
        profit = rev - cogs
        pct = (profit / rev) * 100.0
        return {"metric": "gross_margin", "gross_profit": round(profit, 4), "margin_pct": round(pct, 2),
                "formula": f"({rev} - {cogs}) / {rev} * 100"}
    if m == "net_margin":
        rev, cost = req("revenue"), req("total_cost")
        if rev == 0:
            raise ZeroDivisionError("revenue cannot be zero")
        net = rev - cost
        pct = (net / rev) * 100.0
        return {"metric": "net_margin", "net_profit": round(net, 4), "margin_pct": round(pct, 2),
                "formula": f"({rev} - {cost}) / {rev} * 100"}
    if m == "pct_change":
        old, new = req("old"), req("new")
        if old == 0:
            raise ZeroDivisionError("baseline value cannot be zero")
        diff = new - old
        pct = (diff / old) * 100.0
        return {"metric": "pct_change", "delta": round(diff, 4), "pct_change": round(pct, 2),
                "formula": f"({new} - {old}) / {old} * 100"}
    if m == "conversion_rate":
        buyers, visitors = req("buyers"), req("visitors")
        if visitors == 0:
            raise ZeroDivisionError("visitors cannot be zero")
        pct = (buyers / visitors) * 100.0
        return {"metric": "conversion_rate", "rate_pct": round(pct, 2),
                "formula": f"{buyers} / {visitors} * 100"}
    if m == "aov":
        rev, orders = req("revenue"), req("orders")
        if orders == 0:
            raise ZeroDivisionError("orders cannot be zero")
        return {"metric": "aov", "aov": round(rev / orders, 2), "formula": f"{rev} / {orders}"}
    if m == "breakeven_units":
        fixed, price, var_cost = req("fixed_cost"), req("unit_price"), req("unit_cost")
        contrib = price - var_cost
        if contrib <= 0:
            raise ValueError("unit_price must exceed unit_cost to reach break-even")
        units = math.ceil(fixed / contrib)
        return {"metric": "breakeven_units", "contribution_margin": round(contrib, 4),
                "units": units, "formula": f"ceil({fixed} / ({price} - {var_cost}))"}
    if m == "roi":
        gain, cost = req("gain"), req("cost")
        if cost == 0:
            raise ZeroDivisionError("cost cannot be zero")
        pct = ((gain - cost) / cost) * 100.0
        return {"metric": "roi", "net_gain": round(gain - cost, 4), "roi_pct": round(pct, 2),
                "formula": f"({gain} - {cost}) / {cost} * 100"}
    if m == "eoq":
        demand, order_cost, hold_cost = req("annual_demand"), req("order_cost"), req("holding_cost")
        if demand <= 0 or order_cost <= 0 or hold_cost <= 0:
            raise ValueError("EOQ inputs must be positive")
        qty = math.sqrt((2.0 * demand * order_cost) / hold_cost)
        return {"metric": "eoq", "eoq_units": round(qty, 2),
                "formula": f"sqrt(2 * {demand} * {order_cost} / {hold_cost})"}
    raise ValueError(f"unknown KPI metric: {metric}")


_CURRENCY_PREFIX_RE = re.compile(r"^(rp\.?|idr|usd|eur|gbp|jpy|[$€£¥])\s*", re.I)
_CURRENCY_SUFFIX_RE = re.compile(r"\s*(idr|usd|eur|gbp|jpy)$", re.I)
_SCALE_SUFFIX_RE = re.compile(
    r"\s*(miliar|milyar|billion|bn|b|juta|jt|million|mn|m|ribu|rb|k)\.?$",
    re.I,
)
_SCALE_FACTORS = {
    "rb": 1e3, "ribu": 1e3, "k": 1e3,
    "jt": 1e6, "juta": 1e6, "m": 1e6, "mn": 1e6, "million": 1e6,
    "miliar": 1e9, "milyar": 1e9, "b": 1e9, "bn": 1e9, "billion": 1e9,
}
_CURRENCY_MAP = {
    "rp": "IDR", "rp.": "IDR", "idr": "IDR",
    "$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
    "¥": "JPY", "jpy": "JPY",
}
_NUM_BODY_RE = re.compile(r"^[+\-−]?\d[\d.,]*$")
_INLINE_AMOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9.,])"
    r"\(?[+\-−]?(?:(?:Rp\.?|IDR|USD|EUR|GBP|JPY|[$€£¥])\s*)?"
    r"\d+(?:[.,]\d+)*"
    r"(?:\s*(?:miliar|milyar|billion|bn|juta|jt|million|mn|ribu|rb|k|%|IDR|USD|EUR|GBP|JPY)\b|%)?"
    r"\)?",
    re.I,
)


def parse_amount(raw: str) -> tuple[float | None, str | None]:
    """Parse plain numbers, US/ID thousands separators, currency prefixes, and scale suffixes.

    Returns `(value, unit)` where `unit` is a normalized currency code (`IDR`, `USD`, ...),
    `'%'`, or `None` when unitless. Returns `(None, None)` if malformed or ambiguous.
    """
    s = raw.strip()
    if not s:
        return None, None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    if s.startswith(("-", "−")):
        neg = not neg
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    unit: str | None = None
    m_pre = _CURRENCY_PREFIX_RE.match(s)
    if m_pre:
        unit = _CURRENCY_MAP[m_pre.group(1).lower()]
        s = s[m_pre.end():].strip()
        if s.startswith(("-", "−")):
            neg = not neg
            s = s[1:].strip()

    if s.endswith("%"):
        if unit is not None:
            return None, None
        unit = "%"
        s = s[:-1].strip()

    m_suf_cur = _CURRENCY_SUFFIX_RE.search(s)
    if m_suf_cur:
        suf_unit = _CURRENCY_MAP[m_suf_cur.group(1).lower()]
        if unit is not None and unit != suf_unit:
            return None, None
        unit = suf_unit
        s = s[:m_suf_cur.start()].strip()

    scale = 1.0
    m_scale = _SCALE_SUFFIX_RE.search(s)
    if m_scale:
        scale = _SCALE_FACTORS[m_scale.group(1).lower()]
        s = s[:m_scale.start()].strip()

    if not s or not _NUM_BODY_RE.fullmatch(s):
        return None, None
    if s.endswith((".", ",")) or s.startswith((".", ",")):
        return None, None

    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        if last_comma > last_dot:
            # Indonesian / EU format: 9.750.000,50
            int_part, frac_part = s.split(",", 1)
            if "," in frac_part or "." in frac_part:
                return None, None
            groups = int_part.split(".")
            if not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
                return None, None
            cleaned = "".join(groups) + "." + frac_part
        else:
            # US / EN format: 10,000.50
            int_part, frac_part = s.split(".", 1)
            if "." in frac_part or "," in frac_part:
                return None, None
            groups = int_part.split(",")
            if not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
                return None, None
            cleaned = "".join(groups) + "." + frac_part
    elif has_dot:
        groups = s.split(".")
        if len(groups) >= 3:
            if not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
                return None, None
            cleaned = "".join(groups)
        else:
            # Single dot: treat 3 trailing digits as thousands only for IDR (e.g. Rp18.500)
            if unit == "IDR" and len(groups[1]) == 3 and 1 <= len(groups[0]) <= 3 and groups[0] != "0":
                cleaned = "".join(groups)
            else:
                cleaned = s
    elif has_comma:
        groups = s.split(",")
        if len(groups) >= 3:
            if not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
                return None, None
            cleaned = "".join(groups)
        else:
            # Single comma: 3 trailing digits with non-zero leading digits -> thousands (e.g. 10,000)
            if len(groups[1]) == 3 and 1 <= len(groups[0]) <= 3 and groups[0] != "0":
                cleaned = "".join(groups)
            else:
                cleaned = groups[0] + "." + groups[1]
    else:
        cleaned = s

    try:
        val = float(cleaned) * scale
    except ValueError:
        return None, None
    if not math.isfinite(val):
        return None, None
    return (-val if neg else val), unit


def _parse_number(raw: str) -> float | None:
    val, _ = parse_amount(raw)
    return val


def extract_grounded_numbers(text: str) -> set[float]:
    """Extract all canonical numeric values stated in user text or verified evidence."""
    found: set[float] = set()
    for match in _INLINE_AMOUNT_RE.finditer(text):
        token = match.group(0).strip()
        val, _ = parse_amount(token)
        if val is not None:
            found.add(val)
            found.add(abs(val))
        # Also collect raw unscaled number if scale suffix was attached (e.g. 60 from "Rp60 juta")
        m_plain = re.search(r"\d+(?:[.,]\d+)*", token)
        if m_plain:
            raw_val, _ = parse_amount(m_plain.group(0))
            if raw_val is not None:
                found.add(raw_val)
    return found


_PERIOD_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("week", re.compile(r"^(?:w|wk|week|minggu|mgg)[\s\-_]*(\d+)$", re.I)),
    ("month", re.compile(r"^(?:mo|mon|month|bulan|bln)[\s\-_]*(\d+)$", re.I)),
    ("m_short", re.compile(r"^m[\s\-_]*(\d+)$", re.I)),
    ("quarter", re.compile(r"^(?:q|qtr|quarter|kuartal)[\s\-_]*(\d+)$", re.I)),
    ("period", re.compile(r"^(?:p|per|period|periode)[\s\-_]*(\d+)$", re.I)),
    ("day", re.compile(r"^(?:d|day|h|hari)[\s\-_]*(\d+)$", re.I)),
    ("year", re.compile(r"^(?:y|yr|year|thn|tahun)[\s\-_]*(\d+)$", re.I)),
)
_MONTH_NAMES = {
    "jan": "jan", "january": "jan", "januari": "jan",
    "feb": "feb", "february": "feb", "februari": "feb",
    "mar": "mar", "march": "mar", "maret": "mar",
    "apr": "apr", "april": "apr",
    "may": "may", "mei": "may",
    "jun": "jun", "june": "jun", "juni": "jun",
    "jul": "jul", "july": "jul", "juli": "jul",
    "aug": "aug", "august": "aug", "agu": "aug", "agustus": "aug",
    "sep": "sep", "sept": "sep", "september": "sep",
    "oct": "oct", "october": "oct", "okt": "oct", "oktober": "oct",
    "nov": "nov", "november": "nov",
    "dec": "dec", "december": "dec", "des": "dec", "desember": "dec",
}


def period_aliases(label: str, header: str = "") -> set[str]:
    """Return normalized match keys for a period label (e.g. W1 <-> Week 1, M1 <-> Minggu 1)."""
    clean = " ".join(label.strip().casefold().split())
    if not clean:
        return set()
    keys = {clean}
    hdr = header.strip().casefold()
    if clean in _MONTH_NAMES:
        keys.add(f"cal:{_MONTH_NAMES[clean]}")
    for family, pat in _PERIOD_PATTERNS:
        m = pat.fullmatch(clean)
        if m:
            num = str(int(m.group(1)))
            if family == "m_short":
                # 'M1' means Minggu 1 under a Minggu/Week header, or Month 1 under Month/Bulan
                if any(k in hdr for k in ("minggu", "week", "wk")):
                    keys.add(f"week:{num}")
                elif any(k in hdr for k in ("month", "bulan", "bln")):
                    keys.add(f"month:{num}")
                else:
                    keys.add(f"week:{num}")
                    keys.add(f"month:{num}")
                keys.add(f"m_short:{num}")
            else:
                keys.add(f"{family}:{num}")
    return keys


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Return the headers and raw cells from a CSV, TSV (Excel/Sheets), or Markdown table."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("table must contain a header row and at least one data row")

    rows: list[list[str]] = []
    if lines[0].startswith("|") and lines[0].endswith("|"):
        for ln in lines:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells):
                continue
            rows.append(cells)
    elif "\t" in lines[0] and sum("\t" in ln for ln in lines[:4]) >= 2:
        for ln in lines:
            cells = [c.strip() for c in ln.split("\t")]
            if any(cells):
                rows.append(cells)
    else:
        reader = csv.reader(io.StringIO("\n".join(lines)))
        rows = [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)]

    if len(rows) < 2:
        raise ValueError("no data rows parsed from table")
    return rows[0], rows[1:]


def summarize_table(text: str) -> dict:
    """Parse a CSV or Markdown table and return exact numeric summaries per column."""
    headers, data_rows = parse_table(text)
    col_stats: dict[str, dict] = {}
    for col_idx, col_name in enumerate(headers):
        vals = []
        for r in data_rows:
            if col_idx < len(r):
                v = _parse_number(r[col_idx])
                if v is not None:
                    vals.append(v)
        if vals:
            stat = {
                "count": len(vals),
                "sum": round(sum(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
            if len(vals) >= 2 and vals[0] != 0:
                stat["first_to_last_delta"] = round(vals[-1] - vals[0], 4)
                stat["first_to_last_pct"] = round(((vals[-1] - vals[0]) / abs(vals[0])) * 100.0, 2)
            col_stats[col_name] = stat

    return {
        "row_count": len(data_rows),
        "columns": headers,
        "numeric_summary": col_stats,
    }


@dataclass
class FactLedger:
    """Session-scoped confirmed facts vs hypotheses with stable `[1]`, `[2]` citations."""

    facts: list[tuple[str, str]] = field(default_factory=list)  # (source, fact_text)

    def add(self, text: str, source: str = "client") -> int:
        clean = " ".join(text.strip().split())
        if not clean:
            return len(self.facts)
        for idx, (_, existing) in enumerate(self.facts, start=1):
            if existing.casefold() == clean.casefold():
                return idx
        self.facts.append((source, clean))
        return len(self.facts)

    def format_evidence(self) -> str:
        return "\n".join(f"[{i}] ({src}) {txt}" for i, (src, txt) in enumerate(self.facts, start=1))


_MONETARY_ASK_RE = re.compile(
    r"\b(?:laba\s+(?:bersih|kotor)|keuntungan|biaya\s+sewa|biaya\s+operasional|harga|omzet|pendapatan|"
    r"net\s+profit|gross\s+profit|lease\s+.*rate|rent|cac|customer\s+acquisition\s+cost|cost\s+per)\b",
    re.I,
)
_PERCENT_ASK_RE = re.compile(r"\b(?:berapa\s+persen|persentase|margin|churn\s+rate|conversion\s+rate|diskon\s+rata-rata)\b", re.I)
_PEOPLE_OR_COUNT_UNIT_RE = re.compile(r"\d+(?:[.,]\d+)*\s*(?:orang|pengunjung|people|visitors?|perusahaan|companies|accounts?|transaksi|orders?|paket)\b", re.I)
_HAS_MONEY_OR_PCT_RE = re.compile(r"(?:rp\.?\s*\d|idr\s*\d|\$\s*\d|usd\s*\d|\d+(?:[.,]\d+)?\s*(?:%|persen|percent|juta|ribu|miliar))", re.I)
_ABSTAIN_HINT_RE = re.compile(
    r"(?:tidak\s+(?:disebutkan|menyebutkan|tercantum|tersedia|menjawab|diketahui|tahu)|"
    r"belum\s+(?:tersedia|disebutkan|tercantum|bisa)|"
    r"not\s+(?:specified|mentioned|provided|available|stated|enough)|"
    r"do(?:es)?\s+not\s+(?:answer|contain|mention|specify|provide|state)|don['’]t\s+know|cannot\s+(?:confirm|determine|verify))",
    re.I,
)


def verify_grounded_answer(
    question: str,
    evidence: str,
    answer: str,
    *,
    indonesian: bool = False,
) -> tuple[bool, str | None]:
    """Post-generation grounding check for evidence-based answers.

    Catches two high-confidence hallucination classes before returning to users:
    1. Answering a monetary/percentage question with a plain headcount/order count
       (e.g. claiming net profit is '4.120 orang') with zero money/% in the answer.
    2. Introducing numbers that do not appear anywhere in the question or evidence.
    """
    ans = answer.strip()
    if not ans or _ABSTAIN_HINT_RE.search(ans):
        return True, None
    clean_ans = re.sub(r"\[\d+\]", "", ans)

    # Check 1: unit/category mismatch (asking money/margin/discount% -> answering headcount/companies)
    asks_money_or_pct = bool(_MONETARY_ASK_RE.search(question) or _PERCENT_ASK_RE.search(question))
    if asks_money_or_pct and _PEOPLE_OR_COUNT_UNIT_RE.search(clean_ans) and not _HAS_MONEY_OR_PCT_RE.search(clean_ans):
        fallback = (
            "Kutipan yang tersedia belum menyebutkan angka keuangan atau persentase yang ditanyakan. Bisa bagikan laporan yang memuat data tersebut?"
            if indonesian else
            "The provided excerpt does not state the requested financial figure or percentage. Could you share the report containing that data?"
        )
        return False, fallback

    # Check 2 & 3: ungrounded numeric claims and currency mismatches vs question + evidence
    source_blob = question + "\n" + evidence
    allowed = extract_grounded_numbers(source_blob)
    allowed_currencies: set[str] = set()
    for src_match in _INLINE_AMOUNT_RE.finditer(source_blob):
        _, src_unit = parse_amount(src_match.group(0))
        if src_unit and src_unit != "%":
            allowed_currencies.add(src_unit)

    if allowed:
        for match in _INLINE_AMOUNT_RE.finditer(clean_ans):
            start = match.start()
            # Ignore hyphenated non-data tokens such as "top-3" or "24-hour"
            if start > 0 and clean_ans[start - 1] == "-" and start >= 2 and clean_ans[start - 2].isalpha():
                continue
            if start >= 1 and clean_ans[start] == "-" and clean_ans[start - 1].isalpha():
                continue
            val, ans_unit = parse_amount(match.group(0))
            if val is None:
                continue
            if allowed_currencies and ans_unit and ans_unit != "%" and ans_unit not in allowed_currencies:
                cur_list = ", ".join(sorted(allowed_currencies))
                fallback = (
                    f"Satuan mata uang jawaban tidak sesuai dengan bukti sumber ({cur_list}). Bisa konfirmasi angka dan mata uang pada dokumen sumber?"
                    if indonesian else
                    f"The answer's currency did not match the source evidence ({cur_list}). Please confirm the exact currency from the source document."
                )
                return False, fallback
            if not any(abs(val - a) <= max(1e-6, 1e-4 * abs(a)) or abs(abs(val) - abs(a)) <= max(1e-6, 1e-4 * abs(a))
                       for a in allowed):
                fallback = (
                    "Saya belum bisa memverifikasi angka tersebut dari kutipan yang tersedia. Bisa periksa atau lengkapi dokumen sumbernya?"
                    if indonesian else
                    "I couldn't verify that figure from the provided excerpts. Could you share the exact source numbers?"
                )
                return False, fallback

    return True, None
