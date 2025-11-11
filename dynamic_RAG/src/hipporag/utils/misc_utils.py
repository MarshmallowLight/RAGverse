from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging
from pathlib import Path
from .typing import Triple,Quad
from .llm_utils import filter_invalid_triples
from datetime import datetime, date

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }


def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }

    triple_output_dict = {
        item['idx']: TripleRawOutput(
            chunk_id=item['idx'],
            response=None,
            metadata={},
            triples=item.get('extracted_triples', []),
        )
        for item in corpus_openie_results
    }
    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Tuple[str, ...]]]) -> (List[str], List[List[str]]):
    """Return (graph_nodes, chunk_triple_entities). Accepts (s,r,o) or (s,r,o,t)."""
    chunk_triple_entities = []
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities



def flatten_facts(chunk_triples: List[List[Tuple[str, ...]]]) -> List[Tuple[str, ...]]:
    """Flatten & dedupe. 4-tuples keep t; 3-tuples -> (s,r,o,None)."""
    graph_triples = []
    for triples in chunk_triples:
        for t in triples:
            if isinstance(t, (list, tuple)):
                if len(t) == 4:
                    graph_triples.append(tuple(t))
                elif len(t) == 3:
                    s, r, o = t
                    graph_triples.append((s, r, o, None))
                else:
                    logger.warning(f"Invalid triple length {len(t)}: {t}")
            else:
                logger.warning(f"Invalid triple type: {type(t)} -> {t}")
    return list(set(graph_triples))

# def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
#     graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
#     for triples in chunk_triples:
#         graph_triples.extend([tuple(t) for t in triples])
#     graph_triples = list(set(graph_triples))
#     return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val



def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

def safe_parse_date(d: Any) -> Optional[datetime]:
    """尽力把 YYYY[-MM[-DD]] / 自由文本解析为 datetime；失败返回 None。"""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    s = str(d).strip()
    if not s or s.lower() in {"na", "none", "null", "unknown"}:
        return None

    m = re.match(r"^(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?$", s)
    if m:
        y = int(m.group(1))
        M = int(m.group(2)) if m.group(2) else 1
        d = int(m.group(3)) if m.group(3) else 1
        try:
            return datetime(y, M, d)
        except Exception:
            return None

    try:
        from dateutil import parser
        return parser.parse(s, default=datetime(1970, 1, 1))
    except Exception:
        return None


def extract_anchor_date_from_query(query: str) -> Optional[datetime]:
    """从 query 中抽取日期：优先 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD；再退化为纯 YYYY。"""
    q = query.strip()
    # 1) YYYY[-/.]MM[-/.]DD
    m = re.search(r"\b(\d{4})([-/.])(\d{1,2})\2(\d{1,2})\b", q)
    if m:
        y, M, d = int(m.group(1)), int(m.group(3)), int(m.group(4))
        try:
            return datetime(y, M, d)
        except Exception:
            pass
    # 2) YYYY
    m = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except Exception:
            pass
    # 3) 可选：更自由的解析（带月份英文或杂质），失败无伤大雅
    try:
        from dateutil import parser
        dt = parser.parse(q, fuzzy=True, default=datetime(1970, 1, 1))
        # 避免把“今天 12:30”这种解析成 1970 年
        if dt.year >= 1900:
            return datetime(dt.year, dt.month, dt.day)
    except Exception:
        pass
    return None


def infer_time_direction(query: str, default: Optional[str] = None) -> Optional[str]:
    """
    推断时间方向约束：'before' / 'after' / None。
    使用词边界匹配，兼容标点。
    """
    q = query.lower()

    # 英文 before
    if (re.search(r"\bbefore\b", q)
        or "earlier than" in q
        or "prior to" in q
        or re.search(r"\buntil\b", q)
        or "no later than" in q
        or "on or before" in q):
        return "before"

    # 英文 after
    if (re.search(r"\bafter\b", q)
        or re.search(r"\bsince\b", q)
        or "later than" in q
        or "following" in q
        or "on or after" in q
        or re.search(r"\bfrom\b", q)):  # e.g., "from 2010-03-24"
        return "after"

    # 中文 before
    if any(k in query for k in ["之前", "以前", "早于", "不晚于", "截至"]):
        return "before"

    # 中文 after
    if any(k in query for k in ["之后", "以后", "晚于", "不早于", "自从"]):
        return "after"

    return default


def coerce_to_idx_score_list(query_fact_scores: Any) -> List[Tuple[Any, float]]:
    """统一为 [(idx_or_fact, score)] 列表。"""
    if hasattr(query_fact_scores, "shape"):
        arr = np.asarray(query_fact_scores).reshape(-1)
        return [(i, float(arr[i])) for i in range(arr.shape[0])]
    if isinstance(query_fact_scores, dict):
        return [(k, float(v)) for k, v in query_fact_scores.items()]
    if isinstance(query_fact_scores, (list, tuple)):
        if not query_fact_scores:
            return []
        first = query_fact_scores[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return [(x[0], float(x[1])) for x in query_fact_scores]
        try:
            return [(i, float(s)) for i, s in enumerate(query_fact_scores)]
        except Exception:
            pass
    try:
        return [(i, float(s)) for i, s in enumerate(list(query_fact_scores))]
    except Exception:
        raise ValueError("Unsupported query_fact_scores format.")


def ensure_quad(fact: Union[Tuple, Dict, str]) -> Quad:
    """保证返回 Quad=(s,r,o,t)，无时间则 t = ''。"""
    if isinstance(fact, (list, tuple)):
        if len(fact) == 4:
            s, r, o, t = fact
            return str(s), str(r), str(o), ("" if t is None else str(t))
        if len(fact) == 3:
            s, r, o = fact
            return str(s), str(r), str(o), ""
    if isinstance(fact, dict):
        s = fact.get("s") or fact.get("subj") or fact.get("head") or fact.get("subject")
        r = fact.get("r") or fact.get("rel") or fact.get("relation", "")
        o = fact.get("o") or fact.get("obj") or fact.get("tail") or fact.get("object")
        t = fact.get("t") or fact.get("time") or fact.get("date") or fact.get("timestamp")
        return str(s), str(r), str(o), ("" if t is None else str(t))
    return str(fact), "", "", ""


def fetch_fact_by_index_generic(obj: Any, idx: Any) -> Quad:
    """根据 idx 从 obj 的 fact 容器中取 Quad。"""
    if hasattr(obj, "fact_id_to_fact"):
        rec = obj.fact_id_to_fact[idx]
    elif hasattr(obj, "facts"):
        rec = obj.facts[idx]
    elif hasattr(obj, "fact_table"):
        rec = obj.fact_table[idx]
    elif hasattr(obj, "all_facts"):
        rec = obj.all_facts[idx]
    else:
        raise KeyError("No fact container found on `self` for idx.")
    return ensure_quad(rec)


def time_order_key(
    anchor_dt: Optional[datetime],
    time_dir: Optional[str],
    fact_dt: Optional[datetime],
    neg_score: float
) -> Tuple[int, float, float]:
    """排序 key：(是否违背约束, 与锚点的天数距离, 负分数)。"""
    violate = 0
    if anchor_dt and time_dir and fact_dt:
        if time_dir == "before" and not (fact_dt <= anchor_dt):
            violate = 1
        elif time_dir == "after" and not (fact_dt >= anchor_dt):
            violate = 1

    dist_days = abs((fact_dt - anchor_dt).days) if (anchor_dt and fact_dt) else float("inf")
    return (violate, float(dist_days), float(neg_score))