import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date as _date
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
import ast
import re, datetime as _dt
from itertools import islice
from typing import Iterable
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time,random
import torch
import pickle
import math
from collections import defaultdict, Counter
from pathlib import Path
from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score, evaluate_hit_at_k_f1
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter,TemporalAwareFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .information_extraction.openie_openai import ChunkInfo   # TypedDict 定义就在 openie.py 里
logger = logging.getLogger(__name__)

def triples_to_bound_passages(txt_path: Path) -> List[Dict]:
        """
        读取四元组 <s, r, o, date>，拼成与原先 insert 的句子完全一致的文本，
        聚合为 [{ "content": sentence, "triples": [(s,r,o,date), ...] }, ...]
        """
        sent2trips: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                s, r, o, date = line.rstrip("\n").split("\t")
                # 必须与原先 triples_to_docs 的句式 100% 一致！
                sent = f"{date} {s} {r.replace('_', ' ')} {o}."
                sent2trips[sent].append((s, r, o, date))
        # 变成列表记录
        return [{"content": sent, "triples": trips} for sent, trips in sent2trips.items()]

class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 dynamic = None,
                 temporal_dir = None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint
        if temporal_dir is not None:
            self.global_config.temporal_dir = temporal_dir
            logger.info(f"Loading the temporal model, please wait...")
            with open(temporal_dir, "rb") as f:
                self.temporal_model = pickle.load(f)

        self.global_config.dynamic = dynamic

        self._tok_prompt_sum = 0
        self._tok_completion_sum = 0
        self._tok_total_sum = 0
        self._tok_query_cnt = 0

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None
    
        # Timing buckets
        self.ppr_time = getattr(self, "ppr_time", 0.0)

        # Lazy mapping: "s r o" -> row index in self.fact_embeddings
        self.fact_key_to_index = getattr(self, "fact_key_to_index", None)
                # PPR / weighting hyperparameters
        self._ppr_alpha = 0.6     # weight for rank-based position signal
        self._ppr_beta  = 0.7     # geometric decay for fact ranks: w_j = beta^(j-1)
        self._ppr_gamma = 0.2    # restart probability
        self._ppr_tau   = None    # Dirichlet smoothing; if None -> 1/|R|
        self._ppr_max_iters = 200
        self._ppr_tol       = 1e-5
        self._ppr_topk_rules      = 5
        self._ppr_topk_rule_facts = 5
        # in HippoRAG.__init__(...)
        self._ppr_graph_mode = 'undirected'   # options: 'directed' | 'reversed' | 'undirected' | 'blend'
        self._ppr_edge_merge = 'sum'          # how to merge opposite edges in undirected: 'sum' | 'max' | 'avg' | 'keep'
        self._ppr_blend_lambda = 0.5          # for 'blend': pi = λ * pi_forward + (1-λ) * pi_backward
    def _sparsify_scored_facts(self,
                           scored: list,
                           k: int = 12,
                           max_per_pair: int = 1,
                           max_per_entity: int = 2):
        """
        Reduce redundancy among facts:
        - keep at most `max_per_pair` per (s,o) pair,
        - keep at most `max_per_entity` mentions per entity (as s or o),
        - return at most `k` items overall.
        `scored` is a list of (score, (s,r,o,t)).
        """
        out = []
        seen_pair = defaultdict(int)
        seen_ent  = defaultdict(int)

        for score, f in scored:
            s, r, o, t = self._norm_fact4(f)
            pair = (s, o)

            if seen_pair[pair] >= max_per_pair:
                continue
            if seen_ent[s] >= max_per_entity or seen_ent[o] >= max_per_entity:
                continue

            out.append((score, (s, r, o, t)))
            seen_pair[pair] += 1
            seen_ent[s] += 1
            seen_ent[o] += 1

            if len(out) >= k:
                break
        return out
    
    def _parse_date_soft(self, t):
        """Parse 'YYYY' / 'YYYY-MM' / 'YYYY-MM-DD' (also supports '/'); return datetime.date or None."""
        if not t:
            return None
        s = str(t).strip()
        m = re.match(r"^(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?$", s)
        if not m:
            return None
        y = int(m.group(1)); mth = int(m.group(2) or 1); d = int(m.group(3) or 1)
        try:
            return date(y, mth, d)
        except Exception:
            return None

    def _all_times_for_triple(self, s, r, o):
        """
        Look up all time strings t for a triple (s,r,o) from the temporal model.
        Return a de-duplicated ascending list (parseable dates first, others last).
        """
        tm = getattr(self, "temporal_model", None)
        if tm is None:
            return []
        if hasattr(tm, "fact3_to_rules"):
            rules = tm.fact3_to_rules.get((s, r, o), [])
        elif hasattr(tm, "rules_for_fact"):
            rules = tm.rules_for_fact((s, r, o)) or []
        else:
            rules = []
        if not rules:
            return []

        rule_to_facts = getattr(tm, "rule_to_facts", {})
        ts = []
        for rule in rules:
            for f in rule_to_facts.get(rule, []):
                if isinstance(f, (list, tuple)) and len(f) == 4 and f[:3] == (s, r, o) and f[3]:
                    ts.append(str(f[3]))
                elif isinstance(f, dict) and f.get("s")==s and f.get("r")==r and f.get("o")==o and f.get("t"):
                    ts.append(str(f["t"]))

        seen, uniq = set(), []
        for t in ts:
            if t not in seen:
                uniq.append(t); seen.add(t)

        def _key(x):
            d = self._parse_date_soft(x)
            return (1, x) if d is None else (0, d)  # parseable first (ascending)
        uniq.sort(key=_key)
        return uniq

    def _choose_rep_time(self, fact, anchor_date=None):
        """
        Choose a representative time for a fact:
        1) prefer fact's own t if parseable;
        2) else choose from TM the closest to anchor_date if given;
        3) else choose the latest parseable time from TM;
        Return (time_str or None, parsed_date or None).
        """
        s, r, o = fact[:3]
        # prefer the fact's own time
        if len(fact) >= 4 and fact[3]:
            d = self._parse_date_soft(fact[3])
            if d is not None:
                return str(fact[3]), d

        # fallback to TM
        tlist = self._all_times_for_triple(s, r, o)
        if not tlist:
            return None, None

        if anchor_date is not None:
            # pick closest to anchor
            best_t, best_delta = None, None
            for t in tlist:
                d = self._parse_date_soft(t)
                if d is None:
                    continue
                delta = abs((d - anchor_date).days)
                if best_delta is None or delta < best_delta:
                    best_t, best_delta = t, delta
            if best_t is not None:
                return best_t, self._parse_date_soft(best_t)

        # otherwise pick the latest parseable time
        for t in reversed(tlist):
            d = self._parse_date_soft(t)
            if d is not None:
                return t, d

        return None, None

    def _relation_temporal_sort(self, top_idx, top_facts):
        """
        Keep item 0 as anchor (highest similarity).
        Reorder items 1..n-1 by:
        1) relation match with the anchor (same relation first),
        2) then absolute day distance to the anchor's representative date (smaller is closer),
        3) then original index (stable).
        Facts themselves are NOT mutated here.
        Returns (sorted_idx, sorted_facts, anchor_date).
        """
        if not top_facts or len(top_facts) <= 1:
            return top_idx, top_facts, None

        # anchor
        a_fact = top_facts[0]
        a_rel  = a_fact[1]
        a_t, a_date = self._choose_rep_time(a_fact, anchor_date=None)  # prefer fact's own t
        if a_date is None:
            # fallback: latest parseable time from TM
            tlist = self._all_times_for_triple(*a_fact[:3])
            for t in reversed(tlist):
                d = self._parse_date_soft(t)
                if d is not None:
                    a_t, a_date = t, d
                    break

        # score others
        scored = []  # (rel_mismatch, delta_days or inf, original_index)
        for i in range(1, len(top_facts)):
            f = top_facts[i]
            rel_mismatch = 0 if f[1] == a_rel else 1
            if a_date is not None:
                t_str, d = self._choose_rep_time(f, anchor_date=a_date)
                delta = abs((d - a_date).days) if d is not None else float("inf")
            else:
                delta = float("inf")
            scored.append((rel_mismatch, delta, i))

        scored.sort(key=lambda x: (x[0], x[1], x[2]))  # same relation first; closer in time; stable

        order = [0] + [i for (_, _, i) in scored]
        top_idx_sorted   = [top_idx[j]   for j in order]
        top_facts_sorted = [top_facts[j] for j in order]
        return top_idx_sorted, top_facts_sorted, a_date

    def _build_quads_for_ppr(self, top_facts_sorted, anchor_date, limit=20):
        """
        Build up to 'limit' quads (s,r,o,t_rep). For each fact, attach a representative time
        chosen by _choose_rep_time (closest to anchor_date if available).
        """
        out = []
        for f in top_facts_sorted[:limit]:
            t_str, _ = self._choose_rep_time(f, anchor_date=anchor_date)
            s, r, o = f[:3]
            out.append((s, r, o, t_str))
        return out
   


    # ---------- normalization / rendering ----------
    def _tok(self, s: str) -> str:
        """Lowercase and collapse whitespace to underscores."""
        return re.sub(r"\s+", "_", str(s).strip().lower())

    def _norm_fact4(self, f):
        """Accept 3- or 4-tuple; always return (s, r, o, t)."""
        if len(f) == 4:
            s, r, o, t = f
        elif len(f) == 3:
            s, r, o = f
            t = None
        else:
            raise ValueError(f"fact must be 3 or 4-tuple, got: {f}")
        return (self._tok(s), self._tok(r), self._tok(o), t)

    def _fact_key(self, f4) -> str:
        """Canonical key used to locate the embedding row: 's r o'."""
        s, r, o, _ = self._norm_fact4(f4)
        return f"{s} {r} {o}"

    def _fact_to_doc(self, f):
        """
        将三/四元组转成 LLM 友好的描述：
        On {date}, {s} {r} {o}
        - 不替换下划线，保持你给的例子风格
        - 如果没有日期，降级为 On unknown_time, ...
        - 不在末尾加句号，和你的示例一致
        """
        s, r, o = f[0], f[1], f[2]
        date = f[3] if len(f) >= 4 and f[3] else "unknown_time"

        # 可选的strip，避免意外空格
        s = s.strip()
        r = r.strip()
        o = o.strip()
        date = date.strip()

        return f"On {date}, {s} {r} {o}"

    def _ensure_fact_index_mapping(self, *, needed_keys: Iterable[str] = None, batch_size: int = 50000):
        """
        Build a mapping from normalized fact key 's r o' -> row index in self.fact_embeddings.
        Uses HippoRAG's stores:
        - self.fact_node_keys: list of row ids aligned with fact_embeddings order
        - self.fact_embedding_store.get_rows(ids): returns {id: {"content": "(s, r, o)" or "(s,r,o,t)"}}

        Args:
            needed_keys: optional set/list of keys ('s r o') to resolve; if provided, stop after all are found
            batch_size:  number of rows to read per batch

        The mapping is cached in self.fact_key_to_index.
        """
        # Reuse if already built and non-empty
        if isinstance(getattr(self, "fact_key_to_index", None), dict) and self.fact_key_to_index:
            return

        if not hasattr(self, "fact_node_keys") or len(self.fact_node_keys) == 0:
            logger.warning("No fact_node_keys available; cannot build fact_key_to_index.")
            self.fact_key_to_index = {}
            return

        mapping = {}
        need = set(needed_keys) if needed_keys is not None else None

        total = len(self.fact_node_keys)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = self.fact_node_keys[start:end]

            # Bulk fetch rows
            rows = self.fact_embedding_store.get_rows(batch_ids)  # {id: row_dict}
            for offset, row_id in enumerate(batch_ids):
                row = rows.get(row_id)
                if not row:
                    continue
                cont = row.get("content")
                if cont is None:
                    continue

                # Parse content -> tuple
                try:
                    if isinstance(cont, (tuple, list)):
                        fact = tuple(cont)
                    else:
                        fact = ast.literal_eval(cont)  # safe eval: "('a','b','c')" or "('a','b','c','2005-01-11')"
                    s, r, o, _t = self._norm_fact4(fact)  # normalize: lowercase + spaces->underscores; force 4-tuple
                except Exception:
                    continue  # skip malformed

                key = f"{s} {r} {o}"
                mapping[key] = start + offset  # global row index aligned with self.fact_embeddings

                if need is not None:
                    if key in need:
                        need.remove(key)
                    if not need:
                        self.fact_key_to_index = mapping
                        return

        self.fact_key_to_index = mapping
        if not self.fact_key_to_index:
            logger.warning("Built fact_key_to_index but it is empty after scanning the store.")
    

    # ---------- map Top-K facts -> rules ----------
    def _rules_from_topk_facts(self, topk_facts):
        """Return (rules_set, rule_hits, facts_norm4)."""
        tm = self.temporal_model
        rules_set = set()
        rule_hits = defaultdict(list)
        facts_norm4 = [self._norm_fact4(f) for f in topk_facts]
        for j, f4 in enumerate(facts_norm4, start=1):
            hit_rules = tm.rules_for_fact(f4)   # supports 3/4; we pass 4 (t=None allowed)
            for r in hit_rules:
                rule_hits[r].append(j)
                rules_set.add(r)
        return rules_set, rule_hits, facts_norm4

    # ---------- build restart distribution v ----------
    def _build_restart_distribution(self, rules_set, rule_hits):
        """v_r ∝ (1 - alpha) * Ctilde_r + alpha * Ptilde_r, with Dirichlet smoothing tau."""
        if not rules_set:
            return {}
        tm = self.temporal_model
        alpha, beta = self._ppr_alpha, self._ppr_beta
        cover = {r: len(tm.facts_for_rule(r)) for r in rules_set}
        pos   = {r: sum(beta ** (j - 1) for j in rule_hits[r]) for r in rules_set}
        sum_c = sum(cover.values()) or 1.0
        sum_p = sum(pos.values())   or 1.0
        Ctilde = {r: cover[r] / sum_c for r in rules_set}
        Ptilde = {r: pos[r]   / sum_p for r in rules_set}
        s = {r: (1.0 - alpha) * Ctilde[r] + alpha * Ptilde[r] for r in rules_set}
        tau = (1.0 / len(rules_set)) if (self._ppr_tau is None) else self._ppr_tau
        v = {r: (s[r] + tau) for r in rules_set}
        Z = sum(v.values())
        return {r: v[r] / Z for r in rules_set}

    def _run_ppr_over_rules(self, v, graph_mode: str = None, merge: str = None, blend_lambda: float = None):
        """
        Personalized PageRank over the rule graph with configurable directionality.
        - graph_mode: 'directed' | 'reversed' | 'undirected' | 'blend'
        * directed  : original forward edges only
        * reversed  : use reversed edges only (helps 'before' queries)
        * undirected: symmetrize the graph; for each unordered pair {u,v}, merge weights
        * blend     : run PPR on directed and reversed, then pi = λ*pi_fwd + (1-λ)*pi_rev
        - merge: how to merge opposite edges in 'undirected' mode:
        * 'sum' (default) : w(u,v)_undirected = w(u→v) + w(v→u)
        * 'max'           : max of two directions
        * 'avg'           : (w(u→v)+w(v→u))/2
        * 'keep'          : keep nonzero one if only one exists, else use w(u→v)
        """
        tm = self.temporal_model
        gamma = self._ppr_gamma
        max_iters = self._ppr_max_iters
        tol = self._ppr_tol

        if graph_mode is None:
            graph_mode = getattr(self, "_ppr_graph_mode", "directed")
        if merge is None:
            merge = getattr(self, "_ppr_edge_merge", "sum")
        if blend_lambda is None:
            blend_lambda = getattr(self, "_ppr_blend_lambda", 0.5)

        # ---- collect directed edge weights once ----
        dir_w = {}  # (u,v) -> float weight
        edges = getattr(tm, "edges", {})  # {(u_rule, v_rule): [facts_on_edge] or weight-like}
        for (u, v2), flist in edges.items():
            w = float(len(flist)) if isinstance(flist, (list, set, tuple)) else float(flist) if isinstance(flist, (int, float)) else 1.0
            dir_w[(u, v2)] = dir_w.get((u, v2), 0.0) + w

        def _power_iterate(out_edges, row_sum, nodes):
            # restart distribution v over nodes (already provided)
            pi = {u: 0.0 for u in nodes}
            for r, val in v.items():
                pi[r] = val

            for _ in range(max_iters):
                new_pi = {u: 0.0 for u in nodes}

                for u in out_edges:
                    if row_sum[u] <= 0:
                        continue
                    pu = (1.0 - gamma) * pi.get(u, 0.0)
                    denom = row_sum[u]
                    for (v2, w) in out_edges[u]:
                        new_pi[v2] += pu * (w / denom)

                for r, val in v.items():
                    new_pi[r] += gamma * val

                ssum = sum(new_pi.values()) or 1.0
                new_pi = {u: x / ssum for u, x in new_pi.items()}
                diff = sum(abs(new_pi[u] - pi.get(u, 0.0)) for u in nodes)
                pi = new_pi
                if diff < tol:
                    break
            return pi

        def _build_adj_from_dir_w(mode: str):
            """Return out_edges, row_sum, nodes according to the chosen mode (no blend)."""
            from collections import defaultdict
            out_edges = defaultdict(list)
            row_sum   = defaultdict(float)
            nodes = set()

            if mode == 'directed':
                for (u, v2), w in dir_w.items():
                    out_edges[u].append((v2, w))
                    row_sum[u] += w
                    nodes.add(u); nodes.add(v2)

            elif mode == 'reversed':
                for (u, v2), w in dir_w.items():
                    out_edges[v2].append((u, w))  # reverse
                    row_sum[v2] += w
                    nodes.add(u); nodes.add(v2)

            elif mode == 'undirected':
                # merge opposite directions into one undirected weight
                seen_pairs = set()
                for (u, v2), w_uv in dir_w.items():
                    if (v2, u) in seen_pairs:
                        continue
                    w_vu = dir_w.get((v2, u), 0.0)
                    if merge == 'sum':
                        w = w_uv + w_vu
                    elif merge == 'max':
                        w = max(w_uv, w_vu)
                    elif merge == 'avg':
                        w = 0.5 * (w_uv + w_vu)
                    elif merge == 'keep':
                        w = w_uv if w_uv > 0 else w_vu
                    else:
                        w = w_uv + w_vu  # default to sum

                    # add both directions with same weight
                    if w > 0:
                        out_edges[u].append((v2, w))
                        out_edges[v2].append((u, w))
                        row_sum[u]  += w
                        row_sum[v2] += w
                    nodes.add(u); nodes.add(v2)
                    seen_pairs.add((u, v2)); seen_pairs.add((v2, u))

            else:
                raise ValueError(f"Unknown graph_mode: {mode}")

            # handle dangling: at least self-loop to avoid zero row
            for u in list(nodes):
                if row_sum[u] <= 0:
                    out_edges[u].append((u, 1.0))
                    row_sum[u] = 1.0

            # include restart-only nodes
            nodes |= set(v.keys())
            for r in v.keys():
                if r not in out_edges:
                    out_edges[r].append((r, 1.0))
                    row_sum[r] = 1.0
            return out_edges, row_sum, nodes

        if graph_mode == 'blend':
            # forward
            out_f, row_f, nodes_f = _build_adj_from_dir_w('directed')
            pi_f = _power_iterate(out_f, row_f, nodes_f)
            # backward
            out_b, row_b, nodes_b = _build_adj_from_dir_w('reversed')
            pi_b = _power_iterate(out_b, row_b, nodes_b)
            # convex combination
            nodes = set(pi_f.keys()) | set(pi_b.keys()) | set(v.keys())
            lam = float(blend_lambda)
            pi = {}
            for u in nodes:
                pi[u] = lam * pi_f.get(u, 0.0) + (1.0 - lam) * pi_b.get(u, 0.0)
            # renormalize
            ssum = sum(pi.values()) or 1.0
            for u in list(pi.keys()):
                pi[u] /= ssum
            return pi

        # single-mode (directed / reversed / undirected)
        out_edges, row_sum, nodes = _build_adj_from_dir_w(graph_mode)
        return _power_iterate(out_edges, row_sum, nodes)


    # ---------- expand via PPR using ALL Top-K facts ----------
    def _ppr_expand_from_facts(self, topk_facts):
        """Return list of 4-tuple facts collected from Top PPR rules."""
        tm = self.temporal_model
        rules_set, rule_hits, facts_norm4 = self._rules_from_topk_facts(topk_facts)
        if not rules_set:
            return []
        v  = self._build_restart_distribution(rules_set, rule_hits)
        pi = self._run_ppr_over_rules(v)
        top_rules = sorted(pi.items(), key=lambda kv: kv[1], reverse=True)[: self._ppr_topk_rules]
        cand = set()
        for r, _ in top_rules:
            cand |= set(tm.facts_for_rule(r))  # 4-tuples with real time strings
        return [self._norm_fact4(f) for f in cand]

    # ---------- final: rank merged facts by get_fact_scores and render as docs ----------
    def _rank_rule_augmented_facts_as_docs(self, query: str, top_k_facts, num_to_retrieve: int):
        """
        1) normalize original Top-K
        2) expand via PPR using ALL Top-K as seeds
        3) merge & dedupe (original first)
        4) use get_fact_scores(query) and embedding-row mapping to score candidates
        5) sort desc, take top N, render as docs and aligned scores
        """
        existing_as_4 = [self._norm_fact4(f) for f in top_k_facts]
        try:
            expanded = self._ppr_expand_from_facts(top_k_facts)
        except Exception as e:
            logger.warning("Rule-augmented PPR expansion failed: %s", e)
            expanded = []
        # merge & dedupe (preserve original order first)
        seen = set()
        merged_facts_4 = []
        for f in (existing_as_4 + expanded):
            f4 = self._norm_fact4(f)
            if f4 not in seen:
                merged_facts_4.append(f4)
                seen.add(f4)
        if len(merged_facts_4) == 0:
            return [], []
        # scores for ALL facts
        scores_all = self.get_fact_scores(query)  # np.ndarray, aligned to self.fact_embeddings
        if scores_all is None or len(scores_all) == 0:
            return [], []
        # ensure mapping "s r o" -> row index
        self._ensure_fact_index_mapping()
        if not self.fact_key_to_index:
            logger.warning("fact_key_to_index is empty; cannot map facts to embedding rows.")
            return [], []
        # pick candidate scores
        scored = []
        for f4 in merged_facts_4:
            key = self._fact_key(f4)
            idx = self.fact_key_to_index.get(key, None)
            if idx is None:
                continue
            if 0 <= idx < len(scores_all):
                scored.append((float(scores_all[idx]), f4))
        # scored: List[(score, (s,r,o,t))]  — 全量候选
        if not scored:
            return [], []

        # 1) 先全量排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 2) 设定 QA 专用上限（不直接用 num_to_retrieve，避免过多噪声）
        k_for_qa = getattr(self.global_config, 'dynamic_fact_top_k', 12)
        # 让它不超过 num_to_retrieve
        k_for_qa = min(k_for_qa, num_to_retrieve)

        # 3) 稀疏化以减少冗余（按需调参）
        scored = self._sparsify_scored_facts(
            scored,
            k=k_for_qa,
            max_per_pair=getattr(self.global_config, 'max_facts_per_pair', 1),
            max_per_entity=getattr(self.global_config, 'max_facts_per_entity', 2),
        )

        # 4) 渲染为 docs / scores
        docs   = [self._fact_to_doc(f) for (s, f) in scored]
        scores = [s for (s, f) in scored]
        return docs, scores

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph
   
    

    def load_precomputed_openie(self, txt_path: str, chunks: Dict[str, "ChunkInfo"]):
        # 1) 读 triples，并用与你 triples_to_docs 完全一致的句式生成句子
        sent2triples: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        with Path(txt_path).open("r", encoding="utf-8") as f:
            for line in f:
                s, r, o, date = line.rstrip("\n").split("\t")
                sent = f"{date} {s} {r.replace('_', ' ')} {o}."   # 必须与插入 docs 的句式完全一致
                sent2triples[sent].append((s, r, o, date))

        # 2) 句子内容 → chunk_id（内容哈希），聚合到 cid
        cid2triples: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
        for sent, trips in sent2triples.items():
            cid = compute_mdhash_id(sent, prefix="chunk-")
            cid2triples[cid].extend(trips)

        # 3) 只给这次需要处理的 chunks 填充结果（键对齐，不靠顺序）
        ner_dict, triple_dict = {}, {}
        miss = 0
        for cid, row in chunks.items():
            triples = cid2triples.get(cid, [])
            if not triples:
                miss += 1  # 便于调试
            ents: Set[str] = {t[0] for t in triples} | {t[2] for t in triples}

            ner_dict[cid] = NerRawOutput(
                chunk_id=cid,
                response=None,
                unique_entities=list(ents),
                metadata={"prompt_tokens": 0, "completion_tokens": 0, "cache_hit": True},
            )
            triple_dict[cid] = TripleRawOutput(
                chunk_id=cid,
                response=None,
                triples=triples,
                metadata={"prompt_tokens": 0, "completion_tokens": 0, "cache_hit": True},
            )

        if miss:
            print(f"[openie] {miss}/{len(chunks)} chunks 找不到 triples。"
                f"请确认插入 docs 的句式与这里完全一致（空格/句号/replace('_',' ') 等）。")

        return ner_dict, triple_dict

    
    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')
    
    def index(self, docs: List[str], txt_path):
        """
        绑定式索引：用 txt_path 形成 passage↔triples 绑定，给每个 passage 的 chunk_id 也赋给它的 triples。
        后续照常：实体/事实 embedding、构图、保存。
        """
        logger.info("Indexing Documents")

        # 1) 读取并绑定 passage↔triples
        txt_path = Path(txt_path)
        records = triples_to_bound_passages(txt_path)   # [{content, triples}, ...]
        logger.info(f"Loaded {len(records)} unique passages from {txt_path.name}")

        # （可选）如果你仍想支持外部传入的 docs，就在此校验两者是否一致
        # 否则，建议直接以 records 为准，不再使用传入的 docs 以避免错位
        contents: List[str] = [rec["content"] for rec in records]

        logger.info("Encoding passages (chunk embeddings)")
        # 2) 仅对 content 做 chunk embedding（与原逻辑一致）
        self.chunk_embedding_store.insert_strings(contents)

        # 3) 用内容哈希生成 chunk_id，并直接构造 all_openie_info
        all_openie_info: List[dict] = []
        miss_cnt = 0
        for rec in records:
            passage_text = rec["content"]
            triples = rec["triples"]               # [(s,r,o,date), ...]
            # 用与向量库一致的规则生成 chunk_id（关键！）
            cid = compute_mdhash_id(passage_text, prefix="chunk-")

            # 实体集合（用于后续 entity embedding）
            ents: Set[str] = {t[0] for t in triples} | {t[2] for t in triples}

            # all_openie_info 的标准结构（与 merge_openie_results 产物一致）
            all_openie_info.append({
                "idx": cid,
                "passage": passage_text,
                "extracted_entities": list(ents),
                "extracted_triples": triples,
            })

        logger.info(f"Prepared {len(all_openie_info)} OpenIE-bound entries")
        # print("all_openie_info", all_openie_info[:5])
        # raise SystemExit
        # 4) （可选）保存 openie 结果
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # 5) 标准化为 dict（与原 reformat_openie_results 一致的输出）
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        
        # # 打印前 5 个 key 和对应的内容
        # for i, (cid, triple_out) in enumerate(triple_results_dict.items()):
        #     if i >= 5:
        #         break
        #     print("=" * 50)
        #     print(f"[{i}] chunk_id: {cid}")
        #     print("TripleRawOutput object:", triple_out)
        #     print("triples:", triple_out.triples)
        
        # 6) 准备 data_store
        chunk_ids = [item["idx"] for item in all_openie_info]
        assert len(chunk_ids) == len(ner_results_dict) == len(triple_results_dict)

        # 7) 把 triples 变成后续所需结构
        
        def _clean(x):
            if isinstance(x, str):
                return x.strip().lower()
            return x

        # chunk_triples = [
        #     [(_clean(s), _clean(r), _clean(o)) for (s, r, o) in triple_results_dict[cid].triples]
        #     for cid in chunk_ids
        # ]
        def _norm4(t):
            if len(t) == 4:
                s, r, o, tt = t
                return (_clean(s), _clean(r), _clean(o), tt)
            elif len(t) == 3:
                s, r, o = t
                return (_clean(s), _clean(r), _clean(o), None)
            else:
                return None

        chunk_triples = [
            list(filter(None, [_norm4(t) for t in triple_results_dict[cid].triples]))
            for cid in chunk_ids
        ]
        
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        # print("chunk_triples", chunk_triples[:5])
        facts = flatten_facts(chunk_triples)
        # print("facts", facts)
        logger.info("Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info("Encoding Facts")
        self.fact_embedding_store.insert_strings([str(f) for f in facts])

        logger.info("Constructing Graph")
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()


    # def index(self, docs: List[str], txt_path):
    #     """
    #     Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
    #     based on the given documents and encodes passages, entities and facts separately for later retrieval.

    #     Parameters:
    #         docs : List[str]
    #             A list of documents to be indexed.
    #     """

    #     logger.info(f"Indexing Documents")

    #     logger.info(f"Performing OpenIE")

    #     if self.global_config.openie_mode == 'offline':
    #         logger.info(f"Off line openie_mode")
    #         self.pre_openie(docs)
        
    #     self.chunk_embedding_store.insert_strings(docs)
    #     chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        
    #     all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
    #     # print("chunk_keys_to_process", chunk_keys_to_process)
    #     print("all_openie_info111", all_openie_info[:5])
    #     new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}
    #     # print("new_openie_rows", new_openie_rows)
    #     # exit(0)
    #     if len(chunk_keys_to_process) > 0:
            

    #         # 原来：
    #         new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)

    #         # 现在：
    #         # new_ner_results_dict, new_triple_results_dict = self.load_precomputed_openie(
    #         #     txt_path,
    #         #     new_openie_rows            # 即 chunk_keys_to_process
    #         # )
    #         print("new_triple_results_dict", len(new_triple_results_dict))
            
    #         self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
    #     print("all_openie_info", all_openie_info[:5])
        
    #     if self.global_config.save_openie:
    #         self.save_openie_results(all_openie_info)

    #     ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

    #     assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

    #     # prepare data_store
    #     chunk_ids = list(chunk_to_rows.keys())

    #     chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
    #     print("chunk_triples", chunk_triples)
    #     # exit(0)
    #     entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
    #     facts = flatten_facts(chunk_triples)

    #     logger.info(f"Encoding Entities")
        
    #     self.entity_embedding_store.insert_strings(entity_nodes)

    #     logger.info(f"Encoding Facts")
    #     self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

    #     logger.info(f"Constructing Graph")

    #     self.node_to_node_stats = {}
    #     self.ent_node_to_chunk_ids = {}

    #     self.add_fact_edges(chunk_ids, chunk_triples)
    #     num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

    #     if num_new_chunks > 0:
    #         logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
    #         self.add_synonymy_edges()

    #         self.augment_graph()
    #         self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time
        self.global_config = BaseConfig()
        

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        
        

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        print("Start retrieving!")
        self.get_query_embeddings(queries)

        retrieval_results = []
        
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            # print('query_fact_scores', query_fact_scores[:20])
            # print("query:", query)
            # exit(0)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts_dynamic(query, query_fact_scores)
            rerank_end = time.time()
            self.rerank_time += rerank_end - rerank_start
            
            self.global_config.dynamic = True
            # print('DEBUG top_k_facts', top_k_facts)
            print('DEBUG top_k_facts', top_k_facts)

            fact_key = "usaid sign_formal_agreement ethiopia"
            print(fact_key in self.fact_key_to_index)  # True 表示在 embedding store 里
        
            if self.global_config.dynamic:
                if len(top_k_facts) == 0:
                    logger.info('No facts found after reranking, return DPR results')
                    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                    top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                                for idx in sorted_doc_ids[:num_to_retrieve]]
                    retrieval_results.append(
                        QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                    )
                else:
                    print("Running the dynamic algorithm!")
                    # --- 1) Build merged_facts_4: original Top-K + PPR expansion (dedup, keep original order first) ---
                    try:
                        existing_as_4 = [self._norm_fact4(f) for f in top_k_facts]  # Top-K by similarity (anchors)
                        expanded = self._ppr_expand_from_facts(top_k_facts)         # (your helper) sets self._ppr_topk_rules
                    except Exception as e:
                        logger.warning("Dynamic PPR expansion failed: %s", e)
                        existing_as_4, expanded = [self._norm_fact4(f) for f in top_k_facts], []

                    seen, merged_facts_4 = set(), []
                    for f in (existing_as_4 + expanded):
                        f4 = self._norm_fact4(f)
                        if f4 not in seen:
                            merged_facts_4.append(f4)
                            seen.add(f4)

                    if not merged_facts_4:
                        logger.warning("Dynamic: merged_facts_4 is empty; falling back to legacy DPR/graph.")
                    else:
                        # -----------------------------------------------
                        # NEW STEP-2: build a rule-scoped candidate pool
                        # -----------------------------------------------
                        tm = getattr(self, "temporal_model", None)
                        rule_to_facts = getattr(tm, "rule_to_facts", {}) if tm is not None else {}
                        # --- robustly fetch PPR top rules and normalize to a list of rule ids (order-preserving) ---
                        raw = getattr(self, "_ppr_topk_rules", None)

                        # If it's callable (method/function), call it to obtain the underlying data
                        try:
                            if callable(raw):
                                raw = raw()
                        except Exception:
                            raw = None

                        # If it's a dict, treat keys as rule ids (common shape: {rule_id: score})
                        if isinstance(raw, dict):
                            iter_rules = list(raw.keys())
                        elif raw is None:
                            iter_rules = []
                        else:
                            # Already an iterable or a single value
                            iter_rules = raw

                        # Normalize each item to a rule id
                        norm_ids = []
                        for item in iter_rules if hasattr(iter_rules, "__iter__") and not isinstance(iter_rules, (str, bytes)) else [iter_rules]:
                            rid = None
                            if isinstance(item, (list, tuple)) and len(item) >= 1:
                                # e.g., (rule_id, score)
                                rid = item[0]
                            elif hasattr(item, "rule_id"):
                                rid = getattr(item, "rule_id")
                            elif hasattr(item, "rid"):
                                rid = getattr(item, "rid")
                            else:
                                rid = item
                            norm_ids.append(rid)

                        # Deduplicate but keep original order
                        ppr_rules = list(dict.fromkeys(norm_ids))


                        # Gather all (s,r,o,t) quads under the PPR top rules
                        def _to_quad(g):
                            if isinstance(g, (list, tuple)) and len(g) == 4:
                                return (g[0], g[1], g[2], g[3])
                            if isinstance(g, dict) and all(k in g for k in ("s", "r", "o", "t")):
                                return (g["s"], g["r"], g["o"], g["t"])
                            return None

                        rule_quads = []
                        if ppr_rules and rule_to_facts:
                            qseen = set()
                            for rid in ppr_rules:
                                for g in rule_to_facts.get(rid, []):
                                    q = _to_quad(g)
                                    if not q:
                                        continue
                                    q = self._norm_fact4(q)
                                    if q not in qseen:
                                        rule_quads.append(q)
                                        qseen.add(q)

                        # If PPR didn't yield rules (edge case), fall back to merged_facts_4
                        if not rule_quads:
                            rule_quads = list(merged_facts_4)

                        # -----------------------------------------------
                        # Score candidates inside these rule nodes by semantic similarity
                        # -----------------------------------------------
                        needed_keys = [f"{s} {r} {o}" for (s, r, o, t) in rule_quads]
                        # ensure mapping only for those keys (fast path)
                        self._ensure_fact_index_mapping(needed_keys=needed_keys, batch_size=50000)

                        scores_all = query_fact_scores if (isinstance(locals().get("query_fact_scores", None), np.ndarray)
                                                        and len(query_fact_scores) > 0) else self.get_fact_scores(query)
                        if scores_all is None or len(scores_all) == 0:
                            logger.warning("Dynamic: get_fact_scores returned empty; falling back to legacy DPR/graph.")
                        elif not getattr(self, "fact_key_to_index", None):
                            logger.warning("Dynamic: fact_key_to_index is empty even after ensure; falling back to legacy DPR/graph.")
                        else:
                            # rank rule_quads by similarity score
                            scored_rule_quads = []
                            for f4 in rule_quads:
                                key = f"{f4[0]} {f4[1]} {f4[2]}"
                                idx = self.fact_key_to_index.get(key, None)
                                if idx is None or idx < 0 or idx >= len(scores_all):
                                    continue
                                scored_rule_quads.append((float(scores_all[idx]), f4))
                            scored_rule_quads.sort(key=lambda x: x[0], reverse=True)

                            # take a semantic candidate slice from rule nodes (tunable)
                            cand_slice = getattr(self.global_config, "dynamic_rule_cand_topn", max(num_to_retrieve * 2, 50))
                            cand_facts = [f for (s, f) in scored_rule_quads[:cand_slice]]

                            # -----------------------------------------------
                            # NEW STEP-3: for each candidate, attach nearest 5 neighbors
                            # from the SAME rule node(s), with:
                            #   - same relation,
                            #   - and (same subject OR same object),
                            #   - time closest to the candidate's representative time.
                            # -----------------------------------------------

                            # helpers (rule lookup + neighbor mining)
                            def _rules_for_fact(f4):
                                if tm is None:
                                    return []
                                s, r, o, t = f4
                                # prefer exact (s,r,o,t) if available
                                if hasattr(tm, "fact4_to_rules"):
                                    rs = tm.fact4_to_rules.get((s, r, o, t), [])
                                    if rs:
                                        return rs
                                # fallback to (s,r,o)
                                if hasattr(tm, "fact3_to_rules"):
                                    return tm.fact3_to_rules.get((s, r, o), []) or []
                                if hasattr(tm, "rules_for_fact"):
                                    return tm.rules_for_fact((s, r, o)) or []
                                return []

                            def _closest_neighbors_same_rule(anchor_f4, per_rule_limit=5):
                                """
                                Search facts in the same rule node(s) as anchor_f4,
                                pick neighbors that share relation and (subject or object),
                                keep the closest by |Δdays|. Return at most 5.
                                """
                                if tm is None:
                                    return []
                                a_s, a_r, a_o, a_t = anchor_f4

                                # choose representative time for anchor
                                if hasattr(self, "_choose_rep_time"):
                                    a_t_str, a_date = self._choose_rep_time(anchor_f4, anchor_date=None)
                                else:
                                    a_date = self._parse_date_soft(a_t)
                                    a_t_str = a_t if a_date is not None else None
                                    if a_date is None and hasattr(self, "_all_times_for_triple"):
                                        tlist = self._all_times_for_triple(a_s, a_r, a_o)
                                        for tt in reversed(tlist):
                                            d = self._parse_date_soft(tt)
                                            if d is not None:
                                                a_t_str, a_date = tt, d
                                                break
                                if a_date is None:
                                    return []

                                neighbors = []
                                rules = _rules_for_fact(anchor_f4)
                                for rid in rules:
                                    local = []
                                    for g in rule_to_facts.get(rid, []):
                                        q = _to_quad(g)
                                        if not q:
                                            continue
                                        s2, r2, o2, t2 = q
                                        if r2 != a_r:
                                            continue
                                        if not (s2 == a_s or o2 == a_o):
                                            continue
                                        if q == anchor_f4:
                                            continue
                                        d2 = self._parse_date_soft(t2)
                                        if d2 is None:
                                            continue
                                        delta = abs((d2 - a_date).days)
                                        local.append((delta, (s2, r2, o2, t2)))
                                    # keep closest few from each rule to avoid flooding
                                    local.sort(key=lambda x: x[0])
                                    neighbors.extend([q for _, q in local[:per_rule_limit]])

                                # global uniq + sort by closeness
                                seen_loc, uniq = set(), []
                                for q in neighbors:
                                    if q not in seen_loc:
                                        uniq.append(q); seen_loc.add(q)
                                uniq.sort(key=lambda q: abs((self._parse_date_soft(q[3]) - a_date).days)
                                                    if self._parse_date_soft(q[3]) else 10**9)
                                return uniq[:5]

                            # build final list: candidate -> its neighbors (dedup, keep order), cap to num_to_retrieve
                            final_facts, fset = [], set()
                            for f4 in cand_facts:
                                if f4 not in fset:
                                    final_facts.append(f4); fset.add(f4)
                                try:
                                    neigh = _closest_neighbors_same_rule(f4, per_rule_limit=5)
                                    for qf in neigh:
                                        if len(final_facts) >= num_to_retrieve:
                                            break
                                        if qf not in fset:
                                            final_facts.append(qf); fset.add(qf)
                                except Exception as e:
                                    logger.debug("Neighbor mining failed for %s due to %s", f4, e)
                                if len(final_facts) >= num_to_retrieve:
                                    break

                            # 统一的键（默认按 s,r,o 做去重，忽略时间；如需按四元去重，把 date 拼进去即可）
                            def _fkey(f):
                                s, r, o = f[0], f[1], f[2]
                                return f"{s} {r} {o}"

                            # 记录在 final_facts 与 top_k_facts 中的原始次序（用于稳定排序）
                            final_pos = {}
                            for i, f4 in enumerate(final_facts):
                                final_pos.setdefault(_fkey(f4), i)

                            topk_pos = {}
                            for j, f4 in enumerate(top_k_facts):  # 这里的 top_k_facts 是 _augment_topk_to_quads(...) 的结果
                                topk_pos.setdefault(_fkey(f4), j)

                            # 合并 + 去重（保留首见）
                            combined_map = {}
                            for f4 in final_facts:
                                k = _fkey(f4)
                                if k not in combined_map:
                                    combined_map[k] = f4
                            for f4 in top_k_facts:
                                k = _fkey(f4)
                                if k not in combined_map:
                                    combined_map[k] = f4

                            combined = list(combined_map.values())  # ← 合并去重后

                            # ===== 基于“Top 1 fact”的时间锚点做最后过滤（简版） =====
                            

                            def _parse_date(s):
                                if not s: return None
                                s = str(s).strip()
                                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
                                    try:
                                        d = _dt.datetime.strptime(s, fmt).date()
                                        # 月/年粒度统一补到每月/每年第一天
                                        if fmt in ("%Y-%m", "%Y/%m"): return d.replace(day=1)
                                        if fmt == "%Y": return _dt.date(int(s[:4]), 1, 1)
                                        return d
                                    except ValueError:
                                        pass
                                return None

                            def _time_op_from_query(q: str):
                                """返回 (op, inclusive)；op ∈ {'before','after'} 或 None；inclusive 缺省为 True。"""
                                q = (q or "").lower()
                                # before 类（含当天）
                                if re.search(r'\b(on or before|no later than|not after|by|until|through|before)\b', q):
                                    return "before", True
                                # before 类（不含当天）
                                if re.search(r'\b(prior to|earlier than)\b', q):
                                    return "before", False
                                # after 类（含当天）
                                if re.search(r'\b(on or after|no earlier than|since|from|starting|after)\b', q):
                                    return "after", True
                                # after 类（不含当天）
                                if re.search(r'\b(later than)\b', q):
                                    return "after", False
                                return None, True

                            op, inclusive = _time_op_from_query(query)

                            anchor_date = None
                            if top_k_facts and len(top_k_facts[0]) >= 4:
                                anchor_date = _parse_date(top_k_facts[0][3])

                            if op and anchor_date:
                                KEEP_UNKNOWN_DATES = False  # 无日期的事实是否保留（默认丢弃）
                                def _fact_date(f):
                                    return _parse_date(f[3]) if len(f) >= 4 else None

                                filtered = []
                                for f in combined:
                                    d = _fact_date(f)
                                    if d is None:
                                        if KEEP_UNKNOWN_DATES: filtered.append(f)
                                        continue
                                    if op == "before":
                                        if (d <= anchor_date) if inclusive else (d < anchor_date):
                                            filtered.append(f)
                                    else:  # "after"
                                        if (d >= anchor_date) if inclusive else (d > anchor_date):
                                            filtered.append(f)
                                combined = filtered
                            # ===== 过滤结束，下面继续你的排序逻辑 =====


                            # 原规则排序：
                            #   1) 出现在 final_facts 的优先（0）否则（1）
                            #   2) 若在 final_facts，按其在 final_facts 的先后
                            #   3) 否则按其在 top_k_facts 的先后
                            BIG = 10**12
                            combined.sort(key=lambda f4: (
                                0 if _fkey(f4) in final_pos else 1,
                                final_pos.get(_fkey(f4), BIG),
                                topk_pos.get(_fkey(f4), BIG),
                            ))

                            # 打分函数（用于输出 scores；不改变排序）
                            def _score_for(f4):
                                k = _fkey(f4)
                                idx = self.fact_key_to_index.get(k, None)
                                if idx is not None and 0 <= idx < len(scores_all):
                                    return float(scores_all[idx])
                                return 0.0

                            # === 可选：是否按 (subject, relation) 合并多条事实 ===
                            merge_by_sr = getattr(self, "merge_by_sr", True)  # 也可以改成函数参数传入

                            if merge_by_sr:
                                from collections import OrderedDict
                                groups = OrderedDict()

                                for f in combined:  # combined 已经“合并去重 → 时间过滤 → 排序”完成
                                    s, r, o = f[0].strip(), f[1].strip(), f[2].strip()
                                    d = (f[3].strip() if len(f) >= 4 and f[3] else "unknown_time")
                                    key = (s, r)  # 如需“按基础前缀关系合并”，可把 r 换成 r_base（见下方注释）

                                    if key not in groups:
                                        groups[key] = {
                                            "s": s, "r": r,
                                            "objs": [], "dates": [],
                                            "score": float("-inf")
                                        }
                                    g = groups[key]
                                    # 追加（按出现顺序去重）
                                    if o not in g["objs"]:
                                        g["objs"].append(o)
                                    if d not in g["dates"]:
                                        g["dates"].append(d)
                                    # 组内分数 = 该组中单条事实分数的最大值（保持“最好证据”排序气质）
                                    sc = _score_for(f)
                                    if sc > g["score"]:
                                        g["score"] = sc

                                # 组的顺序 = 首次出现顺序（OrderedDict 保证），符合你现有排序后的先后
                                merged_docs = []
                                merged_scores = []
                                for g in groups.values():
                                    # 合并后的文案：s r o1, o2 on d1, d2
                                    pairs = [f"{obj} @ {dt}" for obj, dt in zip(g["objs"], g["dates"])]
                                    doc_str = f"{g['s']} {g['r']} ({'; '.join(pairs)})"
                                    merged_docs.append(doc_str)
                                    merged_scores.append(g["score"])

                                # 截断
                                docs   = merged_docs[:num_to_retrieve]
                                scores = merged_scores[:num_to_retrieve]

                            else:
                                # 原始逐条输出
                                docs   = [self._fact_to_doc(f4) for f4 in combined[:num_to_retrieve]]
                                scores = [_score_for(f4)       for f4 in combined[:num_to_retrieve]]
                            
                            if docs:
                                retrieval_results.append(QuerySolution(question=query, docs=docs, doc_scores=scores))
                                # IMPORTANT: skip legacy branch for this query
                                print("DEBUG retrieval_results:", len(retrieval_results), '  ', retrieval_results[-1])
                                continue
                            else:
                                logger.warning("Dynamic: final_facts empty after rule-scope ranking; falling back to legacy DPR/graph.")

            else:
                if len(top_k_facts) == 0:
                    logger.info('No facts found after reranking, return DPR results')
                    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                else:
                    logger.info('Processing the graph search')
                    sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                            link_top_k=self.global_config.linking_top_k,
                                                                                            query_fact_scores=query_fact_scores,
                                                                                            top_k_facts=top_k_facts,
                                                                                            top_k_fact_indices=top_k_fact_indices,
                                                                                            passage_node_weight=self.global_config.passage_node_weight)

                top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
                # print("top_k_docs", top_k_docs)
                
                retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
        
     
        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Average Retrieval Time {self.all_retrieval_time/len(queries):.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               qlabels: List[str] = None,
               topk_answers: int = 10) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
        # print("queries:", queries)
        # exit(0)
        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            pred_top_lists = [getattr(qs, "top_answers", [qs.answer]) for qs in queries_solutions]

            hit1_overall, hit1_example = evaluate_hit_at_k_f1(
                gold_answers=gold_answers,
                predicted_topk_answers=pred_top_lists,
                k=1
            )
            hit10_overall, hit10_example = evaluate_hit_at_k_f1(
                gold_answers=gold_answers,
                predicted_topk_answers=pred_top_lists,
                k=10
            )

            if qlabels is None:
                qlabels = ["Single"] * len(queries_solutions)

            idx_single   = [i for i, q in enumerate(qlabels) if str(q).lower().startswith("single")]
            idx_multiple = [i for i, q in enumerate(qlabels) if str(q).lower().startswith("multiple")]

            def _slice(lst, idxs):
                return [lst[i] for i in idxs]

            # 原有 token-F1（QAF1Score）分组
            f1_overall, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[qs.answer for qs in queries_solutions],
                aggregation_fn=np.max)

            f1_single, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=_slice(gold_answers, idx_single),
                predicted_answers=[queries_solutions[i].answer for i in idx_single],
                aggregation_fn=np.max)

            f1_multiple, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=_slice(gold_answers, idx_multiple),
                predicted_answers=[queries_solutions[i].answer for i in idx_multiple],
                aggregation_fn=np.max)

            # Hit@k 分组
            hit1_single, _ = evaluate_hit_at_k_f1(
                gold_answers=_slice(gold_answers, idx_single),
                predicted_topk_answers=_slice(pred_top_lists, idx_single),
                k=1)

            hit1_multiple, _ = evaluate_hit_at_k_f1(
                gold_answers=_slice(gold_answers, idx_multiple),
                predicted_topk_answers=_slice(pred_top_lists, idx_multiple),
                k=1)

            hit10_single, _ = evaluate_hit_at_k_f1(
                gold_answers=_slice(gold_answers, idx_single),
                predicted_topk_answers=_slice(pred_top_lists, idx_single),
                k=10)

            hit10_multiple, _ = evaluate_hit_at_k_f1(
                gold_answers=_slice(gold_answers, idx_multiple),
                predicted_topk_answers=_slice(pred_top_lists, idx_multiple),
                k=10)

            qa_metrics = {
                "tokenF1_overall": f1_overall,  # {"F1": number}
                "tokenF1_by_type": {
                    "Single": f1_single,        # {"F1": number}
                    "Multiple": f1_multiple
                },
                "hitF1_at_1_overall": hit1_overall,   # {"F1": number}
                "hitF1_at_10_overall": hit10_overall,
                "hitF1_at_1_by_type": {
                    "Single": hit1_single,
                    "Multiple": hit1_multiple
                },
                "hitF1_at_10_by_type": {
                    "Single": hit10_single,
                    "Multiple": hit10_multiple
                }
            }

            # ---- 输出“每条 query 平均 token（含 prompt）”----
            if self._tok_query_cnt > 0:
                avg_total      = self._tok_total_sum / self._tok_query_cnt
                avg_prompt     = self._tok_prompt_sum / self._tok_query_cnt
                avg_completion = self._tok_completion_sum / self._tok_query_cnt
                logger.info(f"[Avg Tokens per Query] total={avg_total:.2f} (prompt={avg_prompt:.2f}, completion={avg_completion:.2f})")

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, qa_metrics
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipidia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '
            
            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'multitq'
            print('DEBUG prompt template:', prompt_dataset_name)    
            qa_messages = self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user)
            all_qa_messages.append(qa_messages)
            print('DEBUG qa_messages:', qa_messages)

        
        # print("all_qa_messages", all_qa_messages)
        # exit(0)
        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # ---- 统计 LLM token 用量（以 metadata 为准；拿不到就当 0）----
        prompt_sum = 0
        completion_sum = 0
        total_sum = 0

        for meta in all_metadata:
            # meta 可能为 dict，也可能嵌套；本项目里 openie/qa 元数据键是 prompt_tokens / completion_tokens
            if isinstance(meta, dict):
                pt = int(meta.get("prompt_tokens", 0))
                ct = int(meta.get("completion_tokens", 0))
                tt = int(meta.get("total_tokens", pt + ct))
            else:
                pt = ct = 0
                tt = 0

            prompt_sum += pt
            completion_sum += ct
            total_sum += tt

        # 把本批（这次 qa()）的统计累加到类级累计器
        self._tok_prompt_sum     += prompt_sum
        self._tok_completion_sum += completion_sum
        self._tok_total_sum      += total_sum
        self._tok_query_cnt      += len(all_metadata)  # 一条 metadata 对应一条 query

        #Process responses and extract predicted answers.
        # queries_solutions = []
        # for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
        #     response_content = all_response_message[query_solution_idx]
        #     try:
        #         pred_ans = response_content.split('Answer:')[1].strip()
        #     except Exception as e:
        #         logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
        #         pred_ans = response_content

        #     query_solution.answer = pred_ans
        #     queries_solutions.append(query_solution)
        queries_solutions = []
        for idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[idx]
            top_answers = None
            single_answer = None

            # 优先尝试抓 JSON 列表（我们会在 prompt 里让模型以 `Answers:`+JSON 输出）
            try:
                # 兼容形式1：明确 "Answers:" 前缀
                if 'Answers:' in response_content:
                    after = response_content.split('Answers:', 1)[1]
                    json_text = re.search(r'\[.*\]', after, flags=re.S).group(0)
                    cand = json.loads(json_text)
                    if isinstance(cand, list):
                        top_answers = [str(x).strip() for x in cand if str(x).strip()][:10]

                # 兼容形式2：没写 Answers: 但直接给了一个 JSON 列表
                if top_answers is None:
                    m = re.search(r'\[.*\]', response_content, flags=re.S)
                    if m:
                        cand = json.loads(m.group(0))
                        if isinstance(cand, list):
                            top_answers = [str(x).strip() for x in cand if str(x).strip()][:10]
            except Exception:
                pass

            # 旧逻辑兜底：单一 Answer:
            if top_answers is None:
                try:
                    single_answer = response_content.split('Answer:', 1)[1].strip()
                except Exception:
                    single_answer = response_content.strip()
                top_answers = [single_answer] if single_answer else []

            # 统一写回
            query_solution.answer = single_answer if single_answer is not None else (top_answers[0] if top_answers else "")
            query_solution.top_answers = top_answers
            queries_solutions.append(query_solution)
            

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))
        # print("self.node_to_node_stats", self.node_to_node_stats)
        # print("self.ent_node_to_chunk_ids", self.ent_node_to_chunk_ids)
        # exit(0)
    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
       
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        # print("self.query_to_embedding", self.query_to_embedding)
        
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
        
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            print('DEBUG len(self.fact_embeddings) ', len(self.fact_embeddings))
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])
        print("self.node_name_to_vertex_idx", list(self.node_name_to_vertex_idx.items())[:5])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0
        print("np.count_nonzero(all_phrase_weights)", np.count_nonzero(all_phrase_weights))
        print("len(linking_score_map.keys())", len(linking_score_map.keys()))
        
        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map
    def print_graph_summary(self, sample_size=5):
        g = self.graph
        num_nodes = len(g.vs)
        num_edges = len(g.es)
        print(f"Graph summary:")
        print(f"  - Number of nodes: {num_nodes}")
        print(f"  - Number of edges: {num_edges}")

        # 随机抽取一些节点，或者取前 sample_size 个
        sample_indices = random.sample(range(num_nodes), min(sample_size, num_nodes))
        print(f"\nShowing {len(sample_indices)} sample nodes:")

        for idx in sample_indices:
            v = g.vs[idx]
            node_name = v["name"]
            neighbors = g.neighbors(idx)
            print(f"\nNode {idx}: {node_name}  (degree={len(neighbors)})")
            for neigh_idx in neighbors[:5]:  # 每个节点最多展示5个邻居
                neigh_name = g.vs[neigh_idx]["name"]
                edge_id = g.get_eid(idx, neigh_idx)
                edge_weight = g.es[edge_id]["weight"]
                print(f"   --> {neigh_name}  (weight={edge_weight})")
    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

       
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights
        # print("node_weights:", node_weights)
        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'
        # print("self.graph.es['weight']", self.graph)
        self.print_graph_summary()
        # exit(0)  
        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores
    def _tm(self):
        """取 temporal model，兼容不同字段名"""
        return getattr(self, "temporal_model", getattr(self, "tm", None))

    def _date_key(self, t: str):
        """
        把各种常见时间字符串转成可排序的 key。
        - 返回 (0, date) 表示合法日期；(1, 原串) 表示无法解析（会排在后面）
        - 仅用于排序；不做严格校验。
        """
        if not t:
            return (1, "ZZZ")
        s = str(t).strip()
        # 捕获 YYYY[-/MM[-/DD]]，缺省部分补 01
        m = re.match(r"^(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", s)
        if not m:
            return (1, s)
        y = int(m.group(1))
        mth = int(m.group(2) or 1)
        d = int(m.group(3) or 1)
        try:
            return (0, date(y, mth, d))
        except Exception:
            return (1, s)

    def _quads_for_triple(self, s: str, r: str, o: str):
        """
        从 temporal model 里把所有与 (s,r,o) 完全匹配，且 t 存在的四元组取回并去重。
        输出按时间升序。
        """
        tm = self._tm()
        if tm is None:
            return []

        # 取与 (s,r,o) 相关的规则
        rules = []
        if hasattr(tm, "fact3_to_rules"):
            rules = tm.fact3_to_rules.get((s, r, o), [])
        elif hasattr(tm, "rules_for_fact"):
            rules = tm.rules_for_fact((s, r, o)) or []
        if not rules:
            return []

        quads = []
        rule_to_facts = getattr(tm, "rule_to_facts", {})
        for rule in rules:
            facts = rule_to_facts.get(rule, [])
            for f in facts:
                # 兼容 tuple/list/dict 三种存储
                if isinstance(f, (list, tuple)):
                    if len(f) == 4 and f[0] == s and f[1] == r and f[2] == o and f[3]:
                        quads.append((f[0], f[1], f[2], f[3]))
                elif isinstance(f, dict):
                    if f.get("s") == s and f.get("r") == r and f.get("o") == o and f.get("t"):
                        quads.append((f["s"], f["r"], f["o"], f["t"]))

        # 去重（按完整四元）
        seen, uniq = set(), []
        for q in quads:
            if q not in seen:
                uniq.append(q)
                seen.add(q)

        # 按时间升序；无法解析时间的排在最后
        uniq.sort(key=lambda q: self._date_key(q[3]))
        return uniq

    def _augment_topk_to_quads(self, top_k_facts, per_triple_limit=None, max_total=None, sort_order="asc"):
        """
        把 Top-K 里的三元/四元补成“带时间的四元列表”，按每个三元的时间升序展开。
        - per_triple_limit: 每个 (s,r,o) 最多取多少条时间实例（None=不限制）
        - max_total: 返回的总四元最多条数（None=不限制）
        - sort_order: 'asc' 或 'desc'
        """
        out = []
        for f in top_k_facts:
            s, r, o = f[:3]  # 兼容 (s,r,o) / (s,r,o,None)
            quads = self._quads_for_triple(s, r, o)  # 已经是升序
            if sort_order == "desc":
                quads.reverse()
            if per_triple_limit is not None:
                quads = quads[:per_triple_limit]
            out.extend(quads)
            if max_total is not None and len(out) >= max_total:
                out = out[:max_total]
                break
        return out

    # def rerank_facts_dynamic(self, query: str, query_fact_scores: np.ndarray):
    #     """
    #     Temporal-aware reranker (no dedup):
    #     - enlarge candidate pool (5x)
    #     - normalize/parse to 4-tuples (keep duplicates; sorted by semantic score)
    #     - if the query contains temporal cues (before/after/last/first...), try TemporalAwareFilter
    #     - ALWAYS reorder final Top-K by:
    #         (1) same relation as anchor first,
    #         (2) then absolute day-distance to anchor's representative date (smaller first),
    #         (3) stable by original index
    #     - return top_idx, top_facts and rerank_log (with 'quads_for_ppr': first 20 quads)
    #     """
        
    #     link_top_k: int = int(self.global_config.linking_top_k)
    #     link_top_k = 200  # if you intend to override via config, remove this hard override

    #     if (query_fact_scores is None) or (len(query_fact_scores) == 0) or (len(self.fact_node_keys) == 0):
    #         logger.warning("No facts available for reranking. Returning empty lists.")
    #         return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'quads_for_ppr': []}

    #     # --- 0) config ---
    #     pool_mult = getattr(self.global_config, "linking_pool_multiplier", 5)
    #     pool_size = max(link_top_k * pool_mult, link_top_k)
    #     pre_rerank_cap = max(link_top_k * 2, link_top_k)

    #     # --- 1) take larger semantic pool ---
    #     order = np.argsort(query_fact_scores)[::-1]
    #     cand_indices_sem = order[:pool_size].tolist()
    #     cand_ids = [self.fact_node_keys[idx] for idx in cand_indices_sem if 0 <= idx < len(self.fact_node_keys)]
    #     rows = self.fact_embedding_store.get_rows(cand_ids)

    #     parsed = []
    #     for idx, rid in zip(cand_indices_sem, cand_ids):
    #         row = rows.get(rid)
    #         if not row:
    #             continue
    #         cont = row.get("content")
    #         if cont is None:
    #             continue
    #         try:
    #             f = tuple(cont) if isinstance(cont, (tuple, list)) else ast.literal_eval(cont)
    #             f4 = self._norm_fact4(f)  # -> (s,r,o,t), normalized
    #             parsed.append((idx, f4))
    #         except Exception as e:
    #             logger.debug("Skip fact id=%s parse error: %s", rid, e)

    #     if not parsed:
    #         logger.warning("Parsed candidate list is empty; return empty.")
    #         return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'quads_for_ppr': []}

    #     # keep duplicates: sort by semantic score (desc) without deduplication
    #     parsed_sorted = sorted(
    #         parsed,
    #         key=lambda it: float(query_fact_scores[it[0]]),
    #         reverse=True
    #     )
    #     cand_indices = [idx for (idx, f4) in parsed_sorted][:pre_rerank_cap]
    #     cand_facts   = [f4  for (idx, f4) in parsed_sorted][:pre_rerank_cap]


    #     # --- 2) detect temporal cue (no relation-family template) ---
    #     qlow = query.lower()
    #     has_temporal = any(tok in qlow for tok in [
    #         "before", "after", "last", "first", "earliest", "latest", "prior to", "later than"
    #     ])

    #     # 关系基础前缀：去掉括号后缀（如 "...(such_as_...)"）
    #     def _r_base(r: str) -> str:
    #         import re as _re
    #         return _re.sub(r"\(.*\)$", "", (r or "").strip())

    #     # 从 Top-1 fact 提取两个 subject_hints + 基础 relation
    #     subject_hints = []
    #     r_base = None
    #     if cand_facts:
    #         s_top, r_top, o_top = cand_facts[0][:3]
    #         subject_hints = [s_top, o_top]
    #         r_base = _r_base(r_top)


        
    #     # --- 3) branch: temporal-aware LLM filter or original reranker ---
    #     try:
    #         if has_temporal and cand_facts and subject_hints and r_base:
    #             # —— 工具函数（本地硬过滤）——
    #             def _parse_date(s):
    #                 if not s: return None
    #                 s = str(s).strip()
    #                 for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
    #                     try:
    #                         d = _dt.datetime.strptime(s, fmt).date()
    #                         if fmt in ("%Y-%m", "%Y/%m"): return d.replace(day=1)
    #                         if fmt == "%Y": return _dt.date(int(s[:4]), 1, 1)
    #                         return d
    #                     except ValueError:
    #                         pass
    #                 return None

    #             def _time_op_from_query(q: str):
    #                 q = (q or "").lower()
    #                 # before 类（含当天 / 不含当天）
    #                 if re.search(r'\b(on or before|no later than|not after|by|until|through|before)\b', q): return ("before", True)
    #                 if re.search(r'\b(prior to|earlier than)\b', q):                                        return ("before", False)
    #                 # after 类（含当天 / 不含当天）
    #                 if re.search(r'\b(on or after|no earlier than|since|from|starting|after)\b', q):       return ("after", True)
    #                 if re.search(r'\b(later than)\b', q):                                                   return ("after", False)
    #                 return (None, True)

    #             # 锚点 = Top-1 的日期；若没有日期，则不做时间过滤
    #             anchor_date = _parse_date(cand_facts[0][3]) if len(cand_facts[0]) >= 4 else None
    #             op, inclusive = _time_op_from_query(query)

    #             def _date_keep(d):
    #                 if op is None or anchor_date is None:
    #                     return True
    #                 if d is None:
    #                     return False
    #                 if op == "before":
    #                     return (d <= anchor_date) if inclusive else (d < anchor_date)
    #                 else:  # "after"
    #                     return (d >= anchor_date) if inclusive else (d > anchor_date)

    #             # 对称去重：A↔B 视为同一条，避免同日 A→B / B→A 双份进入
    #             def _canon_key(f4):
    #                 s, r, o = f4[0], f4[1], f4[2]
    #                 d = f4[3] if len(f4) >= 4 else ""
    #                 return (tuple(sorted([s, o])), _r_base(r), d)

    #             # 组合 cand_indices 与 cand_facts，保持原检索顺序
    #             pairs = list(zip(cand_indices, cand_facts))

    #             # 过滤条件：与 Top-1 同基础关系前缀，且包含 subject_hint 实体（无论在 S 还是 O）
    #             def _bucket(ent):
    #                 seen = set()
    #                 out  = []  # (idx, f4, date)
    #                 for idx, f4 in pairs:
    #                     if not f4 or len(f4) < 3:
    #                         continue
    #                     s, r, o = f4[0], f4[1], f4[2]
    #                     if _r_base(r) != r_base:
    #                         continue
    #                     if not (s == ent or o == ent):   # 主客体任一匹配即可
    #                         continue
    #                     d = _parse_date(f4[3]) if len(f4) >= 4 else None
    #                     if not _date_keep(d):            # <—— 硬性时间过滤（保证 “after/before” 生效）
    #                         continue
    #                     key = _canon_key(f4)
    #                     if key in seen:
    #                         continue
    #                     seen.add(key)
    #                     out.append((idx, f4, d))

    #                 # 排序：after ⇒ 升序（最早满足的在前）；before ⇒ 降序；无约束 ⇒ 最近优先
    #                 if op == "after":
    #                     out.sort(key=lambda t: (t[2] is None, t[2]))
    #                 elif op == "before":
    #                     out.sort(key=lambda t: (t[2] is None, t[2]), reverse=True)
    #                 else:
    #                     out.sort(key=lambda t: (t[2] is None, t[2]), reverse=True)
    #                 return out

    #             # 两路各取一半（奇数时第一路多 1）；不足不补
    #             half_A = (link_top_k + 1) // 2
    #             half_B = link_top_k - half_A

    #             bA = _bucket(subject_hints[0])
    #             bB = _bucket(subject_hints[1]) if len(subject_hints) > 1 else []

    #             pickA = bA[:half_A]
    #             pickB = bB[:half_B]

    #             picked_idx   = [i for (i, _, _) in pickA] + [i for (i, _, _) in pickB]
    #             picked_facts = [f for (_, f, _) in pickA] + [f for (_, f, _) in pickB]

    #             # —— 最终排序仍沿用你原本的规则 —— 
    #             top_idx, top_facts, anchor_date2 = self._relation_temporal_sort(picked_idx, picked_facts)
    #             quads_for_ppr = self._build_quads_for_ppr(top_facts, anchor_date2, limit=20)

    #             rerank_log = {
    #                 "facts_before_rerank": cand_facts,
    #                 "facts_after_rerank": top_facts,
    #                 "anchor_date": (anchor_date2.isoformat() if anchor_date2 else None),
    #                 "quads_for_ppr": quads_for_ppr,
    #             }
    #             return top_idx[:link_top_k], top_facts[:link_top_k], rerank_log


    #     except Exception as e:
    #         logger.warning("Temporal-aware rerank failed; falling back. %s", e)
    #         # hard fallback: top-K by semantic
    #         top_idx = cand_indices[:link_top_k]
    #         top_facts = cand_facts[:link_top_k]

    #         # relation-priority + temporal-proximity reorder (no dedup; anchor stays first)
    #         top_idx, top_facts, anchor_date = self._relation_temporal_sort(top_idx, top_facts)
    #         quads_for_ppr = self._build_quads_for_ppr(top_facts, anchor_date, limit=20)

    #         return top_idx, top_facts, {
    #             'facts_before_rerank': cand_facts,
    #             'facts_after_rerank': top_facts,
    #             'anchor_date': (anchor_date.isoformat() if anchor_date else None),
    #             'quads_for_ppr': quads_for_ppr
    #         }

    

    def rerank_facts_dynamic(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        top_k: Optional[int] = None,
        per_triple_limit: int = 5,
        max_total: Optional[int] = None,
        sort_order: str = "asc",
    ) -> Tuple[List[str], List[Quad], Dict[str, Any]]:
        """
        动态 rerank:
        1) 从 query_fact_scores 里找 Top-K fact（按分数降序）
        2) 用 _augment_topk_to_quads 时间补齐
        3) 先按 relation 过滤：仅保留与 Top-1 anchor fact 同 relation 的条目
        4) 再按 query 的 anchor date 与 before/after 时间约束进行过滤和排序
        """

        rerank_start = time.time()
        if top_k is None:
            top_k = getattr(getattr(self, "global_config", object()), "top_k_facts", 200)


        # 映射 's r o' -> 行号
        self._ensure_fact_index_mapping()



        scores_all = query_fact_scores
        if scores_all is None or len(scores_all) == 0:
            return [], [], {}

        # Step 1: 对所有 fact（通过 key）取分
        scored: List[Tuple[float, Quad]] = []
        for key, idx in self.fact_key_to_index.items():
            if 0 <= idx < len(scores_all):
                parts = key.split(" ", 2)  # 你的 key 形如 "s r o"
                if len(parts) != 3:
                    continue
                s, r, o = parts
                scored.append((float(scores_all[idx]), (s, r, o, "")))  # 初始 t 为空字符串

        if not scored:
            return [], [], {}

        scored.sort(key=lambda x: x[0], reverse=True)
        topk_pairs = scored[:top_k]
        topk_facts = [f for (_, f) in topk_pairs]

        print(f"[DEBUG][rerank] total_candidates={len(scored)}, top_k={len(topk_facts)}")

        # 确定锚点 relation（严格使用 Top-1 原始 fact 的 relation）
        anchor_rel: Optional[str] = topk_facts[0][1] if topk_facts else None

        # Step 2: 时间补齐（对 Top-K 做 augment）
        if max_total is None:
            max_total = per_triple_limit * max(1, len(topk_facts))

        augmented_quads: List[Quad] = self._augment_topk_to_quads(
            topk_facts,
            per_triple_limit=per_triple_limit,
            max_total=max_total,
            sort_order=sort_order,
        )
        # 统一 t 为 str
        augmented_quads = [(s, r, o, (t if isinstance(t, str) else ("" if t is None else str(t))))
                        for (s, r, o, t) in augmented_quads]

        augmented_total = len(augmented_quads)
        num_with_time = sum(1 for _, _, _, t in augmented_quads if t and self._parse_date_soft(t) is not None)
        print(f"[DEBUG][augment] after_augment={augmented_total}, with_time={num_with_time}")

        # —— 先按 relation 严格过滤（必须与 Top-1 anchor 的 relation 相同） ——
        if anchor_rel is not None:
            rel_kept_idx = [i for i, q in enumerate(augmented_quads) if q[1] == anchor_rel]
            if not rel_kept_idx:
                # 理论上不会发生（augment 通常保留同一 relation），但加个兜底：
                print(f"[DEBUG][relfilter] kept=0 / {augmented_total} (relation={anchor_rel}); fallback to Top-1 only.")
                # 回退为仅保留 Top-1 原始 fact（可能没有时间戳）
                augmented_quads_rel = [topk_facts[0]]
            else:
                augmented_quads_rel = [augmented_quads[i] for i in rel_kept_idx]
            print(f"[DEBUG][relfilter] kept={len(augmented_quads_rel)} / {augmented_total} (relation={anchor_rel})")
        else:
            augmented_quads_rel = augmented_quads
            print(f"[DEBUG][relfilter] anchor_rel=None, kept={len(augmented_quads_rel)} / {augmented_total} (apply=False)")

        # —— 锚点日期与方向：优先 query，其次全局配置，最后回退为 augment Top-1 的日期 ——


        def _as_date(x):
            if x is None:
                return None
            if isinstance(x, datetime):
                return x.date()
            if isinstance(x, _date):
                return x
            return None

        # 1) query
        anchor_dt_any = extract_anchor_date_from_query(query)
        # 2) config
        if anchor_dt_any is None:
            anchor_cfg = getattr(self, "anchor_date", None) or \
                        getattr(getattr(self, "global_config", object()), "anchor_date", None)
            if anchor_cfg:
                anchor_dt_any = self._parse_date_soft(anchor_cfg)
        # 3) fallback: augment 后的 Top-1（relation 已经过滤，所以拿 relation 过滤后的第 1 条）
        if anchor_dt_any is None and augmented_quads_rel:
            maybe_t = augmented_quads_rel[0][3]
            if maybe_t:
                anchor_dt_any = self._parse_date_soft(maybe_t)

        anchor_dt = _as_date(anchor_dt_any)
        time_dir = infer_time_direction(
            query,
            default=getattr(getattr(self, "global_config", object()), "time_direction", None)
        )
        print(f"[DEBUG][anchor] anchor_date={(anchor_dt.isoformat() if anchor_dt else None)}, "
            f"op={time_dir}, inclusive=True, anchor_relation={anchor_rel}")

        # 预计算：每条的解析日期（统一为 date）与来源分（用于排序）
        base_score_map: Dict[Tuple[str, str, str], float] = {}
        for sc, (s, r, o, _) in topk_pairs:
            base_score_map[(s, r, o)] = sc

        def _base_score_of(quad: Quad) -> float:
            s, r, o, _ = quad
            return base_score_map.get((s, r, o), 0.0)

        parsed_dt_list: List[Optional[_date]] = [
            _as_date(self._parse_date_soft(t)) if t else None
            for (_, _, _, t) in augmented_quads_rel
        ]
        neg_score_list: List[float] = [-_base_score_of(q) for q in augmented_quads_rel]

        # Step 3: 时间过滤（仅当同时有 anchor_dt 与方向时硬筛）
        if anchor_dt is not None and time_dir in {"before", "after"}:
            def _pass(d: Optional[_date]) -> bool:
                if d is None:
                    return False
                return (d <= anchor_dt) if time_dir == "before" else (d >= anchor_dt)

            kept_idx_list = [i for i, d in enumerate(parsed_dt_list) if _pass(d)]
            print(f"[DEBUG][filter] kept={len(kept_idx_list)} / {len(augmented_quads_rel)} (apply=True, dir={time_dir})")

            if len(kept_idx_list) == 0:
                print("[DEBUG][filter] no fact satisfied time constraint; fallback to distance-only ordering.")
                kept_idx_list = list(range(len(augmented_quads_rel)))
                time_filter_applied = False
            else:
                time_filter_applied = True
        else:
            kept_idx_list = list(range(len(augmented_quads_rel)))
            time_filter_applied = False
            print(f"[DEBUG][filter] kept={len(kept_idx_list)} / {len(augmented_quads_rel)} (apply=False)")

        # Step 4: 在保留集合里排序（是否违背约束, 与锚点距离, -score）
        def _order_key(i: int) -> Tuple[int, float, float]:
            violate = 0
            if anchor_dt is not None and time_dir in {"before", "after"} and parsed_dt_list[i] is not None:
                d = parsed_dt_list[i]
                violate = 0 if ((d <= anchor_dt) if time_dir == "before" else (d >= anchor_dt)) else 1
            dist_days = abs((parsed_dt_list[i] - anchor_dt).days) if (anchor_dt is not None and parsed_dt_list[i] is not None) else float("inf")
            return (violate, float(dist_days), float(neg_score_list[i]))

        kept_idx_sorted = sorted(kept_idx_list, key=_order_key)

        # 输出
        top_k_fact_indices = [str(i) for i in kept_idx_sorted]
        top_k_facts = [augmented_quads_rel[i] for i in kept_idx_sorted]

        print(f"[DEBUG][final] returned={len(top_k_facts)}")

        rerank_end = time.time()
        rerank_log: Dict[str, Any] = {
            "num_candidates": len(scored),
            "num_topk": len(topk_facts),
            "num_after_augment": augmented_total,
            "num_with_time": num_with_time,
            "time_filter_applied": time_filter_applied,
            "relation_filter": anchor_rel,
            "num_returned": len(top_k_facts),
            "rerank_time_sec": rerank_end - rerank_start,
        }
        if hasattr(self, "rerank_time"):
            self.rerank_time += rerank_end - rerank_start

        return top_k_fact_indices, top_k_facts, rerank_log



    
    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
        
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    #####DEBUG
    def _debug_first_k(self, iterable, k=5):
        try:
            return iterable[:k]
        except Exception:
            return list(islice(iter(iterable), k))

    def _debug_fetch_fact_by_index(self, idx):
        """
        尝试用多种“常见来源”把 index -> 原始 fact（可能是字符串或三元/四元）
        只用于调试打印，不改变任何逻辑。
        """
        # 1) 并行数组（如果有的话）
        for name in ("fact_node_tuples","fact_tuples","facts","fact_list","fact_values"):
            arr = getattr(self, name, None)
            if arr is not None:
                try:
                    return name, arr[idx]
                except Exception:
                    pass

        # 2) 嵌入库里的文本（常见字段名都试一遍）
        store = getattr(self, "fact_embedding_store", None)
        if store is not None:
            # 2.1 直接按属性列表访问
            for attr in ("texts","contents","items","index_to_text","data"):
                if hasattr(store, attr):
                    try:
                        obj = getattr(store, attr)
                        if isinstance(obj, (list, tuple)):
                            return f"fact_embedding_store.{attr}", obj[idx]
                    except Exception:
                        pass
            # 2.2 常见方法
            for m in ("get_text","get","__getitem__"):
                if hasattr(store, m):
                    try:
                        return f"fact_embedding_store.{m}()", getattr(store, m)(idx)
                    except Exception:
                        pass

        # 3) 实在不行，返回 None
        return "<unknown>", None

    def _debug_parse_to_tuple(self, x):
        """
        尝试把字符串形式的 "(s, r, o, t)" / "(s, r, o)" 解析成 tuple；否则原样返回
        """
        if isinstance(x, str) and x.startswith("(") and x.endswith(")"):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x
    # ===== Helpers (add once in HippoRAG class) =====
    def _r_base(self, r: str) -> str:
        """Strip any suffix like '(such_as_...)' to get relation base."""
        import re
        return re.sub(r"\(.*\)$", "", (r or "").strip())

    def _parse_date(self, s):
        import datetime as _dt
        if not s: return None
        s = str(s).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
            try:
                d = _dt.datetime.strptime(s, fmt).date()
                if fmt in ("%Y-%m", "%Y/%m"): return d.replace(day=1)
                if fmt == "%Y": return _dt.date(int(s[:4]), 1, 1)
                return d
            except ValueError:
                pass
        return None

    def _time_cue(self, query: str):
        """Return (op, inclusive) where op in {'before','after',None}."""
        import re
        q = (query or "").lower()
        # before family
        if re.search(r'\b(on or before|no later than|not after|by|until|through|before)\b', q): return ("before", True)
        if re.search(r'\b(prior to|earlier than)\b', q):                                        return ("before", False)
        # after family
        if re.search(r'\b(on or after|no earlier than|since|from|starting|after)\b', q):       return ("after", True)
        if re.search(r'\b(later than)\b', q):                                                   return ("after", False)
        return (None, True)

    def _selector_from_query(self, query: str):
        """Return 'min' | 'max' | None based on first/last cues."""
        import re
        q = (query or "").lower()
        if re.search(r'\b(first|earliest)\b', q):          return "min"
        if re.search(r'\b(last|latest|most recent)\b', q): return "max"
        return None

    def _date_keep(self, d, anchor_date, op, inclusive):
        """Predicate to keep a date given op/inclusive/anchor."""
        if op is None or anchor_date is None:
            return True
        if d is None:
            return False
        if op == "before":
            return (d <= anchor_date) if inclusive else (d < anchor_date)
        else:  # after
            return (d >= anchor_date) if inclusive else (d > anchor_date)

    def _pick_rep_quad(self, s, r, o, *, anchor_date, op, inclusive, selector):
        """
        From ALL timestamps of (s,r,o), apply time constraint first,
        then pick the representative quad:
        - after  -> min { d >= anchor }
        - before -> max { d <= anchor }
        - none   -> selector 'min'/'max' else default latest
        Return (s,r,o,ds) or None.
        """
        quads = self._quads_for_triple(s, r, o)  # list of (s,r,o,'YYYY-MM-DD'), any order
        if not quads:
            return None
        items = []
        for (ss, rr, oo, ds) in quads:
            d = self._parse_date(ds)
            if d is None:
                continue
            if not self._date_keep(d, anchor_date, op, inclusive):
                continue
            items.append((d, (ss, rr, oo, ds)))
        if not items:
            return None

        if anchor_date is not None and op == "after":
            return min(items, key=lambda t: (t[0] - anchor_date))[1]
        if anchor_date is not None and op == "before":
            return max(items, key=lambda t: (t[0] - anchor_date))[1]

        if selector == "min": return min(items, key=lambda t: t[0])[1]
        if selector == "max": return max(items, key=lambda t: t[0])[1]
        return max(items, key=lambda t: t[0])[1]  # default latest
    


    def _safe_parse_date(d: Any) -> Optional[datetime]:
        """把 YYYY[-MM[-DD]] 或类似格式尽量解析成 datetime；失败则返回 None。"""
        if d is None:
            return None
        if isinstance(d, (datetime, )):
            return d
        if isinstance(d, date):
            return datetime(d.year, d.month, d.day)
        if isinstance(d, (int, float)):
            # 诸如 2012 或 20120726 的场景
            s = str(int(d))
        else:
            s = str(d).strip()

        if not s or s.lower() in {"na", "none", "null", "unknown"}:
            return None

        # 常见格式兜底：YYYY[-MM[-DD]]
        m = re.match(r"^(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?$", s)
        if m:
            y = int(m.group(1))
            M = int(m.group(2)) if m.group(2) else 1
            d = int(m.group(3)) if m.group(3) else 1
            try:
                return datetime(y, M, d)
            except Exception:
                return None

        # 其他自由文本（e.g. "July 26, 2012"）
        try:
            from dateutil import parser  # 如果环境里没有，可以换成更保守解析
            return parser.parse(s, default=datetime(1970, 1, 1))
        except Exception:
            return None

    
   


    def _coerce_to_idx_score_list(
        query_fact_scores: Any
    ) -> List[Tuple[Any, float]]:
        """
        统一把 query_fact_scores 变成 [(idx_or_fact, score)] 列表。
        支持：
        - numpy 1D 向量：与事实表对齐，返回 [(i, score_i)]
        - dict: {idx: score}
        - list/tuple:
            * [(idx, score)] 或 [(fact, score)]
        """
        if hasattr(query_fact_scores, "shape"):  # numpy array
            arr = np.asarray(query_fact_scores).reshape(-1)
            return [(i, float(arr[i])) for i in range(arr.shape[0])]
        if isinstance(query_fact_scores, dict):
            return list(query_fact_scores.items())
        if isinstance(query_fact_scores, (list, tuple)):
            # 简单校验每项长度
            if len(query_fact_scores) == 0:
                return []
            first = query_fact_scores[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return [(x[0], float(x[1])) for x in query_fact_scores]
            # 若直接给的是纯分数列表，则用位置当 idx
            try:
                return [(i, float(s)) for i, s in enumerate(query_fact_scores)]
            except Exception:
                pass
        # 其他情况：尽力转 float 列表
        try:
            as_list = list(query_fact_scores)
            return [(i, float(s)) for i, s in enumerate(as_list)]
        except Exception:
            raise ValueError("Unsupported query_fact_scores format.")


    def _fetch_fact_by_index_generic(obj: Any, idx: Any) -> Tuple[str, str, str, Optional[str]]:
        """
        给一个“对象 obj（通常是 self）”与“事实索引 idx”，尽量取出 (s, r, o, date?)。
        你可以按你的工程实际完善这里的映射。
        优先级：fact_id_to_fact -> facts -> fact_table -> all_facts
        """
        # 1) 如果有显式 map
        if hasattr(obj, "fact_id_to_fact"):
            rec = obj.fact_id_to_fact[idx]  # 支持 idx 为 str/int
        elif hasattr(obj, "facts"):
            rec = obj.facts[idx]
        elif hasattr(obj, "fact_table"):
            rec = obj.fact_table[idx]
        elif hasattr(obj, "all_facts"):
            rec = obj.all_facts[idx]
        else:
            raise KeyError("No fact container found on `self` for idx -> fact mapping.")

        # 适配常见结构：tuple/list/dict
        if isinstance(rec, (list, tuple)):
            if len(rec) == 4:
                s, r, o, t = rec
                return str(s), str(r), str(o), (None if t is None else str(t))
            elif len(rec) == 3:
                s, r, o = rec
                return str(s), str(r), str(o), None
        if isinstance(rec, dict):
            s = rec.get("s") or rec.get("subj") or rec.get("head") or rec.get("subject")
            r = rec.get("r") or rec.get("rel") or rec.get("relation", "")
            o = rec.get("o") or rec.get("obj") or rec.get("tail") or rec.get("object")
            t = rec.get("t") or rec.get("time") or rec.get("date") or rec.get("timestamp")
            return str(s), str(r), str(o), (None if t is None else str(t))

        # 最保守：把整体当字符串
        return str(rec), "", "", None


    def _ensure_quad(fact: Union[Tuple, Dict, str]) -> Tuple[str, str, str, Optional[str]]:
        """把任意三元/四元/字典形式的 fact 规整为 (s, r, o, date_str|None)。"""
        if isinstance(fact, (list, tuple)):
            if len(fact) == 4:
                s, r, o, t = fact
                return str(s), str(r), str(o), (None if t is None else str(t))
            if len(fact) == 3:
                s, r, o = fact
                return str(s), str(r), str(o), None
        if isinstance(fact, dict):
            s = fact.get("s") or fact.get("subj") or fact.get("head") or fact.get("subject")
            r = fact.get("r") or fact.get("rel") or fact.get("relation", "")
            o = fact.get("o") or fact.get("obj") or fact.get("tail") or fact.get("object")
            t = fact.get("t") or fact.get("time") or fact.get("date") or fact.get("timestamp")
            return str(s), str(r), str(o), (None if t is None else str(t))
        # 兜底
        return str(fact), "", "", None


    def _time_order_key(
        anchor_dt: Optional[datetime],
        time_dir: Optional[str],
        fact_dt: Optional[datetime],
        neg_score: float
    ) -> Tuple[int, float, float]:
        """
        排序 key：
        1) 是否满足 before/after 约束（满足=0，不满足=1；None=0）
        2) 与锚点的天数距离（越小越优；无锚点或无时间 => +inf）
        3) 负的分数（为了让原始分更高的更靠前）
        """
        # 满足与否
        violate = 0
        if anchor_dt and time_dir and fact_dt:
            if time_dir == "before" and not (fact_dt <= anchor_dt):
                violate = 1
            elif time_dir == "after" and not (fact_dt >= anchor_dt):
                violate = 1

        # 距离
        if anchor_dt and fact_dt:
            dist_days = abs((fact_dt - anchor_dt).days)
        else:
            dist_days = float("inf")

        return (violate, float(dist_days), float(neg_score))