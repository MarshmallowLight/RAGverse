from collections import defaultdict
from evaluator import Evaluator
from itertools import chain, combinations
import os
import sys
import json
import networkx as nx
from rule import Rule
from correct_assertion import CorrectAssertion
from copy import copy, deepcopy
import _pickle as pickle
import random
from collections import defaultdict
class Temporal_Model:
    '''
    A model consisting of rules which explain a knowledge graph.
    '''
    def __init__(self, model):
        '''
        :graph: the knowledge graph being modeled
        '''
        self.model = model
        
        # creat nodes for rule graph
        self.nodes = list(self.model.rules.keys())
        self.id_2_rule_dict = {}
        self.rule_2_id_dict = {}

        # creat empty edge list
        self.edges = dict()
        # rule -> set of (s,r,o,t)
        self.rule_to_facts = defaultdict(set)
        # (s,r,o) -> set of rule
        self.fact3_to_rules = defaultdict(set)
        # (s,r,o,t) -> set of rule
        self.fact4_to_rules = defaultdict(set)
        for i in range(len(self.nodes)):
            self.id_2_rule_dict[i] = self.nodes[i]
            self.rule_2_id_dict[self.nodes[i]] = i

        self.num_nodes = len(self.id_2_rule_dict)
        self.tensor = set() # (s, r, o, t) 具有前驱的事实集合
        self.label_matrix = set() # (triple, rule) 每个三元组和其能映射到的规则（同样只计算具有前驱的事实）

        self.aft_to_pre = dict()
        self.pre_to_aft = dict()
        self.rule_to_time = dict()
        self.pair_to_pre_rule = dict()
        # ---------- 新增/更新：读取 id 映射，并做 phrase 规范化 ----------
    def _load_id_maps(self):
        """
        从 reproduce/dataset/<graph.name>/ 读取：
        - entity2id.txt
        - relation2id.txt
        - ts_id.txt
        每行格式：<phrase><空白><id> ；例如：
        Investigative Commission (Czech Republic)    4671
        """
        if getattr(self, "_id_maps_loaded", False):
            return
        import os, re
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(ROOT_DIR, "reproduce", "dataset", self.model.graph.name)

        def _read_map(fname):
            path = os.path.join(base_dir, fname)
            d = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # 用正则抓取“任意非空 + 空白 + 末尾数字”
                        m = re.match(r"^(.*\S)\s+(\d+)\s*$", line)
                        if not m:
                            continue
                        phrase = m.group(1)
                        id_ = int(m.group(2))
                        d[id_] = phrase
            return d

        self._ent_id2str = _read_map("entity2id.txt")
        self._rel_id2str = _read_map("relation2id.txt")
        self._ts_id2str  = _read_map("ts_id.txt")
        self._id_maps_loaded = True

    def _normalize_token(self, s):
        """把字符串统一为小写，并将所有空白替换为下划线；非字符串原样返回。"""
        if s is None:
            return None
        if not isinstance(s, str):
            return s
        s = s.strip().lower()
        import re
        s = re.sub(r"\s+", "_", s)
        return s

    def _id2phrase(self, v, kind):
        """
        把元素 v（可能是 int / '123' / phrase）转换为规范后的字符串。
        kind ∈ {'entity','relation','time'}。
        - entity/relation：返回 小写 + 空白→下划线 之后的 phrase
        - time：总是映射为 ts_id.txt 里的“真实时间字符串”，不做小写/下划线转换
        """
        if v is None:
            return None

        # 尝试解析为整数 id（兼容 '123'）
        idnum = None
        if isinstance(v, int):
            idnum = v
        elif isinstance(v, str):
            try:
                idnum = int(v)
            except:
                # 已是 phrase：entity/relation 规范化；time 原样
                if kind in ("entity", "relation"):
                    return self._normalize_token(v)
                else:
                    return v
        else:
            # 例如 numpy.int64
            try:
                idnum = int(v)
            except:
                return self._normalize_token(str(v)) if kind in ("entity","relation") else str(v)

        if kind == "entity":
            phrase = self._ent_id2str.get(idnum, str(v))
            return self._normalize_token(phrase)
        if kind == "relation":
            phrase = self._rel_id2str.get(idnum, str(v))
            return self._normalize_token(phrase)
        if kind in ("time", "ts"):
            # 时间：映射为真实时间字符串（如 '2005-02-09'）
            return self._ts_id2str.get(idnum, str(v))
        return str(v)

    # ---------- 统一 4 元事实（不在这里做小写/下划线转换） ----------
    def _as_fact4(self, fact):
        """统一事实到 4 元 (s,r,o,t) 形式；允许输入为 3 元或 4 元。"""
        if fact is None:
            return None
        if isinstance(fact, tuple):
            if len(fact) == 4:
                return fact
            if len(fact) == 3:
                s, r, o = fact
                return (s, r, o, None)
        return None

    # ---------- 构建 rule↔facts 映射（s/r/o 为“规范化 phrase”；t 为原始字符串） ----------
    def build_fact_rule_mappings(self):
        """
        存储为（全部使用 phrase；s/r/o 统一小写并将空白→下划线；t 为 ts_id.txt 映射出来的真实时间串）：
        - self.rule_to_facts: rule -> {(s_norm, r_norm, o_norm, t_real)}
        - self.fact4_to_rules: (s_norm, r_norm, o_norm, t_real) -> {rule}
        - self.fact3_to_rules: (s_norm, r_norm, o_norm)        -> {rule}
        """
        self._load_id_maps()
        g = self.model.graph
        rule_set = set(self.nodes)

        def _normalize_to_phrase(f4):
            if not f4:
                return None
            s, r, o, t = f4
            s_p = self._id2phrase(s, "entity")
            r_p = self._id2phrase(r, "relation")
            o_p = self._id2phrase(o, "entity")
            t_p = self._id2phrase(t, "time") if t is not None else None  # <--- 时间总是映射为真实时间
            return (s_p, r_p, o_p, t_p)

        # 1) model.rules[rule]
        if hasattr(self.model, "rules") and isinstance(self.model.rules, dict):
            id2edge = getattr(g, "id_to_edge", {}) if hasattr(g, "id_to_edge") else {}
            for rule, items in self.model.rules.items():
                if rule not in rule_set:
                    continue
                for it in items:
                    f = id2edge.get(it) if isinstance(it, int) else it
                    f4 = self._as_fact4(f)
                    f4_norm = _normalize_to_phrase(f4)
                    if f4_norm:
                        self.rule_to_facts[rule].add(f4_norm)

        # 2) graph.candidates[rule]['facts']
        cand = getattr(g, "candidates", {})
        if isinstance(cand, dict):
            for rule in rule_set:
                entry = cand.get(rule)
                if isinstance(entry, dict) and 'facts' in entry:
                    for f in entry['facts']:
                        f4 = self._as_fact4(f)
                        f4_norm = _normalize_to_phrase(f4)
                        if f4_norm:
                            self.rule_to_facts[rule].add(f4_norm)

        # 3) temporal edges: (pre_rule, aft_rule) -> [facts_of_aft]
        if hasattr(self, "edges") and isinstance(self.edges, dict):
            for edge, fact_list in self.edges.items():
                if not isinstance(edge, tuple) or len(edge) < 2:
                    continue
                aft_rule = edge[-1]
                if aft_rule in rule_set and isinstance(fact_list, (list, set)):
                    for f in fact_list:
                        f4 = self._as_fact4(f)
                        f4_norm = _normalize_to_phrase(f4)
                        if f4_norm:
                            self.rule_to_facts[aft_rule].add(f4_norm)

        # 4) 反向索引（键使用规范化后的 phrase + 真实时间）
        for rule, facts in self.rule_to_facts.items():
            for (s_norm, r_norm, o_norm, t_real) in facts:
                self.fact4_to_rules[(s_norm, r_norm, o_norm, t_real)].add(rule)
                self.fact3_to_rules[(s_norm, r_norm, o_norm)].add(rule)


    def rules_for_fact(self, fact):
        """
        输入 fact 可为 3/4 元（可能是 phrase 或 id）。
        - s/r/o：统一小写 + 空白→下划线；
        - t：若给了 id 或数字串，先映射到真实时间再查；若给了已是时间串，直接用。
        """
        f4 = self._as_fact4(fact)
        if f4 is None:
            return set()
        self._load_id_maps()
        s, r, o, t = f4
        # 规范化 s/r/o
        s_norm = self._normalize_token(s) if isinstance(s, str) else self._id2phrase(s, "entity")
        r_norm = self._normalize_token(r) if isinstance(r, str) else self._id2phrase(r, "relation")
        o_norm = self._normalize_token(o) if isinstance(o, str) else self._id2phrase(o, "entity")
        # 规范化 t（映射为真实时间）
        t_real = self._id2phrase(t, "time") if t is not None else None

        out = set()
        if (s_norm, r_norm, o_norm, t_real) in self.fact4_to_rules:
            out |= self.fact4_to_rules[(s_norm, r_norm, o_norm, t_real)]
        if (s_norm, r_norm, o_norm) in self.fact3_to_rules:
            out |= self.fact3_to_rules[(s_norm, r_norm, o_norm)]
        return out

    def facts_for_rule(self, rule):
        """返回某条 rule 对应到的所有 (s_norm, r_norm, o_norm, t_phrase) 事实集合。"""
        return self.rule_to_facts.get(rule, set())



    def iscomplete(self):
        print([len(self.tensor_no_time), len(self.model.graph.triple_list)])
        if len(self.tensor_no_time & self.model.graph.triple_list) != len(self.model.graph.triple_list):
            return True
        return False
    
    def post_process(self):
        for rule in self.pre_to_aft.keys():
            if len(rule) == 4:
                pair = (rule[0], rule[1])
                if pair not in self.pair_to_pre_rule.keys():
                    self.pair_to_pre_rule[pair] = set()
                self.pair_to_pre_rule[pair].add(rule)
            rule_list = self.pre_to_aft[rule]
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size']) if len(g) == 4 else len(self.model.graph.candidates[g[0]]['ca_to_size'])))
            self.pre_to_aft[rule] = rule_list
        
        for rule in self.aft_to_pre.keys():
            rule_list = self.aft_to_pre[rule]
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size']) if len(g) == 4 else len(self.model.graph.candidates[g[0]]['ca_to_size'])))
            self.aft_to_pre[rule] = rule_list
        
        for pair in self.pair_to_pre_rule.keys():
            rule_list = list(self.pair_to_pre_rule[pair])
            rule_list = sorted(rule_list, reverse=True, key= lambda g: (len(g), len(self.model.graph.candidates[g]['ca_to_size'])))
            self.pair_to_pre_rule[pair] = rule_list

    
    def add_edge(self, edge, input_dict, time_dict, type = 'test'):
        if edge in self.edges:
            print('Already added')
            return

        self.edges[edge] = list(input_dict[edge])

        aft_rule = edge[-1]
        if type == 'final':
            #self.model.add_rule(aft_rule)
            if aft_rule not in self.nodes:
                self.nodes.append(edge[-1])
                self.id_2_rule_dict[len(self.nodes)] = aft_rule
                self.rule_2_id_dict[aft_rule] = len(self.nodes)
        if len(edge) == 2:
            pre_rule = edge[0]
            if type == 'final':
                #self.model.add_rule(pre_rule)
                if pre_rule not in self.nodes:
                    self.nodes.append(edge[0])
                    self.id_2_rule_dict[len(self.nodes)] = pre_rule
                    self.rule_2_id_dict[pre_rule] = len(self.nodes)
                if (pre_rule, aft_rule) not in self.rule_to_time.keys():
                    self.rule_to_time[(pre_rule, aft_rule)] = time_dict[edge]
        if len(edge) == 3:
            pre_rule = (edge[0], edge[1])
            if type == 'final':
                if edge[0] not in self.nodes:
                    self.nodes.append(edge[0])
                    self.id_2_rule_dict[len(self.nodes)] = edge[0]
                    self.rule_2_id_dict[edge[0]] = len(self.nodes)
                if edge[1] not in self.nodes:
                    self.nodes.append(edge[1])
                    self.id_2_rule_dict[len(self.nodes)] = edge[1]
                    self.rule_2_id_dict[edge[1]] = len(self.nodes)
                if (pre_rule, aft_rule) not in self.rule_to_time.keys():
                    self.rule_to_time[(pre_rule, aft_rule)] = time_dict[edge]
        
        if aft_rule not in self.aft_to_pre.keys():
            self.aft_to_pre[aft_rule] = set()
        self.aft_to_pre[aft_rule].add(pre_rule)

        if pre_rule not in self.pre_to_aft.keys():
            self.pre_to_aft[pre_rule] = set()
        self.pre_to_aft[pre_rule].add(aft_rule)

        self.make_assertions(edge, input_dict)

    def remove_edge(self, edge, input_dict, time_dict):
        if edge != self.cache['last_updated_edge']:
            print('We can only remove the last added rule.')
            return
        if edge not in self.edges: # make sure the rule is actually there
            return
        # remove rule
        del self.edges[edge]
        if len(edge) == 2:
            self.aft_to_pre[edge[-1]].remove(edge[0])
        if len(edge) == 3:
            self.aft_to_pre[edge[-1]].remove((edge[0], edge[1]))
        if len(self.aft_to_pre[edge[-1]]) == 0:
            del self.aft_to_pre[edge[-1]]
        
        self.undo_assertions(edge, input_dict)

    def make_assertions(self, edge, input_dict):
        '''
        Fills in model's tensor and node label map with assertions of a rule

        :rule: a rule
        '''
        # reset cache
        self.cache = {'last_updated_edge': edge}

        # update cache
        self.cache['new_triples'] = input_dict[edge].difference(self.tensor)
        rule_triple_combine = set([(edge[-1], triple) for triple in input_dict[edge]])
        self.cache['new_rules'] = rule_triple_combine.difference(self.label_matrix)

        self.tensor.update(self.cache['new_triples'])
        self.label_matrix.update(self.cache['new_rules'])

    def undo_assertions(self, egde, input_dict):
        '''
        Removes things from model's tensor and node label map

        :rule: a rule
        '''
        self.tensor.difference_update(self.cache['new_triples'])
        self.label_matrix.difference_update(self.cache['new_rules'])

    def print_stats(self, temporal_model):
        evaluator = Evaluator(self.model.graph)
        val = evaluator.evaluate_temporal(temporal_model)
        print('----- Model stats -----')
        print('L(G,M) = {}'.format(round(val, 2)))
        null_val = evaluator.evaluate_temporal(Temporal_Model(self.model))
        print([val, null_val])
        print('% Bits needed: {}'.format(round((val / null_val) * 100, 2)))
        print('# Rules: {}'.format(len(self.edges)))
        print('% Edges Explained: {}'.format(round(len(self.tensor & self.model.graph.fact_list) / len(self.model.graph.fact_list) * 100, 2)))
        print('-----------------------')
    
    def update(self, rule_pair, span):
        if rule_pair in self.rule_to_time.keys():
            self.rule_to_time[rule_pair] += span
    
    def add_new_rules(self, new_rules, span):
        for new_rule in new_rules:
            if new_rule in self.aft_to_pre.keys():
                continue
            else:
                if (new_rule[0], new_rule[2]) in self.pair_to_pre_rule.keys():
                    pre_rules = self.pair_to_pre_rule[(new_rule[0], new_rule[2])]
                    self.aft_to_pre[new_rule] = pre_rules
                    for pre_rule in pre_rules:
                        if new_rule not in self.pre_to_aft[pre_rule]:
                            self.pre_to_aft[pre_rule].append(new_rule)
                        if (pre_rule, new_rule) not in self.rule_to_time.keys():
                            self.rule_to_time[(pre_rule, new_rule)] = []
                        self.rule_to_time[(pre_rule, new_rule)].append(span)

