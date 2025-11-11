"""
find_paths_with_lines_v3_fix.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
· Based on ANoT model / temporal_model
· Given an entity id, perform forward traversal in the rule-graph
  —— The "next hop" only requires sharing ≥1 entity with the current fact
     and strictly increasing time
· Output: ID version / text version / train.txt line number
"""

# ───────── Parameters ─────────
DATASET_DIR = "data/MultiTQ/kg/tkbc_processed_data"
ENTITY_ID   = 6176         # Iran
MAX_HOP     = 3
SHOW_N      = 5
KEEP_ONLY_LONGER = False   # True → filter out 1-hop
SAVE_TXT    = False
OUT_FILE    = "paths_Iran.txt"
# ─────────────────────────────

# === Uncomment if not deserialized yet ===
import pickle, os
static_pkl   = os.path.join(DATASET_DIR, "static_model_new.pickle")
temporal_pkl = os.path.join(DATASET_DIR, "temporal_model_new.pickle")
model           = pickle.load(open(static_pkl,   "rb"))
temporal_model  = pickle.load(open(temporal_pkl, "rb"))
# ==================================================
print('Model loaded')
from collections import defaultdict
from tqdm import tqdm
import os

print(f"[info] |rules| = {len(model.rules):,}   |edges| = {len(temporal_model.edges):,}")

# ---------- 0. id → name ----------
id2ent, id2rel = {}, {}
def load_map(path, tgt):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                *name, idx = line.rstrip().split()
                tgt[int(idx)] = " ".join(name)
load_map(os.path.join(DATASET_DIR, "entity2id.txt"), id2ent)
load_map(os.path.join(DATASET_DIR, "relation2id.txt"), id2rel)
e_name = lambda x: id2ent.get(x, str(x))
r_name = lambda x: id2rel.get(x, str(x))

# ---------- 0-b. type-id → text ----------
ids_to_labels = model.graph.ids_to_labels
def type_name(tid:int)->str:
    if tid in ids_to_labels:
        return "|".join(r_name(int(r)) for r in ids_to_labels[tid])
    return str(tid)

# ---------- 1. fact ↔ line number ----------
graph_obj = model.graph
fact2line = {fact: idx+1 for idx, fact in graph_obj.id_to_edge.items()}
print(f"[info] stored {len(fact2line):,} fact→line mappings")

# ---------- 2. rule → (fact,line) ----------
rule2facts = defaultdict(list)
print("[info] Building rule2facts ...")
for rule, lst in tqdm(model.rules.items()):
    if not lst:  continue
    sample = lst[0]
    # edge-id
    if isinstance(sample,int):
        for eid in lst:
            if eid in graph_obj.id_to_edge:
                rule2facts[rule].append((graph_obj.id_to_edge[eid], eid+1))
    # tuple
    else:
        for tup in lst:
            if len(tup)==4:
                rule2facts[rule].append((tup, fact2line.get(tup)))
            elif len(tup)==3:
                s,r,o = tup
                rule2facts[rule].append(((s,r,o,None), None))

# Remove duplicates
for r in rule2facts:
    tmp={}
    for f,ln in rule2facts[r]:
        tmp[f]=ln or tmp.get(f)
    rule2facts[r]=[(f,tmp[f]) for f in tmp]
print(f"[info] rules with facts: {len(rule2facts):,}")

# ---------- 3. adjacency list (chain edge) ----------
adj=defaultdict(list)
for k in temporal_model.edges:
    if isinstance(k,tuple) and len(k)==2:
        pre,nxt=k
        adj[pre].append(nxt)
print(f"[info] nodes with out-neighbors: {len(adj):,}")

# ---------- 4. starting nodes ----------
def starting_items(eid:int):
    d=defaultdict(list)
    for r,fl in rule2facts.items():
        for fact,ln in fl:
            if fact[0]==eid or fact[2]==eid:
                d[r].append((fact,ln))
    return d
starts = starting_items(ENTITY_ID)
print(f"[info] entity {ENTITY_ID}({e_name(ENTITY_ID)}) in {len(starts)} start rules")

# ---------- 5. DFS ----------
def dfs_paths(rule0,item0,max_hop:int=3):
    st=[(rule0,item0,0,[(rule0,item0)])]
    while st:
        rule,item,depth,path = st.pop()
        (s,_,o,t),_ = item
        ents_now={s,o}
        if depth==max_hop:
            yield path; continue
        for nr in adj.get(rule,[]):
            for nf,ln in rule2facts.get(nr,[]):
                ns,_,no,nt = nf
                # strictly increasing time
                time_ok = (t is None or nt is None or nt > t)
                # must share at least one entity
                share_any = bool( ents_now & {ns,no} )
                if share_any and time_ok:
                    st.append((nr,(nf,ln),depth+1,path+[(nr,(nf,ln))]))
        if not adj.get(rule):
            yield path

all_paths=[]
for r,fl in starts.items():
    for it in fl:
        all_paths.extend(dfs_paths(r,it,MAX_HOP))
print(f"[info] total raw paths: {len(all_paths):,}")

# (Optional) filter: keep only ≥2-hop and deduplicate
if KEEP_ONLY_LONGER:
    all_paths=[p for p in all_paths if len(p)>=2]
    sig=lambda p:(tuple(x[0] for x in p), tuple(x[1][0][3] for x in p))
    all_paths=list({sig(p):p for p in all_paths}.values())
    print(f"[info] paths (>=2 hop, uniq): {len(all_paths):,}")

# ---------- 6. Stringification ----------
def fact_id_str(f):  return f"({f[0]}) -{f[1]}-> ({f[2]}) @t={f[3]}"
def fact_txt_str(f): return f"({e_name(f[0])}) -{r_name(f[1])}-> ({e_name(f[2])}) @t={f[3]}"
def rule_str(r): ts,rel,to,dr=r; return f"[{type_name(ts)}] -{r_name(rel)}-> [{type_name(to)}] ({dr})"
def path_text(p):
    seg=[]
    for rule,(fact,ln) in p:
        seg.append(f"[ID ] {fact_id_str(fact)}   [line# {ln}]\n"
                   f"[TXT] {fact_txt_str(fact)}\n"
                   f"      ⟹ {rule_str(rule)}")
    return "\n".join(seg)

print(f"\n=== SAMPLE {SHOW_N} PATHS (ID / TEXT / LINE) ===")
for p in all_paths[:SHOW_N]:
    print(path_text(p)); print("-"*80)

# ---------- 7. Save ----------
if SAVE_TXT:
    with open(OUT_FILE,"w",encoding="utf-8") as fout:
        for p in all_paths:
            fout.write(path_text(p)+"\n"+"-"*80+"\n")
    print(f"[saved] {OUT_FILE}")
