graph_name = 'MultiTQ/tkbc_processed_data'
# graph_name = 'MultiTQmini/'
root_path = 'reproduce/dataset/'
dataset = graph_name+'/'

#Build the graph
import pickle
from graph import Graph
graph = Graph(graph_name, idify=False)
pickle.dump(graph, open(root_path + dataset + "graph_new.pickle", "wb"))

# create a Searcher object to search for a model (set of rules) and build the raw model
from searcher import Searcher
import pickle, os

dataset = graph_name+'/'
graph = pickle.load(open(root_path + dataset + "graph_new.pickle",  "rb"))
print("graph loaded")
searcher = Searcher(graph)
print("searcher built")
model = searcher.build_model()
pickle.dump(model, open(root_path + dataset + "static_model_new.pickle", "wb"))
model.print_stats()

# build the temporal model
temporal_model, candidate_p, candidate_t = searcher.build_temporal_model(model)
pickle.dump(temporal_model, open(root_path + dataset + "temporal_model_new.pickle", "wb"))
temporal_model.print_stats(temporal_model)

# # 你从 test_dynamic.py 拿到的 top-k fact（无时间）
# fact3 = ('iraq', 'praise_or_endorse', 'iran')
# print(fact3)
# # 直接反查所有匹配到的 rule（都在 temporal model 的节点里）
# rules = temporal_model.rules_for_fact(fact3)
# for r in rules:
#     # r 是形如 (type_s, rel, type_o, 'out'/'in') 的 rule node
#     print('RULE:', r)

# # 如果想看某条 rule 的事实集合：
# for r in rules:
#     facts = temporal_model.facts_for_rule(r)  # set of (s,r,o,t)
#     print('facts covered by rule:', r, ' -> ', facts)