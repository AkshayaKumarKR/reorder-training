from models.graphsage_mini_graph import SAGE
from models.gat_mini_graph import GAT
from models.gcn_mini_graph import GCN


class Models:
    def get_models():
        return {
            "mini": {
                "GRAPHSAGE": SAGE,
                "GCN": GCN,
                "GAT": GAT,
                },
            }
