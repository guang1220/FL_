import sys
import flwr as fl
from typing import Dict
from randomforestfedtree import FedTree
import socket

class FeatureTree:
    def __init__(self,selected_features,DecisionTreeClassifier):
        self.selected_features=selected_features
        self.DecisionTreeClassifier=DecisionTreeClassifier

def fit_round(node_num: int) -> Dict:
    """Send round number to client."""
    return {"node_num": node_num}

if __name__ == "__main__":
    port = str(sys.argv[1])
    client_num = int(sys.argv[2])

    server_ip = socket.gethostbyname(socket.getfqdn(socket.gethostname())) + ':' + port
    with open('./server_IP', 'a', encoding='utf-8') as f:
        f.write(server_ip + '\n')

    strategy = FedTree(
        on_fit_config_fn=fit_round,
        min_available_clients=client_num,
        min_fit_clients=client_num,
        min_eval_clients=2
    )
    fl.server.start_server("0.0.0.0:"+port,strategy=strategy,config={"num_rounds": 21})
