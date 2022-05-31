import sys
import flwr as fl
import socket

if __name__ == "__main__":
    port = str(sys.argv[1])
    client_num = int(sys.argv[2])

    server_ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))+':'+port
    with open('./server_IP', 'a', encoding='utf-8') as f:
        f.write(server_ip+'\n')

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=client_num,
        min_fit_clients=client_num,
        min_eval_clients=client_num
    )
    fl.server.start_server("0.0.0.0:"+port, strategy=strategy ,config={"num_rounds": 2})
