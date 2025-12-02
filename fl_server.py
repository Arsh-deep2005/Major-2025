# # fl_server.py
# """
# Federated server for Flower (flwr). This server uses a small subclass of FedAvg
# that saves the aggregated weights (as numpy arrays) to disk after each round.
# After FL completes you can load these weights, set them on a Keras model with
# the same architecture, and save the final .h5 model for inference.
# """
# from flwr.server.strategy import FedAvg
# from flwr.server.client_proxy import ClientProxy
# from typing import List, Tuple, Dict, Any
# import flwr as fl
# import pickle
# import numpy as np

# # ---------- Customize FedAvg to save aggregated params ----------
# class SaveAggregatedFedAvg(FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Tuple[List[np.ndarray], Dict[str, Any]]:


#         aggregated = super().aggregate_fit(rnd, results, failures)

#         if aggregated is not None:
#             weights, metrics = aggregated
#             print(f"📌 Saving aggregated weights for round {rnd}")

#             with open("aggregated_weights.pkl", "wb") as f:
#                 pickle.dump(weights, f)

#         return aggregated


# # ---------- Server configuration ----------
# strategy = SaveAggregatedFedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     min_fit_clients=2,
#     min_evaluate_clients=2,
#     min_available_clients=2,
# )

# if __name__ == "__main__":
#     print("🌾 Starting Flower server (with SaveAggregatedFedAvg strategy)...")
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         strategy=strategy,
#         config=fl.server.ServerConfig(num_rounds=5),
#     )
#     print("✅ Flower server stopped (FL rounds complete).")
#     print("ℹ️ Aggregated weights (if any) are stored in aggregated_weights.pkl")
#     print("Next: build the same Keras model architecture and load those weights (see instructions).")





# fl_server.py
"""
Flower server that saves aggregated weights (numpy arrays) after each round
to 'aggregated_weights.pkl'. After FL finishes, convert those weights to an
H5 Keras model using save_aggregated_to_h5.py
"""

import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
from flwr.common import parameters_to_ndarrays
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy


class SaveAggregatedFedAvg(FedAvg):

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:

        print("\n\n========================================================")
        print(f"=================== ROUND {rnd} START ===================")
        print("========================================================\n")

        # Received results count
        print(f"[SERVER] Total clients responded this round: {len(results)}")

        # PRINT DETAILS FOR EACH CLIENT
        for i, (client, fit_res) in enumerate(results):
            if hasattr(fit_res.parameters, "tensors"):
                w_count = len(fit_res.parameters.tensors)
            else:
                w_count = "Unknown"
            print(f"   • Client {i+1}: Returned {w_count} tensors")

        # CALL ORIGINAL FEDAVG
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            params_obj, metrics = aggregated

            print("\n[SERVER] Aggregating client weights using FedAvg...")

            # Convert Parameters object -> list of numpy arrays
            weights = parameters_to_ndarrays(params_obj)

            print("[SERVER] Aggregated Weight Shapes:")
            for idx, w in enumerate(weights):
                print(f"   Layer {idx}: shape = {w.shape}")

            # SAVE AGGREGATED WEIGHTS
            print(f"\n[SERVER] Saving aggregated weights for round {rnd} → aggregated_weights.pkl")
            with open("aggregated_weights.pkl", "wb") as f:
                pickle.dump(weights, f)

            print("[SERVER] ✔ Aggregated weights saved successfully.")

        else:
            print("❌ [SERVER] aggregated returned None — NO WEIGHTS THIS ROUND!")

        print("\n[SERVER] Broadcasting UPDATED global weights back to all clients...")
        print("========================================================")
        print(f"=================== ROUND {rnd} END =====================")
        print("========================================================\n\n")

        return aggregated


if __name__ == "__main__":

    print("\n🌾 ===============================================")
    print("🌾        STARTING FLOWER SERVER (FedAvg)          ")
    print("🌾 ===============================================\n")

    strategy = SaveAggregatedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    print("🌾 [SERVER] Waiting for clients to connect on 127.0.0.1:8080 ...")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    print("\n\n✅ Flower server finished — all FL rounds complete.")
