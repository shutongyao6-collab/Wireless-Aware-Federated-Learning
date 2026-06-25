import math

class Aggregator:
    """
    Handles server-side model parameter aggregation logic, including robust aggregation and asynchronous staleness penalty.
    """
    def __init__(self):
        pass

    @staticmethod
    def aggregate(client_updates, current_global_time, alpha=0.005, async_mode=True):
        """
        Robust aggregation with Staleness Penalty.
        :param client_updates: list of dicts. Each dict contains:
            {
                'state_dict': dict,        # Model weights
                'num_samples': int,        # Local training data size
                'start_time': float,       # Global logical time at start of training
                'arrival_time': float      # Logical arrival time at server
            }
        :param current_global_time: Current global clock time at aggregation trigger
        :param alpha: Time decay coefficient
        :param async_mode: Whether to enable asynchronous staleness penalty
        :return: Aggregated global state dictionary
        """
        if not client_updates:
            return None
            
        # Basic sample-count weighted FedAvg
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        raw_weights = []
        for update in client_updates:
            base_weight = update['num_samples'] / total_samples
            
            if async_mode:
                # Compare "Current Global Time" with "Client's Start Time"
                # In sequential simulation, if a device is very slow, current_global_time 
                # (often max(arrival_times)) will be much larger than its start_time, causing a penalty.
                time_diff = current_global_time - update['start_time']
                time_diff = max(0.0, time_diff)
                
                # Decay weight (exponential decay)
                penalty = math.exp(-alpha * time_diff)
            else:
                # Synchronous mode has no decay penalty for slow computation
                penalty = 1.0
                
            raw_weights.append(base_weight * penalty)
            
        # Normalize penalized weights to ensure the sum is 1
        weight_sum = sum(raw_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in raw_weights]
        else:
            normalized_weights = [1.0 / len(raw_weights) for _ in raw_weights]
            
        # Execute weighted aggregation
        global_state_dict = {}
        first = True
        for update, weight in zip(client_updates, normalized_weights):
            state = update['state_dict']
            if first:
                for key in state.keys():
                    global_state_dict[key] = state[key] * weight
                first = False
            else:
                for key in state.keys():
                    global_state_dict[key] += state[key] * weight
                    
        return global_state_dict
