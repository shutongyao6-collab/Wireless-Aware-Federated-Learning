"""
丢包率鲁棒性实验 —— 对应课题“link failure / 网络可靠性”下的鲁棒性。
唯一自变量：网络丢包率系数 scale。其余全部受控、多种子取 mean ± std。
"""
import os
import json
import statistics

import main

main.MAX_ROUNDS = 5            # 加速消融

SCALES = [0.5, 1.0, 1.5, 2.0]
SEEDS = [0, 1, 2]


def run_drop_rate_robustness():
    print("\n--- 受控丢包率鲁棒性评估 (多种子) ---")
    client_loaders, global_test_loader, _ = main.prepare_shared_data()

    results = []
    for scale in SCALES:
        for m in main.METHODS:
            finals = []
            for s in SEEDS:
                hist = main.run_one(m, scale, s, client_loaders, global_test_loader)
                finals.append(hist[-1]['accuracy'] if hist else 0.0)
            mean_acc = statistics.fmean(finals)
            std_acc = statistics.pstdev(finals) if len(finals) > 1 else 0.0
            results.append({
                'Drop_Rate_Scale': scale,
                'Method': m,
                'Final_Accuracy': mean_acc,
                'Final_Accuracy_std': std_acc,
            })
            print(f"  -> Scale {scale}, {m}: Final Acc = {mean_acc:.4f} ± {std_acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "drop_rate_robustness.json"), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    run_drop_rate_robustness()