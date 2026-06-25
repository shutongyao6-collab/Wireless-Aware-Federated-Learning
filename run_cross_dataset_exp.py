"""
跨数据集稳健性验证 —— 回应 "MNIST 太简单" 的质疑。
在 Fashion-MNIST（更难、更接近真实图像）上重跑同一套受控 2×2 消融，
检验主结论（压缩省通信不掉精度 / 调度省延迟 / 两者互补）是否在更难的任务上依然成立。

控制变量：与 main.py 完全相同的 run_one（相同种子、相同信道与掉线逻辑），
唯一改变的是数据集本身 —— 因此跨数据集对比本身也是受控的。
"""
import os
import json
import statistics

import main

DATASETS = ["MNIST", "FashionMNIST"]   # 唯一改变的维度：数据集
SEEDS = [0, 1, 2]


def run_one_dataset(dataset_name):
    """在指定数据集上跑完整 2×2 消融（多种子），返回每个方法的最终指标。"""
    print(f"\n=== 数据集: {dataset_name} ===")
    client_loaders, global_test_loader, _ = main.prepare_shared_data(dataset_name=dataset_name)

    summary = {}
    for m in main.METHODS:
        accs, comms = [], []
        for s in SEEDS:
            hist = main.run_one(m, 1.0, s, client_loaders, global_test_loader)
            if hist:
                accs.append(hist[-1]["accuracy"])
                comms.append(hist[-1]["total_transmitted_bytes"] / (1024 * 1024))
        mean_acc = statistics.fmean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        mean_mb = statistics.fmean(comms)
        acc_per_mb = mean_acc / mean_mb if mean_mb > 0 else 0.0
        summary[m] = {
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "comm_mb": mean_mb,
            "acc_per_mb": acc_per_mb,
        }
        print(f"  {m:20s}: Acc={mean_acc:.4f}±{std_acc:.4f}, "
              f"Comm={mean_mb:.2f}MB, Acc/MB={acc_per_mb:.3f}")
    return summary


def main_run():
    results = {}
    for ds in DATASETS:
        results[ds] = run_one_dataset(ds)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "cross_dataset.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 打印一个并排对比，便于直接抄进论文
    print("\n========== 跨数据集对比（联合优化 vs 基线）==========")
    for ds in DATASETS:
        b = results[ds]["baseline"]
        j = results[ds]["joint_optimization"]
        ratio = b["comm_mb"] / j["comm_mb"] if j["comm_mb"] > 0 else float("nan")
        print(f"{ds:14s} | 基线 Acc={b['accuracy']:.4f} Comm={b['comm_mb']:.1f}MB | "
              f"联合 Acc={j['accuracy']:.4f} Comm={j['comm_mb']:.1f}MB | "
              f"通信降至 1/{ratio:.1f}")


if __name__ == "__main__":
    main_run()