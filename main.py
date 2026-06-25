"""
无线感知联邦学习 —— 主实验（受控 2×2 消融）
对应课题要求：
  - 自适应带宽/计算资源分配  -> ResourceScheduler（选择 + 本地轮数）
  - 梯度压缩 / 量化           -> client.compress_delta（真实作用于传输增量）
  - 鲁棒的模型聚合            -> Aggregator（陈旧度惩罚，应对异步/掉队）
  - 无线条件                  -> WirelessEnvironment（时延/链路失效/资源不均衡/非IID）
  - 能耗效率                  -> total_energy（计算时间+通信时间 的透明代理）
  - 差分隐私                  -> 单独见 run_dp_exp.py（保持主对比的控制变量干净）
"""
import json
import os
import random
import statistics
from collections import defaultdict

import numpy as np
import torch

from utils.data_loader import get_client_dataloaders
from env.wireless_sim import WirelessEnvironment
from env.resource_alloc import ResourceScheduler
from core.server import FLServer
from core.client import FLClient
from core.aggregator import Aggregator

# ================= 固定实验设置（论文中报告一次，不作为自变量）=================
NUM_CLIENTS = 10
MAX_ROUNDS = 20
ALPHA_DIRICHLET = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 0.05
STALENESS_ALPHA = 0.005

DATA_SEED = 2024
SEEDS = [0, 1, 2]              # 多种子取 mean ± std；快速测试可设 [0]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

METHODS = ["baseline", "only_compression", "only_scheduling", "joint_optimization"]
METHOD_FACTORS = {
    "baseline":           {"调度": False, "压缩": False},
    "only_compression":   {"调度": False, "压缩": True},
    "only_scheduling":    {"调度": True,  "压缩": False},
    "joint_optimization": {"调度": True,  "压缩": True},
}
# ==========================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_shared_data(dataset_name="MNIST"):
    """生成一次非 IID 划分，所有方法/种子共用（控制变量：数据固定）。
    dataset_name: 'MNIST' 或 'FashionMNIST'，用于跨数据集稳健性验证。"""
    set_seed(DATA_SEED)
    client_loaders, global_test_loader, _, client_data_sizes = get_client_dataloaders(
        dataset_name=dataset_name, num_clients=NUM_CLIENTS, alpha=ALPHA_DIRICHLET, batch_size=BATCH_SIZE
    )
    return client_loaders, global_test_loader, client_data_sizes


def run_one(method, scale, seed, client_loaders, global_test_loader,
            dp_sigma=0.0, dp_clip=1.0):
    """
    单次受控仿真。对固定 (scale, seed)，信道/掉线事件/初始模型 都只由 (scale, seed) 决定，
    与 method、dp_sigma 无关 —— 保证不同方法/不同 σ 在完全相同的条件下对比。
    """
    set_seed(seed)
    env = WirelessEnvironment(num_clients=NUM_CLIENTS, base_drop_rate_scale=scale)
    channel_seq = [env.generate_channel_states() for _ in range(MAX_ROUNDS)]
    scheduler = ResourceScheduler()

    set_seed(seed)  # 重置：保证初始全局模型在各对比项间一致
    server = FLServer(device=DEVICE)
    clients = {i: FLClient(i, client_loaders[i], device=DEVICE) for i in range(NUM_CLIENTS)}

    ev_rng = random.Random(10_000 + seed)
    events = {r: {cid: (ev_rng.random(), ev_rng.random()) for cid in range(NUM_CLIENTS)}
              for r in range(1, MAX_ROUNDS + 1)}

    global_time = 0.0
    total_transmitted_bytes = 0.0
    total_energy = 0.0
    history = []

    for round_num in range(1, MAX_ROUNDS + 1):
        channel_states = channel_seq[round_num - 1]
        allocations = scheduler.schedule(channel_states, method=method)
        global_state = server.get_global_model_state()

        client_updates = []
        arrival_times = []

        for cid in range(NUM_CLIENTS):
            alloc = allocations[cid]
            state = channel_states[cid]
            if not alloc['selected']:
                continue
            fail_u, drop_u = events[round_num][cid]
            if fail_u < state['link_failure_prob']:
                continue
            if drop_u < state['drop_rate']:
                continue

            client = clients[cid]
            client.update_model(global_state)
            start_time = global_time

            updated_state, num_samples, loss = client.train(
                epochs=alloc['epochs'], learning_rate=LEARNING_RATE,
                dp_sigma=dp_sigma, dp_clip=dp_clip
            )
            # 真实压缩：对传输增量做 Top-k 稀疏 + 量化（影响精度与通信量）
            updated_state = client.compress_delta(global_state, updated_state, alloc['quantization'])

            compute_time = (alloc['epochs'] * num_samples) / state['compute_power']
            bytes_sent = client.compute_communication_bytes(quantization=alloc['quantization'])
            comm_time = bytes_sent / state['bandwidth_bps']

            arrival_time = start_time + compute_time + comm_time
            arrival_times.append(arrival_time)
            total_transmitted_bytes += bytes_sent
            total_energy += (compute_time + comm_time)  # 能耗代理：设备活跃时间

            client_updates.append({
                'state_dict': updated_state,
                'num_samples': num_samples,
                'start_time': start_time,
                'arrival_time': arrival_time,
            })

        if not client_updates:
            continue

        global_time = max(arrival_times)
        new_global_state = Aggregator.aggregate(
            client_updates, global_time, alpha=STALENESS_ALPHA, async_mode=True
        )
        server.update_global_model(new_global_state)

        acc, val_loss = server.evaluate(global_test_loader)
        history.append({
            'round': round_num,
            'accuracy': acc,
            'val_loss': val_loss,
            'global_time': global_time,
            'total_transmitted_bytes': total_transmitted_bytes,
            'total_energy': total_energy,
            'n_clients': len(client_updates),
        })

    return history


def aggregate_over_seeds(per_seed_histories):
    bucket = defaultdict(lambda: defaultdict(list))
    for hist in per_seed_histories:
        for rec in hist:
            r = rec['round']
            for k in ['accuracy', 'val_loss', 'global_time', 'total_transmitted_bytes', 'total_energy']:
                bucket[r][k].append(rec[k])

    def std(xs):
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    out = []
    for r in sorted(bucket):
        b = bucket[r]
        out.append({
            'round': r,
            'accuracy': statistics.fmean(b['accuracy']),
            'accuracy_std': std(b['accuracy']),
            'val_loss': statistics.fmean(b['val_loss']),
            'global_time': statistics.fmean(b['global_time']),
            'global_time_std': std(b['global_time']),
            'total_transmitted_bytes': statistics.fmean(b['total_transmitted_bytes']),
            'total_transmitted_bytes_std': std(b['total_transmitted_bytes']),
            'total_energy': statistics.fmean(b['total_energy']),
            'total_energy_std': std(b['total_energy']),
            'n_seeds': len(b['accuracy']),
        })
    return out


def run_method(method, scale, seeds, client_loaders, global_test_loader):
    histories = []
    for s in seeds:
        hist = run_one(method, scale, s, client_loaders, global_test_loader)
        if hist:
            f = hist[-1]
            print(f"    [{method}] seed={s}: Acc={f['accuracy']:.4f}, "
                  f"Comm={f['total_transmitted_bytes']/1024/1024:.2f}MB, "
                  f"Latency={f['global_time']:.1f}s")
        histories.append(hist)
    return aggregate_over_seeds(histories)


if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    client_loaders, global_test_loader, client_data_sizes = prepare_shared_data()
    with open(os.path.join(results_dir, "client_data_distribution.json"), 'w') as f:
        json.dump(client_data_sizes, f, indent=4)

    for m in METHODS:
        print(f"\n[{m.upper()}] 多种子受控仿真 (seeds={SEEDS})...")
        agg = run_method(m, scale=1.0, seeds=SEEDS,
                         client_loaders=client_loaders, global_test_loader=global_test_loader)
        with open(os.path.join(results_dir, f"{m}.json"), 'w') as f:
            json.dump(agg, f, indent=4)
        if agg:
            last = agg[-1]
            print(f"  => 最终: Acc={last['accuracy']:.4f}±{last['accuracy_std']:.4f}, "
                  f"Comm={last['total_transmitted_bytes']/1024/1024:.2f}MB")

    print("\n[完成] 主实验完成。接着运行 run_drop_rate_exp.py 与 run_dp_exp.py。")