"""
D0 计算模块。

D0 的含义：
"这个目标的调度操作，在物理层面上的电网上的影响得有多深 b、多广 a，
 然后关联着后面的任务编排至少需要多大的纵深。"

两个分量：
  a — 电气耦合强度：局部导纳矩阵条件数的归一化值，反映"多广"
  b — BFS 拓扑深度：扰动影响的传播距离，反映"多深"
  D0 = max(a_norm, b_norm)，取保守值
"""

import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
import pandapower as pp
import pandapower.topology as top

from config.settings import (
    BFS_VOLTAGE_CHANGE_THRESHOLD,
    CONDITION_NUMBER_UPPER_BOUND,
    H_MAX,
)


def get_neighbor_buses(net, bus_id: int, depth: int = 1) -> set:
    """
    获取目标母线周围 depth 跳范围内的所有母线。
    用 pandapower 内置的 networkx 图做 BFS。
    
    参数:
        net: pandapower 网络
        bus_id: 目标母线 ID
        depth: 搜索跳数
    
    返回:
        包含目标母线及其邻居的母线 ID 集合
    """
    # pandapower 提供了从网络拓扑创建 networkx 图的工具
    # respect_switches=True 表示断开的开关不算连通
    graph = top.create_nxgraph(net, respect_switches=True)

    if bus_id not in graph.nodes:
        raise ValueError(f"母线 {bus_id} 不在网络拓扑图中")

    # BFS 获取 depth 跳以内的所有节点
    # nx.single_source_shortest_path_length 返回 {node: distance} 字典
    import networkx as nx
    distances = nx.single_source_shortest_path_length(graph, bus_id, cutoff=depth)
    return set(distances.keys())


def compute_coupling_strength(net, bus_id: int) -> float:
    """
    计算电气耦合强度 a。
    
  
    1. 取目标母线 + 周围 1 跳邻居，构成局部子网
    2. 从全网导纳矩阵中提取这些母线对应的子矩阵
    3. 计算子矩阵的条件数
    4. 归一化到 [0, 1]
    
    条件数越大 → 矩阵越病态 → 局部电气参数关系越敏感 → 耦合越强 → a 越大
    
    返回:
        a: 归一化耦合强度，范围 [0, 1]
    """
    # 确保潮流已经计算过，这样内部的 _ppc 才有导纳矩阵
    if not hasattr(net, "_ppc") or net._ppc is None:
        pp.runpp(net, numba=False)

    # ---- 提取全网导纳矩阵 (Y-bus) ----
    # pandapower 潮流计算后，Y-bus 存储在内部 pypower 格式中
    # _ppc["internal"]["Ybus"] 是一个 scipy 稀疏矩阵
    ybus = net._ppc["internal"]["Ybus"]

    # ---- pandapower 母线 ID → 内部矩阵索引的映射 ----
    # pandapower 的外部 bus index 和 _ppc 内部的矩阵行列索引不一定一致
    # _pd2ppc_lookups["bus"] 提供了 外部ID → 内部索引 的映射
    bus_lookup = net._pd2ppc_lookups["bus"]

    # 获取局部母线集合（目标 + 1 跳邻居）
    local_buses = get_neighbor_buses(net, bus_id, depth=1)

    # 转换为内部索引
    internal_indices = []
    for b in local_buses:
        if b in bus_lookup:
            idx = bus_lookup[b]
            # bus_lookup 可能返回 numpy int，确保是 Python int
            internal_indices.append(int(idx))

    if len(internal_indices) < 2:
        # 只有一个母线（孤立节点），耦合强度为 0
        return 0.0

    internal_indices = sorted(internal_indices)

    # ---- 提取局部子矩阵 ----
    # 用行列索引切片稀疏矩阵
    ybus_dense = ybus.toarray()
    sub_matrix = ybus_dense[np.ix_(internal_indices, internal_indices)]

    # ---- 计算条件数 ----
    # 条件数 = ||A|| * ||A^(-1)||，用 numpy 计算
    # 对于导纳矩阵，条件数反映了局部电气参数之间的耦合紧密程度
    try:
        cond = np.linalg.cond(sub_matrix)
    except np.linalg.LinAlgError:
        # 奇异矩阵，耦合极强
        return 1.0

    # ---- 归一化到 [0, 1] ----
    a = min(cond / CONDITION_NUMBER_UPPER_BOUND, 1.0)
    return round(a, 6)


def compute_topology_depth(net, bus_id: int) -> int:
    """
    计算 BFS 拓扑深度 b。
    
    "BFS 搜索，找到节点的潮流/电流变化量小于等于阈值时终止，
     量化操作目标设备会影响多远。"
    
    具体实现：
    1. 在目标母线上施加一个小扰动（注入 1MW 负荷）
    2. 重新计算潮流
    3. 逐层 BFS 检查各层母线的电压变化率
    4. 当某一层所有母线的电压变化率都 < 阈值时停止
    5. 返回 BFS 搜索到的层数（即影响深度）
    
    参数:
        net: pandapower 网络（不会被修改，内部用副本）
        bus_id: 目标母线 ID
    
    返回:
        b: 拓扑影响深度（整数，至少为 1）
    """
    import copy
    import networkx as nx

    # ---- 记录原始电压分布 ----
    original_voltages = net.res_bus["vm_pu"].to_dict()

    # ---- 在副本上施加扰动 ----
    net_copy = copy.deepcopy(net)

    # 在目标母线上临时增加 1MW 负荷作为扰动
    pp.create_load(net_copy, bus=bus_id, p_mw=1.0, q_mvar=0.0, name="perturbation")
    try:
        pp.runpp(net_copy, algorithm="nr", init="results", numba=False)
    except pp.powerflow.LoadflowNotConverged:
        # 如果加扰动后潮流不收敛，说明影响极大
        return max(d for _, d in 
                   nx.single_source_shortest_path_length(
                       top.create_nxgraph(net), bus_id).items())

    perturbed_voltages = net_copy.res_bus["vm_pu"].to_dict()

    # ---- BFS 逐层检查电压变化 ----
    graph = top.create_nxgraph(net, respect_switches=True)
    if bus_id not in graph.nodes:
        raise ValueError(f"母线 {bus_id} 不在拓扑图中")

    # 按 BFS 层级分组
    bfs_layers = dict(nx.bfs_successors(graph, bus_id))

    # 用 shortest_path_length 获取每个节点的层级
    distances = nx.single_source_shortest_path_length(graph, bus_id)
    max_possible_depth = max(distances.values()) if distances else 1

    # 按层级分组
    layers = {}
    for node, dist in distances.items():
        if dist == 0:
            continue  # 跳过目标母线本身
        layers.setdefault(dist, []).append(node)

    # 逐层检查：该层所有母线的电压变化率是否都低于阈值
    effective_depth = 0
    for depth in range(1, max_possible_depth + 1):
        if depth not in layers:
            break

        layer_buses = layers[depth]
        max_change = 0.0
        for b in layer_buses:
            if b in original_voltages and b in perturbed_voltages:
                orig_v = original_voltages[b]
                pert_v = perturbed_voltages[b]
                if orig_v > 0:
                    change = abs(pert_v - orig_v) / orig_v
                    max_change = max(max_change, change)

        effective_depth = depth

        # 如果这一层最大变化率已低于阈值，影响衰减完毕，停止
        if max_change < BFS_VOLTAGE_CHANGE_THRESHOLD:
            break

    return max(effective_depth, 1)


def compute_d0(net, bus_id: int) -> dict:
    """
    计算初始深度 D0。
    
    D0 = max(a_normalized, b_normalized)
    其中 a 已经归一化到 [0,1]，b 需要做归一化（除以网络直径）。
    
    返回包含 a、b、D0 的详细字典，方便调试和后续使用。
    """
    import networkx as nx

    # 计算电气耦合强度 a（已归一化到 [0,1]）
    a = compute_coupling_strength(net, bus_id)

    # 计算 BFS 拓扑深度 b（整数）
    b = compute_topology_depth(net, bus_id)

    # 归一化 b：除以网络直径（最远两点间的最短路径长度）
    graph = top.create_nxgraph(net, respect_switches=True)
    try:
        diameter = nx.diameter(graph)
    except nx.NetworkXError:
        # 图不连通时用最大连通分量的直径
        largest_cc = max(nx.connected_components(graph), key=len)
        diameter = nx.diameter(graph.subgraph(largest_cc))
    
    b_norm = round(b / diameter, 6) if diameter > 0 else 0.0

    # D0 取保守值（越大意味着影响范围越广/深，需要更大纵深）
    d0 = round(max(a, b_norm), 6)

    return {
        "bus_id": bus_id,
        "coupling_strength_a": a,
        "topology_depth_b": b,
        "topology_depth_b_norm": b_norm,
        "network_diameter": diameter,
        "d0": d0,
    }


def d0_to_h0(d0: float) -> int:
    """H = ceil(log2(D0)) + 2，且至少 3 层、至多 H_MAX 层。"""
    if d0 <= 0:
        return 3
    return min(H_MAX, max(3, math.ceil(math.log2(d0)) + 2))


def compute_regional_d0(net, target_bus: int) -> dict:
    """Physical D0 for case39: relative 1 MW decay and local Ybus condition."""
    import copy
    import networkx as nx
    from config.settings import CASE39_DECAY_EPS, CASE39_MAX_B

    graph = top.create_nxgraph(net, respect_switches=True)
    distances = nx.single_source_shortest_path_length(graph, target_bus, cutoff=CASE39_MAX_B)
    changed = copy.deepcopy(net)
    pp.create_load(changed, bus=target_bus, p_mw=1.0, q_mvar=0.0)
    pp.runpp(changed, algorithm="nr", init="results", numba=False)
    delta = {int(i): abs(float(changed.res_bus.at[i, "vm_pu"] - net.res_bus.at[i, "vm_pu"]))
             for i in distances}
    first = max((delta[i] for i, hop in distances.items() if hop == 1), default=0.0)
    b = 1
    for hop in range(1, CASE39_MAX_B + 1):
        layer = [delta[i] for i, depth in distances.items() if depth == hop]
        if not layer:
            break
        b = hop
        if first and max(layer) / first < CASE39_DECAY_EPS:
            break
    buses = get_neighbor_buses(net, target_bus, depth=b)
    lookup = net._pd2ppc_lookups["bus"]
    indices = sorted({int(lookup[i]) for i in buses if int(lookup[i]) >= 0})
    ybus = net._ppc["internal"]["Ybus"]
    matrix = ybus[indices, :][:, indices].toarray()
    kappa = float(np.linalg.cond(matrix)) if len(indices) > 1 else 1.0
    d0 = max(math.log10(kappa), b) if math.isfinite(kappa) else float("inf")
    return {"bus_id": target_bus, "topology_depth_b": b, "condition_number": kappa,
            "network_diameter": nx.diameter(graph), "d0": d0,
            "first_hop_change": first, "voltage_changes": delta}


def regional_scope(net, target_buses: list[int], forbidden_regions=()) -> dict:
    """Reachable buses and responsive generators, excluding forbidden operating zones."""
    import copy
    from config.settings import CASE39_INFLUENCE_RATIO
    info = [compute_regional_d0(net, bus) for bus in target_buses]
    buses = set()
    for result in info:
        threshold = CASE39_INFLUENCE_RATIO * result["first_hop_change"]
        buses.update(i for i, change in result["voltage_changes"].items() if change >= threshold)
        buses.add(result["bus_id"])
    forbidden = {int(i) for i in forbidden_regions}
    buses = {i for i in buses if int(net.bus.at[i, "zone"]) not in forbidden}
    responses = {}
    for gen_id in net.gen.index:
        bus = int(net.gen.at[gen_id, "bus"])
        if int(net.bus.at[bus, "zone"]) in forbidden:
            continue
        changed = copy.deepcopy(net)
        changed.gen.at[gen_id, "vm_pu"] += .02
        pp.runpp(changed, algorithm="nr", init="results", numba=False)
        responses[int(gen_id)] = max(abs(float(changed.res_bus.at[target, "vm_pu"] - net.res_bus.at[target, "vm_pu"]))
                                     for target in target_buses)
    peak = max(responses.values(), default=0.0)
    gens = {i for i, response in responses.items() if response >= CASE39_INFLUENCE_RATIO * peak}
    lines = {int(i) for i, row in net.line.iterrows()
             if int(row.from_bus) in buses and int(row.to_bus) in buses}
    trafos = {int(i) for i, row in net.trafo.iterrows()
              if int(row.hv_bus) in buses and int(row.lv_bus) in buses}
    return {"bus": buses, "gen": gens, "line": lines, "trafo": trafos, "d0_info": info}


def regional_tree_depth(scope: dict, net) -> int:
    from config.settings import CASE39_H_MAX
    info = scope["d0_info"]
    b = max(item["topology_depth_b"] for item in info)
    diameter = max(item["network_diameter"] for item in info)
    h0 = 3 + round((CASE39_H_MAX - 3) * b / diameter)
    zones = {int(net.bus.at[i, "zone"]) for i in scope["bus"]}
    levels = {"HV" if net.bus.at[i, "vn_kv"] >= 100 else "MV" if net.bus.at[i, "vn_kv"] >= 10 else "LV"
              for i in scope["bus"]}
    l = int(len(zones) > 1) + int(len(levels) > 1)
    return min(h0, CASE39_H_MAX, 2 + l)
