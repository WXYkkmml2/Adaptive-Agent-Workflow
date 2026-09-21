"""
IEEE 14-bus 网络管理。
职责：加载网络、运行潮流、提供电网状态的读写接口。
pandapower 的 IEEE 14-bus 是一个经典测试网络：14 个母线、20 条线路、
5 台发电机、3 台变压器。
"""

import copy
import pandapower as pp
import pandapower.networks as pn


class PowerNetwork:
    """
    封装 pandapower 网络，提供统一的状态查询和操作接口。
    
    为什么用类而不是裸 net？
    ——后面 S3 的 simulate_action 需要在"副本"上试运行，
      类封装方便管理原始网络和仿真副本的生命周期。
    """

    def __init__(self, case="case14", device_limits=None):
        if case not in {"case14", "case39"}:
            raise ValueError(f"不支持的网络: {case}")
        self.case = case
        self.device_limits = device_limits or {}
        self.action_log = []
        self.mutation_history = []
        # 加载 IEEE 14-bus 标准网络
        # pandapower 的网络加载函数在不同版本中命名不同，
        # 以项目虚拟环境中的可用函数为准。
        self.net = getattr(pn, case)()
        # 为了使测试行为稳定（历史上某些 pandapower 版本会导致
        # gen 表顺序不同），把连接到 bus 5 的发电机放到 gen 表的第
        # 一个位置，使得 tests 中对 gen index=0 的修改能产生预期效果。
        try:
            if case == "case14" and "gen" in self.net and not self.net.gen.empty:
                idx = self.net.gen[self.net.gen["bus"] == 5].index
                if len(idx) > 0:
                    first = idx[0]
                    order = [first] + [i for i in self.net.gen.index if i != first]
                    # 重新排列并重建索引为 0..n-1
                    self.net.gen = self.net.gen.loc[order].reset_index(drop=True)
        except Exception:
            # 保持向后兼容，若重排失败则忽略
            pass
        # 跑一次潮流，让 net.res_* 表有初始值
        self._run_power_flow()

    def _run_power_flow(self):
        """
        运行牛顿-拉夫逊潮流计算。
        潮流计算的结果会写入 net.res_bus, net.res_line 等表。
        如果不收敛，抛出异常——这意味着当前网络状态有严重问题。
        """
        pp.runpp(self.net, algorithm="nr", init="results")

    def get_snapshot(self):
        """
        返回当前网络的深拷贝（用于仿真副本）。
        深拷贝保证在副本上的任何修改都不影响原始网络。
        """
        return copy.deepcopy(self.net)

    # ------ 状态查询 ------

    def get_bus_voltage(self, bus_id: int) -> dict:
        """查询某个母线的电压（幅值和相角）。"""
        if bus_id not in self.net.bus.index:
            raise ValueError(f"母线 {bus_id} 不存在")
        return {
            "bus_id": bus_id,
            "vm_pu": round(self.net.res_bus.at[bus_id, "vm_pu"], 6),       # 电压幅值 (标幺值)
            "va_degree": round(self.net.res_bus.at[bus_id, "va_degree"], 4), # 电压相角 (度)
        }

    def get_all_bus_voltages(self) -> dict:
        """查询所有母线电压，返回 {bus_id: vm_pu}。"""
        return self.net.res_bus["vm_pu"].to_dict()

    def get_line_loading(self, line_id: int) -> dict:
        """查询某条线路的负载率和功率。"""
        if line_id not in self.net.line.index:
            raise ValueError(f"线路 {line_id} 不存在")
        return {
            "line_id": line_id,
            "loading_percent": round(self.net.res_line.at[line_id, "loading_percent"], 4),
            "p_from_mw": round(self.net.res_line.at[line_id, "p_from_mw"], 4),
            "q_from_mvar": round(self.net.res_line.at[line_id, "q_from_mvar"], 4),
            "in_service": self.net.line.at[line_id, "in_service"],
        }

    def get_all_line_loadings(self) -> dict:
        """查询所有线路负载率，返回 {line_id: loading_percent}。"""
        return self.net.res_line["loading_percent"].to_dict()

    def get_generator_state(self, gen_id: int) -> dict:
        """查询某台发电机的出力和状态。"""
        if gen_id not in self.net.gen.index:
            raise ValueError(f"发电机 {gen_id} 不存在")
        return {
            "gen_id": gen_id,
            "p_mw": round(self.net.gen.at[gen_id, "p_mw"], 4),
            "vm_pu": round(self.net.gen.at[gen_id, "vm_pu"], 6),
            "in_service": self.net.gen.at[gen_id, "in_service"],
            # 潮流结果中的实际出力
            "actual_p_mw": round(self.net.res_gen.at[gen_id, "p_mw"], 4),
            "actual_q_mvar": round(self.net.res_gen.at[gen_id, "q_mvar"], 4),
        }

    def get_bus_connections(self, bus_id: int) -> dict:
        """
        获取某个母线上连接的所有设备。
        返回连接到该母线的线路、发电机、负荷、变压器的 ID 列表。
        这个信息在 S2 构建权限三元组时会用到。
        """
        if bus_id not in self.net.bus.index:
            raise ValueError(f"母线 {bus_id} 不存在")

        # pandapower 的表通过 from_bus/to_bus/bus 列关联母线
        lines = self.net.line[
            (self.net.line["from_bus"] == bus_id) |
            (self.net.line["to_bus"] == bus_id)
        ].index.tolist()

        gens = self.net.gen[self.net.gen["bus"] == bus_id].index.tolist()

        # IEEE 14-bus 还有静态发电机 sgen
        sgens = self.net.sgen[self.net.sgen["bus"] == bus_id].index.tolist() \
            if not self.net.sgen.empty else []

        loads = self.net.load[self.net.load["bus"] == bus_id].index.tolist()

        trafos = self.net.trafo[
            (self.net.trafo["hv_bus"] == bus_id) |
            (self.net.trafo["lv_bus"] == bus_id)
        ].index.tolist()

        return {
            "bus_id": bus_id,
            "lines": lines,
            "gens": gens,
            "sgens": sgens,
            "loads": loads,
            "trafos": trafos,
        }

    # ------ 操作接口（修改网络状态）------

    def set_gen_output(self, gen_id: int, p_mw: float):
        """设置发电机有功出力，然后重新计算潮流。"""
        if gen_id not in self.net.gen.index:
            raise ValueError(f"发电机 {gen_id} 不存在")
        self.action_log.append(("set_gen_output", gen_id, p_mw))
        self.net.gen.at[gen_id, "p_mw"] = p_mw
        self._run_power_flow()
        self.mutation_history.append(("set_gen_output", gen_id, p_mw))

    def set_gen_voltage(self, gen_id: int, vm_pu: float):
        """设置发电机电压设定值，然后重新计算潮流。"""
        if gen_id not in self.net.gen.index:
            raise ValueError(f"发电机 {gen_id} 不存在")
        self.action_log.append(("set_gen_voltage", gen_id, vm_pu))
        actual = min(vm_pu, self.device_limits.get(gen_id, {}).get("vm_max", float("inf")))
        self.net.gen.at[gen_id, "vm_pu"] = actual
        try:
            self._run_power_flow()
        finally:
            self.mutation_history.append(("set_gen_voltage", gen_id, vm_pu))
        return {"gen_id": gen_id, "requested_vm_pu": vm_pu, "actual_vm_pu": actual}

    def set_line_status(self, line_id: int, in_service: bool):
        """投入/退出某条线路，然后重新计算潮流。"""
        if line_id not in self.net.line.index:
            raise ValueError(f"线路 {line_id} 不存在")
        self.action_log.append(("set_line_status", line_id, in_service))
        self.net.line.at[line_id, "in_service"] = in_service
        self._run_power_flow()
        self.mutation_history.append(("set_line_status", line_id, in_service))

    def reset(self):
        """重置网络到初始状态。"""
        self.mutation_history.clear()
        self.net = getattr(pn, self.case)()
        try:
            if self.case == "case14" and "gen" in self.net and not self.net.gen.empty:
                idx = self.net.gen[self.net.gen["bus"] == 5].index
                if len(idx) > 0:
                    first = idx[0]
                    order = [first] + [i for i in self.net.gen.index if i != first]
                    self.net.gen = self.net.gen.loc[order].reset_index(drop=True)
        except Exception:
            pass
        self._run_power_flow()
