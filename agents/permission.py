"""
权限三元组管理。

"权限三元组的来源是，实例化智能体时，根据分配给智能体的任务所涉及的设备，
 从电网设备台账和调度权限配置中提取对应的区域标识、电压等级和允许访问的设备类型，
 形成智能体的权限三元组；再和父智能体的权限范围取交集，
 确保子智能体权限不超出父智能体的权限范围，越往下越收窄。"
"""

from dataclasses import dataclass, field
from config.settings import ROOT_PERMISSION


@dataclass
class Permission:
    """
    权限三元组: (区域标识, 电压等级, 允许设备类型)
    
    子智能体的权限 = 父智能体权限 ∩ 任务所需权限
    交集的目的是确保给的是子集，权限只能继承或收窄，不可超出。
    """
    regions: set = field(default_factory=lambda: {"all"})
    voltage_levels: set = field(default_factory=lambda: {"HV", "MV", "LV"})
    device_types: set = field(default_factory=lambda: {"bus", "line", "gen", "trafo", "switch", "load"})
    scope: dict = field(default_factory=dict)

    def intersect(self, child_required: "Permission") -> "Permission":
        """
        取交集，生成子智能体的实际权限。

        例：父节点有 220kV 权限，子任务需要 110kV → 交集为空 → 子节点无权操作。
        这就是原文档说的"父节点都没有这个权限，子节点这个任务就绕开了权限管控"
        """
        # "all" 是通配符，对方有什么就给什么
        if "all" in self.regions:
            new_regions = child_required.regions.copy()
        elif "all" in child_required.regions:
            new_regions = self.regions.copy()
        else:
            new_regions = self.regions & child_required.regions

        new_vl = self.voltage_levels & child_required.voltage_levels
        new_dt = self.device_types & child_required.device_types

        return Permission(
            regions=new_regions,
            voltage_levels=new_vl,
            device_types=new_dt,
            scope={key: (self.scope[key] & child_required.scope[key]
                         if key in self.scope and key in child_required.scope
                         else set(self.scope.get(key, child_required.scope.get(key, set()))))
                   for key in self.scope.keys() | child_required.scope.keys()},
        )

    def covers_device_type(self, device_type: str) -> bool:
        return device_type in self.device_types

    def to_dict(self) -> dict:
        return {
            "regions": sorted(self.regions),
            "voltage_levels": sorted(self.voltage_levels),
            "device_types": sorted(self.device_types),
            "scope": {key: sorted(value) for key, value in self.scope.items()},
        }

    @staticmethod
    def root_permission() -> "Permission":
        """根智能体默认拥有全部权限。"""
        return Permission(
            regions=ROOT_PERMISSION["regions"].copy(),
            voltage_levels=ROOT_PERMISSION["voltage_levels"].copy(),
            device_types=ROOT_PERMISSION["device_types"].copy(),
        )

    @staticmethod
    def from_task(task, net=None) -> "Permission":
        """
        根据任务涉及的设备，推导出该任务所需的最小权限。
        在 IEEE 14-bus 中简化处理：所有设备都在同一区域。
        """
        device_types = {"bus", "line", "trafo"}
        if task.device_type == "gen" or not (task.description or "").strip().startswith(("查询", "检查", "验证")):
            device_types.add("gen")
        if task.device_type:
            device_types.add(task.device_type)
        description = (task.description or "").strip()
        if description.startswith(("仿真", "执行", "调整", "调节", "恢复")) or "发电机" in description:
            device_types.add("gen")

        if net is None:
            return Permission(regions={"all"}, device_types=device_types)
        buses = {int(i) for i in task.devices if i in net.bus.index}
        gens = {int(i) for i in task.devices if i in net.gen.index and task.device_type == "gen"}
        buses.update(int(net.gen.at[i, "bus"]) for i in gens)
        scope = {"bus": buses} if buses else {}
        if task.device_type == "gen":
            scope["gen"] = gens
        regions = {str(int(net.bus.at[i, "zone"])) for i in buses} or {"all"}
        levels = {"HV" if net.bus.at[i, "vn_kv"] >= 100 else "MV" if net.bus.at[i, "vn_kv"] >= 10 else "LV" for i in buses}
        return Permission(regions=regions, voltage_levels=levels or {"HV", "MV", "LV"},
                          device_types=device_types, scope=scope)
