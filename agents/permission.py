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
        )

    def covers_device_type(self, device_type: str) -> bool:
        return device_type in self.device_types

    def to_dict(self) -> dict:
        return {
            "regions": sorted(self.regions),
            "voltage_levels": sorted(self.voltage_levels),
            "device_types": sorted(self.device_types),
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
    def from_task(task) -> "Permission":
        """
        根据任务涉及的设备，推导出该任务所需的最小权限。
        在 IEEE 14-bus 中简化处理：所有设备都在同一区域。
        """
        device_types = set()
        if task.device_type:
            device_types.add(task.device_type)
        # 工具调用可能涉及多种设备
        device_types.update({"bus", "line"})  # 查询类工具几乎都需要

        return Permission(
            regions={"ieee14"},
            voltage_levels={task.voltage_level} if task.voltage_level else {"MV"},
            device_types=device_types,
        )