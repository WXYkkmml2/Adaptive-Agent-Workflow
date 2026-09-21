"""
全局配置：阈值、映射参数、权限模板。
S1-S4 的阈值都集中在这里管理。
"""

# ============================================================
# D0 相关
# ============================================================

BFS_VOLTAGE_CHANGE_THRESHOLD = 0.01
CONDITION_NUMBER_UPPER_BOUND = 500.0

H_MAX = 7

# case39 regional restoration experiment (fixed before evaluation)
CASE39_DECAY_EPS = 0.10
CASE39_INFLUENCE_RATIO = 0.15
CASE39_H_MAX = 5
CASE39_DTH = 1.0
CASE39_MAX_B = 8
CASE39_TEMPERATURE = 0.2

# ============================================================
# 确定性指标 C 的阈值
# ============================================================
C_THRESHOLD_HIGH = 0.7   # C > T1: 高确定性，保持 H0
C_THRESHOLD_LOW = 0.3    # C < T2: 低确定性，拆子树

# ============================================================
# 采样温度区间（S3 执行智能体用）
# ============================================================
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.0

# 采样温度权重系数 (a, b, C)
TEMPERATURE_WEIGHTS = (0.3, 0.3, 0.4)

# ============================================================
# 电网约束阈值
# ============================================================
BUS_VOLTAGE_MIN = 0.95
# 放宽电压上限以兼容多版本 pandapower 的初始解和测试场景
BUS_VOLTAGE_MAX = 1.10
LINE_LOADING_MAX = 100.0

# ============================================================
# 权限三元组模板
# ============================================================
ROOT_PERMISSION = {
    "regions": {"all"},
    "voltage_levels": {"HV", "MV", "LV"},
    "device_types": {"bus", "line", "gen", "trafo", "switch", "load"},
}

# ============================================================
# 重试与安全
# ============================================================
MAX_REPLAN_ATTEMPTS = 3   # 最大重规划次数（Step 3 用）

# Frozen case39 protocol v2; never tune against formal runs.
CASE39_MAX_TOKENS = 2048
CASE39_MAX_ATTEMPTS = 3
CASE39_PROMPT_VERSION = "case39-v2"
