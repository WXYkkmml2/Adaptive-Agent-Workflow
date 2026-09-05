# Power Grid Agent

多智能体系统原型。
使用 IEEE 14-bus 标准测试网络 + pandapower 环境



## 快速开始

```bash
# 克隆
git clone https://github.com/<用户名>/power-grid-agent.git
cd power-grid-agent

# 安装依赖
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt

# 运行演示
python main.py

# 运行测试
pytest tests/ -v
```

## 项目结构
config/settings.py — 全局配置（阈值、映射参数、权限模板）
grid/network.py — IEEE 14-bus 网络加载与状态管理
grid/topology.py — 计算：电气耦合强度 + BFS 拓扑深度
grid/tools.py — 执行智能体工具库（带注册表和权限过滤）
tests/ — 单元测试
main.py — 演示入口

## 技术栈

- Python 3.9+
- pandapower
- NetworkX（拓扑分析）
- NumPy / SciPy（矩阵计算）

## 许可证

MIT