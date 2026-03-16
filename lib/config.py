"""
配置拓扑和流量文件地址
"""
import os

# TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
# TOPOLOGIES_DIR = os.path.join(TL_DIR, "topologies")
# TM_DIR = os.path.join(TL_DIR, "traffic-matrices")



# 适用于卫星数据的路径读取

def get_data_paths(data_name):


    topo = os.path.join(TL_DIR, 'data',data_name, "topologies")
    tm = os.path.join(TL_DIR, 'data',data_name, "traffic-matrices")
    return topo, tm


# 使用示例
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TL_DIR = os.path.join(CURRENT_DIR, "..")
TOPOLOGIES_DIR, TM_DIR = get_data_paths("iridium")
