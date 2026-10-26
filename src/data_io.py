"""陈宇涵时间分析模块的兼容入口，复用全组统一的数据读取规则。"""

from .data_cleaning import read_clean_data

__all__ = ["read_clean_data"]
