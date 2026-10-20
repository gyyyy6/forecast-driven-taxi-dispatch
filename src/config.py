"""YAML 配置加载与路径解析。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """读取 YAML 配置；相对路径基于项目根目录。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {p}")
    return cfg


def resolve(path: str | Path) -> Path:
    """把配置中的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: str | Path) -> Path:
    """创建文件所在目录并返回解析后的路径。"""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_logger(name: str = "project"):
    import logging
    import sys

    # Windows PowerShell 默认用 GBK 解码子进程输出，这里统一为 UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)