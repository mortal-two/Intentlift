"""Lit configuration for the Intent dialect regression tests."""

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "Intent"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.intent_obj_root, "test")

llvm_config.use_default_substitutions()

tool_dirs = [config.intent_tools_dir, config.llvm_tools_dir]
tools = ["intent-opt", "mlir-opt", "FileCheck", "not"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
