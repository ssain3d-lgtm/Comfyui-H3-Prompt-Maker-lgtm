#!/usr/bin/env python3
"""Model discovery must never offer an mmproj as a chat model."""
import importlib.util
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location(
    "h3filter", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp)
sys.modules["h3filter"] = m
sp.loader.exec_module(m)
L = __import__("h3filter.llm_backends", fromlist=["*"])

passed, failures = 0, []


def eq(name, actual, expected):
    global passed
    if actual == expected:
        passed += 1
    else:
        failures.append(f"{name}: expected {expected!r}, got {actual!r}")


openai_ids = ["chat@q4_k_m", "projector@bf16", "embed", "real-chat@bf16"]
metadata = {"data": [
    {"id": "chat@q4_k_m", "type": "llm", "arch": "qwen35"},
    {"id": "projector@bf16", "type": "llm", "arch": "clip"},
    {"id": "embed", "type": "embeddings", "arch": "nomic-bert"},
    {"id": "real-chat@bf16", "type": "llm", "arch": "qwen35"},
]}

eq("clip projector excluded",
   L._filter_lmstudio_chat_models(openai_ids, metadata),
   ["chat@q4_k_m", "real-chat@bf16"])
eq("BF16 is not itself a reason to exclude a model",
   L._filter_lmstudio_chat_models(["real-chat@bf16"], metadata),
   ["real-chat@bf16"])
eq("VLM remains selectable",
   L._filter_lmstudio_chat_models(
       ["vision-model"],
       {"data": [{"id": "vision-model", "type": "vlm", "arch": "qwen2_vl"}]}),
   ["vision-model"])
eq("native v1 variant ids are matched",
   L._filter_lmstudio_chat_models(
       ["chat@q4_k_m", "projector@bf16"],
       {"models": [
           {"key": "chat", "type": "llm", "architecture": "qwen35",
            "variants": ["chat@q4_k_m"]},
           {"key": "projector", "type": "llm", "architecture": "clip",
            "selected_variant": "projector@bf16"},
       ]}),
   ["chat@q4_k_m"])
eq("unknown metadata shape preserves compatibility",
   L._filter_lmstudio_chat_models(openai_ids, {"models": []}), openai_ids)
eq("metadata URLs preserve a reverse-proxy prefix",
   L._lmstudio_metadata_urls("http://host:1234/local/v1"),
   ["http://host:1234/local/api/v1/models", "http://host:1234/local/api/v0/models"])

if failures:
    print("MODEL FILTER TESTS FAILED")
    for failure in failures:
        print(" -", failure)
    raise SystemExit(1)
print(f"{passed} model filter tests passed")
