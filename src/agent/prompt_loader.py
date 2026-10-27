"""
公共提示词加载器

功能:
  1. 从 src/agent/prompts/<agent_name>/<system|user>.md 动态加载提示词
  2. 支持 {var} 风格的变量注入（安全替换，缺变量不报错）
  3. 原始文件内容带内存缓存，避免重复磁盘 IO
  4. 提供 build_messages() 把 system + history + user 一次组装好

目录约定:
  src/agent/prompts/<agent_name>/
    ├── system.md    # 系统提示词
    └── user.md      # 用户提示词模板

用法示例:
  from src.agent.prompt_loader import PromptLoader

  system = PromptLoader.load_system("customer_service")
  user   = PromptLoader.load_user("customer_service", {"rag_context": "..."})

  messages = PromptLoader.build_messages(
      agent_name="customer_service",
      user_text="这个羽绒服有黑色吗",
      variables={"rag_context": "..."},
      history=[{"role": "user", "content": "..."}],
  )
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.logging import log

# 提示词根目录（相对本文件定位，不依赖 cwd）
_PROMPTS_ROOT = Path(__file__).parent / "prompts"

# {var} 占位符正则
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _inject_variables(template: str, variables: dict[str, Any] | None) -> str:
    """安全地替换 {var} 占位符。缺变量时保留原样，避免静默丢失。"""
    if not variables:
        return template

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        # 变量没给 → 保留占位符，方便排查
        log.debug(f"Prompt 变量占位符未替换: {{{key}}}")
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


@lru_cache(maxsize=128)
def _read_raw_prompt(agent_name: str, prompt_type: str, sub_dir: str | None = None) -> str:
    """
    读取原始 .md 文件内容（带缓存）。

    :param agent_name:  agent 目录名（如 "customer_service"）
    :param prompt_type: "system" | "user" | <capability_code>
    :param sub_dir:     子目录（如 "capabilities"），None 时直接放 agent_name/ 下

    文件路径规则:
      prompts/<agent_name>/<prompt_type>.md                 (sub_dir=None)
      prompts/<agent_name>/<sub_dir>/<prompt_type>.md       (sub_dir 有值)
    """
    if sub_dir:
        file_path = _PROMPTS_ROOT / agent_name / sub_dir / f"{prompt_type}.md"
    else:
        file_path = _PROMPTS_ROOT / agent_name / f"{prompt_type}.md"

    if not file_path.exists():
        log.warning(f"⚠️ 提示词文件不存在: {file_path}")
        return (
            f"[{agent_name}/{sub_dir + '/' if sub_dir else ''}{prompt_type}.md 缺失]"
        )

    content = file_path.read_text(encoding="utf-8").strip()
    log.debug(
        f"📖 加载提示词: {agent_name}/{sub_dir + '/' if sub_dir else ''}"
        f"{prompt_type}.md ({len(content)} chars)"
    )
    return content


class PromptLoader:
    """公共提示词加载器"""

    @staticmethod
    def load_system(
        agent_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """加载指定 Agent 的 system prompt（可选变量注入）"""
        raw = _read_raw_prompt(agent_name, "system")
        return _inject_variables(raw, variables)

    @staticmethod
    def load_user(
        agent_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """加载指定 Agent 的 user prompt 模板并注入变量"""
        raw = _read_raw_prompt(agent_name, "user")
        return _inject_variables(raw, variables)

    # ====== 能力描述加载（按意图类别动态注入） ======

    @staticmethod
    def load_capability(
        agent_name: str,
        category_code: str | None,
    ) -> str:
        """
        按二级分类编码加载对应的能力描述文件。
        找不到时 fallback 到 _default.md。
        """
        if not category_code:
            log.debug("🎯 category_code 为空，加载默认能力描述")
            return _read_raw_prompt(agent_name, "_default", sub_dir="capabilities")

        content = _read_raw_prompt(agent_name, category_code, sub_dir="capabilities")
        # 如果返回的是 "缺失" 占位字符串，fallback 到 _default
        if "缺失" in content:
            log.warning(
                f"⚠️ 能力描述文件不存在: capabilities/{category_code}.md，"
                f"fallback 到 _default.md"
            )
            return _read_raw_prompt(agent_name, "_default", sub_dir="capabilities")
        return content

    @staticmethod
    def load_for_intent(
        agent_name: str,
        intent_level_1: str | None = None,
        intent_level_2: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """
        完整加载 system prompt + 能力描述，返回拼好的可直接给 LLM 的内容。

        :param agent_name:      Agent 名
        :param intent_level_1:  一级问题类别名（如 "商品咨询"）
        :param intent_level_2:  二级问题类别编码（如 "GOODS_SPEC"）
        :param variables:       其他 system prompt 占位符变量

        :return: 已注入 category_context 的完整 system prompt
        """
        # 1. 加载能力描述（按二级 code，fallback _default）
        cap_desc = PromptLoader.load_capability(agent_name, intent_level_2)

        # 2. 组装 category_context：一级类别名 + 能力描述
        if intent_level_1:
            category_context = f"【当前意图场景】{intent_level_1}\n\n{cap_desc}"
        else:
            category_context = cap_desc

        # 3. 注入 system prompt
        all_vars = dict(variables or {})
        all_vars["category_context"] = category_context

        return PromptLoader.load_system(agent_name, all_vars)

    @staticmethod
    def build_messages(
        agent_name: str,
        user_text: str,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        system_variables: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """
        一步组装出 LLM 可用的 messages 列表。

        :param agent_name:  提示词所属 agent 名（对应 prompts/<name>/ 目录）
        :param user_text:   用户原始输入（会自动放进 user prompt 的 {user_text} 占位符）
        :param variables:   注入到 user prompt 的变量字典
        :param history:     历史对话（openai 格式: [{"role":"user","content":"..."}]）
        :param system_variables: 注入到 system prompt 的变量字典（和 variables 分开传，语义更清晰）

        :return: messages 列表
        """
        # user_text 自动塞进 variables（调用方也可以自己显式传）
        merged_vars = dict(variables or {})
        if "user_text" not in merged_vars:
            merged_vars["user_text"] = user_text

        system = PromptLoader.load_system(agent_name, system_variables)
        user_prompt = PromptLoader.load_user(agent_name, merged_vars)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        return messages


# ====== 便捷常量：Agent 名，避免硬编码 ======
class AgentNames:
    CUSTOMER_SERVICE = "customer_service"
    INTENT = "intent"
    # 后续加新 Agent 时在这里登记即可
