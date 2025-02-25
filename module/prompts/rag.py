"""
RAG 提示词模板
"""

# 默认模板
DEFAULT_RAG_TEMPLATE = {
    "prompt_template": (
        "请根据以下背景信息回答问题。如果无法从背景信息中得到答案，请明确说明。\n\n"
        "背景信息：\n{context}\n\n"
        "问题：{query}\n\n"
        "回答："
    ),
    "context_format": "[{index}] {text}",
    "context_separator": "\n\n"
}

# 学术场景模板
ACADEMIC_RAG_TEMPLATE = {
    "prompt_template": (
        "作为一位学术专家，请基于以下参考资料回答问题。请使用严谨的学术语言，必要时引用具体段落。"
        "如果资料中没有相关信息，请明确指出。\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{query}\n\n"
        "专业解答："
    ),
    "context_format": "文献{index}（相关度：{score:.2f}）：\n{text}",
    "context_separator": "\n\n---\n\n"
}

# 商业场景模板
BUSINESS_RAG_TEMPLATE = {
    "prompt_template": (
        "作为一位商业顾问，请根据以下商业资料回答问题。请提供具体、可行的建议，并说明理由。"
        "如果资料中信息不足，请说明需要补充哪些信息。\n\n"
        "相关资料：\n{context}\n\n"
        "咨询问题：{query}\n\n"
        "商业建议："
    ),
    "context_format": "资料{index}（参考价值：{score:.2f}）：\n{text}",
    "context_separator": "\n\n---\n\n"
}

# 技术场景模板
TECHNICAL_RAG_TEMPLATE = {
    "prompt_template": (
        "作为一位技术专家，请根据以下技术文档回答问题。请提供准确、可执行的技术说明，必要时包含示例。"
        "如果文档中没有足够信息，请明确指出并提供可能的解决方向。\n\n"
        "技术文档：\n{context}\n\n"
        "技术问题：{query}\n\n"
        "技术解答："
    ),
    "context_format": "文档{index}（相关度：{score:.2f}）：\n{text}",
    "context_separator": "\n\n===\n\n"
}

# 对话场景模板
CONVERSATIONAL_RAG_TEMPLATE = {
    "prompt_template": (
        "你是一个友好、专业的助手。请根据对话历史和参考信息，以自然、连贯的方式回答用户的问题。"
        "如果参考信息中没有相关内容，可以基于常识回答，但请明确指出这一点。"
        "保持回答简洁、易懂，并与之前的对话保持一致性。\n\n"
        "参考信息：\n{context}\n\n"
        "{query}\n\n"
        "回答："
    ),
    "context_format": "信息{index}（相关度：{score:.2f}）：{text}",
    "context_separator": "\n\n"
}

# 总结场景模板
SUMMARY_RAG_TEMPLATE = {
    "prompt_template": (
        "请根据以下材料，对相关内容进行全面但简洁的总结。重点突出关键信息，确保准确性。"
        "如果材料中有矛盾或不确定的信息，请指出。\n\n"
        "原始材料：\n{context}\n\n"
        "总结要求：{query}\n\n"
        "内容总结："
    ),
    "context_format": "材料{index}（重要度：{score:.2f}）：\n{text}",
    "context_separator": "\n\n---\n\n"
} 