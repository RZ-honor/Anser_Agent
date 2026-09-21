"""
问答引擎 - 调用 Qwen API 进行推理作答
含领域 Prompt、Token 预算控制、重试机制、答案后处理
"""
import json as _json
import re
import time
from typing import Optional

from config import (
    DASHSCOPE_API_KEY, QWEN_MODEL, QWEN_BASE_URL,
    TOKEN_PER_QUESTION, EVIDENCE_TOKEN_RATIO, MAX_RETRIES,
)


# ============ 领域 Prompt 模板 ============

DOMAIN_SYSTEM_PROMPTS = {
    "financial_contracts": """你是金融合同分析专家。请严格根据提供的文档证据回答问题。
重点关注：条款编号、违约条件、利率计算、担保条款、偿付安排。
规则：
1. 只根据提供的证据段落作答，不要使用外部知识
2. 仔细阅读每个选项，逐项分析
3. 数值计算请提取数字后用代码验证
4. 多选题需要选完所有正确选项
5. 输出格式：只输出答案字母，如 A 或 ABC""",

    "financial_reports": """你是财务报表分析专家。请严格根据提供的文档证据回答问题。
重点关注：营业收入、净利润、总资产、现金流、每股收益、同比变化、会计政策。
规则：
1. 只根据提供的证据段落作答，不要使用外部知识
2. 仔细阅读每个选项，逐项分析
3. 数值比较时注意单位（元/万元/亿元）和时间范围
4. 多选题需要选完所有正确选项
5. 输出格式：只输出答案字母，如 A 或 ABC""",

    "insurance": """你是保险条款解读专家。请严格根据提供的文档证据回答问题。
重点关注：保障范围、免赔额、赔付条件、等待期、退保规则、现金价值。
规则：
1. 只根据提供的证据段落作答，不要使用外部知识
2. 仔细阅读每个选项，逐项分析
3. 注意区分"保险责任"和"责任免除"
4. 多选题需要选完所有正确选项
5. 输出格式：只输出答案字母，如 A 或 ABC""",

    "regulatory": """你是金融监管法规专家。请严格根据提供的文档证据回答问题。
重点关注：法条引用、适用范围、处罚条款、审批流程、合规要求。
规则：
1. 只根据提供的证据段落作答，不要使用外部知识
2. 仔细阅读每个选项，逐项分析
3. 注意法规的生效时间和适用主体
4. 多选题需要选完所有正确选项
5. 输出格式：只输出答案字母，如 A 或 ABC""",

    "research": """你是行业研报分析专家。请严格根据提供的文档证据回答问题。
重点关注：核心观点、数据来源、投资建议、行业趋势、公司估值。
规则：
1. 只根据提供的证据段落作答，不要使用外部知识
2. 仔细阅读每个选项，逐项分析
3. 注意区分"事实"和"观点/预测"
4. 多选题需要选完所有正确选项
5. 输出格式：只输出答案字母，如 A 或 ABC""",
}

DEFAULT_SYSTEM_PROMPT = """你是一个金融文档问答专家。你的任务是根据提供的文档证据回答问题。

核心规则（必须严格遵守）：
1. 只能使用文档证据中的信息回答，绝对不要使用外部知识
2. 对每个选项，必须在证据中找到明确的支持或反驳依据
3. 如果证据中没有提到某个选项的内容，该选项应判为错误
4. 多选题必须选完所有正确选项（通常2-4个），不要漏选
5. 判断题严格对照证据，证据支持选A（正确），证据不支持或无证据选B（错误）
6. 数值题必须从证据中提取具体数值，不要猜测
7. 输出格式：只输出答案字母，如 A 或 ABC"""

ANSWER_PROMPT_TEMPLATE = """## 文档证据（你的回答必须基于以下证据）

{evidence}

## 问题

{question}

## 选项

{options_text}

## 题目类型：{answer_format}

{one_shot}

请严格根据文档证据回答问题。

**重要提醒**：
1. 仔细阅读证据中的每一句话，找出与问题相关的信息
2. 对每个选项，明确指出证据中支持或反对的依据
3. 如果证据中包含数值，请精确匹配选项中的数值
4. 多选题必须选完所有正确选项，判断题只能选A或B
5. 只输出答案字母，不要输出其他内容"""

# One-shot 示例
ONE_SHOT_MULTI = """## 示例（多选题 - 必须逐项引用证据）

证据：
- 文档A：发行人主体信用评级为AAA，发行金额不超过10亿元，债项评级为AAA/-
- 文档B：发行人主体信用评级为AA+，发行金额不超过5亿元，债项评级为AA+/-

问题：关于两份文档的发行要素，以下哪些描述正确？
选项：
A. 两份文档的发行人主体信用评级均达到AAA级别
B. 第二份文档的本期发行金额上限低于第一份文档
C. 两份文档均明确标注了债项信用评级
D. 第二份文档的发行人主体信用评级为AA+

逐项分析（必须引用证据）：
- A：证据显示文档A是AAA，文档B是AA+，不都是AAA → 错误
- B：证据显示文档A是10亿，文档B是5亿，5亿<10亿 → 正确
- C：证据显示文档A标注了AAA/-，文档B标注了AA+/-，都标注了 → 正确
- D：证据显示文档B是AA+ → 正确

答案：BCD

关键：每个选项都必须在证据中找到依据，不能猜测。
"""

ONE_SHOT_TF = """## 示例（判断题）

证据：
- 文档：发行人承诺将及时、公平地履行信息披露义务。

问题：文档中包含关于发行人将及时、公平地履行信息披露义务的明确承诺条款。

逐项分析：
- 证据中明确提到"发行人承诺将及时、公平地履行信息披露义务"
- 问题陈述与证据一致

答案：A

注意：判断题只有A（正确）和B（错误）两个选项，必须严格对照证据判断。
"""

ONE_SHOT_NUMERIC = """## 示例（数值题 - 必须从证据中提取具体数值）

证据：
- 文档：二零二四年我们的研发投入约为542亿元，同比上升35.68%，累计研发投入超1,800亿元。

问题：比亚迪2024年研发投入约为多少亿元？
选项：
A. 约442亿元
B. 约542亿元
C. 约642亿元
D. 约742亿元

分析：
- 证据明确提到"研发投入约为542亿元"
- 选项B是"约542亿元"，与证据一致

答案：B

关键：数值题必须从证据中找到具体数值，不能猜测。
"""

AUDIT_PROMPT_TEMPLATE = """## 文档证据

{evidence}

## 问题

{question}

## 选项

{options_text}

## 题目类型：{answer_format}

请输出 JSON 格式：
{{
  "answer": "最终答案字母",
  "option_judgements": {{
    "A": {{"verdict": true或false, "quote": "支持该判断的证据片段", "reason": "推理说明"}},
    "B": {{"verdict": true或false, "quote": "...", "reason": "..."}}
  }}
}}

只输出 JSON，不要其他文字。"""


class QAEngine:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.model = model or QWEN_MODEL
        self.client = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        if not self.api_key:
            print("[警告] 未设置 API Key，问答引擎不可用")
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=QWEN_BASE_URL,
            )
        except Exception as e:
            print(f"[错误] 初始化客户端失败: {e}")

    def answer_question(
        self,
        question: str,
        options: dict,
        answer_format: str,
        evidence_chunks: list[dict],
        doc_ids: Optional[list[str]] = None,
        domain: str = "",
        audit_mode: bool = False,
    ) -> dict:
        """
        回答单个问题（含重试、token预算、领域prompt）

        Args:
            question: 问题文本
            options: 选项字典 {"A": "...", "B": "...", ...}
            answer_format: 题型 (mcq/multi/tf)
            evidence_chunks: 证据 chunks [{"text": str, ...}]
            doc_ids: 参考文档 ID（可选）
            domain: 文档领域（用于选择 prompt）
            audit_mode: 是否启用审计模式（输出详细推理和选项判断）

        Returns:
            {"answer", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning", "option_judgements"(可选)}
        """
        if not self.client:
            return self._fallback_answer(question, options, answer_format)

        # 1. 构建证据文本（受 token 预算控制）
        evidence_text = self._build_evidence_bounded(evidence_chunks)

        # 2. 构建选项文本
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

        # 3. 选择领域 prompt
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)

        # 4. 构建 prompt
        format_map = {"mcq": "单选题（选一个）", "multi": "多选题（选所有正确的）", "tf": "判断题"}
        if audit_mode:
            prompt = AUDIT_PROMPT_TEMPLATE.format(
                evidence=evidence_text,
                question=question,
                options_text=options_text,
                answer_format=format_map.get(answer_format, answer_format),
            )
        else:
            # 根据题型选择 one-shot 示例
            if answer_format == "multi":
                one_shot = ONE_SHOT_MULTI
            elif answer_format == "tf":
                one_shot = ONE_SHOT_TF
            elif answer_format == "mcq":
                # 检查是否数值题
                import re as _re
                if _re.search(r'(?:多少|几|比例|率|金额|数量|增长|下降|同比|超过|约为)', question):
                    one_shot = ONE_SHOT_NUMERIC
                else:
                    one_shot = ""
            else:
                one_shot = ""

            prompt = ANSWER_PROMPT_TEMPLATE.format(
                evidence=evidence_text,
                question=question,
                options_text=options_text,
                answer_format=format_map.get(answer_format, answer_format),
                one_shot=one_shot,
            )

        # 5. 调用 API（含重试）
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4000,  # Pro模型思考链路更长，需要足够空间
                )

                content = response.choices[0].message.content or ""
                # 兼容多端思考模型：DeepSeek 用 reasoning_content，AMD Qwen3.6 用 reasoning
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '') or \
                                    getattr(response.choices[0].message, 'reasoning', '') or ""
                # full_content 合并 content + reasoning_content，保存完整推理过程
                # 答案提取优先从 content（最终答案），为空时才用 reasoning_content
                full_content = (content + "\n" + reasoning_content).strip() if reasoning_content else content
                usage = response.usage

                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

                # 6. 后处理答案（保存完整的思考链路和推断链路）
                if audit_mode:
                    # 尝试解析 JSON
                    parsed = self._parse_audit_json(full_content, answer_format)
                    return {
                        "answer": parsed["answer"],
                        "option_judgements": parsed.get("option_judgements", {}),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "reasoning": full_content,
                        "thinking": reasoning_content,  # 思考链路（reasoning_content）
                        "final_output": content,        # 推断链路（content）
                    }
                else:
                    # 答案从 content 提取（最终答案），为空时才用 full_content
                    answer = self._extract_answer(content, answer_format) if content else self._extract_answer(full_content, answer_format)
                    # 多选题验证：从 content 补全，但不强制至少2个（尊重模型基于证据的选择）
                    if answer_format == "multi" and len(answer) < 2:
                        answer = self._validate_multi_answer(answer, content)
                    return {
                        "answer": answer,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "reasoning": full_content,
                        "thinking": reasoning_content,  # 思考链路（reasoning_content）
                        "final_output": content,        # 推断链路（content）
                    }

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"  [重试 {attempt+1}/{MAX_RETRIES}] 等待 {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  [错误] API 调用失败（已重试 {MAX_RETRIES} 次）: {e}")
                    return self._fallback_answer(question, options, answer_format)

    def _build_evidence_bounded(self, chunks: list[dict]) -> str:
        """构建证据文本（受 token 预算控制，适配多路召回）

        优化：条款类chunk不截断（避免关键条款内容丢失），仅截断超长非条款文本
        """
        max_evidence_chars = int(TOKEN_PER_QUESTION * EVIDENCE_TOKEN_RATIO * 3)  # 粗略：1 token ≈ 3 chars
        parts = []
        total_chars = 0

        # 适配多路召回：最多使用15个chunk，但受token预算限制
        for i, chunk in enumerate(chunks[:15], 1):
            doc_id = chunk.get("doc_id", "unknown")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)

            # 识别条款类chunk（包含"第X条"等法律条款标记）
            is_article = bool(re.search(r'第[一二三四五六七八九十百千零0-9]+[条章节款项号]', text))

            # 条款类chunk不截断（保留完整条款），非条款类截断到1200字
            if not is_article and len(text) > 1200:
                text = text[:1200] + "..."

            candidate = f"### 证据{i}（来源: {doc_id}, 相关度: {score:.2f}）\n{text}"
            if total_chars + len(candidate) > max_evidence_chars:
                break
            parts.append(candidate)
            total_chars += len(candidate)

        return "\n\n".join(parts)

    def _extract_answer(self, content: str, answer_format: str) -> str:
        """从模型输出中提取答案字母（改进版）"""
        content = content.strip()

        # 1. 先找 "答案是X" 或 "答案：X" 模式
        answer_pattern = re.search(r'答案[是为：:]\s*([A-D]+)', content, re.IGNORECASE)
        if answer_pattern:
            letters = answer_pattern.group(1).upper()
            if answer_format == "multi":
                return "".join(sorted(set(letters)))
            return letters[0]

        # 2. 找 "选X" 模式
        select_pattern = re.search(r'选\s*([A-D]+)', content, re.IGNORECASE)
        if select_pattern:
            letters = select_pattern.group(1).upper()
            if answer_format == "multi":
                return "".join(sorted(set(letters)))
            return letters[0]

        # 3. 找末尾独立的字母（最可能是最终答案）
        lines = content.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            match = re.search(r'([A-D]+)\s*$', line, re.IGNORECASE)
            if match:
                letters = match.group(1).upper()
                if answer_format == "multi":
                    return "".join(sorted(set(letters)))
                return letters[0]

        # 4. 判断题特殊处理
        if answer_format == "tf":
            # 检查明确的判断词
            if re.search(r'(?:是|正确|对|符合|一致|支持)', content):
                return "A"
            if re.search(r'(?:否|错误|错|不符合|不一致|反驳|不支持)', content):
                return "B"
            # 尝试找AB
            match = re.search(r'\b([AB])\b', content, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 5. 多选题特殊处理
        elif answer_format == "multi":
            letters = re.findall(r'\b([A-D])\b', content, re.IGNORECASE)
            if letters:
                return "".join(sorted(set(l.upper() for l in letters)))

        # 6. 单选题
        else:
            match = re.search(r'\b([A-D])\b', content, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 7. 最后尝试
        letters = re.findall(r'[A-D]', content, re.IGNORECASE)
        if letters:
            if answer_format == "multi":
                return "".join(sorted(set(l.upper() for l in letters)))
            return letters[0].upper()

        # 默认
        if answer_format == "multi":
            return "AB"  # 多选题默认至少2个
        return "A"

    def _validate_multi_answer(self, answer: str, content: str) -> str:
        """验证多选题答案，尊重模型基于证据的选择

        修改：不强制至少2个选项，因为证据不足时模型可能只选1个正确选项
        若 content 中有多个字母，从 content 补全；否则保留模型的选择
        """
        if not answer:
            # 答案为空，从 content 重新提取
            all_letters = set(re.findall(r'\b([A-D])\b', content, re.IGNORECASE))
            if all_letters:
                return "".join(sorted(l.upper() for l in all_letters))
            return "A"  # 兜底
        return answer

    def _parse_audit_json(self, content: str, answer_format: str) -> dict:
        """
        解析审计模式的 JSON 输出，失败时回退到字母提取

        Returns:
            {"answer": str, "option_judgements": dict}
        """
        # 尝试从内容中提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = _json.loads(json_match.group())
                answer = data.get("answer", "")
                # 规范化答案
                if answer:
                    letters = re.findall(r'[A-D]', answer, re.IGNORECASE)
                    if letters:
                        if answer_format == "multi":
                            answer = "".join(sorted(set(l.upper() for l in letters)))
                        else:
                            answer = letters[0].upper()
                    else:
                        answer = self._extract_answer(content, answer_format)
                else:
                    answer = self._extract_answer(content, answer_format)

                option_judgements = data.get("option_judgements", {})
                return {"answer": answer, "option_judgements": option_judgements}
            except (_json.JSONDecodeError, KeyError, TypeError):
                pass

        # JSON 解析失败，回退到字母提取
        return {
            "answer": self._extract_answer(content, answer_format),
            "option_judgements": {},
        }

    def _fallback_answer(
        self, question: str, options: dict, answer_format: str
    ) -> dict:
        """无 API 时的回退方案"""
        return {
            "answer": "A",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning": "fallback",
        }

    def get_total_stats(self) -> dict:
        """获取总计 Token 统计"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


if __name__ == "__main__":
    engine = QAEngine()
    result = engine.answer_question(
        question="比亚迪2025年营业收入是多少？",
        options={"A": "3000亿", "B": "3847亿", "C": "4000亿", "D": "4500亿"},
        answer_format="mcq",
        evidence_chunks=[{"text": "2025年营业收入3,847亿元", "doc_id": "byd_2025", "score": 0.95}],
        domain="financial_reports",
    )
    print(f"答案: {result['answer']}")
    print(f"Token: {result['total_tokens']}")
