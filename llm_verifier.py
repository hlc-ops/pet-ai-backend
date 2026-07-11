"""LLM 视觉复核层(第 3 层级联)

功能:把关键帧发给多模态大模型,让它二次确认"这只猫真的在吃饭吗"

支持:
- 通义千问 Qwen-VL(DashScope,OpenAI 兼容)
- 智谱 GLM-4V

调用节流:同一事件最多 3 次(触发/中期/结束),同一时间最快 3 秒一次。

用法:
    from llm_verifier import LLMVerifier
    verifier = LLMVerifier()  # 从环境变量读配置
    if verifier.available:
        result = verifier.verify_behavior(frame_bgr, "drinking",
                                            animal="cat")
        print(result)  # {"confirmed": True, "reason": "...", "confidence": 0.95}
"""
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import requests

logger = logging.getLogger(__name__)


# ==================== 配置 ====================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "qwen")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or \
    os.environ.get("DASHSCOPE_API_KEY") or ""
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))

# 自动读代理配置(v9+ 新加,避免用户每次手动 set env)
LLM_PROXY = os.environ.get("LLM_PROXY", "")
if LLM_PROXY:
    os.environ.setdefault("HTTP_PROXY", LLM_PROXY)
    os.environ.setdefault("HTTPS_PROXY", LLM_PROXY)

LLM_DEFAULTS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-plus",
    },
}


# ==================== 节流 ====================
LLM_MIN_INTERVAL_SEC = 3.0       # 全局最快 3 秒一次
LLM_MAX_PER_EVENT = 3            # 每个事件最多 3 次


@dataclass
class VerifyResult:
    confirmed: bool
    reason: str
    confidence: float
    latency_ms: int
    provider: str


class LLMVerifier:
    """LLM 视觉复核客户端(单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inited = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_inited", False):
            return
        self._inited = True

        provider = LLM_PROVIDER.lower()
        defaults = LLM_DEFAULTS.get(provider, LLM_DEFAULTS["qwen"])
        self.provider = provider
        self.base_url = LLM_BASE_URL or defaults["base_url"]
        self.model = LLM_MODEL or defaults["model"]
        self.api_key = LLM_API_KEY

        self.available = bool(self.api_key)
        if self.available:
            logger.info(
                f"✅ LLM 复核层已启用: provider={provider} "
                f"model={self.model}")
        else:
            logger.warning(
                "⚠️ LLM 复核未启用: 缺少 LLM_API_KEY / DASHSCOPE_API_KEY")

        self._last_call_time = 0.0
        self._event_call_count = {}   # event_id -> count

    def can_call(self, event_id: str) -> bool:
        """检查是否允许调用(节流)"""
        if not self.available:
            return False
        # 全局节流
        if time.time() - self._last_call_time < LLM_MIN_INTERVAL_SEC:
            return False
        # 单事件次数限制
        if self._event_call_count.get(event_id, 0) >= LLM_MAX_PER_EVENT:
            return False
        return True

    def verify_behavior(
            self,
            frame_bgr,
            event_type: str,
            animal_class: str = "cat",
            event_id: str = "default",
            extra_context: str = "") -> Optional[VerifyResult]:
        """
        输入:
            frame_bgr: OpenCV BGR ndarray
            event_type: 'drinking' / 'excretion' 等
            animal_class: 'cat' / 'dog' / ...
            event_id: 事件唯一 ID(用于节流)
        输出:
            VerifyResult 或 None(节流拒绝 / API 失败)
        """
        if not self.can_call(event_id):
            return None

        self._last_call_time = time.time()
        self._event_call_count[event_id] = \
            self._event_call_count.get(event_id, 0) + 1

        # 图像转 base64:
        # 1) 先降到 max 720p 边长,避免大分辨率生成 MB 级 payload
        # 2) JPEG 质量 55(替换原 85):测下来本机对 >800KB 的 SSL POST 会 EOF
        h, w = frame_bgr.shape[:2]
        max_side = 720
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
        _, buf = cv2.imencode(".jpg", frame_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 55])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # 构建 prompt
        cn_names = {
            "cat": "猫", "dog": "狗", "monkey": "猴子",
            "other_primate": "灵长类",
        }
        cn_event = {
            "drinking": "喝水或吃食物",
            "excretion": "排泄(大便或小便)",
        }
        animal_cn = cn_names.get(animal_class, animal_class)
        event_cn = cn_event.get(event_type, event_type)

        prompt = f"""你是宠物行为分析助手。请判断图片里的{animal_cn}是否正在{event_cn}。

要求:
1. 严格按 JSON 格式回答,不要额外说明
2. confirmed 字段:true = 确实在做该行为, false = 不是
3. reason 字段:简短理由(20 字内)
4. confidence 字段:0-1 之间的置信度

额外上下文:{extra_context}

只输出 JSON:
{{"confirmed": true/false, "reason": "...", "confidence": 0.9}}"""

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        t0 = time.time()
        # 强制直连:阿里云 Qwen 是国内域名,不走本机代理
        # trust_env=False 忽略 HTTP_PROXY/HTTPS_PROXY 环境变量
        # verify=False 兜底本机 Clash/V2Ray 开启系统代理 MITM 时的 SSL 报错
        session = requests.Session()
        session.trust_env = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # 本机 Clash TUN 模式会 SSLEOFError,重试 3 次退避
        last_err = None
        resp = None
        for attempt in range(3):
            try:
                resp = session.post(
                    url, json=payload, headers=headers,
                    timeout=LLM_TIMEOUT, verify=False)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
        if resp is None:
            logger.warning(f"❌ LLM 网络重试 3 次全失败: {last_err}")
            return None
        try:
            latency_ms = int((time.time() - t0) * 1000)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # 尝试 JSON 解析(有时 LLM 会带 markdown code block)
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            parsed = json.loads(content)

            result = VerifyResult(
                confirmed=bool(parsed.get("confirmed", False)),
                reason=str(parsed.get("reason", ""))[:80],
                confidence=float(parsed.get("confidence", 0.5)),
                latency_ms=latency_ms,
                provider=self.provider,
            )
            logger.info(
                f"🤖 LLM 复核 [{event_id}]: "
                f"{result.confirmed} conf={result.confidence:.2f} "
                f"({result.reason}) {latency_ms}ms")
            return result
        except Exception as e:
            logger.warning(f"❌ LLM 复核失败: {e}")
            return None


_instance = None


def get_verifier() -> LLMVerifier:
    global _instance
    if _instance is None:
        _instance = LLMVerifier()
    return _instance
