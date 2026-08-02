"""Client for Veo 3 video generation via an OpenAI-compatible chat gateway."""

from __future__ import annotations

from typing import Any, Dict, List

from ..api_video_client import OpenAiVideoSynthesis


class Veo3Synthesis(OpenAiVideoSynthesis):
    """Veo3 生成合成类，提供统一的接口用于音视频生成。

    负责 API 调用和模型推理相关的工作。
    """

    MODEL = "veo3.1"

    def __init__(self, endpoint: str, api_key: str, logger=None) -> None:
        """Veo is reached through a caller-chosen gateway, so ``endpoint`` is required."""
        super().__init__(endpoint=endpoint, api_key=api_key, logger=logger)

    def _invoke_chat_completion(self, *, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 Chat Completions API 并返回响应对象。"""
        return self.client.chat.completions.create(model=self.MODEL, messages=messages)

    def generate_t2av(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """文本到视频生成（T2V）。

        Args:
            processed_data: 处理后的数据（来自 operator），包含已构建好的
                ``user_content``。
            kwargs: 其他参数（保留以兼容接口）。
        """
        del kwargs
        return self._generate(processed_data)

    def generate_i2av(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """图像到视频生成（I2V）。

        与 T2V 走同一个 chat 接口：图像已由 operator 编入 ``user_content``。
        """
        del kwargs
        return self._generate(processed_data)

    def _generate(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """把 operator 准备好的 ``user_content`` 作为单条用户消息发出。"""
        return self._invoke_chat_completion(
            messages=[{"role": "user", "content": processed_data.get("user_content", [])}]
        )

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """按输入自动选择 T2V 或 I2V 并提交生成任务。

        Args:
            processed_data: 处理后的数据（来自 operator）。
            task_type: "auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频。

        Returns:
            含 ``task_type`` 与 ``result`` 的字典。

        Raises:
            ValueError: 任务类型不支持，或 i2av 缺少 ``images``。
        """
        images = processed_data.get("images", None)

        if task_type == "auto":
            task_type = "i2av" if images is not None else "t2av"

        if task_type == "i2av":
            if images is None:
                raise ValueError("i2av 任务需要提供 images 参数")
            result = self.generate_i2av(processed_data=processed_data, **kwargs)
        elif task_type == "t2av":
            result = self.generate_t2av(processed_data=processed_data, **kwargs)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        return {
            "task_type": task_type,
            "result": result,
        }


__all__ = ["Veo3Synthesis"]
