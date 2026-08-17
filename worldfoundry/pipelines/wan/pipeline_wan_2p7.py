"""Wan 2P7 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
import time
from typing import Optional, Dict, Any

from ...operators.wan_2p7_operator import Wan2p7Operator
from ...synthesis.visual_generation.wan.wan_2p7_synthesis import Wan2p7Synthesis
from ..api_runtime import resolve_api_key


_API_KEY_ENV = ("DASHSCOPE_API_KEY", "WAN_API_KEY", "ALIYUN_API_KEY")

logger = logging.getLogger(__name__)


class Wan2p7Pipeline(PipelineABC):
    """
    Wan2.7 API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[Wan2p7Operator] = None,
        synthesis_model: Optional[Wan2p7Synthesis] = None,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.7")
        self.endpoint = endpoint
        self.api_key = api_key
        self.operator = operator
        self.synthesis_model = synthesis_model

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> "Wan2p7Pipeline":
        """Initialize API client credentials and runtime endpoints."""
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.7")
        synthesis_model = Wan2p7Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = Wan2p7Operator()
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key,
        )

    def process(
        self,
        prompt: str,
        images: Optional[str] = None,
        last_frame: Optional[str] = None,
        audio_url: Optional[str] = None,
        first_clip: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: Dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        processed_perception = self.operator.process_perception(
            images=images,
            last_frame=last_frame,
            audio_url=audio_url,
            first_clip=first_clip,
            **kwargs,
        )
        processed_data["media"] = processed_perception["media"]
        processed_data["images"] = processed_perception["images"]
        processed_data["last_frame"] = processed_perception["last_frame"]
        processed_data["audio_url"] = processed_perception["audio_url"]
        processed_data["first_clip"] = processed_perception["first_clip"]

        return processed_data

    def _extract_task_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the task identifier from a service response."""
        return response.get("output", {}).get("task_id")

    def _extract_task_status(self, response: Dict[str, Any]) -> str:
        """Query and extract the task status from a service response."""
        return response.get("output", {}).get("task_status", "")

    def _extract_video_url(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract video url for Wan2p7Pipeline."""
        return response.get("output", {}).get("video_url")

    def _poll_task_status(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_retries: int = 120,
    ) -> Dict[str, Any]:
        """Poll task status for Wan2p7Pipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        completed_statuses = {"SUCCEEDED"}
        failed_statuses = {"FAILED", "CANCELED"}
        last_status = None

        for _ in range(max_retries):
            task_info = self.synthesis_model.get_task(task_id)
            task_status = self._extract_task_status(task_info).upper()

            if task_status and task_status != last_status:
                logger.info("Wan2.7 task %s: %s", task_id, task_status)
                last_status = task_status

            if task_status in completed_statuses:
                return task_info
            if task_status in failed_statuses:
                return task_info

            time.sleep(poll_interval)

        raise TimeoutError(f"Wan2.7 task {task_id} polling timed out.")

    def __call__(
        self,
        prompt: str,
        images: Optional[str] = None,
        last_frame: Optional[str] = None,
        audio_url: Optional[str] = None,
        first_clip: Optional[str] = None,
        task_type: str = "i2av",
        model: Optional[str] = None,
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        wait: bool = True,
        poll_interval: int = 10,
        max_retries: int = 120,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data = self.process(
            prompt=prompt,
            images=images,
            last_frame=last_frame,
            audio_url=audio_url,
            first_clip=first_clip,
            **kwargs,
        )

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            model=model,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )

        response = result["response"]
        task_id = self._extract_task_id(response)
        result["task_id"] = task_id

        if wait and task_id:
            response = self._poll_task_status(
                task_id=task_id,
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
            result["response"] = response

        result["task_status"] = self._extract_task_status(result["response"])
        result["video_url"] = self._extract_video_url(result["response"])

        if output_path and result["video_url"]:
            saved_path = self.synthesis_model.download_video(
                result["video_url"],
                output_path,
            )
            result["output_path"] = saved_path

        return result

    def get_operator(self) -> Optional[Wan2p7Operator]:
        """Get operator for Wan2p7Pipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[Wan2p7Synthesis]:
        """Get synthesis model for Wan2p7Pipeline."""
        return self.synthesis_model
