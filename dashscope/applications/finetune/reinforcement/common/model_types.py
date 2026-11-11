# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class StatusType(str, Enum):
    PENDING = "PENDING"
    SUSPENDED = "SUSPENDED"
    SUCCEEDED = "SUCCEEDED"
    CANCELED = "CANCELED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    HEALTH = "healthy"

    def __str__(self):
        return self.value


class FunctionType(str, Enum):
    ROLLOUT = "rollout"
    REWARD = "reward"
    GROUP_REWARD = "group_reward"

    def __str__(self):
        return self.value


class DataSourceType(str, Enum):
    FILE_ID = "file_id"
    DOWNLOAD_URL = "download_url"
    OSS_MOUNT = "oss_mount"

    def __str__(self):
        return self.value


class DatasetsType(str, Enum):
    TRAINING = "training"
    TESTING = "testing"
    VALIDATION = "validation"

    def __str__(self):
        return self.value


class TrainingType(str, Enum):
    TRAINING_TYPE = "reinforcement"
    PG_OPD = "pg_opd"

    def __str__(self):
        return self.value


class RLType(str, Enum):
    GRPO = "GRPO"
    GRPOLoRA = "GRPOLoRA"
    PPO = "PPO"
    FinetuneLoRA = "FinetuneLoRA"
    FinetuneSFT = "FinetuneSft"
    DPO = "DPO"
    DPOLoRA = "DPOLoRA"
    Pretrain = "Pretrain"

    def __str__(self):
        return self.value


class FileSpec(BaseModel):
    path: str = ""
    descriptions: Optional[str] = ""


class Status(BaseModel):
    task: StatusType = StatusType.SUCCEEDED
    name: str = ""
    code: int = 200
    message: str = ""

    @property
    def success(self) -> bool:
        return self.task == StatusType.SUCCEEDED


class RequestFC(BaseModel):
    name: str = ""
    unique_key: str = ""
    func: str = ""
    code_url: str = ""


class ResponseFC(BaseModel):
    """FC result with optional upstream failure context.

    ``request_id`` and ``body`` retain the original FaaS rejection without
    replacing the SDK's existing numeric ``status.code``. The raw ``body``
    may contain credentials and is excluded from the model's repr.
    """

    status: Status = Status()
    output: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    body: Optional[Dict[str, Any]] = Field(default=None, repr=False)


class RequestTuning(BaseModel):
    model: str = ""  # Foundation model
    training_file_ids: List[str] = []
    validation_file_ids: Optional[List[str]] = []
    rolloutId: str = ""
    rewardIds: List[str] = []
    hyper_parameters: Optional[Dict[str, str]] = {}
    training_type: str = TrainingType.TRAINING_TYPE
    job_name: Optional[str] = ""
    model_name: Optional[str] = ""  # Post-tests model name


class ResponseTuning(BaseModel):
    status: Status = Status()
    output: Optional[Dict[str, Any]] = None
