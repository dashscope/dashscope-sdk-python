# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional, List, Any, Dict
from pydantic import BaseModel


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
    status: Status = Status()
    output: Optional[Dict[str, Any]] = None


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
