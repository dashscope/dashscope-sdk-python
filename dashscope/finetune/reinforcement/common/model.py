# -*- coding: utf-8 -*-
from __future__ import annotations

# Standard Library
import os
import re
import shutil
import tempfile
import asyncio

# Third-party Libraries
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from typing_extensions import Self, Annotated
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    ConfigDict,
    Tag,
    field_validator,
)
import yaml

# Local Application
from dashscope.finetune.reinforcement.common.constants import (
    FC_API_KEY,
    FC_LOAD_API,
    FC_QUERY_API,
    FC_REGISTER_REWARD_API,
    FC_REGISTER_ROLLOUT_API,
    FC_REGISTER_GROUP_REWARD_API,
    FC_UPLOAD_OSS_API,
    FC_LAYER_NAME,
    FC_REQUIREMENTS_FILE,
    FC_LAYER_USED,
    FC_LAYER_CREATE_API,
    FC_LAYER_QUERY_API,
    LOG_LEVEL,
)
from dashscope.finetune.reinforcement.common.model_types import (
    FileSpec,
    FunctionType,
    DatasetsType,
    DataSourceType,
    RequestFC,
    ResponseFC,
    Status,
    StatusType,
    TrainingType,
)
from dashscope.finetune.reinforcement.component.data import (
    RewardInput,
    RewardOutput,
    RolloutInput,
    RolloutOutput,
    GroupRewardInput,
    GroupRewardOutput,
)
from dashscope.finetune.reinforcement.common.utils import (
    check_file,
    client_fc,
    create_deployment_files,
    deep_mask,
    generate_random_id,
    upload_zip_to_oss_and_by_signed_url,
    zip_dir,
    to_bailian_data,
    get_filepath_classname,
    get_func_type_id,
    get_weights_from_file,
    deep_remove_none,
)
from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.common.errors import (
    InputError,
    OutputError,
    OSSConnectionError,
    OSSUploadError,
    RegistrationError,
    FunctionLoadError,
    InstanceWarmupError,
    InstanceQueryError,
    FunctionLayerError,
    ValidationError,
    IOErrorWithCode,
    ValueErrorWithCode,
)
from dashscope.common.error_registry import (
    INVALID_REQUEST,
    INTERNAL_ERROR,
    SERVICE_UNAVAILABLE,
)


class MountStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: Optional[str] = None
    bucket: Optional[str] = None
    file_path: Optional[str] = None


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: DatasetsType = DatasetsType.TRAINING

    data_source_type: Optional[DataSourceType] = DataSourceType.FILE_ID
    file_name: Optional[str] = None
    file_id: Optional[str] = None
    download_url: Optional[str] = None
    mount_storage: Optional[MountStorage] = None

    async def upload_dataset(self) -> Optional[str]:
        if (
            self.data_source_type == DataSourceType.FILE_ID
            and self.file_name is not None
        ):
            try:
                file_id = await to_bailian_data(
                    [FileSpec(path=self.file_name)],
                )
                if file_id and isinstance(file_id, List) and len(file_id) > 0:
                    self.file_id = file_id[0]

            except Exception as e:
                raise OSSUploadError(
                    "Failed to upload datasets",
                    error_code=2061,
                    public_error=SERVICE_UNAVAILABLE,
                ) from e

        return self.file_id


class Datasets(BaseModel):
    name: str = None
    datasets: List[Dataset] = None

    @classmethod
    async def upload_datasets(
        cls,
        training_files: Union[List[str], str] = None,
        validation_files: Union[List[str], str] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        training_files = (
            training_files
            if isinstance(training_files, List)
            else [training_files]
        )
        validation_files = (
            validation_files
            if isinstance(validation_files, List)
            else [validation_files]
        )

        training_filespecs = [FileSpec(path=f) for f in training_files]
        validation_filespecs = [FileSpec(path=f) for f in validation_files]

        uploaded_training_ids = None
        uploaded_validation_ids = None
        try:
            if training_files:
                uploaded_training_ids = await to_bailian_data(
                    training_filespecs,
                )
            if validation_files:
                uploaded_validation_ids = await to_bailian_data(
                    validation_filespecs,
                )

        except Exception as e:
            raise OSSUploadError(
                "Failed to upload datasets",
                error_code=2062,
                public_error=SERVICE_UNAVAILABLE,
            ) from e

        return uploaded_training_ids, uploaded_validation_ids


class TrainingDataset(Dataset):
    type: DatasetsType = DatasetsType.TRAINING


class ValidationDataset(Dataset):
    type: DatasetsType = DatasetsType.VALIDATION


class FoundationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = None


class Training(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TrainingType = TrainingType.TRAINING_TYPE
    hyper_parameters: Dict[str, Any] = None
    resources: Dict[str, Any] = None

    @field_validator("hyper_parameters", "resources", mode="before")
    @classmethod
    def to_dict_values(cls, v: Dict) -> Dict[str, str]:
        """Convert all values in the dict to strings."""
        if v is None:
            return None
        return dict(v.items())


class Observability(BaseModel):
    ...


class Models(BaseModel):
    @classmethod
    def load_from_dict(cls, data: dict, **kwargs) -> Self:
        """Create an instance directly from a dictionary. Keys must match
        constructor parameters."""
        try:
            data.update(kwargs)
            return cls(**data)
        except Exception as e:
            raise IOErrorWithCode(
                "Failed to load from dict",
                error_code=1002,
                public_error=INTERNAL_ERROR,
            ) from e

    @classmethod
    def load_from_yaml(cls, file_path: str, **kwargs) -> Self:
        """Load a YAML file and create an instance."""
        try:
            check_file(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                d = yaml.load(f.read(), Loader=yaml.SafeLoader)
                d.update(kwargs)
            logger.info(f"Loaded from YAML: {file_path}")
        except Exception as e:
            raise IOErrorWithCode(
                f"Failed to load YAML file: {file_path}",
                error_code=1001,
                path=file_path,
                public_error=INTERNAL_ERROR,
            ) from e

        return cls.load_from_dict(d)

    def to_yaml(self, file_path: str, overwrite: bool = True) -> None:
        path = Path(file_path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {file_path}, use overwrite=True to "
                f"force overwrite",
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_dict = self.model_dump(mode="json")
            # Remove keys whose value is None or empty string for a cleaner
            # YAML output
            model_dict = deep_remove_none(model_dict)
            logger.debug(
                f"The struct of Models class: "
                f""
                f"{model_dict if LOG_LEVEL=='DEBUG' else deep_mask(model_dict)}",  # noqa: E501
            )

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    model_dict,
                    f,
                    encoding="utf-8",
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            raise IOErrorWithCode(
                "Failed to write file",
                error_code=1003,
                public_error=INTERNAL_ERROR,
            ) from e


class FunctionComponentModel(BaseModel):
    """Model representing function component configuration and operations."""

    model_config = ConfigDict(extra="forbid")

    zipdir: str = Field(
        default="./",
        description="Local directory path containing function code",
    )
    classpath: Optional[str] = Field(
        default="",
        description="Entrypoint class path for the function",
    )

    filepath: str = Field(
        default="",
        description="Main Python filepath containing function logic",
    )
    classname: str = Field(
        default="",
        description="Entrypoint class name for the function",
    )

    requirements_path: Optional[str] = Field(
        default=FC_REQUIREMENTS_FILE,
        description="Specify Python dependencies",
    )
    extra_files: Optional[List[str]] = Field(
        default=[],
        description="Additional deployment files required for function "
        "execution",
    )

    oss_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for OSS storage resource",
    )
    oss_signed_url: Optional[str] = Field(
        default=None,
        description="Pre-signed URL for OSS bucket access",
    )

    def generate_id(self, func_type: FunctionType) -> str:
        """Generate unique OSS resource identifier."""
        self.oss_id = generate_random_id(func_type.value)
        logger.debug(
            f"Generated OSS ID | Type: {func_type.name}, ID: {self.oss_id}",
        )
        return self.oss_id

    async def get_oss(self, oss_id: Optional[str] = None) -> str:
        """Retrieve OSS signed URL for deployment package."""
        try:
            if oss_id:
                self.oss_id = oss_id

            result = await client_fc(
                FC_API_KEY,
                FC_UPLOAD_OSS_API,
                {"unique_key": self.oss_id},
            )
            self.oss_signed_url = result.get("output", {}).get("url", "")
            if not self.oss_signed_url:
                raise OSSConnectionError(
                    f"Empty OSS URL received: {result}",
                    error_code=2001,
                    public_error=SERVICE_UNAVAILABLE,
                )

            logger.debug(
                f"Obtained OSS signed URL | ID: {self.oss_id}, "
                f"URL: {self.oss_signed_url}",
            )
            return self.oss_signed_url

        except OSSConnectionError:
            raise
        except Exception as e:
            raise OSSConnectionError(
                "Failed to obtain OSS URL",
                error_code=2002,
                public_error=SERVICE_UNAVAILABLE,
            ) from e

    async def create_layer(
        self,
        name: str = FC_LAYER_NAME,
        requirements_file: str = FC_REQUIREMENTS_FILE,
        # oss_signed_url: str = None,
    ) -> str:
        """Retrieve OSS signed URL for deployment package."""
        layer_code = None
        try:
            # Validate requirements file if provided
            layer_name = name + "-" + generate_random_id()[:8]
            requirements = None
            if requirements_file and requirements_file.strip():
                req_path = os.path.join(self.zipdir, requirements_file)
                check_file(req_path)
                logger.debug(f"Found requirements file: {req_path}")
                with open(req_path, "r", encoding="utf-8") as f:
                    requirements = f.read()
            # oss_signed_url = oss_signed_url or self.oss_signed_url

            result = await client_fc(
                FC_API_KEY,
                FC_LAYER_CREATE_API,
                {
                    "layer_name": layer_name,
                    "requirements_content": requirements,
                    # "signed_url": oss_signed_url,
                },
            )
            if result.get("status", {}).get("code", 500) != 200:
                raise FunctionLayerError(
                    f"Function layer create failed: {result}",
                )

            layer_code = result.get("output", {}).get("layer_code", "")
            logger.debug(
                f"Create function layer | layer-name: {layer_name} | "
                f"layer_code: {layer_code}",
            )

        except Exception as e:
            raise FunctionLayerError(
                "Function layer create failed",
                error_code=2013,
                public_error=SERVICE_UNAVAILABLE,
            ) from e

        return layer_code

    async def to_oss(
        self,
        func_type: FunctionType,
        signed_url: Optional[str] = None,
    ) -> None:
        """Upload function package to OSS storage."""
        url = signed_url or self.oss_signed_url

        try:
            # Create deploy files
            create_deployment_files(
                functype=func_type,
                dirpath=self.zipdir,
                filepath=self.filepath,
                classname=self.classname,
                requirements_path=self.requirements_path,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".zip",
                delete=False,
            ) as tmp:
                zip_dir(
                    output_zip=tmp.name,
                    dirpath=self.zipdir,
                    extra_files=self.extra_files,
                    rw_type="w",
                )

                await upload_zip_to_oss_and_by_signed_url(url, tmp.name)
                logger.debug(
                    f"Package uploaded | Size: {os.path.getsize(tmp.name)} "
                    f"bytes, Files: "
                    f"{len(self.extra_files) + 1 if self.extra_files else 1}",
                )

                self.clean_temp_files(tmp.name)

        except Exception as e:
            raise OSSUploadError(
                "Package upload failed",
                error_code=2003,
                endpoint=url or "",
                public_error=SERVICE_UNAVAILABLE,
            ) from e

        return

    async def get_layer(self, layer_code: str) -> str:
        """Get FC layer status."""
        try:
            i = 0
            while i < 3:
                result = await client_fc(
                    FC_API_KEY,
                    FC_LAYER_QUERY_API,
                    {
                        "layer_code": layer_code,
                    },
                    "GET",
                )
                status = result.get("output", {}).get("status", "")
                if status != "SUCCESS":
                    i += 1
                    logger.debug(
                        f"Load function layer({i}) | layer_code:"
                        f" {layer_code} | Status: {status}",
                    )
                    await asyncio.sleep(10)
                else:
                    break

            logger.debug(
                f"Load function layer | layer_code: {layer_code} | Status:"
                f" {status}",
            )
            if status != "SUCCESS":
                raise FunctionLayerError(
                    f"Function layer create failed: {status}",
                    error_code=2014,
                    public_error=SERVICE_UNAVAILABLE,
                )

            return status

        except Exception as e:
            logger.warning(
                f"Load function layer failed | layer_code: {layer_code}, "
                f"Error: {str(e)}",
            )

        return "SUCCESS"

    def clean_temp_files(self, tmp_path: str) -> None:
        """Cleanup temporary deployment files."""
        try:
            for f in [tmp_path]:
                if os.path.exists(f):
                    if os.path.isfile(f):
                        os.remove(f)
                    else:
                        shutil.rmtree(f)
        except Exception as e:
            logger.warning(f"Temp file cleanup failed: {str(e)}")

    def split_classpath(self):
        self.filepath, self.classname = get_filepath_classname(self.classpath)

    def get_sub_function_weights(self):
        if not self.filepath and self.classpath:
            self.split_classpath()
        return get_weights_from_file(self.filepath, self.classname)


class FunctionComponentRuntime(BaseModel):
    """Runtime configuration for a function component"""

    model_config = ConfigDict(extra="forbid")

    layer_code: Optional[str] = None
    """Code of function layer"""

    cpu: Optional[float] = None
    """Number of CPU cores allocated (in vCPU units)"""

    memory_size: Optional[int] = None
    """Memory allocation size (in MB)"""

    disk_size: Optional[int] = None
    """Disk storage capacity (in MB)"""

    concurrency: Optional[int] = None
    """Maximum number of concurrent executions per instance (unitless count)"""

    capacity: Optional[int] = None
    """Current number of active instances (unitless count)"""

    max_capacity: Optional[int] = None
    """Maximum allowed instances for scaling (unitless count)"""

    min_capacity: Optional[int] = None
    """Minimum required instances for scaling (unitless count)"""

    memory_scale_threshold: Optional[float] = None
    """Memory utilization percentage threshold for scaling (0-100%)"""

    concurrency_scale_threshold: Optional[float] = None
    """Concurrency utilization percentage threshold for scaling (0-100%)"""

    enable_vpc_config: Optional[bool] = None
    """Flag indicating if VPC configuration is enabled (boolean)"""

    security_group_id: Optional[str] = None
    """ID of the security group (string identifier)"""

    switch_ids: Optional[List[str]] = None
    """List of network switch IDs (string identifiers)"""

    vpc_id: Optional[str] = None
    """Virtual Private Cloud ID (string identifier)"""

    vpc_role: Optional[str] = None
    """IAM role for VPC access (string identifier)"""

    enable_log: Optional[bool] = None
    """Flag indicating if logging is enabled (boolean)"""

    env: Optional[Dict[str, Any]] = None
    """
    Environment variables for the function runtime environment.

    Key-value pairs where:
    - Key: Environment variable name (string)
    - Value: Environment variable value (any serializable type)

    Example:
        {
            "ENABLE_TRAJECTORY": True
        }

    Best Practices:
    1. Use UPPER_SNAKE_CASE for variable names
    2. Keep values as strings when possible for maximum compatibility
    3. Avoid storing sensitive secrets directly - use secret management systems
    """


class AgenticRLFunctionComponent(Models, BaseModel):
    """Main class managing function component lifecycle operations."""

    type: FunctionType = Field(
        default=FunctionType.ROLLOUT,
        description="Type of function component",
    )
    name: Optional[str] = Field(default=None, description="Function name")
    timeout: Optional[int] = Field(
        default=None,
        description="Function timeout",
    )

    # for register
    fcmodel: Optional[FunctionComponentModel] = Field(
        default_factory=FunctionComponentModel,
        description="Function component model",
    )
    # for load
    runtime: Optional[FunctionComponentRuntime] = Field(
        default=None,
        description="Function component runtime",
    )

    entity_id: Optional[str] = Field(
        default=None,
        description="System-generated registration identifier",
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Deployed instance identifier",
    )
    instance_status: Optional[int] = Field(
        default=-1,
        description="Current instance state (-1=Unknown, 0=Initialized, "
        "1=Deploying, 2=Active)",
    )
    instance_url: Optional[str] = Field(
        default=None,
        description="Endpoint URL for deployed instance",
    )
    instance_token: Optional[str] = Field(
        default=None,
        description="Authentication token for instance access",
    )

    model_config = ConfigDict(extra="forbid")

    # pylint: disable=too-many-branches
    async def register(
        self,
        oss_id: Optional[str] = None,
        oss_url: Optional[str] = None,
    ) -> ResponseFC:
        """Register function in the deployment system."""
        try:
            # Handle OSS configuration
            if oss_id and oss_url:
                self.fcmodel.oss_id = oss_id
                self.fcmodel.oss_signed_url = oss_url
            else:
                self.fcmodel.generate_id(self.type)
                await self.fcmodel.get_oss()

            # Split: classpath to filepath & classname
            if self.fcmodel.classpath:
                self.fcmodel.split_classpath()

            # Upload
            await self.fcmodel.to_oss(
                func_type=self.type,
                signed_url=self.fcmodel.oss_signed_url,
            )

            # Create function layer
            if FC_LAYER_USED:
                if self.runtime is None:
                    self.runtime = FunctionComponentRuntime()
                self.runtime.layer_code = await self.fcmodel.create_layer()

        except FunctionLayerError as e:
            root = e
            while root.__cause__:
                root = root.__cause__
            return ResponseFC(
                status=Status(
                    task=StatusType.FAILED,
                    name="DeploymentError",
                    code=524,
                    message=f"Function layer deployment failed: {root}",
                ),
                output={},
            )
        except Exception as e:
            root = e
            while root.__cause__:
                root = root.__cause__
            return ResponseFC(
                status=Status(
                    task=StatusType.FAILED,
                    name="DeploymentError",
                    code=525,
                    message=f"Function deployment failed: {root}",
                ),
                output={},
            )

        try:
            # Register
            request = RequestFC(
                unique_key=self.fcmodel.oss_id,
                name=self.fcmodel.filepath,
                func=self.fcmodel.classname,
                code_url=self.fcmodel.oss_signed_url,
            )

            if self.type == FunctionType.ROLLOUT:
                endpoint = FC_REGISTER_ROLLOUT_API
            elif self.type == FunctionType.REWARD:
                endpoint = FC_REGISTER_REWARD_API
            elif self.type == FunctionType.GROUP_REWARD:
                endpoint = FC_REGISTER_GROUP_REWARD_API
            else:
                raise RegistrationError(
                    f"Not exist type: {self.type.name}",
                    error_code=2011,
                    public_error=INVALID_REQUEST,
                )

            result = await client_fc(
                FC_API_KEY,
                endpoint,
                request.model_dump(),
            )
            func_type_id = get_func_type_id(self.type)
            self.entity_id = result.get("output", {}).get(func_type_id, "")

            if not self.entity_id:
                raise RegistrationError(
                    f"Empty entity ID received: {result}",
                    error_code=2012,
                    public_error=INTERNAL_ERROR,
                )

            logger.info(
                f"Function registered | Type: {self.type.name}, "
                f"ID: {self.entity_id}, OSS: {self.fcmodel.oss_id}",
            )

            return ResponseFC(
                status=Status(
                    task=StatusType.SUCCEEDED,
                    name="FunctionRegistered",
                    code=200,
                    message=f"{self.type.name} function registered "
                    f"successfully",
                ),
                output={"entity_id": self.entity_id},
            )

        except Exception as e:
            root = e
            while root.__cause__:
                root = root.__cause__
            return ResponseFC(
                status=Status(
                    task=StatusType.FAILED,
                    name="DeploymentError",
                    code=521,
                    message=f"Full deployment failed: {root}",
                ),
                output={},
            )

    async def load(
        self,
        entity_id: Optional[str] = None,
        runtime: Optional[FunctionComponentRuntime] = None,
        warmup: bool = False,
    ) -> ResponseFC:
        """Load and initialize a registered function instance."""
        try:
            # Resolve target registration ID
            target_entity_id = entity_id or self.entity_id
            if not target_entity_id:
                raise ValueErrorWithCode(
                    "No valid registration ID provided",
                    error_code=2021,
                    public_error=INVALID_REQUEST,
                )

            if FC_LAYER_USED:
                if self.runtime.layer_code is None:
                    raise ValueErrorWithCode(
                        "layer_code is required when FC_LAYER_USED is enabled",
                        error_code=2022,
                        public_error=INVALID_REQUEST,
                    )
                await self.fcmodel.get_layer(
                    layer_code=self.runtime.layer_code,
                )

            # Load function instance
            runtime_obj = runtime or self.runtime
            runtime_dict = (
                deep_remove_none({**runtime_obj.model_dump()})
                if runtime_obj
                else {}
            )
            job_id = generate_random_id()
            url = f"{FC_LOAD_API}/jobId-{job_id}/{target_entity_id}"

            result = await client_fc(FC_API_KEY, url, runtime_dict)
            self.instance_id = result.get("output", {}).get("instanceId", "")
            if not self.instance_id:
                raise FunctionLoadError(
                    f"Empty instance ID received: {result}",
                    error_code=2023,
                    public_error=INTERNAL_ERROR,
                )

            self.instance_url = result.get("output", {}).get("trigger_url", "")
            self.instance_token = result.get("output", {}).get(
                "trigger_token",
                "",
            )
            if (not self.instance_url) or (not self.instance_token):
                raise FunctionLoadError(
                    "Missing instance URL or token",
                    error_code=2024,
                    public_error=INTERNAL_ERROR,
                )

            logger.info(
                f"Instance initialized | EntityID: {target_entity_id}, "
                f"InstanceID: {self.instance_id}, "
                f"Endpoint: {self.instance_url}, "
                f"Response: "
                f"{result if LOG_LEVEL=='DEBUG' else deep_mask(result)}",
            )

        except Exception as e:
            logger.debug(
                f"Instance initialization failed | EntityID:"
                f" {target_entity_id}, "
                f"Error: {str(e)}",
            )
            return ResponseFC(
                status=Status(
                    task=StatusType.FAILED,
                    name="FunctionLoadError",
                    code=522,
                    message=f"Instance initialization failed: {str(e)}",
                ),
                output={},
            )

        # Perform instance warmup if requested
        if warmup:
            try:
                if not self.instance_url.startswith(("http://", "https://")):
                    raise ValueErrorWithCode(
                        "Invalid instance URL format",
                        error_code=2025,
                        public_error=INVALID_REQUEST,
                    )

                url = f"{self.instance_url.rstrip('/')}/health"
                result = await client_fc(
                    self.instance_token,
                    url,
                    {},
                    "GET",
                )
                status = result.get("status", str(StatusType.UNKNOWN))
                if status != StatusType.HEALTH:
                    raise InstanceWarmupError(
                        f"Health check failed: {result}",
                        error_code=2026,
                        instance_url=url,
                        public_error=SERVICE_UNAVAILABLE,
                    )

                logger.info(
                    f"Instance warmup completed | Instance:"
                    f" {self.instance_id if self.instance_id else 'N/A'}",
                )

            except Exception as e:
                logger.debug(
                    f"Warmup failed | InstanceID: {self.instance_id}, "
                    f"Error: {str(e)}",
                )
                return ResponseFC(
                    status=Status(
                        task=StatusType.FAILED,
                        name="InstanceWarmupError",
                        code=511,
                        message=f"Instance warmup failed: {str(e)}",
                    ),
                    output={"instance_id": self.instance_id},
                )

        return ResponseFC(
            status=Status(
                task=StatusType.SUCCEEDED,
                name="InstanceReady",
                code=200,
                message="Function instance ready for requests",
            ),
            output={
                "instance_id": self.instance_id,
                "endpoint": self.instance_url,
                "status": 2,  # 2 = Active status
            },
        )

    @classmethod
    async def query(cls, instance_id: str) -> ResponseFC:
        """Retrieve current status of a function instance."""
        try:
            if not instance_id:
                raise InputError(
                    "No instance ID available for query",
                    error_code=2031,
                )

            url = f"{FC_QUERY_API}/{instance_id}"
            result = await client_fc(FC_API_KEY, url, {})
            status = result.get("output", {}).get("status", -1)
            if status == -1:
                raise InstanceQueryError(
                    f"Invalid status received: {result}",
                    error_code=2032,
                )

            logger.debug(
                f"Status query completed | InstanceID: {instance_id} | "
                f"Status: {status}.",
            )

        except Exception as e:
            logger.debug(
                f"Status query failed | InstanceID: {instance_id}, "
                f"Error: {str(e)}",
            )
            return ResponseFC(
                status=Status(
                    task=StatusType.FAILED,
                    name="InstanceQueryError",
                    code=523,
                    message=f"Status query failed: {str(e)}",
                ),
                output={"instance_id": instance_id},
            )

        return ResponseFC(
            status=Status(
                task=StatusType.SUCCEEDED,
                name="InstanceStatus",
                code=200,
                message="Instance status retrieved",
            ),
            output=result,
        )

    @classmethod
    async def verify_function(
        cls,
        input_data: Union[RolloutInput, RewardInput, GroupRewardInput],
        instance_id: Optional[str] = None,
        instance_url: Optional[str] = None,
        instance_token: Optional[str] = None,
    ) -> Dict:
        """Validate deployed function functionality."""
        try:
            # Get instance metadata
            if instance_id is None:
                raise ValueErrorWithCode(
                    "instance_id is required for verification",
                    error_code=2041,
                )
            result = await cls.query(instance_id)
            if result.status.task != StatusType.SUCCEEDED:
                raise InstanceQueryError(
                    "Status query failed",
                    error_code=2042,
                )
            instance_url = instance_url or result.output.get("output", {}).get(
                "trigger_url",
                "",
            )
            instance_token = instance_token or result.output.get(
                "output",
                {},
            ).get("trigger_token", "")
            if (not instance_url) or (not instance_token):
                raise OutputError(
                    "No instance url/token provided",
                    error_code=2043,
                )

            input_data_dict = input_data.model_dump(
                mode="json",
                exclude_none=True,
            )
            if (
                "model_resource" in input_data_dict
                and "api_key" in input_data_dict["model_resource"]
            ):
                input_data_dict["model_resource"][
                    "api_key"
                ] = input_data.model_resource.api_key.get_secret_value()

            # Execute test request
            response = await client_fc(
                instance_token,
                f"{instance_url}/api/v1",
                input_data_dict,
            )

            # Validate response format
            if isinstance(input_data, RolloutInput):
                validator = RolloutOutput
            elif isinstance(input_data, RewardInput):
                validator = RewardOutput
            elif isinstance(input_data, GroupRewardInput):
                validator = GroupRewardOutput
            else:
                raise ValidationError(
                    "Unsupported input type",
                    error_code=2044,
                )

        except Exception as e:
            raise ValidationError(
                "Function verification failed",
                error_code=2045,
            ) from e

        try:
            resp_status = response.get("status", {})
            if isinstance(resp_status, dict):
                status_code = resp_status.get("code", 200)
                if status_code != 200:
                    error_msg = resp_status.get("message", "Unknown error")
                    fc_message = response.get("Message", "")
                    if fc_message:
                        match = re.search(
                            r"(\w+(?:Error|Exception|Warning): .+)",
                            fc_message,
                        )
                        if match:
                            error_msg = match.group(1)
                    raise ValidationError(
                        error_msg,
                        error_code=2046,
                    )

            validated = validator.model_validate(response)

            logger.info(
                f"Validation succeeded | Input: "
                f"{input_data_dict if LOG_LEVEL == 'DEBUG' else deep_mask(input_data_dict)}, "  # noqa: E501  # pylint: disable=line-too-long
                f"Output: {validated.model_dump_json()}, "
                f"Status: {response.get('status', StatusType.SUCCEEDED)}",
            )
            return validated.model_dump()

        except Exception as e:
            raise ValidationError(
                "Function output validation failed",
                error_code=2047,
            ) from e


class RolloutFunctionComponent(AgenticRLFunctionComponent):
    """Rollout function component with type fixed as ROLLOUT."""

    type: FunctionType = Field(
        default=FunctionType.ROLLOUT,
        description="Type of function component",
    )

    # PLACEHOLDER
    weight: Optional[float] = Field(
        default=None,
        description="[PLACEHOLDER] Function weight mapping. This field is "
        "currently not used by the system.",
    )
    reward_metric_weight: Optional[Dict[str, float]] = Field(
        default=None,
        description="[PLACEHOLDER] Reward metric weight mapping. This field "
        "is currently not used by the system.",
    )


class RewardFunctionComponent(AgenticRLFunctionComponent):
    """Reward function component with type fixed as REWARD."""

    type: FunctionType = Field(
        default=FunctionType.REWARD,
        description="Type of function component",
    )
    weight: Optional[float] = Field(
        default=None,
        description="Function weight",
    )
    reward_metric_weight: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reward metric weight mapping",
    )

    async def register(
        self,
        oss_id: Optional[str] = None,
        oss_url: Optional[str] = None,
    ) -> ResponseFC:
        if not self.reward_metric_weight:
            self.reward_metric_weight = self.fcmodel.get_sub_function_weights()
        result = await super().register(oss_id=oss_id, oss_url=oss_url)
        return result


class GroupRewardFunctionComponent(AgenticRLFunctionComponent):
    """Group reward function component with type fixed as GROUP_REWARD."""

    type: FunctionType = Field(
        default=FunctionType.GROUP_REWARD,
        description="Type of function component",
    )
    weight: Optional[float] = Field(
        default=None,
        description="Function weight",
    )
    reward_metric_weight: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reward metric weight mapping",
    )

    async def register(
        self,
        oss_id: Optional[str] = None,
        oss_url: Optional[str] = None,
    ) -> ResponseFC:
        if not self.reward_metric_weight:
            self.reward_metric_weight = self.fcmodel.get_sub_function_weights()
        result = await super().register(oss_id=oss_id, oss_url=oss_url)
        return result


_FUNCTYPE_CLASS_MAP = {
    FunctionType.ROLLOUT: RolloutFunctionComponent,
    FunctionType.REWARD: RewardFunctionComponent,
    FunctionType.GROUP_REWARD: GroupRewardFunctionComponent,
}


def _fc_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "rollout")
    return getattr(v, "type", "rollout").value


FunctionComponentUnion = Annotated[
    Union[
        Annotated[RolloutFunctionComponent, Tag("rollout")],
        Annotated[RewardFunctionComponent, Tag("reward")],
        Annotated[GroupRewardFunctionComponent, Tag("group_reward")],
    ],
    Discriminator(_fc_discriminator),
]


class TuningModel(Models, BaseModel):
    """Core configuration model for managing model tuning tasks."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="agentic-rl", min_length=1, max_length=256)
    model: FoundationModel = FoundationModel()
    functions: List[FunctionComponentUnion] = Field(default_factory=list)
    datasets: List[Dataset] = Field(default_factory=list)
    training: Training = Training()
    observability: Optional[Observability] = Observability()

    # pylint: disable=too-many-branches
    async def register_functions(
        self,
        lazy_load: Optional[bool] = True,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
    ]:
        """Register function compute components (functions) for the tuning
        job."""
        entity_rollout_ids = []
        entity_reward_ids = []
        entity_group_reward_ids = []
        instance_rollout_ids = []
        instance_reward_ids = []
        instance_group_reward_ids = []

        try:
            for fc in self.functions:
                if fc.entity_id:
                    entity_id = fc.entity_id
                else:
                    reg_result = await fc.register()
                    if reg_result.status.success:
                        entity_id = reg_result.output.get("entity_id", "")
                        if not entity_id:
                            raise RegistrationError(
                                "Empty entity ID after registration",
                                error_code=2051,
                            )
                        logger.debug(
                            f"Registered new function component: "
                            f"Type={fc.type.value}, RegisterID={entity_id}",
                        )
                    else:
                        raise RegistrationError(
                            reg_result.status.message,
                            error_code=2052,
                        )

                if fc.type == FunctionType.ROLLOUT:
                    entity_rollout_ids.append(entity_id)
                elif fc.type == FunctionType.REWARD:
                    entity_reward_ids.append(entity_id)
                elif fc.type == FunctionType.GROUP_REWARD:
                    entity_group_reward_ids.append(entity_id)

                if not lazy_load:
                    load_result = await fc.load(entity_id=entity_id)
                    if load_result.status.success:
                        instance_id = load_result.output.get("instance_id", "")
                        if not instance_id:
                            raise FunctionLoadError(
                                "Empty instance ID after load",
                                error_code=2053,
                            )
                        logger.debug(
                            f"Loaded function component instance: "
                            f"RegisterID={entity_id}, InstanceID"
                            f"={instance_id}",
                        )
                        if fc.type == FunctionType.ROLLOUT:
                            instance_rollout_ids.append(instance_id)
                        elif fc.type == FunctionType.REWARD:
                            instance_reward_ids.append(instance_id)
                        elif fc.type == FunctionType.GROUP_REWARD:
                            instance_group_reward_ids.append(instance_id)
                    else:
                        raise FunctionLoadError(
                            f"Load failed: {load_result}",
                            error_code=2054,
                        )

        except Exception as e:
            if hasattr(e, "error_code"):
                raise
            raise RegistrationError(
                "Function component registration failed",
                error_code=2055,
            ) from e

        return (
            entity_rollout_ids,
            entity_reward_ids,
            entity_group_reward_ids,
            instance_rollout_ids,
            instance_reward_ids,
            instance_group_reward_ids,
        )

    async def upload_datasets(
        self,
        training_files: Union[List[str], str] = None,
        validation_files: Union[List[str], str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Register and validate training/validation datasets."""
        uploaded_training_ids = []
        uploaded_validation_ids = []
        try:
            if training_files:
                self.datasets = []
                for f in training_files:
                    ds = Dataset(
                        type=DatasetsType.TRAINING,
                        data_source_type=DataSourceType.FILE_ID,
                        file_name=f,
                    )
                    self.datasets.append(ds)

                if validation_files:
                    for f in validation_files:
                        ds = Dataset(
                            type=DatasetsType.VALIDATION,
                            data_source_type=DataSourceType.FILE_ID,
                            file_name=f,
                        )
                        self.datasets.append(ds)

            # Perform dataset validation and upload
            for ds in self.datasets:
                ds_id = await ds.upload_dataset()
                if ds.type == DatasetsType.TRAINING and ds_id is not None:
                    uploaded_training_ids.append(ds_id)
                elif ds.type == DatasetsType.VALIDATION and ds_id is not None:
                    uploaded_validation_ids.append(ds_id)

            logger.info(
                f"Successfully datasets registration: "
                f"{len(uploaded_training_ids)} training, "
                f"{len(uploaded_validation_ids)} validation",
            )

        except Exception as e:
            raise OSSUploadError(
                "Dataset registration failed",
                error_code=2063,
            ) from e

        return uploaded_training_ids, uploaded_validation_ids

    def get_entity_ids(self, functype: FunctionType):
        ids = []
        for fc in self.functions:
            if functype == fc.type and fc.entity_id:
                ids.append(fc.entity_id)
        return ids

    def get_runtimes(self, functype: FunctionType):
        runtimes = []
        for fc in self.functions:
            if functype == fc.type and fc.runtime:
                runtimes.append(fc.runtime.model_dump())
        return runtimes

    def get_names(self, functype: FunctionType):
        """Get name values for function components of the given type."""
        names = []
        for fc in self.functions:
            if functype == fc.type:
                names.append(getattr(fc, "name", None))
        return names

    def set_names(self, functype: FunctionType):
        """Set name values for function components of the given type."""
        for fc in self.functions:
            if functype == fc.type and not fc.name:
                fc.name = "-".join((str(fc.type), generate_random_id()[:8]))
                logger.debug(
                    f"Generate a random name: {fc.name} for {functype}",
                )

    def get_weights(self, functype: FunctionType):
        """Get weight values for function components of the given type."""
        weights = []
        for fc in self.functions:
            if functype == fc.type:
                weights.append(getattr(fc, "weight", None))
        return weights

    def get_timeouts(self, functype: FunctionType):
        """Get timeout values for function components of the given type."""
        timeouts = []
        for fc in self.functions:
            if functype == fc.type:
                timeouts.append(getattr(fc, "timeout", None))
        return timeouts

    def get_reward_metric_weights(self, functype: FunctionType):
        """Get reward_metric_weight values for function components of the
        given type."""
        metric_weights = []
        for fc in self.functions:
            if functype == fc.type and functype == FunctionType.REWARD:
                metric_weights.append(
                    getattr(fc, "reward_metric_weight", None),
                )
        return metric_weights

    def combine_ids_runtimes(
        self,
        functype: FunctionType,
        ids: Union[List[str], str] = None,
        runtimes: Union[List[Dict[str, Any]], Dict[str, Any]] = None,
        id_str: str = None,
    ):
        if ids:
            ids = [ids] if isinstance(ids, str) else ids
        if runtimes:
            runtimes = [runtimes] if isinstance(runtimes, Dict) else runtimes
        function_ids = ids or self.get_entity_ids(functype)
        function_runtimes = runtimes or self.get_runtimes(functype) or []
        assert isinstance(
            function_runtimes,
            list,
        ), "function_runtimes must be a list"

        self.set_names(functype)
        function_names = self.get_names(functype)
        function_weights = self.get_weights(functype)
        function_timeouts = self.get_timeouts(functype)
        function_metric_weights = self.get_reward_metric_weights(functype)

        id_str = id_str or get_func_type_id(functype)
        functions = []
        for i in range(  # pylint: disable=consider-using-enumerate
            len(function_ids),
        ):
            function = {id_str: function_ids[i]}

            # Add name if present (for reward/group_reward types)
            if i < len(function_names) and function_names[i] is not None:
                function["name"] = function_names[i]

            # Add weight if present (for reward/group_reward types)
            if i < len(function_weights) and function_weights[i] is not None:
                function["weight"] = function_weights[i]

            # Add timeout if present (for reward/group_reward types)
            if i < len(function_timeouts) and function_timeouts[i] is not None:
                function["timeout"] = function_timeouts[i]

            # Add reward_metric_weight if present (for reward/group_reward
            # types)
            if (
                i < len(function_metric_weights)
                and function_metric_weights[i] is not None
            ):
                function["reward_metric_weight"] = function_metric_weights[i]

            # Merge runtime config
            if function_runtimes and i <= len(function_runtimes) - 1:
                runtime_config = function_runtimes[i].copy()
                if "env" in runtime_config and isinstance(
                    runtime_config["env"],
                    Dict,
                ):
                    for key, value in runtime_config["env"].items():
                        if isinstance(value, bool):
                            runtime_config["env"][key] = str(value).lower()

                function.update(runtime_config)

            functions.append(function)

        return functions

    def add_function_components(
        self,
        functype: FunctionType,
        classpaths: Optional[Union[List[str], str]] = None,
        entity_ids: Optional[
            Union[List[str], str]
        ] = None,  # Prefer entity_ids over classpaths when available
        runtimes: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        names: Optional[Union[List[str], str]] = None,
        weights: Optional[Union[List[float], float]] = None,
        timeouts: Optional[Union[List[int], int]] = None,
        reward_metric_weights: Optional[
            Union[List[Dict[str, float]], Dict[str, float]]
        ] = None,
        workspace_dir: Optional[str] = "./",
    ):
        classpaths = (
            [classpaths] if isinstance(classpaths, str) else classpaths
        )
        entity_ids = (
            [entity_ids] if isinstance(entity_ids, str) else entity_ids
        )
        runtimes = [runtimes] if isinstance(runtimes, Dict) else runtimes
        names = [names] if isinstance(names, str) else names
        weights = [weights] if isinstance(weights, float) else weights
        timeouts = [timeouts] if isinstance(timeouts, int) else timeouts
        reward_metric_weights = (
            [reward_metric_weights]
            if isinstance(reward_metric_weights, Dict)
            else reward_metric_weights
        )

        len_classpaths = len(classpaths) if classpaths else 0
        len_entity_ids = len(entity_ids) if entity_ids else 0
        len_runtimes = len(runtimes) if runtimes else 0
        len_names = len(names) if names else 0
        len_weights = len(weights) if weights else 0
        len_timeouts = len(timeouts) if timeouts else 0
        len_reward_metric_weights = (
            len(reward_metric_weights) if reward_metric_weights else 0
        )

        if len_classpaths == 0 and len_entity_ids == 0:
            logger.warning(
                f"The inputs of classpaths and entity_ids for "
                f"{functype} are none.",
            )
            return []

        cls = _FUNCTYPE_CLASS_MAP.get(functype, AgenticRLFunctionComponent)

        if (
            len_entity_ids > 0
        ):  # Prefer entity_ids over classpaths when available
            assert entity_ids is not None
            for i in range(len_entity_ids):
                self.functions.append(
                    cls(
                        type=functype,
                        entity_id=entity_ids[i],
                        runtime=(
                            FunctionComponentRuntime(**runtimes[i])
                            if runtimes and i < len_runtimes
                            else None
                        ),
                        name=names[i] if names and i < len_names else None,
                        weight=(
                            weights[i] if weights and i < len_weights else None
                        ),
                        timeout=(
                            timeouts[i]
                            if timeouts and i < len_timeouts
                            else None
                        ),
                        reward_metric_weight=(
                            reward_metric_weights[i]
                            if reward_metric_weights
                            and i < len_reward_metric_weights
                            else None
                        ),
                    ),
                )
        else:
            assert classpaths is not None
            for i in range(len_classpaths):
                self.functions.append(
                    cls(
                        type=functype,
                        fcmodel=FunctionComponentModel(
                            zipdir=workspace_dir,
                            classpath=classpaths[i],
                        ),
                        runtime=(
                            FunctionComponentRuntime(**runtimes[i])
                            if runtimes and i < len_runtimes
                            else None
                        ),
                        name=names[i] if names and i < len_names else None,
                        weight=(
                            weights[i] if weights and i < len_weights else None
                        ),
                        timeout=(
                            timeouts[i]
                            if timeouts and i < len_timeouts
                            else None
                        ),
                        reward_metric_weight=(
                            reward_metric_weights[i]
                            if reward_metric_weights
                            and i < len_reward_metric_weights
                            else None
                        ),
                    ),
                )

        return self.functions

    def check_function_names(self) -> bool:
        """
        Check for duplicate function component names.

        Returns:
            True if all names are unique, False if duplicates found.
        """
        seen_names = {}
        duplicate_found = False

        for index, fc in enumerate(self.functions):
            if not hasattr(fc, "name"):
                logger.error(
                    f"Function component at index {index} is missing a "
                    f"'name' attribute",
                )
                duplicate_found = True
                continue

            name = fc.name
            if name in seen_names:
                logger.error(
                    f"Duplicate function name '{name}' found: "
                    f"Original at index {seen_names[name]}, duplicate at "
                    f"index {index}",
                )
                duplicate_found = True
            else:
                seen_names[name] = index

        if duplicate_found:
            logger.error(
                "Duplicate function names detected. All function names must "
                "be unique.",
            )
            return False

        logger.debug("All function names are unique.")
        return True


class AgenticRLTuning(Models, BaseModel):
    """Main interface class for model tuning operations."""

    tuning_id: Optional[str] = None
    tuning: TuningModel = TuningModel()
