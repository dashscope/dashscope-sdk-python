# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# multimodal conversation request directive


class RequestToRespondType:
    TRANSCRIPT = "transcript"
    PROMPT = "prompt"


# multimodal conversation response directive
RESPONSE_NAME_TASK_STARTED = "task-started"
RESPONSE_NAME_RESULT_GENERATED = "result-generated"
RESPONSE_NAME_TASK_FINISHED = "task-finished"

RESPONSE_NAME_TASK_FAILED = "TaskFailed"
RESPONSE_NAME_STARTED = "Started"
RESPONSE_NAME_STOPPED = "Stopped"
RESPONSE_NAME_STATE_CHANGED = "DialogStateChanged"
RESPONSE_NAME_REQUEST_ACCEPTED = "RequestAccepted"
RESPONSE_NAME_SPEECH_STARTED = "SpeechStarted"
# Server sends this event when ASR speech endpoint is detected,
# optional event
RESPONSE_NAME_SPEECH_ENDED = "SpeechEnded"
# AI voice response starts, SDK prepares to receive audio
RESPONSE_NAME_RESPONDING_STARTED = "RespondingStarted"
RESPONSE_NAME_RESPONDING_ENDED = "RespondingEnded"  # AI voice response ends
RESPONSE_NAME_SPEECH_CONTENT = (
    "SpeechContent"  # User speech recognition text, full streaming output
)
RESPONSE_NAME_RESPONDING_CONTENT = (
    "RespondingContent"  # System output text, full streaming output
)
RESPONSE_NAME_ERROR = "Error"  # Server-side error during dialog
RESPONSE_NAME_HEART_BEAT = "HeartBeat"  # Heartbeat message
