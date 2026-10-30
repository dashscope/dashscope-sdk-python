# -*- coding: utf-8 -*-
# dialog_state.py

from enum import Enum


class DialogState(Enum):
    """
    Dialog state enumeration class, defining the possible states
    of a dialog bot.

    Attributes:
        IDLE (str): Bot is in idle state.
        LISTENING (str): Bot is listening to user input.
        THINKING (str): Bot is thinking.
        RESPONDING (str): Bot is generating or responding.
    """

    IDLE = "Idle"
    LISTENING = "Listening"
    THINKING = "Thinking"
    RESPONDING = "Responding"


class StateMachine:
    """
    State machine class for managing bot state transitions.

    Attributes:
        current_state (DialogState): Current state.
    """

    def __init__(self):
        # Set initial state to IDLE when initializing the state machine
        self.current_state = DialogState.IDLE

    def change_state(self, new_state: str) -> None:
        """
        Change the current state to the specified new state.

        Args:
            new_state (str): The new state to switch to.

        Raises:
            ValueError: If attempting to switch to an invalid state.
        """
        if new_state in [state.value for state in DialogState]:
            self.current_state = DialogState(new_state)
        else:
            raise ValueError("Invalid state type")

    def get_current_state(self) -> DialogState:
        """
        Get the current state.

        Returns:
            DialogState: The current state.
        """
        return self.current_state
