"""
history_manager.py
------------------
Robust State History Engine for Comic-Lab:
Provides deep snapshotting, multi-step Undo (Ctrl+Z) and Redo (Ctrl+Y / Ctrl+Shift+Z)
across Stage 2 (Bubble Mask Editor / Scissors / Brush) and Stage 3 (Translation / Text Editing).
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st


MAX_HISTORY = 25


def init_history_state():
    """Ensures undo and redo stacks are initialized in st.session_state."""
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []


def sync_bubble_widget_states():
    """Syncs trans_*, active_*, nudge_* widget session keys with current bubbles list."""
    bubbles = st.session_state.get("bubbles", [])
    for b in bubbles:
        bid = b.get("bubble_id") if isinstance(b, dict) else getattr(b, "bubble_id", None)
        if bid is not None:
            t_val = b.get("uzbek_translation", "") if isinstance(b, dict) else getattr(b, "uzbek_translation", "")
            act_val = b.get("is_active", True) if isinstance(b, dict) else getattr(b, "is_active", True)
            nudge_val = b.get("font_size_offset", 0) if isinstance(b, dict) else getattr(b, "font_size_offset", 0)
            st.session_state[f"trans_{bid}"] = t_val
            st.session_state[f"active_{bid}"] = act_val
            st.session_state[f"nudge_{bid}"] = nudge_val


def push_undo_snapshot(description: str = "Tahrir"):
    """
    Captures a deep copy of current bubbles state onto the undo stack.
    Clears redo stack on any new mutation.
    Caps history at MAX_HISTORY snapshots.
    """
    init_history_state()

    bubbles = st.session_state.get("bubbles", [])
    if not bubbles:
        return

    # Deep copy bubbles to ensure independence from subsequent in-place mutations
    snapshot = {
        "description": description,
        "timestamp": time.time(),
        "bubbles": copy.deepcopy(bubbles),
        "analysis_version": st.session_state.get("analysis_version", 0)
    }

    st.session_state.undo_stack.append(snapshot)
    if len(st.session_state.undo_stack) > MAX_HISTORY:
        st.session_state.undo_stack.pop(0)

    # Any new action invalidates previously undone actions
    st.session_state.redo_stack.clear()


def undo_action() -> Tuple[bool, str]:
    """
    Reverts to the most recent snapshot on the undo stack.
    Pushes current state onto redo stack before restoring.
    Returns (success, description).
    """
    init_history_state()

    if not st.session_state.undo_stack:
        return False, "Bekor qilish uchun amallar mavjud emas"

    current_bubbles = st.session_state.get("bubbles", [])
    snapshot = st.session_state.undo_stack.pop()

    # Save current state to redo stack
    redo_snapshot = {
        "description": snapshot.get("description", "Tahrir"),
        "timestamp": time.time(),
        "bubbles": copy.deepcopy(current_bubbles),
        "analysis_version": st.session_state.get("analysis_version", 0)
    }
    st.session_state.redo_stack.append(redo_snapshot)
    if len(st.session_state.redo_stack) > MAX_HISTORY:
        st.session_state.redo_stack.pop(0)

    # Restore bubbles from snapshot
    st.session_state.bubbles = copy.deepcopy(snapshot["bubbles"])
    sync_bubble_widget_states()

    # Clear rendered cache so changes re-render immediately
    st.session_state.cleaned_page = None
    st.session_state.rendered_image = None

    # Increment analysis version to force canvas and display re-render
    st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1

    return True, snapshot.get("description", "Tahrir")


def redo_action() -> Tuple[bool, str]:
    """
    Re-applies the most recently undone action from the redo stack.
    Returns (success, description).
    """
    init_history_state()

    if not st.session_state.redo_stack:
        return False, "Qaytarish uchun amallar mavjud emas"

    current_bubbles = st.session_state.get("bubbles", [])
    snapshot = st.session_state.redo_stack.pop()

    # Save current state to undo stack
    undo_snapshot = {
        "description": snapshot.get("description", "Tahrir"),
        "timestamp": time.time(),
        "bubbles": copy.deepcopy(current_bubbles),
        "analysis_version": st.session_state.get("analysis_version", 0)
    }
    st.session_state.undo_stack.append(undo_snapshot)
    if len(st.session_state.undo_stack) > MAX_HISTORY:
        st.session_state.undo_stack.pop(0)

    # Restore bubbles from snapshot
    st.session_state.bubbles = copy.deepcopy(snapshot["bubbles"])
    sync_bubble_widget_states()

    # Clear rendered cache so changes re-render immediately
    st.session_state.cleaned_page = None
    st.session_state.rendered_image = None

    # Increment analysis version to force canvas and display re-render
    st.session_state.analysis_version = st.session_state.get("analysis_version", 0) + 1

    return True, snapshot.get("description", "Tahrir")


def can_undo() -> bool:
    """Returns True if there is at least one action that can be undone."""
    return bool(st.session_state.get("undo_stack"))


def can_redo() -> bool:
    """Returns True if there is at least one action that can be redone."""
    return bool(st.session_state.get("redo_stack"))


def get_undo_description() -> Optional[str]:
    """Returns the description of the action that will be undone, or None."""
    stack = st.session_state.get("undo_stack", [])
    return stack[-1].get("description") if stack else None


def get_redo_description() -> Optional[str]:
    """Returns the description of the action that will be redone, or None."""
    stack = st.session_state.get("redo_stack", [])
    return stack[-1].get("description") if stack else None
