"""
Ambiance and conversational constraint presets for realistic acoustic and dialogue simulation.

Defines diverse social, environmental, and emotional presets configuring LLM prompt
guidelines, turn-taking overlap dynamics, shouting/laughter probabilities, and background
acoustic ambient noise levels.
"""

# Import Modules
from dataclasses import dataclass

import random


@dataclass
class AmbiancePreset:
    """
    Configuration parameters governing a distinct conversational ambiance and social setting.
    """

    name: str
    display_name: str
    description: str
    prompt_guidelines: str
    overlap_rate: float
    shout_rate: float
    laugh_rate: float
    ambient_snr_db: float
    num_constraint_words: int


# Comprehensive registry of rich ambiance presets
AMBIANCE_PRESETS: dict[str, AmbiancePreset] = {
    "casual_chit_chat": AmbiancePreset(
        name="casual_chit_chat",
        display_name="Casual Chit-Chat",
        description="Relaxed friends catching up in a living room with organic banter.",
        prompt_guidelines=(
            "The conversation is relaxed, friendly, and natural. Friends are catching up about "
            "their daily lives, sharing funny stories, and joking around. The tone is warm."
        ),
        overlap_rate=0.20,
        shout_rate=0.02,
        laugh_rate=0.05,
        ambient_snr_db=38.0,
        num_constraint_words=3,
    ),
    "kitchen_cooking": AmbiancePreset(
        name="kitchen_cooking",
        display_name="Kitchen Cooking",
        description="People cooking dinner, discussing recipes, ingredients, and kitchen timers.",
        prompt_guidelines=(
            "The speakers are in a kitchen cooking dinner together. They discuss recipes, ask to "
            "pass spices or utensils, check oven temperatures, and coordinate meal timing."
        ),
        overlap_rate=0.22,
        shout_rate=0.03,
        laugh_rate=0.04,
        ambient_snr_db=36.0,
        num_constraint_words=3,
    ),
    "workplace_meeting": AmbiancePreset(
        name="workplace_meeting",
        display_name="Workplace Meeting",
        description=(
            "Colleagues discussing sprint deadlines, project deliverables, and technical issues."
        ),
        prompt_guidelines=(
            "A professional business meeting or sprint review. The speakers discuss technical "
            "deliverables, timelines, architecture decisions, and budget considerations."
        ),
        overlap_rate=0.15,
        shout_rate=0.01,
        laugh_rate=0.03,
        ambient_snr_db=42.0,
        num_constraint_words=3,
    ),
    "heated_argument": AmbiancePreset(
        name="heated_argument",
        display_name="Heated Argument",
        description="Passionate disagreement with frequent interruptions and objections.",
        prompt_guidelines=(
            "A passionate debate or argument. The speakers strongly disagree, talk over each "
            "other, raise voices, express frustration, and cut each other off with counterpoints."
        ),
        overlap_rate=0.35,
        shout_rate=0.08,
        laugh_rate=0.02,
        ambient_snr_db=38.0,
        num_constraint_words=2,
    ),
    "party_celebration": AmbiancePreset(
        name="party_celebration",
        display_name="Party Celebration",
        description="High-energy party atmosphere with toasts, laughter, jokes, and lively banter.",
        prompt_guidelines=(
            "A lively party celebration. High energy, cheerful toasts, storytelling, loud "
            "laughter, and enthusiastic congratulations between people having fun."
        ),
        overlap_rate=0.28,
        shout_rate=0.05,
        laugh_rate=0.08,
        ambient_snr_db=35.0,
        num_constraint_words=3,
    ),
    "smart_assistant_household": AmbiancePreset(
        name="smart_assistant_household",
        display_name="Smart Assistant Household",
        description=(
            "Family conversation interspersed with Alexa/Echo commands for timers and smart lights."
        ),
        prompt_guidelines=(
            "A smart home living space where speakers chat while occasionally invoking smart "
            "assistant commands (e.g. 'Echo, set a timer', 'Alexa, what is the weather')."
        ),
        overlap_rate=0.20,
        shout_rate=0.02,
        laugh_rate=0.04,
        ambient_snr_db=38.0,
        num_constraint_words=3,
    ),
    "gaming_session": AmbiancePreset(
        name="gaming_session",
        display_name="Gaming Session",
        description="Co-op multiplayer session with tactical callouts, excitement, and panic.",
        prompt_guidelines=(
            "Players in a multiplayer co-op game coordinating strategies, yelling tactical "
            "callouts, reacting to surprises, cheering victories, and groaning over mistakes."
        ),
        overlap_rate=0.30,
        shout_rate=0.06,
        laugh_rate=0.05,
        ambient_snr_db=36.0,
        num_constraint_words=2,
    ),
    "interview_podcast": AmbiancePreset(
        name="interview_podcast",
        display_name="Interview / Podcast",
        description="Structured host-guest interview with probing questions and follow-ups.",
        prompt_guidelines=(
            "A podcast or interview recording. The host asks insightful, probing questions and "
            "the guest provides detailed anecdotes and reflections. Turn-taking is orderly."
        ),
        overlap_rate=0.12,
        shout_rate=0.01,
        laugh_rate=0.02,
        ambient_snr_db=44.0,
        num_constraint_words=3,
    ),
    "late_night_philosophy": AmbiancePreset(
        name="late_night_philosophy",
        display_name="Late-Night Philosophy",
        description=(
            "Quiet late-night existential conversation with thought experiments and pauses."
        ),
        prompt_guidelines=(
            "Late-night introspective conversation. The atmosphere is quiet, contemplative, and "
            "reflective. Speakers discuss existence, memory, time, and deep philosophical ideas."
        ),
        overlap_rate=0.10,
        shout_rate=0.01,
        laugh_rate=0.02,
        ambient_snr_db=45.0,
        num_constraint_words=4,
    ),
    "family_dinner": AmbiancePreset(
        name="family_dinner",
        display_name="Family Dinner",
        description="Multi-generational family gathering around the table sharing news and food.",
        prompt_guidelines=(
            "A family dinner around the table. Multiple generations discussing school, work, "
            "passing bread and dishes, with affectionate interjections and cross-table chatter."
        ),
        overlap_rate=0.25,
        shout_rate=0.03,
        laugh_rate=0.05,
        ambient_snr_db=37.0,
        num_constraint_words=3,
    ),
    "academic_defense": AmbiancePreset(
        name="academic_defense",
        display_name="Academic Defense",
        description="Rigorous intellectual critique and thesis defense with formal vocabulary.",
        prompt_guidelines=(
            "A formal academic committee defense. Examiners raise rigorous methodological "
            "critiques and the candidate defends hypotheses, analytical models, and findings."
        ),
        overlap_rate=0.12,
        shout_rate=0.01,
        laugh_rate=0.02,
        ambient_snr_db=44.0,
        num_constraint_words=4,
    ),
    "emergency_rush": AmbiancePreset(
        name="emergency_rush",
        display_name="Emergency Rush",
        description=(
            "Urgent rush to catch a flight or find missing passports with hurried instructions."
        ),
        prompt_guidelines=(
            "People in a hurry packing to catch a flight or train. Fast-paced, rushed questions, "
            "searching for missing keys, urgency, shouting across rooms, and checking clocks."
        ),
        overlap_rate=0.32,
        shout_rate=0.06,
        laugh_rate=0.03,
        ambient_snr_db=36.0,
        num_constraint_words=2,
    ),
}


def get_all_ambiance_presets() -> dict[str, AmbiancePreset]:
    """
    Retrieve mapping of all available ambiance presets.

    Returns:
        dict[str, AmbiancePreset]: Presets dictionary keyed by name.
    """

    return dict(AMBIANCE_PRESETS)


def get_preset_names() -> list[str]:
    """
    Retrieve sorted list of all valid ambiance preset names.

    Returns:
        list[str]: Sorted list of preset identifier strings.
    """

    return sorted(list(AMBIANCE_PRESETS.keys()))


def get_ambiance_preset(name: str) -> AmbiancePreset:
    """
    Retrieve a specific ambiance preset by name or select randomly if 'random'.

    Args:
        name (str): Identifier or 'random'.

    Returns:
        AmbiancePreset: Selected ambiance preset instance.
    """

    clean_name: str = name.strip().lower()

    if clean_name == "random":
        random_key: str = random.choice(list(AMBIANCE_PRESETS.keys()))
        return AMBIANCE_PRESETS[random_key]

    if clean_name in AMBIANCE_PRESETS:
        return AMBIANCE_PRESETS[clean_name]

    # Fallback to casual chit-chat if not found
    return AMBIANCE_PRESETS["casual_chit_chat"]
