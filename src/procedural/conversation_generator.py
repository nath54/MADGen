"""
Procedural conversation timeline generator simulating multi-person interactions.

Coordinates parallel conversational groups, natural turn-taking dynamics,
dialogue interruptions and overlaps, and dynamic emotional vocal styles.
"""

# Import Modules
import random

from src.procedural.dialogue_bank import sample_dialogue_text
from src.config.models import (
    VocalStyle,
    PersonaConfig,
    UtteranceConfig,
    ConversationalStyle,
)


def partition_conversational_groups(
    personas: list[PersonaConfig],
) -> list[list[PersonaConfig]]:
    """
    Partition personas into small conversational cliques of two or three people.

    Args:
        personas (list[PersonaConfig]): Complete roster of scene personas.

    Returns:
        list[list[PersonaConfig]]: List of conversational groups.
    """

    # If small party, everyone is in the same group
    if len(personas) <= 3:
        return [personas]

    # Shuffle copies to create organic cliques
    shuffled_pool: list[PersonaConfig] = list(personas)
    random.shuffle(shuffled_pool)

    # Split into groups of 2 or 3
    groups: list[list[PersonaConfig]] = []
    index: int = 0
    while index < len(shuffled_pool):
        remaining: int = len(shuffled_pool) - index
        group_size: int = 3 if remaining >= 5 or remaining == 3 else 2
        groups.append(shuffled_pool[index : index + group_size])
        index += group_size

    return groups


def select_vocal_style_and_category(
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> tuple[VocalStyle, str]:
    """
    Select vocal style and dialogue category based on emotional probabilities.

    Args:
        shout_rate (float): Probability of shouting or raised voice.
        laugh_rate (float): Probability of laughter or chuckles.
        style_preset (ConversationalStyle): Overall conversational scene style.

    Returns:
        tuple[VocalStyle, str]: Selected vocal style and dialogue bank category.
    """

    # Roll probability die
    roll: float = random.random()

    # Check shouting probability
    if roll < shout_rate or style_preset == ConversationalStyle.ARGUMENT:
        return (VocalStyle.SHOUTING, "shouting")

    # Check laughter probability
    if roll < (shout_rate + laugh_rate) or style_preset == ConversationalStyle.PARTY:
        return (VocalStyle.LAUGHTER, "laughter")

    # Check assistant command probability in assistant mode
    if style_preset == ConversationalStyle.ASSISTANT and random.random() < 0.35:
        return (VocalStyle.NORMAL, "assistant")

    return (VocalStyle.NORMAL, "general")


def estimate_utterance_duration(text: str) -> float:
    """
    Estimate acoustic speech duration based on word count.

    Args:
        text (str): Spoken phrase.

    Returns:
        float: Estimated duration in seconds.
    """

    # Average speaking speed: ~0.32 seconds per word with punctuation pauses
    word_count: int = max(1, len(text.split()))
    duration: float = word_count * 0.32 + 0.3
    return duration


def calculate_turn_timing(
    prev_end_s: float,
    overlap_rate: float,
) -> tuple[float, bool]:
    """
    Determine start timestamp for the next turn, checking for an interruption.

    Args:
        prev_end_s (float): Termination timestamp of preceding turn.
        overlap_rate (float): Probability of cutting into the preceding turn.

    Returns:
        tuple[float, bool]: Pair of (start_timestamp, is_interruption).
    """

    # Interruption cuts into ongoing speech
    if random.random() < overlap_rate and prev_end_s > 1.0:
        cut_duration_s: float = random.uniform(0.3, 1.0)
        start_time_s: float = max(0.4, prev_end_s - cut_duration_s)
        return (start_time_s, True)

    # Normal turn-taking gap
    pause_s: float = random.uniform(0.2, 0.7)
    return (prev_end_s + pause_s, False)


def create_dialogue_turn(
    speaker: PersonaConfig,
    start_time_s: float,
    vocal_style: VocalStyle,
    category: str,
) -> tuple[UtteranceConfig, float]:
    """
    Sample utterance and compute its termination timestamp.

    Args:
        speaker (PersonaConfig): Speaker uttering the phrase.
        start_time_s (float): Start timestamp in seconds.
        vocal_style (VocalStyle): Emotional vocal style.
        category (str): Dialogue intent category.

    Returns:
        tuple[UtteranceConfig, float]: Configured utterance and end timestamp.
    """

    # Sample text and calculate duration
    text: str = sample_dialogue_text(language=speaker.language, category=category)
    duration_s: float = estimate_utterance_duration(text)

    utterance: UtteranceConfig = UtteranceConfig(
        text=text,
        start_time_s=round(start_time_s, 2),
        language=speaker.language,
        vocal_style=vocal_style,
    )

    return (utterance, start_time_s + duration_s)


def execute_single_turn(
    group: list[PersonaConfig],
    prev_end_s: float,
    last_speaker_id: str | None,
    overlap_rate: float,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> tuple[float, str]:
    """
    Execute a single turn cycle for a conversational group.

    Args:
        group (list[PersonaConfig]): Group speakers.
        prev_end_s (float): Preceding turn end timestamp.
        last_speaker_id (str | None): Identifier of previous turn speaker.
        overlap_rate (float): Overlap probability.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Style preset.

    Returns:
        tuple[float, str]: New turn end timestamp and speaker identifier.
    """

    # Select speaker
    candidates: list[PersonaConfig] = [p for p in group if p.id != last_speaker_id]
    speaker: PersonaConfig = random.choice(candidates if candidates else group)

    # Style and timing selection
    style, cat = select_vocal_style_and_category(shout_rate, laugh_rate, style_preset)
    start_t, is_cut = calculate_turn_timing(prev_end_s, overlap_rate)

    if is_cut and style == VocalStyle.NORMAL:
        style = VocalStyle.INTERRUPTION
        cat = "interruption"

    utt, end_t = create_dialogue_turn(speaker, start_t, style, cat)
    speaker.utterances.append(utt)

    return (end_t, speaker.id)


def generate_group_timeline(
    group: list[PersonaConfig],
    max_duration_s: float,
    overlap_rate: float,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> None:
    """
    Generate turn-taking dialogue exchanges for a single conversational group.

    Args:
        group (list[PersonaConfig]): Personas participating in this conversation group.
        max_duration_s (float): Termination boundary for dialogue generation in seconds.
        overlap_rate (float): Probability of interruptions and speech overlaps.
        shout_rate (float): Probability of shouting segments.
        laugh_rate (float): Probability of laughing segments.
        style_preset (ConversationalStyle): Preset scenario style.
    """

    # Track current group clock timestamp and previous speaker
    prev_end_s: float = random.uniform(0.3, 1.2)
    last_speaker_id: str | None = None

    # Generate sequential turns until sequence boundary is reached
    while prev_end_s < (max_duration_s - 2.0):
        prev_end_s, last_speaker_id = execute_single_turn(
            group=group,
            prev_end_s=prev_end_s,
            last_speaker_id=last_speaker_id,
            overlap_rate=overlap_rate,
            shout_rate=shout_rate,
            laugh_rate=laugh_rate,
            style_preset=style_preset,
        )


def generate_conversations_for_personas(
    personas: list[PersonaConfig],
    duration_s: float,
    overlap_rate: float = 0.35,
    shout_rate: float = 0.15,
    laugh_rate: float = 0.20,
    style_preset: ConversationalStyle = ConversationalStyle.MIXED,
) -> None:
    """
    Procedurally generate parallel conversations, interruptions, and styles for personas.

    Args:
        personas (list[PersonaConfig]): Personas to populate with utterances.
        duration_s (float): Desired audio duration in seconds.
        overlap_rate (float): Interruption probability.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational context.
    """

    # Clear any preexisting utterances
    for persona in personas:
        persona.utterances.clear()

    # Partition personas into conversational cliques
    groups: list[list[PersonaConfig]] = partition_conversational_groups(personas)

    # Generate timeline turns for each conversational group in parallel
    for group in groups:
        generate_group_timeline(
            group=group,
            max_duration_s=duration_s,
            overlap_rate=overlap_rate,
            shout_rate=shout_rate,
            laugh_rate=laugh_rate,
            style_preset=style_preset,
        )
