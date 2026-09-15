"""
Procedural conversation timeline generator simulating multi-person interactions.

Coordinates parallel conversational groups, natural turn-taking dynamics,
dialogue interruptions and overlaps, and dynamic emotional vocal styles.
"""

# Import Modules
import random

from src.llm.llm_types import GeneratedTurn
from src.llm.dialogue_generator import LLMDialogueGenerator
from src.procedural.dialogue_bank import sample_dialogue_text
from src.procedural.ambiance_presets import AmbiancePreset
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


def generate_fallback_turns(
    group: list[PersonaConfig],
    target_turns: int,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> list[GeneratedTurn]:
    """
    Generate conversational turns from offline dialogue bank matching vocal style probabilities.

    Args:
        group (list[PersonaConfig]): Personas in the group.
        target_turns (int): Target number of turns.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational style preset.

    Returns:
        list[GeneratedTurn]: Synthesized turns list.
    """

    turns: list[GeneratedTurn] = []
    last_speaker_id: str | None = None

    for _ in range(target_turns):
        candidates: list[PersonaConfig] = [p for p in group if p.id != last_speaker_id]
        speaker: PersonaConfig = random.choice(candidates if candidates else group)
        last_speaker_id = speaker.id

        style, cat = select_vocal_style_and_category(shout_rate, laugh_rate, style_preset)
        text: str = sample_dialogue_text(language=speaker.language, category=cat)
        turns.append(GeneratedTurn(speaker_id=speaker.id, text=text, vocal_style=style.value))

    return turns


def sequence_llm_turns_on_timeline(
    group: list[PersonaConfig],
    turns: list[GeneratedTurn],
    group_id: int = 0,
    max_duration_s: float | None = None,
    overlap_rate: float = 0.35,
) -> None:
    """
    Chronologically sequence dialogue turns onto the group timeline.

    Args:
        group (list[PersonaConfig]): Group personas participating in dialogue.
        turns (list[GeneratedTurn]): List of dialogue turns.
        group_id (int): Conversation clique identifier.
        max_duration_s (float | None): Optional timeline boundary limit in seconds.
        overlap_rate (float): Interruption and speech overlap probability.
    """

    persona_map: dict[str, PersonaConfig] = {p.id: p for p in group}
    prev_end_s: float = random.uniform(0.3, 1.0)

    for turn_idx, turn in enumerate(turns):
        if max_duration_s is not None and prev_end_s >= (max_duration_s - 1.5):
            break

        speaker: PersonaConfig = persona_map.get(turn.speaker_id, group[0])
        start_t, is_cut = calculate_turn_timing(prev_end_s, overlap_rate)

        # Parse vocal style
        style_enum: VocalStyle = VocalStyle.NORMAL
        if turn.vocal_style == "shouting":
            style_enum = VocalStyle.SHOUTING
        elif turn.vocal_style == "laughter":
            style_enum = VocalStyle.LAUGHTER
        elif turn.vocal_style == "interruption" or is_cut:
            style_enum = VocalStyle.INTERRUPTION

        dur_s: float = estimate_utterance_duration(turn.text)
        utterance: UtteranceConfig = UtteranceConfig(
            text=turn.text,
            start_time_s=round(start_t, 2),
            language=speaker.language,
            vocal_style=style_enum,
            group_id=group_id,
            turn_order=turn_idx,
        )
        speaker.utterances.append(utterance)
        prev_end_s = start_t + dur_s


def generate_group_timeline(
    group: list[PersonaConfig],
    target_turns: int,
    group_id: int,
    overlap_rate: float,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
    llm_generator: LLMDialogueGenerator | None = None,
    ambiance: AmbiancePreset | None = None,
    constraint_words: list[str] | None = None,
    max_duration_s: float | None = None,
) -> None:
    """
    Generate turn-taking dialogue exchanges for a single conversational group.

    Args:
        group (list[PersonaConfig]): Personas participating in this conversation group.
        target_turns (int): Target number of dialogue turns.
        group_id (int): Conversation clique identifier.
        overlap_rate (float): Probability of interruptions and speech overlaps.
        shout_rate (float): Probability of shouting segments.
        laugh_rate (float): Probability of laughing segments.
        style_preset (ConversationalStyle): Preset scenario style.
        llm_generator (LLMDialogueGenerator | None): Optional LLM dialogue generator.
        ambiance (AmbiancePreset | None): Optional ambiance preset.
        constraint_words (list[str] | None): Optional thematic constraint keywords.
        max_duration_s (float | None): Optional boundary limit in seconds.
    """

    turns: list[GeneratedTurn] = []

    # If LLM generator is active and ambiance is set, attempt LLM dialogue generation
    if llm_generator is not None and ambiance is not None:
        turns = llm_generator.generate_group_dialogue(
            group=group,
            ambiance=ambiance,
            constraint_words=constraint_words if constraint_words is not None else [],
            target_turns_count=target_turns,
        )

    # Fallback to rule-based dialogue bank if turns empty
    if not turns:
        turns = generate_fallback_turns(
            group=group,
            target_turns=target_turns,
            shout_rate=shout_rate,
            laugh_rate=laugh_rate,
            style_preset=style_preset,
        )

    sequence_llm_turns_on_timeline(
        group=group,
        turns=turns,
        group_id=group_id,
        max_duration_s=max_duration_s,
        overlap_rate=overlap_rate,
    )


def _calculate_turns_per_group(
    min_sentences: int,
    num_groups: int,
    duration_s: float | None,
) -> int:
    """
    Calculate number of turns per group satisfying min_sentences and duration.

    Args:
        min_sentences (int): Minimum total sentences required.
        num_groups (int): Number of conversation groups.
        duration_s (float | None): Optional duration ceiling in seconds.

    Returns:
        int: Number of dialogue turns to generate per group.
    """

    turns_per_group: int = (min_sentences + num_groups - 1) // max(1, num_groups)
    if duration_s is not None:
        turns_from_dur: int = max(3, int(duration_s / 3.2))
        return max(turns_per_group, turns_from_dur)

    return max(5, turns_per_group)


def _ensure_minimum_sentences(
    personas: list[PersonaConfig],
    groups: list[list[PersonaConfig]],
    min_sentences: int,
    overlap_rate: float,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
    duration_s: float | None,
) -> None:
    """
    Append additional turns if total utterances fall below min_sentences threshold.

    Args:
        personas (list[PersonaConfig]): Complete roster of personas.
        groups (list[list[PersonaConfig]]): Active conversation groups.
        min_sentences (int): Minimum required sentence count.
        overlap_rate (float): Overlap probability.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational style preset.
        duration_s (float | None): Optional timeline limit.
    """

    total: int = sum(len(p.utterances) for p in personas)
    if total < min_sentences:
        deficit: int = min_sentences - total
        extra: list[GeneratedTurn] = generate_fallback_turns(
            group=groups[0],
            target_turns=deficit,
            shout_rate=shout_rate,
            laugh_rate=laugh_rate,
            style_preset=style_preset,
        )
        sequence_llm_turns_on_timeline(
            group=groups[0],
            turns=extra,
            group_id=0,
            max_duration_s=duration_s,
            overlap_rate=overlap_rate,
        )


def generate_conversations_for_personas(
    personas: list[PersonaConfig],
    duration_s: float | None = None,
    min_sentences: int = 100,
    overlap_rate: float = 0.35,
    shout_rate: float = 0.15,
    laugh_rate: float = 0.20,
    style_preset: ConversationalStyle = ConversationalStyle.MIXED,
    llm_generator: LLMDialogueGenerator | None = None,
    ambiance: AmbiancePreset | None = None,
    constraint_words: list[str] | None = None,
) -> None:
    """
    Procedurally generate parallel conversations, interruptions, and styles for personas.

    Args:
        personas (list[PersonaConfig]): Personas to populate with utterances.
        duration_s (float | None): Optional desired audio duration in seconds.
        min_sentences (int): Minimum required total sentences across all personas.
        overlap_rate (float): Interruption probability.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational context.
        llm_generator (LLMDialogueGenerator | None): Optional LLM dialogue generator.
        ambiance (AmbiancePreset | None): Optional ambiance preset.
        constraint_words (list[str] | None): Optional thematic constraint keywords.
    """

    # Clear any preexisting utterances
    for p in personas:
        p.utterances.clear()

    # Partition personas into conversational cliques
    groups: list[list[PersonaConfig]] = partition_conversational_groups(personas)
    turns: int = _calculate_turns_per_group(min_sentences, len(groups), duration_s)

    # Generate timeline turns for each conversational group in parallel
    for idx, grp in enumerate(groups):
        generate_group_timeline(
            group=grp,
            target_turns=turns,
            group_id=idx,
            overlap_rate=overlap_rate,
            shout_rate=shout_rate,
            laugh_rate=laugh_rate,
            style_preset=style_preset,
            llm_generator=llm_generator,
            ambiance=ambiance,
            constraint_words=constraint_words,
            max_duration_s=duration_s,
        )

    _ensure_minimum_sentences(
        personas=personas,
        groups=groups,
        min_sentences=min_sentences,
        overlap_rate=overlap_rate,
        shout_rate=shout_rate,
        laugh_rate=laugh_rate,
        style_preset=style_preset,
        duration_s=duration_s,
    )
