"""
Multilingual conversational dialogue repository and utterance sampling.

Provides categorized speech corpora across English, French, Spanish, and German,
supporting natural chit-chat, smart assistant requests, interruptions, shouting,
and laughter interjections.
"""

# Import Modules
import random

DIALOGUE_BANK: dict[str, dict[str, list[str]]] = {
    "en": {
        "general": [
            "I was thinking about the project we discussed yesterday.",
            "The weather has been unusually warm for this time of year.",
            "Have you seen the latest updates on the new model architecture?",
            "I am planning to cook a special dinner this weekend.",
            "We definitely need to schedule another brainstorming session soon.",
            "That sounds like a very promising direction to explore.",
            "I read an interesting article about spatial acoustics earlier.",
            "Let us review the experimental data before making a final decision.",
        ],
        "assistant": [
            "Alexa, what is the traffic like towards the airport?",
            "Echo, set a timer for fifteen minutes please.",
            "Alexa, play some relaxing acoustic jazz in the living room.",
            "Echo, can you add milk and olive oil to my shopping list?",
            "Alexa, what is the temperature in the backyard right now?",
            "Echo, turn off the kitchen lights and lock the front door.",
        ],
        "shouting": [
            "Hey, watch out for that cup on the counter!",
            "Can someone turn down the volume over there?",
            "No way, you cannot be serious about that!",
            "Hurry up or we are going to miss the train!",
            "Listen to me, this is really important!",
        ],
        "laughter": [
            "Haha, that is honestly the funniest thing I have heard all week!",
            "Hahaha, I cannot believe you actually did that!",
            "Haha, oh wow, that caught me completely off guard!",
            "Hehe, that explains everything, doesn't it?",
        ],
        "interruption": [
            "Wait, hold on a second, let me just add something.",
            "Sorry to cut in, but that is not quite right.",
            "Hold on, let me stop you right there for a moment.",
            "Actually, before you continue, can I clarify one point?",
        ],
    },
    "fr": {
        "general": [
            "Je pensais justement au projet dont nous avons parlé hier.",
            "Le temps est particulièrement agréable aujourd'hui.",
            "Tu as vu les derniers résultats de l'expérience acoustique ?",
            "Je vais préparer un bon café avant de commencer la réunion.",
            "Nous devrions organiser une session de travail collectif.",
            "C'est une excellente idée qui mérite d'être développée.",
            "J'ai lu une étude très intéressante sur la séparation des voix.",
            "Prenons le temps de vérifier tous les paramètres du modèle.",
        ],
        "assistant": [
            "Alexa, donne-moi les prévisions météo pour cet après-midi.",
            "Echo, mets un réveil pour demain matin à sept heures.",
            "Alexa, mets un peu de musique classique s'il te plaît.",
            "Echo, ajoute du pain et du beurre à ma liste de courses.",
            "Alexa, quelle est la distance entre Paris et Lyon ?",
            "Echo, éteins les lumières du salon.",
        ],
        "shouting": [
            "Attention à la tasse sur la table !",
            "Baissez un peu la musique s'il vous plaît !",
            "Mais non, ce n'est absolument pas ce que j'ai dit !",
            "Dépêchez-vous, on va être en retard pour le train !",
            "Écoutez-moi deux minutes, c'est primordial !",
        ],
        "laughter": [
            "Haha, c'est tellement drôle, je ne m'y attendais pas du tout !",
            "Hahaha, non mais c'est pas possible d'avoir fait ça !",
            "Haha, tu m'as vraiment fait rire avec cette histoire !",
            "Hihi, ça explique beaucoup de choses en fait !",
        ],
        "interruption": [
            "Attends, laisse-moi t'interrompre une seconde !",
            "Pardon mais je ne suis pas d'accord avec ce point.",
            "Attends deux minutes, écoute ce que je voulais dire.",
            "En fait, avant que tu finisses, je voulais préciser un détail.",
        ],
    },
    "es": {
        "general": [
            "Estaba pensando en la propuesta que comentamos ayer.",
            "El clima está bastante agradable esta tarde.",
            "¿Has revisado los últimos datos del experimento acústico?",
            "Voy a preparar un café antes de seguir con el trabajo.",
            "Deberíamos organizar una reunión para coordinar las tareas.",
            "Me parece una solución muy interesante para este problema.",
        ],
        "assistant": [
            "Alexa, ¿qué tiempo va a hacer mañana por la mañana?",
            "Echo, pon una alarma para las ocho en punto.",
            "Alexa, pon algo de música tranquila en la sala.",
            "Echo, añade café y manzanas a la lista de la compra.",
        ],
        "shouting": [
            "¡Cuidado con el vaso que está en el borde!",
            "¡Por favor, bajen un poco el volumen!",
            "¡No me lo puedo creer, estás bromeando!",
            "¡Dense prisa que vamos a llegar tarde!",
        ],
        "laughter": [
            "¡Jajaja, qué divertido, no me lo esperaba para nada!",
            "¡Jajaja, es que no tiene ningún sentido!",
            "¡Jaja, me hiciste reír mucho con ese comentario!",
        ],
        "interruption": [
            "¡Espera un segundo, déjame decir una cosa!",
            "Perdona que te interrumpa, pero no es exactamente así.",
            "¡Oye, espera, déjame terminar esta idea primero!",
        ],
    },
    "de": {
        "general": [
            "Ich habe über das Projekt nachgedacht, das wir gestern besprochen haben.",
            "Das Wetter ist heute überraschend sonnig.",
            "Hast du die neuen Ergebnisse der akustischen Simulation gesehen?",
            "Wir sollten bald ein gemeinsames Treffen dazu vereinbaren.",
        ],
        "assistant": [
            "Alexa, wie wird das Wetter morgen Vormittag?",
            "Echo, stelle einen Wecker auf sieben Uhr morgens.",
            "Alexa, spiele bitte entspannende Klaviermusik im Wohnzimmer.",
        ],
        "shouting": [
            "Achtung, pass auf die Tasse auf dem Tisch auf!",
            "Macht bitte die Musik ein bisschen leiser!",
            "Das kann doch jetzt wirklich nicht wahr sein!",
        ],
        "laughter": [
            "Hahaha, das ist wirklich unglaublich witzig!",
            "Haha, das habe ich jetzt absolut nicht erwartet!",
        ],
        "interruption": [
            "Warte mal kurz, lass mich das kurz klarstellen!",
            "Entschuldigung, aber da muss ich kurz einhaken!",
        ],
    },
}


def get_supported_languages() -> list[str]:
    """
    Retrieve list of language codes available in the dialogue repository.

    Returns:
        list[str]: Language codes (e.g. ['en', 'fr', 'es', 'de']).
    """

    return sorted(list(DIALOGUE_BANK.keys()))


def sample_dialogue_text(
    language: str,
    category: str = "general",
) -> str:
    """
    Sample a random utterance text for a given language and dialogue category.

    Args:
        language (str): Two-letter language code ('en', 'fr', 'es', 'de').
        category (str): Category ('general', 'assistant', 'shouting', 'laughter', 'interruption').

    Returns:
        str: Selected utterance string.
    """

    # Fallback to English if language not supported
    lang_bank: dict[str, list[str]] = DIALOGUE_BANK.get(language, DIALOGUE_BANK["en"])

    # Fallback to general category if requested category missing
    utterances: list[str] = lang_bank.get(category, lang_bank["general"])

    # Return random selection
    return random.choice(utterances)
