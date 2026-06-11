"""L1 helper: capture functional behavior baselines for the regex modules
BEFORE relocating their RU/UK data into the lang packs, so the parity test can
assert the public behavior is unchanged after relocation."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "_regex_parity_baseline.json"

NAME_PROBES = [
    "Меня зовут Алексей",  # Меня зовут Алексей
    "мене звати Олег",                     # мене звати Олег
    "My name is Bob",
    "I am Alice",
    "I'm Carol",
    "user name is Dave",
    "Я Максим",                                                         # Я Максим
    "random text no name here",
    "живёт в Париже",                           # живёт в Париже
]
SELF_PROBES = ["я тут", "меня", "мене",
               "мені", "мной", "I am", "my code", "random"]

QSPLIT_PROBES = [
    "какой порт и почему мы выбрали его",  # какой порт и почему мы выбрали его
    "X потому что Y",                                            # X потому что Y
    "foo а также bar",                                                          # foo а также bar
    "what port and why did we pick it",
    "single question only",
]

CSTATE_PROBES = [
    # current-state (RU/UK/EN, positive)
    "сейчас живу в Киеве",
    "теперь живу в Берлине",
    "переехал в Берлин",
    "переехала в Прагу",
    "зараз живу в Києві",
    "сейчас работаю в Anthropic",
    "теперь работает в Google",
    "I now live in Tampa",
    "I currently live in Boston",
    "user currently works at Acme",
    "my current city is Madrid",
    "I just moved to Denver",
    # negation / unknown (user-subject)
    "I no longer live in Warsaw",
    "я больше не живу в Варшаве",
    "the user does not currently live in Warsaw; current city is unknown.",
    "I don't work at Google anymore",
    "я уже не работаю в Яндексе",
    "my current employer is unknown",
    "мой текущий город неизвестен",
    # third-party — must NOT fire (the A1 correctness cases)
    "Alice relocated to Berlin",
    "Alice no longer lives in Paris",
    "my sister doesn't work at Google anymore",
    "Алиса больше не живёт в Париже",
    # subject-cue probes (broad has_user_subject_cue)
    "I learned that Alice moved",
    "пользователь переехал",
    "we use pnpm not npm",
    "the deployment runs on port 5432",
]


def cap_user_names():
    import pmb.reasoning.user_names as U
    return {
        "detect": {p: sorted(U.detect_user_names([p])) for p in NAME_PROBES},
        "looks": {p: U.looks_like_name_statement(p) for p in NAME_PROBES},
        "self_re": {p: bool(U._SELF_RE.search(p)) for p in SELF_PROBES},
        "self_markers": sorted(U.SELF_MARKERS),
    }


def cap_query_split():
    import pmb.reasoning.query_split as Q
    fn = getattr(Q, "split_compound_query", None) or getattr(Q, "split_query", None)
    res = {}
    if fn:
        for p in QSPLIT_PROBES:
            try:
                res[p] = fn(p)
            except Exception as e:
                res[p] = f"ERR:{e}"
    return res


def cap_attributes():
    import pmb.reasoning.attributes as A
    return {
        "detect_current_state": {p: A.detect_current_state(p) for p in CSTATE_PROBES},
        "detect_negated_state": {p: A.detect_negated_state(p) for p in CSTATE_PROBES},
        "has_user_subject_cue": {p: A.has_user_subject_cue(p) for p in CSTATE_PROBES},
    }


INTENT_PROBES = [
    # RU
    "когда я последний раз был в спортзале", "что я делал вчера",
    "почему мы выбрали Postgres", "какой у меня план", "где я записал пароль",
    "кто такой Алекс", "что мы только что обсуждали", "что мы сейчас делаем",
    "какие у меня цели", "мои задачи", "что осталось доделать",
    "какие правила проекта", "какие есть уроки", "исправь баг в auth",
    "отрефактори модуль", "привет", "спасибо", "ок",
    # UK
    "коли я це робив", "що я зробив", "які у мене цілі", "які правила",
    "що ми щойно обговорювали", "виправ помилку", "дякую", "привіт",
    # EN
    "what did I do yesterday", "why did we choose Postgres", "who is Alice",
    "what are we doing right now", "my open goals", "what's left to do",
    "do we have a rule about commits", "refactor the auth module", "hi", "thanks",
    "the api runs on port 5432", "I live in Tampa",
]


def cap_intents():
    from pmb.hooks.auto_recall import detect_intents
    return {p: detect_intents(p, known_projects=set()) for p in INTENT_PROBES}


snap = {
    "user_names": cap_user_names(),
    "query_split": cap_query_split(),
    "attributes": cap_attributes(),
    "intents": cap_intents(),
}
OUT.write_text(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
print("captured ->", OUT)
for k, v in snap.items():
    print(" ", k, "keys:", list(v.keys()))
