"""
Общие утилиты: нормализация текста, стемминг, разбор фильтров и параметров.

Используем PyStemmer (Snowball-стеммер для русского, open-source, BSD) —
он быстрый (C-реализация) и не требует скачивания моделей.
"""
import re
from functools import lru_cache

import Stemmer

_STEM_RU = Stemmer.Stemmer("russian")
_STEM_EN = Stemmer.Stemmer("english")

# Токены: последовательности букв/цифр. Дефисы и точки разбивают слова
# («видео-съёмка» -> «видео», «съемка»).
_TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
_LAT_RE = re.compile(r"[a-z]")

# Короткие частотные слова, которые не несут смысла для поиска услуг.
STOPWORDS = set(
    """
    и в во на по с со к ко о об от до за из у для при про без над под а но или
    же ли бы то не ни это как так что все всё вы мы я он она они их его ее её
    наш ваш свой мой ваша наша также очень где когда который которые
    """.split()
)


def normalize(text: str) -> str:
    """Нижний регистр, ё->е, всё кроме букв/цифр превращаем в пробелы."""
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е")


@lru_cache(maxsize=2_000_000)
def stem_word(w: str) -> str:
    """Стемминг одного слова. Латиница — английским стеммером, кириллица — русским."""
    if _LAT_RE.search(w):
        return _STEM_EN.stemWord(w)
    return _STEM_RU.stemWord(w)


def tokenize(text: str, stem: bool = True, drop_stop: bool = True) -> list:
    toks = _TOKEN_RE.findall(normalize(text))
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [stem_word(t) for t in toks]
    return toks


def stem_text(text: str, max_chars: int | None = None) -> str:
    """Текст -> строка стемов через пробел (удобно для sklearn-векторайзеров)."""
    if max_chars is not None and isinstance(text, str):
        text = text[:max_chars]
    return " ".join(tokenize(text))


# ---------------------------------------------------------------------------
# Разбор фильтров поиска (search_infm_params_text).
# Фильтр — это склейка пар «ключ значение» без разделителей, например:
#   «Тип услуги Маникюр, педикюр Вид услуги Красота, здоровье».
# Нас интересуют ключи «Вид услуги», «Тип услуги», «Тип услуги автосервиса».
# В параметрах объявления они записаны в том же виде, поэтому значение
# фильтра можно проверить подстрокой «<ключ> <значение>» в item_infm_params_text.
# Чтобы корректно вырезать значение, перечисляем известные ключи фильтров
# (ключ заканчивает значение предыдущего ключа).
# ---------------------------------------------------------------------------
FILTER_KEYS = [
    "Тип услуги автосервиса", "Тип автосервиса", "Вид услуги", "Тип услуги",
    "Рейтинг пользователя", "Кто оказывает услуги", "Как вы работаете",
    "Где вы оказываете услуги", "Срочная услуга (мультистатус)", "Сортировка для URL",
    "Онлайн-запись", "Поиск по слотам", "Услуга", "Услуги", "Направление",
    "Чем вы занимаетесь", "Предмет или специальность", "Марка авто",
    "Опыт работы", "Гарантия", "Работаете с юрлицами и ИП", "Исполнителей в команде",
    "Тип помещения", "Выезд за город", "Где снимаете", "Сколько человек может участвовать",
]
_KEYS_ALT = "|".join(re.escape(k) for k in sorted(FILTER_KEYS, key=len, reverse=True))
_FILTER_RE = re.compile(
    rf"({_KEYS_ALT})(?:\s+(?!(?:{_KEYS_ALT})(?:\s|$))(.*?))?(?=\s+(?:{_KEYS_ALT})(?:\s|$)|$)"
)


def parse_filters(s: str) -> dict:
    """Возвращает {ключ: [значения]} для строки фильтров."""
    out = {}
    if not isinstance(s, str) or not s.strip():
        return out
    for k, v in _FILTER_RE.findall(s):
        v = (v or "").strip()
        if v:
            out.setdefault(k, []).append(v)
    return out


# Ключи, по которым фильтр почти всегда совпадает с параметрами объявления
# (проверено на train: 93–99% совпадений).
STRICT_FILTER_KEYS = ("Вид услуги", "Тип услуги", "Тип услуги автосервиса")


def filter_constraints(s: str) -> list:
    """Список подстрок, которые должны встречаться в параметрах объявления."""
    f = parse_filters(s)
    cons = []
    for k in STRICT_FILTER_KEYS:
        for v in f.get(k, []):
            cons.append(f"{k} {v}")
    return cons
