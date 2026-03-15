import re
import inflect

p = inflect.engine()


roman_map = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000
}

def roman_to_int(s):

    s = s.upper()

    total = 0
    prev = 0

    for char in reversed(s):

        if char not in roman_map:
            return s.lower()

        val = roman_map[char]

        if val < prev:
            total -= val
        else:
            total += val

        prev = val

    return total

def expand_roman(text):

    pattern = r"\b[IVXLCDMivxlcdm]+\b"

    def repl(match):

        roman = match.group()

        value = roman_to_int(roman)

        if isinstance(value, int):
            return p.number_to_words(value)

        return roman.lower()

    return re.sub(pattern, repl, text)

def expand_clause(text, keep_as_number = True):

    pattern = r"(\d+)\(([\dA-Za-z]+)\)"

    def repl(match):

        left = int(match.group(1))
        if not keep_as_number:
          left = p.number_to_words(left)
        else:
          left = str(left)

        right = match.group(2)

        if right.isdigit():
            right = int(right)

            if not keep_as_number:
                right = p.number_to_words(right)
            else:
                right  = str(right)

        else:
            right = " ".join(list(right.lower()))

        return f"{left} {right}"

    return re.sub(pattern, repl, text)

def year_to_words(year):

    year = int(year)

    if 2010 <= year <= 2099:
        return "twenty " + p.number_to_words(year % 100)

    if 2000 <= year < 2010:
        remainder = year % 100
        if remainder == 0:
            return "two thousand"
        return "two thousand " + p.number_to_words(remainder)

    if 1900 <= year < 2000:
        remainder = year % 100
        if remainder == 0:
            return "nineteen hundred"
        return "nineteen " + p.number_to_words(remainder)

    return p.number_to_words(year)

def expand_years(text):

    pattern = r"\b(19\d\d|20\d\d)\b"

    def repl(match):
        return year_to_words(match.group())

    return re.sub(pattern, repl, text)

def expand_numbers(text):

    pattern = r"\b\d+\b"

    def repl(match):
        return p.number_to_words(int(match.group()))

    return re.sub(pattern, repl, text)

def expand_percentages(text):

    pattern = r"(\d+)\s*%"

    def repl(match):
        number = int(match.group(1))
        return f"{p.number_to_words(number)} percent"

    return re.sub(pattern, repl, text)

def normalize_text(text):

    text = expand_clause(text)
    text = expand_percentages(text)
    text = expand_roman(text)
    text = expand_years(text)
    text = expand_numbers(text)
    text = remove_ellipses(text)
    text = expand_abbreviations(text)

    text = text.lower()

    text = re.sub(r"-", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def expand_abbreviations(text):

    pattern = r"\b[A-Z]{2,}\b"

    def repl(match):
        word = match.group()
        return " ".join(list(word.lower()))

    return re.sub(pattern, repl, text)

def remove_ellipses(text):

    text = re.sub(r"\.{2,}", " ", text)

    return text


if __name__ == "__main__":
    
    examples = [
        "Section 2(19) of the Act of 2018",
        "11(6A) of the Arbitration Act",
        "Volume V page 50",
        "Judgment delivered in 2008",
        "Case filed in 2012",
        "99% of all time"
        "The court has to look into ABC case.."
    ]

    for e in examples:
        print(e)
        print(normalize_text(e))
        print()