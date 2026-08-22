import re
import unicodedata


class TextCleaner:
    """
    Cleans text extraction artifacts from legal PDFs while strictly
    preserving legal language, section numbers, titles, and definitions verbatim.
    """

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Clean PDF extraction artifacts while strictly preserving all legal terminology and numbering.
        """
        if not raw_text:
            return ""

        # Normalize unicode (NFKC to resolve weird ligatures like 'fi', 'fl', fractions, etc.)
        text = unicodedata.normalize("NFKC", raw_text)

        # Replace non-breaking spaces and zero-width spaces with standard space
        text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

        # Normalize diverse quotes and dashes to standard ASCII without changing words
        text = re.sub(r'[\u2018\u2019\u201a\u201b]', "'", text)
        text = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', text)
        text = re.sub(r'[\u2013\u2014]', "-", text)

        # Remove null characters and control chars except standard newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Normalize trailing whitespace on each line
        lines = [line.strip() for line in text.splitlines()]

        cleaned_lines = []
        for line in lines:
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            # Collapse multiple spaces and tabs into single space within the line
            cleaned_line = re.sub(r'[ \t]+', ' ', line)
            cleaned_lines.append(cleaned_line)

        # Join lines preserving logical paragraph separations
        result = "\n".join(cleaned_lines).strip()

        # Collapse 3 or more consecutive newlines into double newlines
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result
