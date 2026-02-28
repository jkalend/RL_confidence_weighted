"""Prompts for entity extraction with Few-Shot CoT."""

ENTITY_EXTRACTION_SYSTEM = """You are an expert at Named Entity Recognition. Extract all named entities from the given text. For each entity, provide its text span and fine-grained type (e.g., person-actor, location-city, organization-company). Output ONLY a JSON array, no other text."""

ENTITY_EXTRACTION_FEW_SHOT = """Examples:

Input: "Apple Inc. was founded by Steve Jobs in Cupertino."
Output: [{"text": "Apple Inc.", "type": "organization-company"}, {"text": "Steve Jobs", "type": "person-founder"}, {"text": "Cupertino", "type": "location-city"}]

Input: "The conference will be held at MIT in Boston next March."
Output: [{"text": "MIT", "type": "organization-university"}, {"text": "Boston", "type": "location-city"}, {"text": "March", "type": "date-month"}]
"""


def build_entity_extraction_prompt(sentence: str, use_cot: bool = True) -> str:
    """Build Few-Shot CoT prompt for entity extraction."""
    cot_instruction = (
        "First, reason about the context and entity types. "
        if use_cot
        else ""
    )
    return f"""{ENTITY_EXTRACTION_SYSTEM}
{ENTITY_EXTRACTION_FEW_SHOT}

Input: "{sentence}"
{cot_instruction}Output:"""
