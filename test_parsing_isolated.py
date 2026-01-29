import re
import json
from pydantic import BaseModel, Field
from typing import List

# Mocking the Pydantic models from medical_agent.py
class MajorConditions(BaseModel):
    key: str
    value: str
    start_date: str
    end_date: str
    status: str

class DiagnosisExtraction(BaseModel):
    major_conditions: List[MajorConditions]

# Mocking the parser.parse behavior
def mock_parser_parse(text: str):
    # Strip markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)

def test_parsing_logic(last_message: str):
    print(f"\nTesting message: {last_message}")
    try:
        # Step 3 logic from run_medical_agent
        try:
            parsed_json = mock_parser_parse(last_message)
            return DiagnosisExtraction(**parsed_json)
        except Exception as e:
            print(f"Direct parse failed: {e}")
            # Attempt to find JSON in the string if direct parsing fails
            json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
            if json_match:
                try:
                    parsed_json = mock_parser_parse(json_match.group())
                    return DiagnosisExtraction(**parsed_json)
                except Exception as e2:
                    print(f"Regex match parse failed: {e2}")
            return "Failed to parse"
    except Exception as e:
        return f"Outer failure: {e}"

if __name__ == "__main__":
    test_cases = [
        '{"major_conditions": [{"key": "Diabetes", "value": "Type 2", "start_date": "01-01-2020", "end_date": "ongoing", "status": "ongoing"}]}',
        '```json\n{"major_conditions": [{"key": "Hypertension", "value": "High blood pressure", "start_date": "05-12-2018", "end_date": "ongoing", "status": "ongoing"}]}\n```',
        'Some intro text here... {"major_conditions": [{"key": "IBS", "value": "Stomach issues", "start_date": "10-10-2021", "end_date": "ongoing", "status": "ongoing"}]} some outro text.'
    ]

    for case in test_cases:
        result = test_parsing_logic(case)
        print(f"Result: {result}")
