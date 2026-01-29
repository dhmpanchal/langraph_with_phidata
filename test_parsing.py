from medical_agent import parser, DiagnosisExtraction
import re

def test_parsing():
    mock_responses = [
        # Perfect JSON
        '{"major_conditions": [{"key": "Diabetes", "value": "Type 2", "start_date": "01-01-2020", "end_date": "ongoing", "status": "ongoing"}]}',
        # JSON with markdown
        '```json\n{"major_conditions": [{"key": "Hypertension", "value": "High blood pressure", "start_date": "05-12-2018", "end_date": "ongoing", "status": "ongoing"}]}\n```',
        # JSON with extra text
        'Based on the records: {"major_conditions": [{"key": "IBS", "value": "Stomach issues", "start_date": "10-10-2021", "end_date": "ongoing", "status": "ongoing"}]} - End of report.'
    ]

    for i, resp in enumerate(mock_responses):
        print(f"\nTesting Response {i+1}:")
        try:
            # Replicating logic from run_medical_agent
            try:
                parsed_json = parser.parse(resp)
            except Exception as e:
                print(f"Direct parse failed, trying regex: {e}")
                json_match = re.search(r'\{.*\}', resp, re.DOTALL)
                if json_match:
                    parsed_json = parser.parse(json_match.group())
                else:
                    raise e
            
            result = DiagnosisExtraction(**parsed_json)
            print(f"Successfully parsed: {result}")
        except Exception as e:
            print(f"Failed to parse: {e}")

if __name__ == "__main__":
    test_parsing()
