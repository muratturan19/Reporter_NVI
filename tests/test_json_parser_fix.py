from json_parser_fix import parse_json_from_response


def test_parse_json_with_unescaped_quotes_in_value():
    response = '''
    {
        "query": "Detailed insights on "Industry 4.0" adoption?",
        "region": "Global"
    }
    '''

    parsed = parse_json_from_response(response)

    assert parsed["query"] == 'Detailed insights on "Industry 4.0" adoption?'
    assert parsed["region"] == "Global"
