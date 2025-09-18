# -*- coding: utf-8 -*-
"""Debug test - planlama aşamasını test eder"""

import asyncio
from main_report_agent import MainReportAgent


async def debug_planning():
    """Sadece planlama aşamasını test et"""
    print("🔍 Debug test başlatılıyor...")

    agent = MainReportAgent()

    # Test planlama
    topic = "Yapay zeka ajanlarının üretim firmalarında kullanım alanları"

    messages = agent.planner_prompt.format_messages(
        topic=topic,
        research_data="Test araştırma verisi"
    )

    print("📝 Gönderilen prompt:")
    for msg in messages:
        print(f"Role: {msg.__class__.__name__}")
        print(f"Content: {msg.content}")
        print("-" * 50)

    try:
        response = await agent.llm.ainvoke(messages)
        print("🤖 Model yanıtı:")
        print(response.content)
        print("=" * 70)

        # Parser test
        from json_parser_fix import parse_json_from_response

        result = parse_json_from_response(response.content)
        print("✅ Parse edilmiş JSON:")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_planning())
