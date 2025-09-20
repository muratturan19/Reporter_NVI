from collections import OrderedDict

import httpx
import pytest
from langchain_core.messages import AIMessage

from provider_manager import TavilySearchProvider
from researcher_agent import ResearcherAgent


@pytest.fixture
def anyio_backend() -> str:  # pragma: no cover - test konfigürasyonu
    return "asyncio"


@pytest.mark.anyio
async def test_tavily_search_normalizes_invalid_topic(monkeypatch):
    provider = TavilySearchProvider()
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured_payloads = []

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:  # pragma: no cover - basit stub
            return None

        def json(self) -> dict:
            return {"results": [], "answer": "mock"}

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured_payloads.append(json)
            topic = json.get("topic")
            if topic not in TavilySearchProvider.VALID_TOPICS:
                request = httpx.Request("POST", url)
                response = httpx.Response(400, request=request)
                raise httpx.HTTPStatusError("Bad Request", request=request, response=response)
            return DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    result = await provider.search("example query", topic="🚨 Breaking", max_results=3)

    assert result.error is None
    assert captured_payloads
    assert captured_payloads[0]["topic"] == "general"


@pytest.mark.anyio
async def test_researcher_agent_maps_topic_to_valid_choice():
    class DummyLLM:
        async def ainvoke(self, *args, **kwargs):  # pragma: no cover - test double
            return AIMessage(content="")

    class DummySearchTool:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, args):
            self.calls.append(args)
            return "ok"

    dummy_tool = DummySearchTool()
    agent = ResearcherAgent(DummyLLM(), dummy_tool)

    plan = OrderedDict(
        [
            (
                "foundation",
                {"title": "Foundation", "queries": ["sample query"]},
            )
        ]
    )

    results, tool_messages = await agent._execute_query_plan("Finans🔥", plan, variant="initial")

    assert dummy_tool.calls
    assert dummy_tool.calls[0]["topic"] == "general"
    assert dummy_tool.calls[0]["queries"] == ["sample query"]

    assert "foundation" in results
    assert results["foundation"][0]["result"] == "ok"
    assert tool_messages
