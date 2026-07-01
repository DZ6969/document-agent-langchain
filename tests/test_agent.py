"""The agent graph is built with a dummy key — no network call is made."""


def test_agent_graph_builds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-for-tests")
    from doc_agent.agent import build_agent

    agent = build_agent("Invoice total: 100 MXN")
    nodes = agent.get_graph().nodes
    assert "model" in nodes
    assert "tools" in nodes
