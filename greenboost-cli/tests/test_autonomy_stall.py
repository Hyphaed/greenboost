"""The stall brake stops circling, not a chain of cheap but new tool results."""
from greenboost_cli.core.autonomy import AutonomyState, STALL_MIN_CONTINUES


def _chain(st: AutonomyState, results):
    """Run STALL_MIN_CONTINUES+2 tiny continuations, each with one tool result."""
    st.note_chain_start()
    out = []
    for i in range(STALL_MIN_CONTINUES + 2):
        st.note_tool_result("Bash", results(i))
        st.note_output_tokens(60)
        out.append(st.decide_continue("I'll check the next one now."))
        if out[-1][0]:
            st.note_continue(out[-1][1])
    return out


def test_cheap_calls_with_new_results_keep_going():
    st = AutonomyState()
    assert all(go for go, _ in _chain(st, lambda i: f"clocks line {i}"))


def test_same_result_repeated_is_still_a_stall():
    st = AutonomyState()
    res = _chain(st, lambda i: "no files matched")
    assert not res[-1][0] and "going in circles" in res[-1][1]
