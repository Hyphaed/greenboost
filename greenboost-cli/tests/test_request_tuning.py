from greenboost_cli.inference.adapters import DEFAULT_SAMPLING, apply_request_tuning, sampling_for


def _k(settings):
    kw = {}
    apply_request_tuning(kw, settings)
    return kw


def test_code_safe_sampling_is_sent_by_default():
    kw = _k({})
    assert kw["temperature"] == 0.6 and kw["top_p"] == 0.95
    eb = kw["extra_body"]
    assert eb["top_k"] == 20 and eb["min_p"] == 0.05
    assert eb["repeat_penalty"] == 1.0 and kw["presence_penalty"] == 0.0 and kw["frequency_penalty"] == 0.0
    assert eb["dry_multiplier"] == 0.8 and eb["dry_base"] == 1.75 and eb["dry_allowed_length"] == 2
    assert "dry_penalty_last_n" not in eb                      # server default = whole context
    assert eb["chat_template_kwargs"] == {"enable_thinking": False}


def test_user_sampling_overrides_key_by_key_and_false_disables():
    assert _k({"sampling": {"temperature": 0.2}})["temperature"] == 0.2
    assert _k({"sampling": {"temperature": 0.2}})["extra_body"]["top_k"] == 20
    kw = _k({"sampling": False})
    assert "temperature" not in kw and "top_k" not in kw["extra_body"]


def test_effort_only_applies_while_thinking_is_on():
    assert "reasoning_effort" not in _k({"reasoning_effort": "high"})["extra_body"]["chat_template_kwargs"]
    t = _k({"enable_thinking": True, "reasoning_effort": "high"})["extra_body"]["chat_template_kwargs"]
    assert t == {"reasoning_effort": "high"}
    assert "reasoning_effort" not in _k({"enable_thinking": True, "reasoning_effort": "bogus"})["extra_body"].get("chat_template_kwargs", {})


def test_preserve_thinking_only_sent_when_set_and_existing_extra_body_kept():
    assert "preserve_thinking" not in _k({})["extra_body"]["chat_template_kwargs"]
    eb = _k({"preserve_thinking": False, "extra_body": {"foo": 1}})["extra_body"]
    assert eb["foo"] == 1 and eb["chat_template_kwargs"]["preserve_thinking"] is False


def test_default_dict_is_not_mutated():
    before = dict(DEFAULT_SAMPLING)
    sampling_for({"sampling": {"temperature": 9}})
    _k({"sampling": {"top_k": 1}})
    assert DEFAULT_SAMPLING == before
