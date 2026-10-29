"""_gb_boot_param_presence must not report a param missing just because it
sits directly against the closing quote of a bash double-quoted assignment.

Observed live 2026-09-22: boot_tuning_persisted read 0.83 (5/6) on every
single check, permanently, not because any tuning had actually regressed,
but because 99-greenboost.cfg's GRUB_CMDLINE_LINUX_DEFAULT line ends with
`...workqueue.power_efficient=0"` , no trailing space before the closing
quote. The old boundary `(?!\\S)` treated that `"` as part of the token and
never matched. Real content on disk was 6/6; the metric reported 5/6.
"""
from __future__ import annotations

import gb_semantics as gs


_STATIC = ["transparent_hugepage=always", "skew_tick=1", "numa_balancing=disable",
           "workqueue.power_efficient=0"]
_RANGE = ['rcu_nocbs=[^ "]*', 'nohz_full=[^ "]*']


def test_last_param_before_closing_quote_counts_as_present():
    text = (
        'GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} '
        'transparent_hugepage=always skew_tick=1 rcu_nocbs=16-31 nohz_full=4-7 '
        'numa_balancing=disable workqueue.power_efficient=0"'
    )
    present, total, missing = gs._gb_boot_param_presence(text, _STATIC, _RANGE)
    assert missing == []
    assert present == total == 6


def test_param_immediately_after_opening_quote_also_counts():
    text = 'GRUB_CMDLINE_LINUX_DEFAULT="transparent_hugepage=always skew_tick=1"'
    present, total, missing = gs._gb_boot_param_presence(text, _STATIC[:2], [])
    assert missing == []
    assert present == total == 2


def test_a_genuinely_absent_param_is_still_reported_missing():
    text = 'GRUB_CMDLINE_LINUX_DEFAULT="skew_tick=1"'
    present, total, missing = gs._gb_boot_param_presence(text, _STATIC, _RANGE)
    assert "transparent_hugepage=always" in missing
    assert present == 1
    assert total == 6
