import sys

import pytest

sys.path.append("../src")

from scsd.ldmhelper import LDMHelper

test_prompt = "A cartoon of a AI (robot)1.5 helping a (human)1.5 in the office to manage the manufacturing scheule. masterpiece. 4k"


# @pytest.mark.skip(reason="test takes too long")
def test_ldmhelper_turbo():
    ldm = LDMHelper(mode="turbo")
    ldm.txt2img(test_prompt, batch_size=8, topic="ai_robot")


@pytest.mark.skip(reason="test takes too long")
def test_ldmhelper_basic():
    ldm = LDMHelper(mode="basic")
    ldm.txt2img(test_prompt, batch_size=1, topic="ai_robot")


@pytest.mark.skip(reason="test takes too long")
def test_ldmhelper_refine():
    ldm = LDMHelper(mode="refine")
    ldm.txt2img(test_prompt, batch_size=1, topic="ai_robot")
