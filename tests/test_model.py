import sys

import pytest

sys.path.append("../src")

from scsd.pipeline import SDPipeline

test_prompt = "A cartoon of a AI (robot)1.5 helping a (human)1.5 in the office to manage the manufacturing scheule. masterpiece. 4k"


@pytest.mark.skip(reason="test takes too long")
def test_pipeline_turbo():
    sdp = SDPipeline()
    sdp.set_pipeline("turbo")
    sdp.set_prompt(test_prompt)
    images = sdp.generate_batch(batch_size=8)

    print(sdp.ldm_info)
    for i in range(len(images)):
        images[i].save("tests/output/test_turbo_{}.jpg".format(i))


def test_change_mode():
    sdp = SDPipeline()
    sdp.set_pipeline("turbo")
    sdp.set_prompt(test_prompt)
    images = sdp.generate_batch(batch_size=8)

    sdp.set_pipeline("basic")
    sdp.set_prompt(test_prompt)
    images = sdp.generate_batch(batch_size=1)


@pytest.mark.skip(reason="test takes too long")
def test_pipeline_basic():
    sdp = SDPipeline()
    sdp.set_pipeline("basic")
    sdp.set_prompt(test_prompt)
    images = sdp.generate_batch(batch_size=1)

    print(sdp.ldm_info)
    for i in range(len(images)):
        images[i].save("tests/output/test_basic_{}.jpg".format(i))


@pytest.mark.skip(reason="test takes too long")
def test_pipeline_refine():
    sdp = SDPipeline()
    sdp.set_pipeline("refine")
    sdp.set_prompt(test_prompt)
    images = sdp.generate_batch(batch_size=1)

    print(sdp.ldm_info)
    for i in range(len(images)):
        images[i].save("tests/output/test_refine_{}.jpg".format(i))
