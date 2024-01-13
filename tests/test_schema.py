import sys

import pytest

sys.path.append("../src")

from scsd.schema import LDMInfo, load_ldminfo, load_default_ldminfo


def test_schema():
    config_path = "src/scsd/settings/sdxl_basic.yaml"
    ldm_info = load_ldminfo(config_path)

    print(ldm_info)


def test_load():
    ldm_info = load_default_ldminfo("turbo")
    print(ldm_info)
