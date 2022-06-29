import os.path

import create_mil_data
import common


def test_root_dir():
    print(common.BASE_DIR)


def test_create_mil():
    create_mil_data.create_data(f"{common.BASE_DIR}/data_example/space",
                                f"{common.BASE_DIR}/seeds/space")


