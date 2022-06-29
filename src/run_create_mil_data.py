import os.path

import create_mil_data
import common
if __name__ == "__main__":
    create_mil_data.create_data(f"{common.BASE_DIR}/data_example/space",
                                f"{common.BASE_DIR}/seeds/space")