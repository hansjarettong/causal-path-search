import sys

sys.path.append("..")

import pytest
import pandas as pd
from lingam_mmi import kNN_LiNGAM_MMI, HSIC_LiNGAM_MMI, PWLR_LiNGAM_MMI
from lingam_zdd import kNN_LiNGAM_ZDD, HSIC_LiNGAM_ZDD, PWLR_LiNGAM_ZDD
from data_generating_process import generate_data


@pytest.mark.parametrize(
    "method_mmi, method_zdd",
    [
        (kNN_LiNGAM_MMI, kNN_LiNGAM_ZDD),
        (HSIC_LiNGAM_MMI, HSIC_LiNGAM_ZDD),
        (PWLR_LiNGAM_MMI, PWLR_LiNGAM_ZDD),
    ],
)
@pytest.mark.parametrize("num_features", [3,4,5])
def test_ordering_consistency(method_mmi, method_zdd, num_features):
    # Generate data
    df = pd.DataFrame(generate_data(num_features, sample_size=200))

    # Get ordering from ZDD
    zdd_ordering = method_zdd().fit(df).get_shortest_path_ordering()

    # Get ordering from MMI
    mmi_ordering = method_mmi().fit(df).causal_order_

    # Assert that the orderings are the same
    assert zdd_ordering == mmi_ordering
