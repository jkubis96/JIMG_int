import os

import matplotlib.figure as mpl_fig
import pandas as pd
import pytest

import jimg_int.config as cfg
from jimg_int.intensity import FeatureIntensity, IntensityAnalysis, test_data


@pytest.fixture(autouse=True)
def disable_display():
    cfg._DISPLAY_MODE = False


def test_test_data_runs():
    test_data()
    assert os.path.exists("test_data")


def test_feature_intensity_setup():
    fi = FeatureIntensity()

    assert isinstance(fi.current_metadata, tuple)

    previous = fi.current_metadata
    fi.set_projection("avg")
    fi.set_correction_factorn(0.2)

    assert fi.current_metadata == fi.current_metadata
    assert previous != fi.current_metadata


@pytest.mark.parametrize("group", ["ctrl", "dise"])
def test_loading_images_and_masks(group):
    base = f"test_data/intensity/{group}"

    fi = FeatureIntensity()
    fi.set_projection("avg")
    fi.set_correction_factorn(0.2)

    # 3D image
    fi.load_image_3D(os.path.join(base, "image.tiff"))
    assert fi.input_image is not None

    # mask
    fi.load_mask_(os.path.join(base, "mask_1.png"))
    assert fi.mask is not None

    # normalization mask
    fi.load_normalization_mask_(os.path.join(base, "background_1.png"))
    assert fi.background_mask is not None


@pytest.mark.parametrize("group", ["ctrl", "dise"])
def test_run_calculations(group):
    base = f"test_data/intensity/{group}"

    fi = FeatureIntensity()
    fi.set_projection("avg")
    fi.set_correction_factorn(0.2)

    fi.load_image_3D(os.path.join(base, "image.tiff"))
    fi.load_mask_(os.path.join(base, "mask_1.png"))
    fi.load_normalization_mask_(os.path.join(base, "background_1.png"))

    fi.run_calculations()
    results = fi.get_results()

    assert "intensity" in results.keys()
    assert len(results["intensity"]) > 0


def test_save_results():
    fi = FeatureIntensity()
    fi.set_projection("avg")
    fi.set_correction_factorn(0.2)

    fi.load_image_3D("test_data/intensity/ctrl/image.tiff")
    fi.load_mask_("test_data/intensity/ctrl/mask_1.png")
    fi.load_normalization_mask_("test_data/intensity/ctrl/background_1.png")

    fi.run_calculations()

    fi.save_results(
        path="",
        mask_region="brain",
        feature_name="Feature1",
        individual_number=1,
        individual_name="CTRL",
    )

    assert os.path.exists("CTRL_1_brain_Feature1.int")


def test_concatenate():
    fi = FeatureIntensity()

    fi.current_metadata

    fi.set_projection(projection="avg")

    fi.set_correction_factorn(factor=0.2)

    fi.load_image_3D(path="test_data/intensity/ctrl/image.tiff")

    fi.load_mask_(path="test_data/intensity/ctrl/mask_1.png")

    fi.load_normalization_mask_(path="test_data/intensity/ctrl/background_1.png")

    fi.run_calculations()

    fi.save_results(
        path="",
        mask_region="brain",
        feature_name="Feature1",
        individual_number=1,
        individual_name="CTRL",
    )

    fi = FeatureIntensity()

    fi.set_projection(projection="avg")

    fi.set_correction_factorn(factor=0.2)

    fi.load_image_3D(path="test_data/intensity/dise/image.tiff")

    fi.load_mask_(path="test_data/intensity/dise/mask_1.png")

    fi.load_normalization_mask_(path="test_data/intensity/dise/background_1.png")

    fi.run_calculations()

    fi.save_results(
        path="",
        mask_region="brain",
        feature_name="Feature1",
        individual_number=1,
        individual_name="DISEASE",
    )

    fi.concatenate_intensity_data(directory="", name="example_data")

    assert os.path.exists("example_data_Feature1_brain.csv")


def test_intensity_analysis_hist():

    ia = IntensityAnalysis()

    input_data = pd.read_csv("example_data_Feature1_brain.csv")

    data = ia.df_to_percentiles(
        data=input_data,
        group_col="individual_name",
        values_col="norm_intensity",
        sep_perc=1,
    )

    results = ia.hist_compare_plot(
        data=data,
        queue=["CTRL", "DISEASE"],
        tested_value="avg",
        p_adj=True,
        txt_size=20,
    )

    assert isinstance(results, mpl_fig.Figure)
