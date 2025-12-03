import copy
import io
import math
import os
import pickle
import re
import sys
import tarfile
import tkinter as tk
from itertools import combinations

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tifffile as tiff
from joblib import Parallel, delayed
from scipy import stats
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import chi2_contingency
from tqdm import tqdm

sys.stdout = io.StringIO()


def umap_html(umap_result, width=1000, height=1200):
    """
    Create an interactive HTML UMAP scatter plot.

    Parameters
    ----------
    umap_result : pandas.DataFrame or dict-like
        UMAP embedding containing at least:
        - `0` : array-like, UMAP dimension 1
        - `1` : array-like, UMAP dimension 2
        - `'clusters'` : array-like, assigned cluster labels

    width : int, optional
        Width of the output figure in pixels. Default is 1000.

    height : int, optional
        Height of the output figure in pixels. Default is 1200.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Interactive Plotly scatter plot object visualizing UMAP with colored clusters.
    """

    fig = px.scatter(
        x=umap_result[0],
        y=umap_result[1],
        color=umap_result["clusters"],
        labels={"color": "Cells"},
        template="simple_white",
        width=width,
        height=height,
        render_mode="svg",
        color_discrete_sequence=px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24,
    )

    fig.update_xaxes(title_text="UMAP 1")
    fig.update_yaxes(title_text="UMAP 2")

    return fig


def umap_static(umap_result, width=10, height=13):
    """
    Create a static matplotlib UMAP scatter plot.

    Parameters
    ----------
    umap_result : pandas.DataFrame or dict-like
        UMAP projection containing:
        - `0` : array-like, UMAP dimension 1
        - `1` : array-like, UMAP dimension 2
        - `'clusters'` : array-like, cluster assignments

    width : float, optional
        Width of the figure in inches. Default is 10.

    height : float, optional
        Height of the figure in inches. Default is 13.

    Returns
    -------
    matplotlib.figure.Figure
        Static matplotlib figure representing the UMAP embedding with clusters.
    """

    plotly_colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    num_colors = len(plotly_colors)

    fig = plt.figure(figsize=(width, height))

    cluster_counts = {
        label: np.sum(umap_result["clusters"] == label)
        for label in np.unique(umap_result["clusters"])
    }

    sorted_labels = sorted(
        cluster_counts, key=lambda label: cluster_counts[label], reverse=True
    )

    color_map = {
        label: plotly_colors[i % num_colors] for i, label in enumerate(sorted_labels)
    }

    for label in sorted_labels:
        subset = umap_result[umap_result["clusters"] == label]
        plt.scatter(
            subset[0],
            subset[1],
            c=[color_map[label]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=20,
            edgecolor="black",
            linewidths=0.1,
        )

    plt.xlabel("UMAP 1", fontsize=14)
    plt.ylabel("UMAP 2", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    return fig


def test_data(path=""):
    """
    Download and extract test data from Google Drive.

    This function downloads a compressed archive containing example test data
    and extracts it into the specified directory. The data is fetched using
    a direct Google Drive link. If the download or extraction fails, an
    error message is printed.

    Parameters
    ----------
    path : str, optional
        Destination directory where the test dataset will be downloaded and
        extracted. Defaults to the current working directory.

    Notes
    -----
    - The downloaded file is named ``test_data.tar.gz``.
    - The archive is extracted into ``<path>/test_data``.
    - In case of any failure (download or extraction), the function prints
      an informative message instead of raising an exception.
    """

    try:

        file_name = "test_data.tar.gz"

        file_name = os.path.join(path, file_name)

        url = "https://drive.google.com/uc?id=1MhzhleMP7iTzlBVW8eP5sFaonJdg1a3T"

        gdown.download(url, file_name, quiet=False)

        # Unzip

        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=path)

        print(
            f"\nTest data downloaded succesfully -> {os.path.join(path, 'test_data')}"
        )

    except:

        print(
            "\nTest data could not be downloaded. Please check your connection and try again!"
        )


def prop_plot(df_pivot, chi_df):
    """
    Create a stacked bar plot of proportional data with post-hoc significance annotations.

    Parameters
    ----------
    df_pivot : pandas.DataFrame
        Pivot table where rows represent categories (e.g., compartments) and columns
        represent groups. Values are counts or frequencies.

    chi_df : pandas.DataFrame
        DataFrame containing pairwise Chi-square test results with an added
        'Significance_Label' column (e.g., '***', '**', '*', 'ns') for each pair
        of groups. Typically output from `chi_pairs` and `get_significance_label`.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object containing the stacked bar plot.

    Notes
    -----
    - The function converts raw counts to percentages per group for visualization.
    - Each pairwise comparison and its significance label is displayed as a text box
      next to the plot.
    - Colors are assigned using the 'viridis' colormap by default.
    - The plot is configured for clarity with labeled axes, legend, and appropriately
      sized text.
    """

    df_pivot_perc = df_pivot.div(df_pivot.sum(axis=0), axis=1) * 100

    posthoc_text = "\n".join(
        [
            f"{row['Group 1']} → {row['Group 2']}: {row['Significance_Label']}"
            for _, row in chi_df.iterrows()
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    df_pivot_perc.T.plot(kind="bar", stacked=True, ax=ax, cmap="viridis")

    ax.set_ylabel("Percentage (%)", fontsize=16)
    ax.set_xlabel("Groups", fontsize=16)

    ax.tick_params(axis="both", labelsize=12)

    ax.legend(
        title="Compartment", loc="upper left", bbox_to_anchor=(1.02, 1.05), fontsize=14
    )

    plt.figtext(
        0.93,
        0.6,
        posthoc_text,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
    )

    return fig


def get_significance_label(p_value):
    """
    Return a standard significance label based on a p-value.

    Parameters
    ----------
    p_value : float
        The p-value for which the significance label should be determined.

    Returns
    -------
    str
        A significance marker commonly used in statistical reporting:

        - '***' : p < 0.001
        - '**'  : p < 0.01
        - '*'   : p < 0.05
        - 'ns'  : not significant (p ≥ 0.05)

    Notes
    -----
    This helper function is typically used for annotating statistical test
    results in tables or visualizations. Thresholds follow conventional
    statistical notation for significance levels.
    """

    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def chi_pairs(df_pivot):
    """
    Compute pairwise Chi-square tests for all combinations of groups in a pivoted dataframe.

    Parameters
    ----------
    df_pivot : pandas.DataFrame
        A pivot table where rows represent categories and columns represent groups.
        Values should be counts (frequencies). The function will add +1 to each cell
        to avoid zero counts during chi-square computation.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing pairwise Chi-square test results with the following columns:
        - 'Group 1' : str
            Name of the first group in the pair.
        - 'Group 2' : str
            Name of the second group in the pair.
        - 'Chi²' : float
            The Chi-square statistic for the comparison.
        - 'p-value' : float
            The p-value of the Chi-square test.

    Notes
    -----
    The function compares every possible pair of columns using `scipy.stats.chi2_contingency`.
    Yates' correction is applied by default unless disabled in the SciPy version used.
    A value of 1 is added to all cells to avoid issues with zero frequencies.
    """

    group_pairs = list(combinations(df_pivot.columns, 2))

    posthoc_results = []

    for group1, group2 in group_pairs:
        sub_table = df_pivot.T.loc[[group1, group2]] + 1
        chi2, p, dof, expected = chi2_contingency(sub_table)

        posthoc_results.append(
            {"Group 1": group1, "Group 2": group2, "Chi²": chi2, "p-value": p}
        )

    posthoc_df = pd.DataFrame(posthoc_results)

    return posthoc_df


def statistic(input_df, sets=None, metadata=None, n_proc=10):
    """
    Compute statistical comparison between cell groups or clusters.

    This function performs differential feature analysis between either:
    (1) every group vs. all other groups (default mode), or
    (2) two user-defined groups specified in ``sets``.

    The analysis includes:
    - Mann–Whitney U test
    - Percentage of non-zero values
    - Means and standard deviations
    - Effect size metric (ESM)
    - Benjamini–Hochberg FDR correction
    - Fold-change and log2 fold-change


    Parameters
    ----------
    input_df : pandas.DataFrame
        Input feature matrix where rows represent features and columns represent cells.
        The function transposes this table internally, treating columns as features.

    sets : dict or None, optional
        Mode selection:
        - ``None`` (default): each unique label in ``metadata['sets']`` is compared
          against all remaining groups.
        - ``dict``: must contain exactly two keys, each mapping to a list of labels
          belonging to each comparison group. Example:
          ``{'A': ['T1', 'T2'], 'B': ['C1', 'C2']}``.

    metadata : pandas.DataFrame, optional
        Metadata containing at least a column ``'sets'`` with group labels
        corresponding to columns of ``input_df``.

    n_proc : int, optional
        Number of parallel processes used for statistical computation.
        Default is ``10``.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing statistical results for each feature, including:

        - ``feature`` : str
        - ``p_val`` : float
        - ``adj_pval`` : float
        - ``pct_valid`` : float
        - ``pct_ctrl`` : float
        - ``avg_valid`` : float
        - ``avg_ctrl`` : float
        - ``sd_valid`` : float
        - ``sd_ctrl`` : float
        - ``esm`` : float
        - ``FC`` : float
        - ``log(FC)`` : float
        - ``norm_diff`` : float
        - ``valid_group`` : str
        - ``-log(p_val)`` : float

        If ``sets`` is ``None``, results for each group are concatenated.

        Returns ``None`` in case of errors or invalid parameters.

    Notes
    -----
    - Columns containing only zeros are automatically removed.
    - p-values equal for both groups produce ``p_val = 1``.
    - Benjamini–Hochberg correction is applied separately within each group comparison.
    - Fold-change is stabilized using a small, data-derived ``low_factor``.
    - Uses ``Mann–Whitney U`` test with ``alternative='two-sided'``.

    Raises
    ------
    None
        All exceptions are caught internally and printed as messages.

    Examples
    --------
    >>> df = pd.DataFrame(...)
    >>> meta = pd.DataFrame({'sets': [...]})
    >>> stat = statistic(df, metadata=meta)
    >>> stat.head()

    >>> # Compare two groups explicitly
    >>> sets = {'A': ['Type1'], 'B': ['Type2']}
    >>> stat = statistic(df, sets=sets, metadata=meta, n_proc=4)
    """
    try:
        offset = 1e-100

        def stat_calc(choose, feature_name):
            target_values = choose.loc[choose["DEG"] == "target", feature_name]
            rest_values = choose.loc[choose["DEG"] == "rest", feature_name]

            pct_valid = (target_values > 0).sum() / len(target_values)
            pct_rest = (rest_values > 0).sum() / len(rest_values)

            avg_valid = np.mean(target_values)
            avg_ctrl = np.mean(rest_values)
            sd_valid = np.std(target_values, ddof=1)
            sd_ctrl = np.std(rest_values, ddof=1)
            esm = (avg_valid - avg_ctrl) / np.sqrt(((sd_valid**2 + sd_ctrl**2) / 2))

            if np.sum(target_values) == np.sum(rest_values):
                p_val = 1.0
            else:
                _, p_val = stats.mannwhitneyu(
                    target_values, rest_values, alternative="two-sided"
                )

            return {
                "feature": feature_name,
                "p_val": p_val,
                "pct_valid": pct_valid,
                "pct_ctrl": pct_rest,
                "avg_valid": avg_valid,
                "avg_ctrl": avg_ctrl,
                "sd_valid": sd_valid,
                "sd_ctrl": sd_ctrl,
                "esm": esm,
            }

        # Transpose the input DataFrame
        choose = input_df.copy().T

        if sets is None:
            print("\nAnalysis started...")
            print("\nComparing each type of cell to others...")
            final_results = []

            if len(set(metadata["sets"])) > 1:
                choose.index = metadata["sets"]

            indexes = list(choose.index)

            for c in set(indexes):
                print(f"Calculating statistics for {c}")

                choose.index = indexes
                choose["DEG"] = np.where(choose.index == c, "target", "rest")

                valid = ",".join(set(choose.index[choose["DEG"] == "target"]))
                choose = choose.loc[
                    :, (choose != 0).any(axis=0)
                ]  # Remove all-zero columns

                # Parallel computation
                results = Parallel(n_jobs=n_proc)(
                    delayed(stat_calc)(choose, feature)
                    for feature in tqdm(choose.columns[choose.columns != "DEG"])
                )

                # Convert results to DataFrame
                combined_df = pd.DataFrame(results)
                combined_df["valid_group"] = valid
                combined_df.sort_values(by="p_val", inplace=True)

                # Adjusted p-values using Benjamini-Hochberg method
                num_tests = len(combined_df)
                combined_df["adj_pval"] = np.minimum(
                    1, (combined_df["p_val"] * num_tests) / np.arange(1, num_tests + 1)
                )

                combined_df["-log(p_val)"] = -np.log10(offset + combined_df["p_val"])
                low_factor = (
                    np.min(
                        (combined_df["avg_valid"] + combined_df["avg_ctrl"])[
                            combined_df["avg_valid"] + combined_df["avg_ctrl"] > 0
                        ]
                    )
                    * 0.95
                )

                combined_df["FC"] = (combined_df["avg_valid"] + low_factor) / (
                    combined_df["avg_ctrl"] + low_factor
                )
                combined_df["log(FC)"] = np.log2(combined_df["FC"])
                combined_df["norm_diff"] = (combined_df["avg_valid"] + low_factor) - (
                    combined_df["avg_ctrl"]
                )

                final_results.append(combined_df)

            print("\nAnalysis has finished!")
            return pd.concat(final_results, ignore_index=True)

        elif isinstance(sets, dict):
            print("\nAnalysis started...")
            print("\nComparing groups...")

            group_list = list(sets.keys())
            choose.index = metadata["sets"]

            inx = sorted([item for sublist in sets.values() for item in sublist])
            choose = choose.loc[inx]

            full_df = pd.DataFrame()
            for n, g in enumerate(group_list):
                print(f"Calculating statistics for {g}")

                rest_indices = [
                    idx
                    for i, group in enumerate(group_list)
                    if i != n
                    for idx in sets[group]
                ]

                choose["DEG"] = np.where(
                    choose.index.isin(sets[g]),
                    "target",
                    np.where(choose.index.isin(rest_indices), "rest", "drop"),
                )

                choose = choose[choose["DEG"] != "drop"]

                valid = g
                choose = choose.loc[
                    :, (choose != 0).any(axis=0)
                ]  # Remove all-zero columns

                # Parallel computation
                results = Parallel(n_jobs=n_proc)(
                    delayed(stat_calc)(choose, feature)
                    for feature in tqdm(choose.columns[choose.columns != "DEG"])
                )

                # Convert results to DataFrame
                combined_df = pd.DataFrame(results)
                combined_df["valid_group"] = valid
                combined_df.sort_values(by="p_val", inplace=True)

                # Adjusted p-values using Benjamini-Hochberg method
                num_tests = len(combined_df)
                combined_df["adj_pval"] = np.minimum(
                    1, (combined_df["p_val"] * num_tests) / np.arange(1, num_tests + 1)
                )

                combined_df["-log(p_val)"] = -np.log10(offset + combined_df["p_val"])
                low_factor = (
                    np.min(
                        (combined_df["avg_valid"] + combined_df["avg_ctrl"])[
                            combined_df["avg_valid"] + combined_df["avg_ctrl"] > 0
                        ]
                    )
                    * 0.95
                )

                combined_df["FC"] = (combined_df["avg_valid"] + low_factor) / (
                    combined_df["avg_ctrl"] + low_factor
                )
                combined_df["log(FC)"] = np.log2(combined_df["FC"])
                combined_df["norm_diff"] = (
                    combined_df["avg_valid"] - combined_df["avg_ctrl"]
                )

                full_df = pd.concat([full_df, combined_df])

            print("\nAnalysis has finished!")
            return full_df

        else:
            print("\nInvalid parameters. Please check the input.")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


# UTILS JIMG


def save_image(image, path_to_save):
    """
    Save an image to disk.

    Parameters
    ----------
    image : np.ndarray
        Input image array to be saved.

    path_to_save : str
        Output file path including the filename.
        Must include one of the following extensions:
        ``.png``, ``.tiff`` or ``.tif``.

    Returns
    -------
    str
        Path to the saved file.
    """

    try:
        if (
            len(path_to_save) == 0
            or ".png" not in path_to_save
            or ".tiff" not in path_to_save
            or ".tif" not in path_to_save
        ):
            print(
                "\nThe path is not provided or the file extension is not *.png, *.tiff or *.tif"
            )
        else:
            cv2.imwrite(path_to_save, image)

    except:
        print("Something went wrong. Check the function input data and try again!")


def load_tiff(path_to_tiff: str):
    """
    Load a *.tiff image and ensure it is in 16-bit format.

    Parameters
    ----------
    path_to_tiff : str
        Path to the *.tiff file.

    Returns
    -------
    np.ndarray
        Loaded image stack, converted to 16-bit if necessary.
    """

    try:
        stack = tiff.imread(path_to_tiff)

        if stack.dtype != "uint16":

            stack = stack.astype(np.uint16)

            for n, _ in enumerate(stack):

                min_val = np.min(stack[n])
                max_val = np.max(stack[n])

                stack[n] = ((stack[n] - min_val) / (max_val - min_val) * 65535).astype(
                    np.uint16
                )

                stack[n] = np.clip(stack[n], 0, 65535)

        return stack

    except:
        print("Something went wrong. Check the function input data and try again!")


def z_projection(tiff_object, projection_type="avg"):
    """
    Perform Z-projection on a stacked (3D) image.

    Parameters
    ----------
    tiff_object : np.ndarray
        Input stacked 3D image (e.g., loaded with `load_tiff()`).

    projection_type : str
        Type of Z-axis projection. Options: 'avg', 'median', 'min', 'max', 'std'.

    Returns
    -------
    np.ndarray
        Resulting 2D projection image.
    """

    try:

        if projection_type == "avg":
            img = np.mean(tiff_object, axis=0).astype(np.uint16)
        elif projection_type == "max":
            img = np.max(tiff_object, axis=0).astype(np.uint16)
        elif projection_type == "min":
            img = np.min(tiff_object, axis=0).astype(np.uint16)
        elif projection_type == "std":
            img = np.std(tiff_object, axis=0).astype(np.uint16)
        elif projection_type == "median":
            img = np.median(tiff_object, axis=0).astype(np.uint16)

        return img

    except:
        print("Something went wrong. Check the function input data and try again!")


def clahe_16bit(img, kernal=(100, 100)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    img : np.ndarray
        Image after CLAHE enhancement.

    kernel : tuple
        Size of the CLAHE tile grid used for processing, e.g., (100, 100).
    """

    try:

        img = img.copy()

        img8bit = img.copy()

        min_val = np.min(img8bit)
        max_val = np.max(img8bit)

        img8bit = ((img8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=kernal)
        img8bit = clahe.apply(img8bit)

        img8bit = img8bit / 255

        img = img * img8bit

        min_val = np.min(img)
        max_val = np.max(img)

        img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

        return img

    except:
        print("Something went wrong. Check the function input data and try again!")


def equalizeHist_16bit(image_eq):
    """
    Apply global histogram equalization to an image.

    Parameters
    ----------
    image_eq : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Image after global histogram equalization.
    """

    try:

        image = image_eq.copy()

        min_val = np.min(image)
        max_val = np.max(image)

        scaled_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        eq_image = cv2.equalizeHist(scaled_image)

        eq_image_bin = eq_image / 255

        image_eq_16 = image * eq_image_bin
        image_eq_16 = (image_eq_16 / np.max(image_eq_16)) * 65535
        image_eq_16[image_eq_16 > (65535 / 2)] += 65535 - np.max(image_eq_16)
        image_eq_16 = image_eq_16.astype(np.uint16)

        return image_eq_16

    except:
        print("Something went wrong. Check the function input data and try again!")


def load_image(path):
    """
    Load an image and convert it to 16-bit if necessary.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Loaded 16-bit image.
    """

    try:

        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

        # convert to 16 bit (the function are working on 16 bit images!)
        if img.dtype != "uint16":

            min_val = np.min(img)
            max_val = np.max(img)

            img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

            img = np.clip(img, 0, 65535)

        return img

    except:
        print("Something went wrong. Check the function input data and try again!")


def rotate_image(img, rotate: int):
    """
    Rotate an image by a specified angle.

    Parameters
    ----------
    img : np.ndarray
        Image to rotate.

    rotate : int
        Degree of rotation. Available options: 90, 180, 270.

    Returns
    -------
    np.ndarray
        Rotated image.
    """

    try:

        if rotate == 0:
            r = 0
        elif rotate == 90:
            r = -1
        elif rotate == 180:
            r = 2
        elif rotate == 180:
            r = 2
        elif rotate == 270:
            r = 1
        else:
            print("Wrong argument - rotate!")
            return None

        img = img.copy()

        img = np.rot90(img.copy(), k=r)

        return img

    except:
        print("Something went wrong. Check the function input data and try again!")


def mirror_image(img, rotate: str):
    """
    Mirror an image along specified axes.

    Parameters
    ----------
    img : np.ndarray
        Image to mirror.

    rotate : str
        Type of mirroring to apply. Options:
            'h'  - horizontal mirroring
            'v'  - vertical mirroring
            'hv' - both horizontal and vertical mirroring

    Returns
    -------
    np.ndarray
        Mirrored image.
    """

    try:

        if rotate == "h":
            img = np.fliplr(img.copy())
        elif rotate == "v":
            img = np.flipud(img.copy())
        elif rotate == "hv":
            img = np.flipud(np.fliplr(img.copy()))
        else:
            print("Wrong argument - rotate!")
            return None

        return img

    except:
        print("Something went wrong. Check the function input data and try again!")


# validatet UTILS


def merge_images(image_list: list, intensity_factors: list = []):
    """
    Merge multiple image projections from different channels.

    Parameters
    ----------
    image_list : list of np.ndarray
        List of images to merge. All images must have the same shape and size.

    intensity_factors : list of float
        Intensity scaling factors for each image in `image_list`. Base value is 1.
            - Values < 1 decrease intensity.
            - Values > 1 increase intensity.

    Returns
    -------
    np.ndarray
        Merged image after applying intensity scaling.
    """

    try:

        result = None

        if len(intensity_factors) == 0:

            intensity_factors = []
            for bt in range(len(image_list)):
                intensity_factors.append(1)

        for i, image in enumerate(image_list):
            if result is None:
                result = image.astype(np.uint64) * intensity_factors[i]
            else:
                result = cv2.addWeighted(
                    result, 1, image.astype(np.uint64) * intensity_factors[i], 1, 0
                )

        result = np.clip(result, 0, 65535)

        result = result.astype(np.uint16)

        return result

    except:
        print("Something went wrong. Check the function input data and try again!")


def adjust_img_16bit(
    img,
    color="gray",
    max_intensity: int = 65535,
    min_intenisty: int = 0,
    brightness: int = 1000,
    contrast=1.0,
    gamma=1.0,
):
    """
    Manually adjust image parameters and return the adjusted image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    color : str
        Image color channel. Options: ['green', 'blue', 'red', 'yellow', 'magenta', 'cyan'].

    max_intensity : int
        Upper threshold for pixel values. Pixels exceeding this value are set to `max_intensity`.

    min_intensity : int
        Lower threshold for pixel values. Pixels below this value are set to 0.

    brightness : int, optional
        Image brightness adjustment. Typical range: [900-2000]. Default is 1000.

    contrast : float or int, optional
        Image contrast adjustment. Typical range: [0-5]. Default is 1.

    gamma : float or int, optional
        Gamma correction factor. Typical range: [0-5]. Default is 1.

    Returns
    -------
    np.ndarray
        Adjusted image after applying brightness, contrast, gamma, and intensity thresholds.
    """

    try:

        img = img.copy()

        img = img.astype(np.uint64)

        img = np.clip(img, 0, 65535)

        # brightness
        if brightness != 1000:
            factor = -1000 + brightness
            side = factor / abs(factor)
            img[img > 0] = img[img > 0] + ((img[img > 0] * abs(factor) / 100) * side)
            img = np.clip(img, 0, 65535)

        # contrast
        if contrast != 1:
            img = ((img - np.mean(img)) * contrast) + np.mean(img)
            img = np.clip(img, 0, 65535)

        # gamma
        if gamma != 1:

            max_val = np.max(img)

            image_array = img.copy() / max_val

            image_array = np.clip(image_array, 0, 1)

            corrected_array = image_array ** (1 / gamma)

            img = corrected_array * max_val

            del image_array, corrected_array

            img = np.clip(img, 0, 65535)

        img = np.nan_to_num(img, nan=0, posinf=65535, neginf=0)
        max_val = np.max(img)
        if max_val > 0:
            img = ((img / max_val) * 65535).astype(np.uint16)

        # max intenisty
        if max_intensity != 65535:
            img[img >= max_intensity] = 65535

        # min intenisty
        if min_intenisty != 0:
            img[img <= min_intenisty] = 0

        img_gamma = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)

        if color == "green":
            img_gamma[:, :, 1] = img

        elif color == "red":
            img_gamma[:, :, 2] = img

        elif color == "blue":
            img_gamma[:, :, 0] = img

        elif color == "magenta":
            img_gamma[:, :, 0] = img
            img_gamma[:, :, 2] = img

        elif color == "yellow":
            img_gamma[:, :, 1] = img
            img_gamma[:, :, 2] = img

        elif color == "cyan":
            img_gamma[:, :, 0] = img
            img_gamma[:, :, 1] = img

        elif color == "gray":
            img_gamma[:, :, 0] = img
            img_gamma[:, :, 1] = img
            img_gamma[:, :, 2] = img

        return img_gamma

    except:
        print("Something went wrong. Check the function input data and try again!")


def get_screan():
    """
    Get the current screen width and height.

    Returns
    -------
    tuple of int
        screen_width, screen_height
    """

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.destroy()

    return screen_width, screen_height


def resize_to_screen_img(img_file, factor=1):
    """
    Resize an input image to fit the screen size, optionally scaled by a factor.

    Parameters
    ----------
    img_file : np.ndarray
        Input image to be resized.

    factor : float, optional
        Scaling factor applied to the screen dimensions before resizing the image. Default is 1.

    Returns
    -------
    np.ndarray
        Resized image that fits within the screen dimensions while maintaining aspect ratio.
    """

    screen_width, screen_height = get_screan()

    screen_width = int(screen_width * factor)
    screen_height = int(screen_height * factor)

    h = int(img_file.shape[0])
    w = int(img_file.shape[1])

    if screen_width < w:
        h = img_file.shape[0]
        w = img_file.shape[1]

        ww = int((screen_width / w) * w)
        hh = int((screen_width / w) * h)

        img_file = cv2.resize(img_file, (ww, hh))

        h = img_file.shape[0]
        w = img_file.shape[1]

    if screen_height < h:
        h = img_file.shape[0]
        w = img_file.shape[1]

        ww = int((screen_height / h) * w)
        hh = int((screen_height / h) * h)

        img_file = cv2.resize(img_file, (ww, hh))

    return img_file


def display_preview(image):
    """
    Quickly preview an image using a display window.

    Parameters
    ----------
    image : np.ndarray
        Input image to be displayed.

    Notes
    -----
        The function displays the image in a window. Does not return a value.
    """
    try:

        res_sc = resize_to_screen_img(image.copy(), factor=0.8)

        cv2.imshow("Display", res_sc)

        cv2.waitKey(0) & 0xFF

        cv2.destroyAllWindows()

    except:
        print("Something went wrong. Check the function input data and try again!")


## flow_JIMG_functions


class ImageTools:
    """
    Collection of utility functions for image preprocessing, adjustment,
    merging, and stitching.

    This class provides standalone static methods for operations such as
    histogram equalization, CLAHE enhancement, intensity adjustments,
    image merging based on weighted ratios, and horizontal stitching.
    These tools are used internally by `ImagesManagement` but can also be
    applied independently for generic image-processing workflows.

    Notes
    -----
    All methods operate on NumPy arrays and expect images in standard
    OpenCV format. Some functions assume 16-bit images unless stated
    otherwise.

    Examples
    --------
    >>> from ImageTools import ImageTools
    >>> img = ImageTools.equalize_hist_16bit(img)
    >>> merged = ImageTools.merge_images([img1, img2], [0.5, 0.5])
    """

    def get_screan(self):
        """
        Return the current screen size.

        Returns
        -------
        screen_width : int
            Width of the screen in pixels.

        screen_height : int
            Height of the screen in pixels.
        """

        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        root.destroy()

        return screen_width, screen_height

    def resize_to_screen_img(self, img_file, factor=0.5):
        """
        Resize an input image to a scaled version of the current screen size.

        Parameters
        ----------
        img_file : np.ndarray
            Input image to be resized.

        factor : int
            Scaling factor applied to the screen dimensions.

        Returns
        -------
        resized_image : np.ndarray
            Resized image adjusted to the scaled screen size.
        """

        screen_width, screen_height = self.get_screan()

        screen_width = int(screen_width * factor)
        screen_height = int(screen_height * factor)

        h = int(img_file.shape[0])
        w = int(img_file.shape[1])

        if screen_width < w or screen_width * 0.3 > w:
            h = img_file.shape[0]
            w = img_file.shape[1]

            ww = int((screen_width / w) * w)
            hh = int((screen_width / w) * h)

            img_file = cv2.resize(img_file, (ww, hh))

            h = img_file.shape[0]
            w = img_file.shape[1]

        if screen_height < h or screen_height * 0.3 > h:
            h = img_file.shape[0]
            w = img_file.shape[1]

            ww = int((screen_height / h) * w)
            hh = int((screen_height / h) * h)

            img_file = cv2.resize(img_file, (ww, hh))

        return img_file

    def load_JIMG_project(self, project_path):
        """
        Load a JIMG project from a `.pjm` file.

        Parameters
        ----------
        file_path : str
            Path to the project file with `.pjm` extension.

        Returns
        -------
        project : object
            Loaded project object.

        Raises
        ------
        ValueError
            If the provided file does not have a `.pjm` extension.
        """

        if ".pjm" in project_path:
            with open(project_path, "rb") as file:
                app_metadata_tmp = pickle.load(file)

            return app_metadata_tmp

        else:
            print("\nProvide path to the project metadata file with *.pjm extension!!!")

    def ajd_mask_size(self, image, mask):
        """
        Adjusts the size of a mask to match the dimensions of a given image.

        Parameters
        ----------
        image : np.ndarray
            Reference image whose size the mask should match. Can be 2D or 3D.

        mask : np.ndarray
            Mask image to be resized.

        Returns
        -------
        np.ndarray
            Resized mask matching the input image dimensions.
        """

        try:
            mask = cv2.resize(mask, (image.shape[2], image.shape[1]))
        except:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        return mask

    def load_image(self, path_to_image):
        """
        Load an image from the specified file path.

        Parameters
        ----------
        path_to_image : str
            Path to the image file.

        Returns
        -------
        image : np.ndarray
            Loaded image as a NumPy array.
        """

        image = load_image(path_to_image)
        return image

    def load_3D_tiff(self, path_to_image):
        """
        Load a 3D image from a TIFF file.

        Parameters
        ----------
        path_to_image : str
            Path to the 3D TIFF image file.

        Returns
        -------
        image : np.ndarray
            Loaded 3D image as a NumPy array.
        """

        image = load_tiff(path_to_image)

        return image

    def load_mask(self, path_to_mask):
        """
        Load a mask image.

        Parameters
        ----------
        path_to_mask : str
            Path to the mask image file.

        Returns
        -------
        mask : np.ndarray
            Loaded mask image as a NumPy array.
        """

        mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        return mask

    def save(self, image, file_name):
        """
        Save an image to disk.

        Parameters
        ----------
        image : np.ndarray
            Image data to be saved.

        file_name : str
            Output file path including the desired image extension (e.g., ".png", ".jpg").

        """

        cv2.imwrite(filename=file_name, img=image)

    # calculation methods

    def drop_dict(self, dictionary, key, var, action=None):
        """
        Filters elements from a dictionary based on a condition applied to a specified key.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing lists or arrays under each key.

        key : str
            The key in the dictionary on which the filtering condition will be applied.

        var : numeric
            Value to compare against each element in dictionary[key].

        action : str, optional
            Comparison operator as string: '<=', '>=', '==', '<', '>'.
            Default is None, which raises an error.

        Returns
        -------
        dict
            A new dictionary with elements removed where the condition matches.
        """

        dictionary = copy.deepcopy(dictionary)
        indices_to_drop = []
        for i, dr in enumerate(dictionary[key]):

            if isinstance(dr, np.ndarray):
                dr = np.mean(dr)

            if action == "<=":
                if var <= dr:
                    indices_to_drop.append(i)
            elif action == ">=":
                if var >= dr:
                    indices_to_drop.append(i)
            elif action == "==":
                if var == dr:
                    indices_to_drop.append(i)
            elif action == "<":
                if var < dr:
                    indices_to_drop.append(i)
            elif action == ">":
                if var > dr:
                    indices_to_drop.append(i)
            else:
                print("\nWrong action!")
                return None

        for key, value in dictionary.items():
            dictionary[key] = [
                v for i, v in enumerate(value) if i not in indices_to_drop
            ]

        return dictionary

    # modified for gradation (separation near nucleus)

    def create_mask(self, dictionary, image):
        """
        Creates a mask image with gradated intensity values for each coordinate set in a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing a 'coords' key with a list of arrays of coordinates.

        image : np.ndarray
            Base image to define the shape of the mask.

        Returns
        -------
        np.ndarray
            Mask image with uint16 gradated intensity.
        """

        image_mask = np.zeros(image.shape)

        arrays_list = copy.deepcopy(dictionary["coords"])

        if len(arrays_list) > 0:

            initial_val = math.floor((2**16 - 1) / 4)
            intensity_list = ((2**16 - 1) - math.floor((2**16 - 1) / 4)) / len(
                arrays_list
            )

            gradation = 1
            for arr in arrays_list:
                image_mask[arr[:, 0], arr[:, 1]] = initial_val + (
                    gradation * intensity_list
                )
                gradation += 1

        return image_mask.astype("uint16")

    def min_max_histograme(self, image):
        """
        Calculates histogram-based minimum and maximum intensity percentiles in an image.

        Parameters
        ----------
        image : np.ndarray
            Input image for histogram analysis.

        Returns
        -------
        min_val : float
            Minimum intensity percentile above zero.

        max_val : float
            Maximum intensity percentile based on histogram gradient.

        df : pd.DataFrame
            DataFrame containing quantiles, corresponding intensity values, and cumulative percents.
        """

        q = []
        val = []
        perc = []

        max_val = image.shape[0] * image.shape[1]

        for n in range(0, 100, 5):
            q.append(n)
            val.append(np.quantile(image, n / 100))
            sum_val = np.sum(image < np.quantile(image, n / 100))
            pr = sum_val / max_val
            perc.append(pr)

        df = pd.DataFrame({"q": q, "val": val, "perc": perc})

        min_val = 0
        for i in df.index:
            if df["val"][i] != 0 and min_val == 0:
                min_val = df["perc"][i]

        max_val = 0
        df = df[df["perc"] > 0]
        df = df.sort_values("q", ascending=False).reset_index(drop=True)

        for i in df.index:
            if i > 1 and df["val"][i] * 1.5 > df["val"][i - 1]:
                max_val = df["perc"][i]
                break
            elif i == len(df.index) - 1:
                max_val = df["perc"][i]

        return min_val, max_val, df
