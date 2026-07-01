### JIMG_int – Python library for marker intensity and distribution analysis

![Python version](https://img.shields.io/badge/python-%E2%89%A53.12%20%7C%20%3C3.13-blue?logo=python&logoColor=white.png)
![License](https://img.shields.io/badge/license-GPLv3-blue)
![Docs](https://img.shields.io/badge/docs-available-blueviolet)
</br>

<p align="right">
    <img src="https://github.com/jkubis96/Logos/blob/main/logos/jbs_current.png?raw=true" alt="drawing" width="250" />
    <img src="https://github.com/jkubis96/Logos/blob/main/logos/jbi_current.png?raw=true" alt="drawing" width="250" />
</p>

</br>

### Author: Jakub Kubiś 

<div align="left">
 Institute of Bioorganic Chemistry<br />
 Polish Academy of Sciences<br />
</div>


## Description


<div align="justify">


***JIMG_int*** is a Python library designed to quantify and analyze marker intensity and spatial distribution in images. It is particularly useful for biological imaging, immunofluorescence, or any field requiring analysis of labeled markers in microscopy or medical images.


The library support the [JIMG](https://github.com/jkubis96/JIMG) image processing tool, specifically tailored for analyzing high-resolution confocal microscope images [Opera-Phoenix](https://www.revvity.com/product/opera-phenix-plus-system-hh14001000?srsltid=AfmBOoohz1LiEemNbG4SJnaEtScwr16MyFL8Ulf9NyDDEAffV2NLJXoe) and other technologies. 

It provides algorithms for measuring the intensity of specific protein markers using customizable image masks on high-resolution microscope images. These measurements are normalized using a background mask for consistent data comparison. The collected intensity data can be statistically analyzed to detect differences in marker localization, occurrence, and intensity.

 

</div>

</br>



<br />

## 📚 Table of Contents
- 1.[Installation](#installation)
- 2.[Documentation](#doc)
- 3.[Example pipelines](#epip)
  - 3.1. [Marker intensity features extraction](#nacm)
    - 3.1.1 [Adjusting parameters and image loading](#nacm1)
    - 3.1.2 [ROI mask loading](#nacm2)
    - 3.1.3 [Normalization mask loading](#nacm3)
    - 3.1.4 [Extracting intensity values](#nacm4)
    - 3.1.5 [Saving intensity data](#nacm5)
    - 3.1.6 [APerforming analysis pipeline for the subsequent image](#nacm6)
    - 3.1.7 [Combining the experimental data](#nacm7)
  - 3.2. [Analyzing the intensity and spatial distribution of the marker](#miacmda)
    - 3.2.1 [Loading experimental data](#miacmda1)
    - 3.2.2 [Quantitative analysis of marker intensity and spatial distribution](#miacmda2)
    - 3.2.3 [Visualizing the data using a histogram distribution](#miacmda3)

    

<br />

<br />

# 1. Installation <a id="installation"></a>

#### In command line write:

```
pip install jimg-int
```

# 2. Documenation <a id="doc"></a>

Documentation for classes and functions is available here 👉 [Documentation 📄](https://jkubis96.github.io/JIMG_int/jimg_int.html)


# 3. Example pipelines <a id="epip"></a>

If you want to run the examples, you must download the test data. To do this, use:

```
from jimg_ncd.nuclei import test_data

test_data()
```


<br />

#### 3.1 Marker intensity features extraction <a id="nacm"></a>


##### 3.1.1 Adjusting parameters and image loading <a id="nacm1"></a>


```
from jimg_int.intensity import FeatureIntensity

# Select intenity are data for 1st Image - healthy

# initiate class
fi = FeatureIntensity()

# check current metadata
fi.current_metadata

# if required, change parameters
fi.set_projection(projection="avg")

fi.set_correction_factorn(factor=0.2)

# fi.set_scale(scale = 0.5)
# fi.set_selection_list(rm_list = [2,5,6,7])
# OR
# load JIMG project where scale and rm_lis is set in project metadata
# fi.load_JIMG_project_(path = '')
# for more information go to: https://github.com/jkubis96/JIMG
# rm_list and scale can be omitted

# load image
fi.load_image_3D(path="test_data/intensity/ctrl/image.tiff")

# or 1D image after projection, be sure that image was not adjusted, for analysis should be use !RAW! image
# fi.load_image_(path)
```
<br/>

***Analysed image projection (after projection with JIMG)***

* input image in this case is raw 3D-image in *.tiff format

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/ctrl.bmp" alt="drawing" width="600" />
</p>

<br/>

##### 3.1.2 ROI mask loading<a id="nacm2"></a>

```
fi.load_mask_(path = 'test_data/intensity/ctrl/mask_1.png')
```
<br/>

***Analysed image region mask***


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/ctrl_mask.bmp" alt="drawing" width="600" />
</p>

<br/>

##### 3.1.3 Normalization mask loading<a id="nacm3"></a>

```
fi.load_normalization_mask_(path = 'test_data/intensity/ctrl/background_1.png')
```
<br/>

***Normalization region mask (reversed)***


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/ctrl_back.bmp" alt="drawing" width="600" />
</p>

<br/>

##### 3.1.4 Extracting intensity values <a id="nacm4"></a>

```
# strat calculations
fi.run_calculations()


# get results
results = fi.get_results()
```

##### 3.1.5 Saving intensity data <a id="nacm5"></a>

```
# save results for further analysis, ensuring each feature 
# is stored in a separate directory (single directory 
# should contain data with the same 'feature_name'),
# this setup allows running fi.concatenate_intensity_data() 
# in the specific directory of each feature
# while preventing errors from incorrect feature concatenation

fi.save_results(path = os.getcwd(), 
             mask_region = 'brain', 
             feature_name = 'Feature1', 
             individual_number = 1, 
             individual_name = 'CTRL')
```

##### 3.1.6 Performing analysis pipeline for the subsequent image <a id="nacm6"></a>

* Apply steps 3.1.1–3.1.5 to the subsequent image to support comparison analysis.

```
# Select intenity are data for 2st Image - disease

# initiate class
fi = FeatureIntensity()

fi.set_projection(projection="avg")

fi.set_correction_factorn(factor=0.2)

fi.load_image_3D(path="test_data/intensity/dise/image.tiff")

###############################################################################

fi.load_mask_(path="test_data/intensity/dise/mask_1.png")

###############################################################################

fi.load_normalization_mask_(path="test_data/intensity/dise/background_1.png")

###############################################################################

fi.run_calculations()

results = fi.get_results()

fi.save_results(
    path="",
    mask_region="brain",
    feature_name="Feature1",
    individual_number=1,
    individual_name="DISEASE",
)
```

***Analysed image projection (after projection with JIMG)***

* input image in this case is raw 3D-image in *.tiff format

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/dis.bmp" alt="drawing" width="600" />
</p>

<br/>

***Normalization region mask (reversed)***


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/dis_back.bmp" alt="drawing" width="600" />
</p>

<br/>

***Analysed image region mask***


<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/documentation/fig/Intensity/dis_mask.bmp" alt="drawing" width="600" />
</p>



##### 3.1.7 Combining the experimental data <a id="nacm7"></a>


```
# concatenate data of experiment 1 & 2
fi.concatenate_intensity_data(directory="", name="example_data")
```

<br/>

#### 3.2 Analyzing the intensity and spatial distribution of the marker  <a id="miacmda"></a>


#### 3.2.1 Loading experimental data  <a id="miacmda1"></a>


```
import pandas as pd
from jimg_int.intensity import IntensityAnalysis

# initiate class
ia = IntensityAnalysis()

input_data = pd.read_csv("example_data_Feature1_brain.csv")

# check columns
input_data.head()
```

#### 3.2.2 Quantitative analysis of marker intensity and spatial distribution <a id="miacmda2"></a>

```
data = ia.df_to_percentiles(
    data=input_data,
    group_col="individual_name",
    values_col="norm_intensity",
    replication_col = 'individual_number',
    sep_perc=1,
)

stats = ia.get_stats(data, tested_value = 'avg')
```

#### 3.2.3 Visualizing the data using a histogram distribution <a id="miacmda3"></a>

```
results = ia.hist_compare_plot(
    data=stats, queue=["CTRL", "DISEASE"], p_adj=True, txt_size=20
)
```
<br/>

***Results of intensity comparison analysis (region under the mask)***

<p align="center">
<img  src="https://raw.githubusercontent.com/jkubis96/JIMG_int/refs/heads/main/fig/Intensity/compare_result.bmp" alt="drawing" width="600" />
</p>

<br/>

```
results.savefig('example_results.svg', format = 'svg', dpi = 300, bbox_inches = 'tight')
```

<br />
<br />

### Have fun JBS


