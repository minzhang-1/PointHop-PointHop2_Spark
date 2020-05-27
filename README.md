# A fast and low memory requirement version of [PointHop](https://github.com/minzhang-1/PointHop) and [PointHop++](https://github.com/minzhang-1/PointHop2)
Created by [Min Zhang](https://github.com/minzhang-1)

### Introduction
This work is an improved implementation of our [PointHop method](https://arxiv.org/abs/1907.12766) and [PointHop++ method](https://arxiv.org/abs/2002.03281), which is built upon Apache Spark. With 12 cores (Intel (R) core ™ i7-5930k CPU @ 3.5GHZ), PointHop finishes in 20 minutes using less than 12GB memory, and PointHop++ finishes in 40 minutes using less than 14GB memory. 

In this repository, we release code and data for training a baseline of PointHop and PointHop++ classification network on point clouds sampled from 3D shapes.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2020pointhop,
	  title={PointHop: An Explainable Machine Learning Method for Point Cloud Classification},
	  author={Zhang, Min and You, Haoxuan and Kadam, Pranav and Liu, Shan and Kuo, C-C Jay},
	  journal={IEEE Transactions on Multimedia},
	  year={2020},
	  publisher={IEEE}
	}

	@article{zhang2020pointhop++,
	  title={PointHop++: A Lightweight Learning Model on Point Sets for 3D Classification},
	  author={Zhang, Min and Wang, Yifan and Kadam, Pranav and Liu, Shan and Kuo, C-C Jay},
	  journal={arXiv preprint arXiv:2002.03281},
	  year={2020}
	}

### Installation

The code has been tested with Python 2.7 and 3.5, Java 8.0. You may need to install h5py, sklearn, pickle and pyspark packages.

To check your java version:
```bash
java --version
```

To install pyspark for Python:
```bash
sudo pip install pyspark
```

If you are using Python 3. You may need to set up your configuarion.
```bash
PYSPARK_PYTHON=/usr/bin/python3
PYSPARK_DRIVER_PYTHON=/usr/bin/python3
```

### Usage
To train and test a single PointHop model without ensemble to classify point clouds sampled from 3D shapes:

    python3 pointhop_spark.py

To train and test a single PointHop++ model without feature selection and ensemble to classify point clouds sampled from 3D shapes:

    python3 pointhop2_spark.py

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.


