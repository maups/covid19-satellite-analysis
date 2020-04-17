![Header](/assets/header.jpg)
======

The COVID-19 outbreak forced governments worldwide to impose lockdowns and quarantines over their population to prevent virus transmission. As a consequence, there are disruptions in human and economic activities all over the globe. The recovery process is also expected to be rough. Economic activities impact social behaviors, which leave signatures in satellite images that can be automatically detected and classified.  Satellite imagery can support the decision-making of analysts and policymakers by providing a different kind of visibility into the unfolding economic changes. Such information can be useful both during the crisis and also as we recover from it. In this work, we use a deep learning approach that combines strategic location sampling and an ensemble of lightweight convolutional neural networks (CNNs) to recognize specific elements in satellite images and compute economic indicators based on it, automatically. This CNN ensemble framework ranked third place in the US Department of Defense xView challenge, the most advanced benchmark for object detection in satellite images. We show the potential of our framework for temporal analysis using the US IARPA Function Map of the World (fMoW) dataset. We also show results on real examples of different sites before and after the COVID-19 outbreak to demonstrate possibilities. Among the future work is the possibility that with a satellite image dataset that samples a region at a weekly (or biweekly) frequency, we can generate more informative temporal signatures that can predict future economic states. [ [FULL REPORT](https://arxiv.org/abs/2004.07438) ]

## Authors

- Rodrigo Minetto
- Mauricio Pamplona Segundo
- Gilbert Rotich
- Sudeep Sarkar

## Downloading models and running

Make sure you install all prerequisites before running the following commands:

```
$ ./download_models.sh
$ run_inference.sh
```

Models are under the following license: https://github.com/DIUx-xView/xView1_baseline/blob/master/LICENSE

Credits for images are given in the report.

## Citing

If you find the code in this repository useful in your research, please consider citing:
```
@misc{minetto2020measuring,
    title={Measuring Human and Economic Activity from Satellite Imagery to Support City-Scale Decision-Making during COVID-19 Pandemic},
    author={Rodrigo Minetto and Mauricio Pamplona Segundo and Gilbert Rotich and Sudeep Sarkar},
    year={2020},
    eprint={2004.07438},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
