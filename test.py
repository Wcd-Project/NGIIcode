# -------------------------------------------------------------------------------------
# A Bidirectional Focal Atention Network implementation based on
# https://arxiv.org/abs/1909.11416.
# "Focus Your Atention: A Bidirectional Focal Atention Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2019
# -------------------------------------------------------------------------------------
"""test"""

from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DATA_PATH = "/home/gpuadmin/DatasetsPy2/data"
evaluation.evalrank("/home/gpuadmin/ZNDMT8-20201107/PT1/runs/f30k_precomp/model_best.pth.tar", split="test")

