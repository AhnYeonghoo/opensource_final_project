import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from bs4 import BeautifulSoup


df = pd.read_excel('./국방예산추이.xlsx' )
df = pd.DataFrame(df)
print(df)