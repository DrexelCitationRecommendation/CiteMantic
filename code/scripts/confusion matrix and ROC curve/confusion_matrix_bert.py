# %%
import pandas as pd
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# %%
"""
Transfer learning for claim prediction using Discourse CRF model
"""
import sys
sys.path.insert(0, '..')
# sys.path.insert(0, '/Users/kchu/Documents/Projects/Senior Project/Claim Extraction/detecting-scientific-claim-master/')

import os
cwd = os.getcwd()
print('Current directory:', cwd)

# %%
# df_val = pd.read_csv('./biobert_y_true_pred_test.csv')
# y_true_val = np.array(df_val['y_true'])
# y_pred_val = np.array(df_val['y_pred'])

# cm = metrics.confusion_matrix(y_true_val, y_pred_val)
# print(cm)
# # plt.imshow(cm, cmap='binary')
# df_cm = pd.DataFrame(cm, index = ['true', 'false'],
#                   columns = ['true', 'false'])
# plt.figure(figsize = (6, 6))
# plt.xlabel('Predicted')
# plt.ylabel('True')
# sn.heatmap(df_cm, annot=True, fmt='g')
# plt.show()

# fpr, tpr, threshold = metrics.roc_curve(y_true_val, y_pred_val)
# roc_auc = metrics.auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

df_test = pd.read_csv('./biobert_y_true_pred_test.csv')
y_true_test = np.array(df_test['y_true'])
y_pred_test = np.array(df_test['y_pred'])

cm = metrics.confusion_matrix(y_true_test, y_pred_test)
print(cm)
df_cm = pd.DataFrame(cm, index = ['true', 'false'],
                  columns = ['true', 'false'])
plt.figure(figsize = (6, 6))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

fpr, tpr, threshold = metrics.roc_curve(y_true_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()