#%%
import pandas as pd
df_abnormal = pd.read_csv('ecg_dataset/ptbdb_abnormal.csv', header=None)
df_normal = pd.read_csv('ecg_dataset/ptbdb_normal.csv', header=None)
df = pd.concat([df_abnormal, df_normal], ignore_index=True)
df

#%%
import matplotlib.pyplot as plt
import numpy as np

def plt_normal(df):
  ypoints = df[df[187] < 1].iloc[:, :-1].sample(1).squeeze().to_numpy()
  plt.plot(ypoints, color = 'green', label='normal')
  plt.legend()
  plt.show()

plt_normal(df)

# %%

def plt_abnormal(df):
  ypoints = df[df[187] > 0].iloc[:, :-1].sample(1).squeeze().to_numpy()
  plt.plot(ypoints, color = 'red', label='abnormal')
  plt.legend()
  plt.show()

plt_abnormal(df)


#%%
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

def plt_gaf_normal(df):
    X = df[df[187] < 1].iloc[:, :-1].to_numpy()
    gaf = GramianAngularField(image_size=187)
    im_normal = gaf.fit_transform(X)
    plt.axis('off')
    for i in range(im_normal.shape[0]):
      img = plt.imshow(im_normal[i])
      plt.savefig(f"task1-normal-img/normal-img-{i}.png", bbox_inches='tight')    

plt_gaf_normal(df)

#%%
def plt_gaf_abnormal(df):
    X = df[df[187] > 0].iloc[:, :-1].to_numpy()
    gaf = GramianAngularField(image_size=187)
    im_normal = gaf.fit_transform(X)
    plt.axis('off')
    for i in range(im_normal.shape[0]):
      img = plt.imshow(im_normal[i])
      plt.savefig(f"task1-abnormal-img/abnormal-img-{i}.png", bbox_inches='tight')    

plt_gaf_abnormal(df)

# def plt_gaf_abnormal(df):
#     X = df[df[187] > 0].iloc[:, :-1].sample(1).to_numpy()
#     gaf = GramianAngularField(image_size=187)
#     im_normal = gaf.fit_transform(X)
#     plt.axis('off')
#     img = plt.imshow(im_normal[0])
#     #plt.savefig("test.png", bbox_inches='tight')
    
# plt_gaf_abnormal(df)



#%%



# from sklearn.model_selection import train_test_split
# y = df.iloc[:,-1:].squeeze().to_numpy()
# X = df.iloc[:, :-1].to_numpy()
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# #%%
# from pyts.image import GramianAngularField
# import matplotlib.pyplot as plt

# gaf = GramianAngularField(image_size=24)
# im_train = gaf.fit_transform(X_train)
# im_test = gaf.transform(X_test)

# # plot one image
# plt.imshow(im_train[0])
# plt.show()
