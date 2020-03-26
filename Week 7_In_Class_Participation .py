#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.style.use('seaborn-whitegrid')


# In[5]:


X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]


# In[6]:


plt.scatter(X,Y)
plt.xlim(0,1000)
plt.ylim(0,100)


# In[7]:


#scatter plot color
plt.scatter(X,Y, s=60, c='red', marker='^')


# In[8]:


#change axes range
plt.xlim(0,1000)
plt.ylim(0,100)


# In[9]:


#add title
plt.title('Relationship Between Temperature and Iced Coffee Sales')


# In[10]:


#add x and y labels
plt.xlabel('Sold Cofee')
plt.ylabel('Temperature in Fahrenheit')


# # Importing Numpy and Calling Its Functions

# In[11]:


import matplotlib.pyplot as plt
import numpy as np


# In[12]:


plt.style.use('seaborn-whitegrid')


# In[13]:


#Create an Empty Figure
fig = plt.figure()
ax = plt.axes()
x=np.linspace(0,10,1000) #returns evenly spaces numbers over a specified interval
ax.plot(x,np.sin(x))
plt.plot(x, np.sin(x))
plt.plot(x,np.cos(x))

#set the x and y range
plt.xlim(0,11)
plt.ylim(-2,2)
plt.axis('tight')

#add title
plt.title('Plotting data using sin and cos')


# # Seaborn

# In[14]:


import pandas as pd


# In[15]:


import seaborn as sns


# # Plotly

# In[16]:


import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[17]:


x = np.random.randn(2000)
y = np.random.randn(2000)


# In[18]:


iplot([go.Histogram2dContour(x=x, y=y,
contours=dict (coloring='heatmap')),
go.Scatter(x=x, y=y, mode='markers',
marker=dict(color='white', size=3,
opacity=0.3))], show_link=False)


# In[19]:


import plotly.offline as offline
import plotly.graph_objs as go


# In[20]:


offline.plot({'data': [{'y': [14, 22, 30,44]}],
         'layout': {'title': 'Offline Plotly', 'font':
         dict(size=16)}}, image='png')


# In[21]:


from plotly import __version__


# In[22]:


from plotly.offline import download_plotlyjs, plot, iplot##, init_notebook_mode(connected=True)
print(__version__)


# In[23]:


import plotly.graph_objs as go
plot([go.Scatter(x=[95, 77, 84], y=[75, 67, 56])])


# In[25]:


import pandas as pd 
import numpy as np
df = pd.DataFrame(np.random.randn(200,6),index= pd.date_range('1/9/2009', periods=200), columns= list('ABCDEF'))
df.plot(figsize=(20, 10)).legend(bbox_to_anchor=(1, 1))


# In[26]:


df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb','March','April', 'May'])
df.plot.bar(figsize=(20, 10)).legend(bbox_to_anchor=(1.1, 1))


# In[28]:


import pandas as pd
df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb','March','April', 'May']) 
df.plot.bar(stacked=True,
figsize=(20, 10)).legend(bbox_to_anchor=(1.1, 1))


# In[29]:


df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb',
'March','April', 'May']) 
df.plot.barh(stacked=True,
figsize=(20, 10)).legend(bbox_to_anchor=(1.1, 1))


# In[31]:


df = pd.DataFrame(np.random.rand(20,5), columns=['Jan','Feb','March','April', 'May'])
df.plot.hist(bins= 20, figsize=(10,8)).legend
bbox_to_anchor=(1.2, 1)


# In[32]:


df=pd.DataFrame({'April':np.random.randn(1000)+1,'May':np.random.
randn(1000),'June': np.random.randn(1000) - 1}, columns=['April',
'May', 'June'])
df.hist(bins=20)


# In[33]:


df = pd.DataFrame(np.random.rand(20,5),
columns=['Jan','Feb','March','April', 'May'])
df.plot.box()


# In[37]:


df = pd.DataFrame(np.random.rand(20,5),
columns= ['Jan','Feb','March','April', 'May'])
df.plot.area(figsize=(6, 4)).legend(bbox_to_anchor=(1.3, 1))


# In[38]:


df = pd.DataFrame(np.random.rand(20,5),columns= ['Jan','Feb', 'March','April', 'May'])
df.plot.scatter(x='Feb', y='Jan', title='Temperature over two months')


# In[ ]:




