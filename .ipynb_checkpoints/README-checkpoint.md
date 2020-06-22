### Data Science


  ...in Python and R.

##### Data Science is a field that can be broadly defined that is made up of following activities.
Source: [50 years of Data Science](http://courses.csail.mit.edu/18.337/2015/docs/50YearsDataScience.pdf)

1. Data Exploration and Preparation: involves cleaning data and manipulating it for further analysis.
2. Data Representation and Transformation: Several different forms of representing data. Tabular structures in the first course, text data in the fourth course, and graph-based data in the last course.
3. Computing with Data: Pipelining and how data scientist need to be able to work with different languages for
different parts on an analysis project.

Unlike enterprise software projects, where you might use one language for
implementing all of the functionality you need.
Modern data science projects can span many different languages and
computing paradigms.
Knowing when to use the right tool for the job is an important attribute. 

4. Data Modeling: Predictive modeling, generative modeling.

5. Data visualization and Presentation: Charting, graphing, 3D visualization and interactive environments.


#### Tool that is used:

 Jupyter LAB
 
 - Drag operations
 - Realtime Markdown preview and Python kernel
     - Create console for the following code by right clicking and selecting Create Console for the editor
     - Execute the kernel by pressing shift enter LINE BY LINE. Note: Please refrain from using multiline input as kernel will not be able to parse it. (As Python is an interpreter)
     
 
 Following is the Python section:
 
 ```python
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np


data = {'x':np.random.rand(100),'y':np.random.rand(100),'color':np.random.rand(100),'size': 100.0*np.random.rand(100)}
df = pd.DataFrame(data)
df.head()
```
```python
style.use('seaborn-dark-palette')

plt.scatter('x', 'y', c='color', s='size', data=df,cmap=plt.cm.Blues)
plt.xlabel('x')
plt.xlabel('y')

plt.title("The data that we collected")
 ```