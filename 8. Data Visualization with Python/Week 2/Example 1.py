import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Sales data
data = {
    'Category': ['Electronics', 'Electronics', 'Electronics',
                 'Furniture', 'Furniture', 'Furniture',
                 'Clothing', 'Clothing', 'Clothing'],
    'Subcategory': ['Laptops', 'Smartphones', 'Tablets',
                    'Chairs', 'Tables', 'Sofas',
                    'Men', 'Women', 'Kids'],
    'Sales': [120000, 80000, 30000,
              50000, 40000, 20000,
              70000, 90000, 40000]
}
df = pd.DataFrame(data)
# Creating the treemap
fig = px.treemap(
    df,
    path=['Category', 'Subcategory'],
    values='Sales',
    title='Sales Data Treemap'
)
fig.show()


import plotly.express as px
import pandas as pd
# Sales data
data = {
    'Category': ['Category 1', 'Category 1', 'Category 2', 'Category 2', 'Category 3'],
    'Subcategory': ['Subcategory 1A', 'Subcategory 1B', 'Subcategory 2A', 'Subcategory 2B', 'Subcategory 3A'],
    'Value': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)
# Creating the treemap
fig = px.treemap(
    df,
    path=['Category', 'Subcategory'],
    values='Value',
    title='Sales Data Treemap'
)
fig.show()


df_pivot = pd.read_excel("./pivot_chart.xlsx")
pivot_table = df_pivot.pivot_table(index='Date', columns=['Category','Subcategory'], values='Sales', aggfunc=np.sum)

pivot_table.plot(kind='bar', figsize=(14, 8))
plt.title('Sales Summary of IT Products by Category and Subcategory')
plt.xlabel('Quarters')
plt.ylabel('Total Sales')
plt.grid(False)
plt.legend(title=('Category', 'Subcategory'), bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
