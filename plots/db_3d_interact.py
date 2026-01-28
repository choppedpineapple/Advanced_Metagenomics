import plotly.express as px
fig = px.scatter_3d(
    df, x='PC1', y='PC2', z='PC3', 
    color='Cluster', text='Sequence',
    title='Interactive DBSCAN DNA Clusters'
)
fig.show()
