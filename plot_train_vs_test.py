# usage
# python plot_train_vs_test.py relative_dir model_loss_all.txt
# example
# python plot_train_vs_test.py tennis_gpt_rnn model_loss_all.txt
import pandas as pd
import plotly.express as px
import sys
import os
import plotly
# Assuming your CSV file is in arg1

out = sys.argv[1]
file = sys.argv[2]
file_path = os.path.join('.\\',out,file)
print(file_path)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Create a line plot using Plotly Express
fig = px.line(df, x='step', y=['train_loss', 'test_loss'], labels={'value': 'Loss'}, title='Train Loss vs Test Loss')

# Show the plot
#fig.show()
out_path = os.path.join('.\\',out,'train_vs_test.html')
plotly.offline.plot(fig,filename=out_path)
