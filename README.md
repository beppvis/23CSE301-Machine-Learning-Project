## Regression 
- [CalCOFI dataset](https://www.kaggle.com/datasets/sohier/calcofi/data)
- [Metadata](https://calcofi.org/data/oceanographic-data/bottle-database/)

### Instructions
1) Download the dataset
2) copy the `cast.csv` file to `data` folder


## Clustering
Using [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

### Instructions
1) Download the dataset , which is a xlxs file
2) place it `data/clustering` folder
3) execute all notebook blocks

### Customer segmentation app

The included Streamlit app retrains the notebook's 5-cluster K-Means model from
`data/clustering/Online Retail.xlsx`, then assigns manually entered customer
details to a cluster and explains the segment.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter days since the last purchase, number of purchases, total spend, and the
smallest/largest order values. The app validates that the order values are
consistent before making a prediction.
