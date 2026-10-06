# Assignment 6: Self Supervised and Contrastive Learning on STL10

ResNet18 with no pretrained weights in every part. SEED = 5330.

| Part | Method | Test accuracy |
|---|---|---|
| A | Supervised, 500 labels, 15 epochs | 0.3783 |
| B | Rotation pretraining on 100,000 unlabeled images, frozen encoder, linear layer | 0.4476 |
| C | SimCLR on 20,000 unlabeled images, tau 0.2, frozen encoder, linear layer | 0.5401 |

Part D shows top 5 cosine nearest neighbors for three test queries with each encoder. The analysis is the last cell of the notebook.

## Files

* `notebooks/assignment6_ssl_stl10.ipynb`, the notebook with outputs
* `scripts/build_notebook.py`, generates the notebook

## Rerunning

Requires `torch`, `torchvision`, `matplotlib` and `jupyter`. STL10 downloads to `data/` on first run.

```
python scripts/build_notebook.py
cd notebooks
jupyter nbconvert --to notebook --execute --inplace assignment6_ssl_stl10.ipynb
```
