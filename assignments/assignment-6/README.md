# Assignment 6: Self Supervised and Contrastive Learning on STL10

ResNet18 with no pretrained weights in every part. SEED = 5330.

| Part | Method | Test accuracy |
|---|---|---|
| A | Supervised, 500 labels, 15 epochs | 0.3783 |
| B | Rotation pretraining on 100,000 unlabeled images, frozen encoder, linear layer | 0.4476 |
| C | SimCLR on 20,000 unlabeled images, tau 0.2, frozen encoder, linear layer | 0.5401 |

Part D shows top 5 cosine nearest neighbors for three test queries with each encoder. The analysis is below and in the last cell of the notebook.

## Files

* `notebooks/assignment6_ssl_stl10.ipynb`, the notebook with outputs

## Rerunning

Requires `torch`, `torchvision`, `matplotlib` and `jupyter`. STL10 downloads to `data/` on first run.

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace assignment6_ssl_stl10.ipynb
```

## Analysis

| Model | Linear test accuracy | Precision at 5 |
|---|---|---|
| Supervised (Part A) | 0.3783 | 0.3347 |
| Rotation (Part B) | 0.4476 | 0.4191 |
| SimCLR (Part C) | **0.5401** | **0.4720** |

**Best model.** SimCLR performed best on both metrics. Its loss asks two augmented views of one image to match while being pushed away from every other image in the batch, so the encoder must keep features that survive cropping, flipping and color changes: object shape and parts. Those are the features that separate the ten classes, so a linear layer can read the class out of them.

**Rotation.** Predicting rotation needs some knowledge of objects (where the sky, legs or head usually are), so it beats the supervised baseline. But the task can also be solved with low level cues such as horizon lines, lighting direction and black borders, and it reached 89% rotation accuracy without needing fine class detail. Its features are therefore less class specific than SimCLR's.

**Supervised.** The supervised model saw only 500 labeled images. ResNet18 has about 11 million parameters, so it memorized the training set (loss near 0.2 to 0.4) and generalized poorly. It never used the 100,000 unlabeled images that the other two models learned from.

**Possible improvements.**
* Supervised: stronger augmentation (random crops, color jitter), weight decay, a learning rate schedule, and more epochs.
* Rotation: predict all four rotations of every image per step, train longer, and probe an earlier layer, whose features are usually more general than the last block, which specializes in rotation.
* SimCLR: train on all 100,000 images for many more epochs, use larger batches (more negatives), and add Gaussian blur. SimCLR normally trains for hundreds of epochs.
* All: a smaller stem for 96 pixel inputs (3x3 first convolution, no max pool) keeps more spatial detail.

**Nearest neighbor visualizations.** Counting correct neighbors over the three queries gives SimCLR 10 of 15, Supervised 9 and Rotation 7, matching the precision at 5 ranking across the full test set. SimCLR returned five ships for the ship query and four deer for the deer query, matching on the object itself despite changes in color and background. Rotation neighbors mostly match by layout and orientation: the ship query pulled in three airplanes with a similar horizontal shape and a horizon line. Supervised neighbors are a mix of right class and similar looking scenes (airplanes for the ship, a cat and a dog for the deer). SimCLR's weak result on the monkey query (dogs and a cat with similar brown fur) shows its features still rely partly on color and texture after only 20 epochs. Overall, SimCLR embeddings group images by semantic class best, Rotation embeddings group them by geometry and scene layout, and the Supervised embeddings are limited by having seen only 500 images.
