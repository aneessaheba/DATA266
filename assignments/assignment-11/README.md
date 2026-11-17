# Assignment 11: CLIP Caption Similarity and Zero Shot Classification on CIFAR 10

Model: CLIP ViT B/32 from OpenAI, loaded with Hugging Face transformers. Data: CIFAR 10 test set.

## Part 1: Caption vs images

Caption "a photo of a dog" compared against 1000 test images, 100 from each class. All of the top 8 matches are dogs. Mean similarity is highest for the dog class at 0.259 and lowest for ship and truck at 0.195.

## Part 2: Zero shot classification

One prompt per class, "a photo of a {class}", on the full 10,000 image test set. Overall accuracy is 0.888.

Per class accuracy:

* airplane 0.901
* automobile 0.976
* bird 0.888
* cat 0.860
* deer 0.765
* dog 0.884
* frog 0.774
* horse 0.976
* ship 0.955
* truck 0.901

The confusion matrix and the discussion of the results are in the notebook.

## Files

* notebooks/assignment11_clip.ipynb, the notebook with outputs

## Rerunning

Uncomment the first cell to install packages, then run all cells. CIFAR 10 is downloaded automatically.
