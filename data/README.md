# Sample images (for the demo apps)

The "Learn" tab shows random MRI images from here, so it needs some images.

You only need a small set (25 test images per class):

1. In the Kaggle training notebook, run `training/03_make_sample_set.py` as a new cell.
2. Download `samples.zip` from the Kaggle Output panel and unzip it here, so you get:

```
data/samples/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

These folders are in `.gitignore`, so they won't be pushed to GitHub.
