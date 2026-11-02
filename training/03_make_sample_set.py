# Paste as a NEW CELL at the end of the Kaggle training notebook.
# Makes a small demo set (25 test images per class) and zips it for download.
# Download /kaggle/working/samples.zip and unzip it into data/samples/ in the project.
import shutil, os
N_PER_CLASS = 25
out = "/kaggle/working/samples"
shutil.rmtree(out, ignore_errors=True)
for cls in CLASS_NAMES:
    os.makedirs(f"{out}/{cls}", exist_ok=True)
    for p in test_df[test_df.label == cls].sample(N_PER_CLASS, random_state=0)["path"]:
        shutil.copy(p, f"{out}/{cls}/")
shutil.make_archive("/kaggle/working/samples", "zip", out)
print("Done:", os.path.getsize("/kaggle/working/samples.zip") / 1e6, "MB")
