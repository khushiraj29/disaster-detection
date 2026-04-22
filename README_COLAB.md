# 🛡️ Google Colab Training Bridge

Training this model locally without a GPU might take 1-2 hours. In Colab, it takes **under 10 minutes**.

## 🚀 Fast Training Workflow

I have prepared everything for you to move this project to the cloud.

### Step 1: Upload the Project Bundle
I have created a special zip file for you: `disaster_project_colab.zip` in your project root.
1. Open [Google Colab](https://colab.research.google.com/).
2. Click the **Files** icon (folder) on the left sidebar.
3. Click the **Upload to session storage** button.
4. Select `disaster_project_colab.zip` from your computer.

### Step 2: Unzip and Setup
Run this in a Colab cell:
```python
!unzip disaster_project_colab.zip
!pip install -r requirements.txt
```

### Step 3: Train at Light Speed
Run this in a cell to start training:
```python
!python3 download_dataset.py  # Builds the dataset in the cloud
!python3 src/train.py         # Trains the model using Colab GPU
```

### Step 4: Get your Model Back
Once training is done, your model will be in `models/disaster_model_final.h5`. You can download it directly from the Files sidebar.

---

## 🎛️ Need a ready-to-use Notebook?
I've also created a fully interactive notebook for you: `notebooks/fast_colab_train.ipynb`.
You can upload this `.ipynb` file to Colab directly!
