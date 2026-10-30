# Notebooks

Use Google Colab for GPU runs (Runtime -> Change runtime type -> GPU).

Minimal Colab setup:

```python
!git clone https://github.com/<owner>/explainable-hinglish-hate-speech.git
%cd explainable-hinglish-hate-speech
!pip install -q -r requirements.txt

# Mount Google Drive to keep datasets and checkpoints between sessions
from google.colab import drive
drive.mount('/content/drive')
# e.g. copy data:  !cp -r /content/drive/MyDrive/hate_project_data/* data/raw/

!python -m src.train_transformer --config configs/muril_hasoc.yaml
```

Notebook naming: `NN_short_description.ipynb`, e.g. `01_data_exploration.ipynb`.
Clear large outputs before committing a notebook.
