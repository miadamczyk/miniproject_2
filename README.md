# Explainable Artificial Intelligence (XAI)

Notebooki przedstawione w reozytrium wprowadzają do technik wyjaśnialności modeli sztucznej inteligencji w wizji komputerowej.

## Opis projektu

Ten projekt to zestaw materiałów w formie Jupyter Notebook, który krok po kroku wprowadza w zagadnienia interpretowalności głębokich sieci neuronowych. Projekt skupia się na metodach pozwalających zrozumieć, dlaczego model podjął konkretną decyzję.


## Struktura projektu

```text
xai-project/
├── 1_Wstęp.ipynb          # Teoria XAI i praktyczna implementacja Grad-CAM
├── 2_ProtoPNet.ipynb      # Architektura "This Looks Like That" (prototypy)
├── 3_LIME.ipynb           # Lokalne wyjaśnienia niezależne od modelu
├── 4_EPIC.ipynb           # Wyjaśnianie sieci pretrenowanych przez prototypy
├── images/                # Folder z grafikami wykorzystywanymi w materiałach
├── requirements.txt       # Zależności Python 
├── environment.yml        # Środowisko Conda
└── README.md              # Dokumentacja projektu
```

## Instalacja

### Opcja 1: Conda (zalecane)

```bash
conda env create -f environment.yml
conda activate xai-notebooks
```

### Opcja 2: pip

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Uruchomienie

```bash
jupyter notebook -nazwa_notebooka-
```

## Zawartość notebooków

### 1_Wstęp
1. Wprowadzienie do XAI w CV.
2. Przedstawienie metod: 
   - Saliency Maps
   - Concept Bottleneck Models
   - Prototype Methods
3. Obowiązowe zadanie nr 1 - implementacja GradCAM.

### 2_ProtoPNet
1. Przedstawienie modelu.
2. Implementacja oraz trening.
3. Prezentacja wyników działania modelu.
4. Nieobowiązkowe zadanie nr 2 - pytania o ProtoPNet.

### 3_LIME

### 4_EPIC

## Wymagania

- numpy>=1.24.0,<2.0.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- scikit-learn>=1.2.0
- opencv-python
- torch>=2.0.0
- torchvision>=0.15.0
- lime>=0.2.0
- jupyter>=1.0.0
- ipykernel>=6.0.0
- tqdm

## Autorzy

Miłosz Adamczyk, Yury Sarosek 
