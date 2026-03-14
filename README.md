# Few-Shot Learning

Interaktywny notebook edukacyjny wprowadzający do technik Few-Shot Learning.

## Opis projektu

Ten projekt to zestaw materiałów dydaktycznych w formie **Jupyter Notebook**, który krok po kroku wprowadza w:

1. **Metric-Based Few-Shot Learning** – klasyfikacja nowych klas przy użyciu metod metrycznych opartych na porównywaniu embeddingów przy dostępnej małej ilości danych.
2. **Meta-Learning** – wprowadzenie do MAML (Model-Agnostic Meta-Learning) i nauki szybkiej adaptacji do nowych zadań.

## Struktura projektu

```
miniproject/
├── metric-based.ipynb      # Notebook z wprowadzeniem i metodami metric-based
├── meta-learning.ipynb     # Notebook z meta-learning
├── requirements.txt        # Zależności Python (pip)
├── environment.yml         # Środowisko Conda
├── README.md               # Ten plik
└── Images/                 # Folder ze zdjęciami wykorzysywanymi w notebookach
```

## Instalacja

### Opcja 1: Conda (zalecane)

```bash
conda env create -f environment.yml
conda activate fsl-notebook
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
jupyter notebook metric-based.ipynb
```

```bash
jupyter notebook meta-learning.ipynb
```

## Zagadnienia poruszane w notebookach

### Few-Shot Learning
- Czym jest Few-Shot Learning?
- Dlaczego jest przydatne?
- Kluczowe pojęcia: N-way K-shot, Support Set, Query Set

### Metric-Based Few-Shot Learning
- Metody Metric-Based - podstawy

### Prototypical Networks, Matching Networks, Siamese Networks
- Teoria i implementacja
- Przykład na CIFAR-10 z pretrenowanym ResNet-18
- Wizualizacje t-SNE i prównanie z baseline - Nearest Neighbor

### Meta-Learning i MAML (Model-Agnostic Meta-Learning)
- Czym jest meta-learning?
- Inner loop vs Outer loop
- Implementacja MAML od podstaw
- Porównanie z metodami metric-based

### Ćwiczenia
- **Ćwiczenie 1 (obowiązkowe):** Odpowiedz na pytanie dotyczące wizualizacji TSNE
- **Cwiczenie 2 (dodatkowe):** Zaimplementuj kontekstowy Matching Network
- **Ćwiczenie 3 (obowiązkowe):** Zaimplementuj Inner-Loop Adaptation (MAML)
- **Ćwiczenie 4 (dodatkowe)** Porównaj metody

## Wymagania

- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.2.0
- numpy >= 1.24.0,<2.0.0
- jupyter >= 1.0.0
- ipykernel >= 6.0.0

## Bibliografia

- [xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision](https://worldline.com/en/home/main-navigation/resources/blogs/2021/ever-heard-of-the-ai-black-box-problem) (dostęp 13.03.2026)
- [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) (dostęp 13.03.2026)
- [This Looks Like That: Deep Learning for Interpretable Image Recognition](https://learnopencv.com/intro-to-gradcam/) (dostęp 13.03.2026)
- [What is a Convex Function?](https://thedatacopywriter.substack.com/p/what-is-a-convex-function) (dostęp 13.03.2026)

## Autorzy

Miłosz Adamczyk, Piotr Bednarski
