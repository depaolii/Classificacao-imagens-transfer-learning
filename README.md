# Classificação de Imagens com Transfer Learning

Classificador automático de **11 órgãos do corpo humano** em imagens de CT scan, utilizando Transfer Learning com DenseNet121 e o dataset OrganAMNIST (MedMNIST) + VÍDEO EXPLICATIVO de 5min

---

## Tecnologias Utilizadas

| Tecnologia | Versão / Descrição |
|---|---|
| Python | 3.x |
| TensorFlow / Keras | 2.x |
| DenseNet121 | Backbone pré-treinado (ImageNet) |
| MedMNIST | Dataset OrganAMNIST |
| Google Colab | Ambiente de execução (GPU T4) |
| Matplotlib | Visualização das curvas de treino |

---

## Descrição do Projeto

Este projeto aplica **Transfer Learning com redes neurais convolucionais (CNN)** para classificação automática de imagens médicas, utilizando o dataset **OrganAMNIST** da biblioteca MedMNIST.

### Backbone: DenseNet121

O **DenseNet121** foi escolhido pelos seguintes motivos:

- **Arquitetura densa:** conecta todas as camadas entre si, preservando detalhes finos essenciais para diagnóstico médico
- **Comprovado em imagens médicas:** base do projeto CheXNet (Stanford), que superou radiologistas na detecção de pneumonia em raio-X
- **Eficiente:** menos parâmetros que VGG16 e ResNet50, reduzindo o risco de overfitting

### Dataset: OrganAMNIST

| Característica | Detalhe |
|---|---|
| Fonte | MedMNIST (`pip install medmnist`) |
| Tipo | Imagens de CT scan |
| Classes (11) | Bexiga, Fêmur esq/dir, Coração, Rim esq/dir, Fígado, Pulmão esq/dir, Pâncreas, Baço |
| Resolução original | 28×28 px → redimensionado para 224×224 px |
| Download | Automático via código |

---

## Estratégia de Treinamento

### Fase 1 — Backbone congelado

- DenseNet121 completamente congelado (pesos ImageNet preservados)
- Apenas a camada de classificação final é treinada
- **5 épocas** | Learning Rate: `1e-3` (Adam)

### Fase 2 — Fine-tuning

- Últimas **10 camadas** do DenseNet121 descongeladas
- Learning Rate reduzido para `1e-5` (100× menor, para não destruir o conhecimento pré-treinado)
- **5 épocas**

---

## Estrutura do Repositório

```
/
├── CNN_Transfer_learning.ipynb     # Notebook principal (Google Colab)
├── README.md                       # Este arquivo
├── curva_accuracy.png              # Curva de Accuracy — Fase 1
├── curva_loss.png                  # Curva de Loss — Fase 1
├── curva_accuracy_finetune.png     # Curva de Accuracy — Fine-tuning
├── curva_loss_finetune.png         # Curva de Loss — Fine-tuning
└── link_video.txt                  # Link do vídeo explicativo (YouTube)
```

---

## Como Executar

1. Acesse [Google Colab](https://colab.research.google.com)
2. Faça upload do arquivo `CNN_Transfer_learning.ipynb`
3. Ative a GPU: `Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU`
4. Execute todas as células em ordem

> Nenhuma configuração adicional necessária. O dataset é baixado automaticamente pelo código.

---

## Features

- Download automático do dataset OrganAMNIST via MedMNIST
- Pipeline otimizado com `tf.data.Dataset` para evitar estouro de memória (OOM)
- Conversão automática de escala de cinza para RGB (`as_rgb=True`)
- Treinamento em duas fases: feature extraction + fine-tuning
- Plotagem e salvamento automático das curvas de loss e accuracy
- Modelo salvo no formato `.keras` para reutilização

---

## Dificuldades Encontradas

- **Erro OOM (Out of Memory):** ao redimensionar as ~34 mil imagens para 224×224 de uma vez, a memória da GPU estourava. Resolvido com pipeline `tf.data.Dataset` processando em batches de 32
- **Escala de cinza → RGB:** o OrganAMNIST é originalmente em escala de cinza, mas o DenseNet121 exige 3 canais. Resolvido com o parâmetro `as_rgb=True`

---

## Vídeo Explicativo

Assista ao vídeo de apresentação do projeto (5–7 min) no YouTube:

**[Assistir no YouTube](https://youtu.be/S2ZtBMpjeqw)**

---

## Referência

- [MedMNIST](https://medmnist.com/)
- [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [DenseNet — CheXNet (Stanford)](https://stanfordmlgroup.github.io/projects/chexnet/)
