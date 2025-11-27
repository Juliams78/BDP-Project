# 🔮 Fake News Detection using Spark + LSTM

## 🎯 Purpose
The primary objective is to replicate the experiment conducted on the paper [**Real-Time Fake News Detection Using Big Data Analytics and Deep Neural Network (IEEE, 2023)**](https://doi.org/10.1109/TCSS.2023.3309704) as 
a requirement for the final project of the discipline Big Data Programming for the Big Data course at Chungbuk National University.

The paper describes a Fake News Detection Model using LSTM and Deep Neural Network, with the intention of using the model on real-time fake news detection in social networks, 
where time is an important feature to avoid the spread of fake-news and misinformation. 

The experiment conducted in this project was conducted in a smaller scale and using different datasets from [Kaggle](https://www.kaggle.com/) in two different environments. The first environment being Google Colab and the other being the 
Docker Desktop installed in a local machine. 

---

## 📂 Google Colab
- `Telecom_X_Final.ipynb` → notebook principal com todo o pipeline de análise, modelagem e avaliação.  
- `dados.csv` → conjunto de dados após o pré-processamento.  

---

## ⚙️ Preparação dos Dados
- **Classificação das variáveis**:  
  - Variáveis categóricas (ex.: forma de pagamento, tipo de contrato).  
  - Variáveis numéricas (ex.: tenure, valor mensal da fatura).  

- **Pré-processamento**:  
  - Normalização/escala para variáveis numéricas, quando necessário.  
  - Codificação de variáveis categóricas com *one-hot encoding* (`pd.get_dummies`).  
  - Separação dos dados em **treino (70%)** e **teste (30%)**.  

- **Justificativas de modelagem**:  
  - **Regressão Logística** escolhida por sua interpretabilidade.  
  - **Random Forest** aplicada para capturar relações não lineares e identificar variáveis mais importantes.  

---

## 📊 Análise Exploratória (EDA)
Durante a EDA foram gerados gráficos e insights, como:  
- Distribuição de clientes por tipo de contrato e sua relação com churn.  
- Boxplots comparando valor da fatura mensal entre clientes que cancelaram ou não.  
- Gráficos de barras mostrando a influência da forma de pagamento na taxa de churn.  

Essas visualizações ajudaram a direcionar a modelagem e entender os fatores mais relevantes para evasão.

---
Foram avaliados dois modelos de Machine Learning: **Regressão Logística** e **Random Forest**.  
Os principais resultados obtidos foram:

| Modelo               | Acurácia | Precisão | Recall | F1-score | AUC-ROC |
|----------------------|----------|----------|--------|----------|---------|
| Regressão Logística  | 75 %   | 52 %   | 81 % | 63 %   | 0.84    |
| Random Forest        | 78 % | 58 % | 60 % | 59 % | 0.82 |
  

🔎 **Insights**:  
Embora tenha demonstrado pior acurácia, o modelo de **Regressão Logística** apresentou o melhor desempenho geral, equilibrando recall e AUC-ROC, o que é essencial para prever corretamente os clientes que realmente irão cancelar.
