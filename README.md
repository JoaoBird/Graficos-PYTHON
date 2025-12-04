# 📊 Dashboard e Relatório em Python para Regressão Linear e Não Linear

Este projeto oferece um **dashboard interativo** e um **relatório automático** para análise estatística, regressão linear, polinomial e exponencial.  
Ele combina `pandas`, `numpy`, `sklearn`, `scipy`, `statsmodels`, `plotly` e **Streamlit** para visualização dinâmica dos dados.

---

## 📁 Arquivo principal
`dashboard_regressao.py`

---

## ▶️ Como usar (via terminal)

### **1️⃣ Instale as dependências:**
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels scipy plotly streamlit openpyxl
pip install seaborn
```

---

### **2️⃣ Coloque o arquivo de dados na mesma pasta:**
```
tabelinha.xlsx
```

---

### **3️⃣ Execute o relatório estático (opcional):**
```bash
python dashboard_regressao.py --report
```

---

### **4️⃣ Execute o dashboard interativo (Streamlit):**
```bash
streamlit run dashboard_regressao.py
```

---

## 🧠 O que o script faz
- Carrega e trata os dados automaticamente  
- Faz análise estatística completa  
- Executa:
  - Regressão linear  
  - Regressão polinomial  
  - Ajuste exponencial  
- Gera gráficos interativos com **Plotly**  
- Exibe tudo em um dashboard **Streamlit**  
- Gera relatório automático quando executado com `--report`

---

## 🔎 Observação importante
O **Streamlit** exige chamadas diretas da API dentro do arquivo quando executado com:

```bash
streamlit run dashboard_regressao.py
```

Para gerar **apenas o relatório**, use a flag:

```bash
python dashboard_regressao.py --report
```

---

