# ChallengeAluraTelecomX_Part_2

Este trabalho trata de visualizar e analisar a evasão de clientes na empresa Telecom X.

Na primeira parte tivemos a extração dos dados.

Utilizando os dados tratados do desafio anterior, foi feito o upload do arquivo no ambiente do google collab.

Em seguida foi feito a importação das bibliotecas, ajuste no aumento da visualização das colunas e a leitura do arquivo:

`import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
pd.set_option('display.max_columns', None)
dados = pd.read_csv('dados_tratados.csv')`

Apesar do arquivo estar previamente tratado ele ainda continha dados aninhados então foi feito o destrinchamento dos dados e criação de colunas ja fazendo alteração dos seus valores sendo "0" para "não" e "1" para "sim" já adiantando o processo de encoding:

`dados['NoPhoneService'] = dados['MultipleLines'].apply(lambda x: 0 if x in ['0', '1'] else 1)
dados['FiberOptic'] = dados['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
dados['MultipleLines'] = dados['MultipleLines'].replace('No phone service', '0')
dados['DSL'] = dados['InternetService'].apply(lambda x: 1 if x == 'DSL' else 0)
dados['FiberOptic'] = dados['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
dados['NoInternetService'] = dados['InternetService'].apply(lambda x: 1 if x == "0" else 0)
dados['InternetService'] = dados['InternetService'].replace('DSL', 1)
dados['InternetService'] = dados['InternetService'].replace('Fiber optic', 1)
dados['Male'] = dados['gender'].apply(lambda x: 1 if x == 'Male' else 0)
dados['Female'] = dados['gender'].apply(lambda x: 1 if x == 'Female' else 0)
dados['Month-to-month'] = dados['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
dados['One year'] = dados['Contract'].apply(lambda x: 1 if x == 'One year' else 0)
dados['Two year'] = dados['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)
dados['Eletronic check'] = dados['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
dados['Mailed check'] = dados['PaymentMethod'].apply(lambda x: 1 if x == 'Mailed check' else 0)
dados['Bank transfer'] = dados['PaymentMethod'].apply(lambda x: 1 if x == 'Bank transfer (automatic)' else 0)
dados['Credit card'] = dados['PaymentMethod'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)`

Depois foi feita a exclusão de algumas colunas:

`try:
    dados = dados.drop(columns=['customerID', 'gender', 'Contract'])
except KeyError:
    display(dados.head())
try:
    dados = dados.drop(columns=['PaymentMethod'])
except KeyError:
    display(dados.head())`

Depois desse tratamento agora o arquivo possui 7043 linhas × 31 colunas

Após isso foi gerado a proporção de clientes ativos e que deram evasão junto verificado se há uma desiquilíbrio entre as classes de margem de erro de 0.5%

`total_churn = dados['Churn'].value_counts()
proporcao_evasao = total_churn[1] / total_churn.sum()
proporcao_ativo = total_churn[0] / total_churn.sum()
imbalance_threshold = 0.005
display(total_churn)
print(f"Proporção de clientes que evadiram (Churn): {proporcao_evasao:.2%}")
print(f"Proporção de clientes que permaneceram ativos: {proporcao_ativo:.2%}")
if proporcao_evasao < imbalance_threshold or proporcao_ativo < imbalance_threshold:
    print("\nHá um desequilíbrio significativo entre as classes Churn e Não Churn.")
else:
    print("\nNão há um desequilíbrio significativo entre as classes Churn e Não Churn.")`

Foi constatado que não há nenhum desiquilíbrio significativo permitindo assim a continuação do projeto.

Proporção de clientes que evadiram (Churn): 26.54%
Proporção de clientes que permaneceram ativos: 73.46%

Também pode ser observado que aproximadamente a cada 4(quatro) clientes 1(um) deles é evasão.
Isso mostra que a perda de clientes está muito alta para os parâmetros da empresa.

Foi feita a tentativa de aplicação de SMOTE, porém algumas colunas ainda tinham valores não númericos.
Resumidamente o tratamento realizado foi:

* As colunas categóricas 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV' e 'StreamingMovies' foram convertidas com sucesso para representações numéricas (0s, 1s e 2s).
* A coluna 'Charges.Total' foi convertida para um formato numérico, tratando strings vazias ao substituí-las por NaN e, em seguida, preenchendo esses valores ausentes com a mediana da coluna.
* Após as etapas de limpeza e transformação de dados, todas as colunas no DataFrame foram confirmadas como sendo de tipos de dados numéricos (int64, float64 ou int32).
* O SMOTE foi aplicado com sucesso ao conjunto de dados, equilibrando a distribuição da classe 'Churn' através da sobreamostragem da classe minoritária.

SMOTE:

`X = dados.drop('Churn', axis=1)
y = dados['Churn']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Nova distribuição da classe Churn após SMOTE:")
display(y_resampled.value_counts())`

A distribuição atualizada ficou: 

Churn	
0	5174
1	5174

Fazendo assim as amostras estarem devidamente equilibradas.
Foi realizada a padronização dos dados utilizando StandardScaler:

`X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Dados de treino e teste divididos e features padronizadas.")
print("Forma de X_train_scaled:", X_train_scaled.shape)
print("Forma de X_test_scaled:", X_test_scaled.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)`

A separação ficou em 25% para teste e 75% para treinamento do modelo

Depois disso, a criação e exibição da matriz de correlação através de:

`correlation_matrix = dados.corr()
display(correlation_matrix)`

Em seguida da criação do mapa de calor das variáveis:

`plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()`

<img width="1233" height="947" alt="image" src="https://github.com/user-attachments/assets/b7ae2bbe-7abb-481c-b011-3ac419783835" />

Com base no mapa assim podemos realizar algumas análises: 

Varíaveis que aumentam as chances de evasão:

* Month-to-month (0.405): Clientes com contrato mensal têm uma correlação positiva notável com a evasão. Isso sugere que contratos de curto prazo estão fortemente associados a uma maior probabilidade de sair. No gráfico abaixo podemos ver uma simples relação.

<img width="704" height="547" alt="image" src="https://github.com/user-attachments/assets/e574393a-8e3f-45aa-b74c-d2ddde5509bf" />

* FiberOptic (0.308): Clientes com serviço de internet fibra óptica também apresentam uma correlação considerável com a evasão. Isso pode indicar problemas de qualidade ou custo associados a esse serviço. Segue o gráfico para uma melhor compreensão.

<img width="704" height="555" alt="image" src="https://github.com/user-attachments/assets/a54dd586-d98f-413f-a8a3-f236837e8fc8" />

* Eletronic check (0.302): Clientes que usam cheque eletrônico como forma de pagamento mostram uma correlação com a evasão. Isso pode estar relacionado a problemas com o processo de pagamento ou a um perfil de cliente mais propenso a evasão que prefere essa forma de pagamento. O gráfico abaixo em barra estacada demostra o que foi analisado.

<img width="704" height="549" alt="image" src="https://github.com/user-attachments/assets/11d79cb9-1bc8-4502-b0e3-41e106db3477" />


Varíaveis que diminuem as chances de evasão:

* tenure (-0.352): A duração do tempo como cliente (tenure) tem uma correlação negativa forte com a evasão. Isso é esperado, pois clientes que estão há mais tempo tendem a ser mais leais. Abaixo temos o gráfico box plot, em que temos a seguinte análise:
    * Comparação das Medianas: A linha central (mediana) na caixa para clientes que Não Evadiram (0) está em um valor de "tenure" consideravelmente maior do que a linha central na caixa para clientes que Evadiram (1). Isso sugere que a metade dos clientes que não evadiram tem um tempo de permanência maior do que a metade dos clientes que evadiram.
    * Distribuição Geral: A caixa e os "bigodes" (whiskers) para a classe "Não Evadiu (0)" estão localizados em valores de "tenure" mais altos no geral. Isso indica que a maioria dos clientes de longo prazo tende a não evadir.
Clientes que Evadiram (1): A caixa para clientes que Evadiram (1) está concentrada em valores de "tenure" mais baixos. Isso significa que a maioria dos clientes que evadem o fazem relativamente cedo em seu relacionamento com a empresa.

<img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/f3484cb3-7fef-42b1-8021-2d9713b5c559" />
    
    O box plot visualmente confirma a forte correlação negativa que vimos na matriz de correlação. Clientes com um tempo de permanência (tenure) mais longo têm uma probabilidade menor de evadir, enquanto clientes com "tenure" curto são mais propensos ao churn. Esta variável é claramente um fator importante na previsão da evasão de clientes.
 
* Two year (-0.302): Clientes com contrato de dois anos apresentam uma correlação negativa forte com a evasão. Contratos de longo prazo indicam um compromisso maior e menor probabilidade de sair.
* NoInternetService (-0.228): Clientes sem serviço de internet têm uma correlação negativa moderada com a evasão, o que faz sentido, pois a falta do serviço de internet remove um possível ponto de atrito que leva à evasão.
*OnlineSecurity (-0.171) e TechSupport (-0.165): Clientes que usam segurança online e suporte técnico mostram correlações negativas moderadas com a evasão, sugerindo que esses serviços de valor agregado ajudam a reter clientes.

Portanto as varíaveis mais importantes a princípio são:

* Month-to-month
* FiberOptic
* Eletronic check
* tenure
* Two year
* NoInternetService
* OnlineSecurity
* TechSupport

Agora vamos partir para os modelos:

Eu escolhi a Regressão Logística e Support Vector Machine (SVM)

A regressão logística é um bom ponto de partida para entender quais features são mais importantes na previsão de churn. Como já padronizamos os dados, a otimização do modelo será mais eficiente, enquanto os dados estão padronizados, o desempenho do SVM (que é sensível à escala) será otimizado. É uma boa opção para tentar capturar relações não lineares que a Regressão Logística pode não identificar.

O código da Regressão Logística ficou:

`logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
print("Avaliação do Modelo de Regressão Logística:")
print("Acurácia:", accuracy_score(y_test, y_pred_logistic))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_logistic))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_logistic))`

Obteve as seguintes informações:

<img width="373" height="268" alt="image" src="https://github.com/user-attachments/assets/dfc5586c-3f88-4bc0-8331-a2aa8bbda6a1" />

Enquanto o código Support Vector Machine(SVM) ficou:

`svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("Avaliação do Modelo Support Vector Machine (SVM):")
print("Acurácia:", accuracy_score(y_test, y_pred_svm))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_svm))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_svm))`

Obtive as seguintes informações:

<img width="371" height="265" alt="image" src="https://github.com/user-attachments/assets/5f411d3f-873f-4e60-9fb8-5e7a22ee8ef8" />

Entre os dois modelos foi constatado que o modelo SVM teve um melhor desempenho, mesmo sendo pouca a diferença, em relação a acurácia precision e recall.

Sobre o overfitting ou underfitting, foi feito um código para analisar o treinamento e então fazer a comparação

Regressão Logística

`y_pred_logistic_train = logistic_model.predict(X_train_scaled)
print("Acurácia (Treino):", accuracy_score(y_train, y_pred_logistic_train))
print("\nMatriz de Confusão (Treino):\n", confusion_matrix(y_train, y_pred_logistic_train))
print("\nRelatório de Classificação (Treino):\n", classification_report(y_train, y_pred_logistic_train))`

e foi obtido isso:

<img width="371" height="265" alt="image" src="https://github.com/user-attachments/assets/54bb5e21-ede4-4071-b310-c9971c57b5b8" />

SVM

`y_pred_svm_train = svm_model.predict(X_train_scaled)
print("Acurácia (Treino):", accuracy_score(y_train, y_pred_svm_train))
print("\nMatriz de Confusão (Treino):\n", confusion_matrix(y_train, y_pred_svm_train))
print("\nRelatório de Classificação (Treino):\n", classification_report(y_train, y_pred_svm_train))`

E foi obtido isto:

<img width="376" height="265" alt="image" src="https://github.com/user-attachments/assets/b864e874-2269-4464-847c-d20eb613b0c2" />


Podemos analisar que não houve um overfitting severo já que a diferença foi de 1 à 2 pontos percentuais, em relação a underfitting o desempenho no conjunto de teste não é baixo (acima de 84% de acurácia), o que indica que não houve underfitting significativo. Os modelos foram capazes de aprender as relações nos dados de forma adequada.

---

#Conclusão


Com base na análise exploratória, incluindo a matriz de correlação e as visualizações gráficas, e considerando a natureza dos modelos preditivos utilizados (Regressão Logística e SVM), afirmamos os principais fatores que influenciam a evasão de clientes neste dataset:

1.  **Contrato Mensal ('Month-to-month'):** Este é um dos fatores com maior correlação positiva com a evasão. Clientes com contratos mensais têm uma flexibilidade maior para cancelar o serviço, o que se reflete em uma probabilidade significativamente maior de evadir em comparação com aqueles com contratos de um ou dois anos. A Regressão Logística, sendo um modelo linear, capturaria essa relação direta, enquanto o SVM também seria capaz de modelar essa influência, mesmo que existam interações não lineares com outras features.

2.  **Serviço de Fibra Óptica ('FiberOptic'):** Clientes que utilizam o serviço de fibra óptica também demonstram uma tendência maior à evasão. Embora seja um serviço de alta velocidade, a evasão associada a ele pode estar relacionada a expectativas de serviço, preço ou problemas técnicos específicos da infraestrutura de fibra óptica. A correlação positiva observada é um indicativo forte, e ambos os modelos considerariam essa feature em suas previsões.

3.  **Método de Pagamento ('Eletronic check'):** O uso do cheque eletrônico como método de pagamento está fortemente correlacionado positivamente com a evasão. Isso pode sugerir que clientes que utilizam este método de pagamento podem ter menos engajamento com a empresa ou encontrar o processo menos conveniente, levando a uma maior probabilidade de cancelamento. Tanto a Regressão Logística quanto o SVM utilizariam essa feature como um preditor importante.

4.  **Tempo de Permanência ('tenure'):** O tempo de permanência do cliente é um dos fatores com maior correlação negativa com a evasão. Clientes com maior tempo de permanência são, em geral, mais leais e menos propensos a evadir. A relação inversa observada no box plot é clara: clientes que evadem tendem a ter um tempo de permanência muito menor. Esta relação linear e a diferença nas distribuições seriam bem capturadas por ambos os modelos.

5.  **Serviços de Segurança Online ('OnlineSecurity') e Suporte Técnico ('TechSupport'):** A presença destes serviços está correlacionada negativamente com a evasão. Isso sugere que clientes que assinam serviços adicionais como segurança online e suporte técnico tendem a ter um maior nível de satisfação ou dependência dos serviços da empresa, diminuindo a probabilidade de evadir. A inclusão dessas features nos modelos ajudaria a identificar clientes com menor risco de evasão.

Com base nos principais fatores de evasão identificados na análise anterior, propomos as seguintes estratégias de retenção de clientes, direcionadas a mitigar o risco de churn associado a cada fator:

1.  **Fator de Evasão: Contrato Mensal ('Month-to-month')**
    - **Estratégia de Retenção:** Oferecer incentivos para clientes com contrato mensal migrarem para planos de longo prazo (um ou dois anos), como descontos na mensalidade, benefícios adicionais de serviço (ex: upgrade de velocidade de internet, canais de TV premium gratuitos por um período), ou descontos em dispositivos.
    - **Justificativa:** Contratos de longo prazo reduzem a flexibilidade de cancelamento e, como visto na análise, estão associados a uma menor taxa de evasão. Incentivar essa migração pode aumentar a lealdade do cliente.

2.  **Fator de Evasão: Serviço de Fibra Óptica ('FiberOptic')**
    - **Estratégia de Retenção:** Investir na melhoria da qualidade do serviço de fibra óptica e no suporte técnico especializado para estes clientes. Realizar pesquisas de satisfação específicas para usuários de fibra óptica para identificar e resolver pontos problemáticos. Oferecer pacotes com serviços de valor agregado (segurança, backup) para aumentar o engajamento.
    - **Justificativa:** A maior evasão entre usuários de fibra óptica pode indicar problemas de serviço ou expectativa. Melhorar a experiência e oferecer serviços complementares pode aumentar a satisfação e a retenção.

3.  **Fator de Evasão: Método de Pagamento ('Eletronic check')**
    - **Estratégia de Retenção:** Promover ativamente métodos de pagamento automáticos e mais convenientes (transferência bancária, cartão de crédito) oferecendo pequenos descontos ou bônus para quem optar por essas formas de pagamento. Simplificar o processo de pagamento online ou oferecer opções de débito automático com vantagens.
    - **Justificativa:** O cheque eletrônico pode ser um indicador de menor engajamento ou maior atrito no processo de pagamento. Facilitar e incentivar métodos automáticos pode reduzir a probabilidade de churn relacionada a questões de pagamento.

4.  **Fator de Evasão: Tempo de Permanência ('tenure' baixo)**
    - **Estratégia de Retenção:** Implementar um programa de 'onboarding' robusto e proativo para novos clientes nos primeiros meses de serviço. Incluir comunicação regular para verificar a satisfação, oferecer suporte na configuração de serviços e apresentar todos os benefícios do plano. Oferecer um pequeno incentivo ou check-in proativo antes do final do primeiro ano.
    - **Justificativa:** Clientes com menor tempo de permanência são mais propensos a evadir. Um programa de 'onboarding' eficaz pode aumentar o engajamento inicial, resolver problemas rapidamente e construir um relacionamento mais forte nos primeiros meses críticos.

5.  **Fator de Evasão: Ausência de Serviços Adicionais (Segurança Online e Suporte Técnico)**
    - **Estratégia de Retenção:** Oferecer pacotes de serviços que incluam segurança online e suporte técnico a preços competitivos, talvez com um período experimental gratuito para que os clientes experimentem o valor. Educar os clientes sobre os benefícios desses serviços para sua segurança e tranquilidade online.
    - **Justificativa:** A presença desses serviços está associada à menor evasão. Clientes que se sentem mais seguros e bem suportados tendem a permanecer com a empresa. Promover esses serviços pode aumentar a percepção de valor e a lealdade.

Essas estratégias visam abordar os principais motivadores de evasão identificados, focando em ações direcionadas para os grupos de clientes mais propensos a cancelar o serviço. A implementação e o acompanhamento da eficácia dessas estratégias podem ser cruciais para melhorar as taxas de retenção de clientes.

---

### Principais Descobertas

Este projeto de previsão de evasão de clientes analisou dados de telecomunicações para identificar os fatores que mais contribuem para o churn e desenvolver modelos preditivos.
As principais descobertas da análise exploratória e da modelagem foram:
- Fatores como **contrato mensal, serviço de fibra óptica, método de pagamento por cheque eletrônico** e **baixo tempo de permanência (tenure)** estão fortemente associados a uma maior probabilidade de evasão.
- Por outro lado, clientes com **contratos de longo prazo (dois anos)** e que assinam serviços adicionais como **segurança online e suporte técnico** tendem a ter uma menor taxa de evasão.
- Para a modelagem preditiva, foram treinados modelos de Regressão Logística e SVM.
- Após o balanceamento das classes com SMOTE e a padronização das features, ambos os modelos apresentaram **desempenho satisfatório** no conjunto de teste, com acurácias em torno de 84%.
- A análise de overfitting/underfitting indicou que os modelos **generalizaram bem** para dados não vistos, sem sinais significativos de sobreajuste ou subajuste.
- As estratégias de retenção propostas são direcionadas a mitigar os riscos associados aos fatores de evasão identificados, como oferecer incentivos para contratos de longo prazo, melhorar a experiência com fibra óptica, promover métodos de pagamento automáticos e fortalecer o programa de 'onboarding' para novos clientes.

---

### Próximos Passos Possíveis

Para continuar a aprimorar o modelo de previsão de evasão e as estratégias de retenção, sugerimos os seguintes próximos passos:
1.  **Explorar Outros Modelos de Machine Learning:** Avaliar o desempenho de outros algoritmos de classificação, como Random Forest, Gradient Boosting (e.g., XGBoost, LightGBM) ou redes neurais, que podem capturar padrões mais complexos nos dados.
2.  **Otimização de Hiperparâmetros:** Realizar a otimização dos hiperparâmetros dos modelos treinados (Regressão Logística e SVM) usando técnicas como GridSearchCV ou RandomizedSearchCV para encontrar a melhor configuração e potencialmente melhorar o desempenho.
3.  **Análise Mais Aprofundada de Fatores Específicos:** Conduzir análises mais detalhadas sobre a interação entre os fatores de evasão, como investigar por que clientes de fibra óptica evadem mais, ou analisar o comportamento de evasão dentro de segmentos específicos de clientes.
4.  **Engenharia de Features Adicionais:** Criar novas features a partir dos dados existentes que possam fornecer informações adicionais relevantes para a previsão de evasão (e.g., frequência de uso de serviços, histórico de problemas técnicos reportados).
5.  **Validação Cruzada:** Utilizar técnicas de validação cruzada para obter uma estimativa mais robusta do desempenho do modelo e reduzir a dependência de uma única divisão treino/teste.
6.  **Implementação e Monitoramento:** Se o modelo for implementado em produção, estabelecer um sistema de monitoramento contínuo para acompanhar o desempenho do modelo em dados reais e ajustar as estratégias de retenção conforme necessário.
