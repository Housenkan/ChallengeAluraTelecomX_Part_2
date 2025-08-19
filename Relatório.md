# ChallengeAluraTelecomX_Part_2

Este trabalho trata de visualizar e analisar a evasão de clientes na empresa Telecom X.

Na primeira parte tivemos a extração dos dados.

Utilizando os dados tratados do desafio anterior, foi feito o upload do arquivo no ambiente do google collab.

Em seguida foi feito a importação das bibliotecas, ajuste no aumento da visualização das colunas e a leitura do arquivo:

`import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
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

* tenure (-0.352): A duração do tempo como cliente (tenure) tem uma correlação negativa forte com a evasão. Isso é esperado, pois clientes que estão há mais tempo tendem a ser mais leais.
* Two year (-0.302): Clientes com contrato de dois anos apresentam uma correlação negativa forte com a evasão. Contratos de longo prazo indicam um compromisso maior e menor probabilidade de sair.
* NoInternetService (-0.228): Clientes sem serviço de internet têm uma correlação negativa moderada com a evasão, o que faz sentido, pois a falta do serviço de internet remove um possível ponto de atrito que leva à evasão.
*OnlineSecurity (-0.171) e TechSupport (-0.165): Clientes que usam segurança online e suporte técnico mostram correlações negativas moderadas com a evasão, sugerindo que esses serviços de valor agregado ajudam a reter clientes.

Portanto as varíaveis mais importantes a princípio são:

Month-to-month
FiberOptic
Eletronic check
tenure
Two year
NoInternetService
OnlineSecurity
TechSupport



