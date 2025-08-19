# ChallengeAluraTelecomX_Part_2

Este trabalho trata de visualizar e analisar a evasão de clientes na empresa Telecom X.

Na primeira parte tivemos a extração dos dados.

Utilizando os dados tratados do desafio anterior, foi feito o upload do arquivo no ambiente do google collab.

Em seguida foi feito a importação das bibliotecas, ajuste no aumento da visualização das colunas e a leitura do arquivo:

`import pandas as pd
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






