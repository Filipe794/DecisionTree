from sklearn import set_config

# Transformadores devolvem DataFrame => colunas mantêm o nome entre as etapas.
# Fica aqui para valer em qualquer import de src (inclusive ao carregar um pipeline.joblib).
set_config(transform_output="pandas")
