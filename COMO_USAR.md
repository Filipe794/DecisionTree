1. trocar a seed - testar alem do 42
2. testar outras divisoes - 80/20 / 65/35
3. fazer k-fold

usar k=5, 6 no máximo. justificar: k = 5 ou 10 são os valores usuais. Com essa base, 5 é um bom equilíbrio: cada fold de validação tem ~107 linhas, o suficiente para uma estimativa razoável. Com k = 10 os folds ficam com ~54 linhas e a métrica de cada um fica mais ruidosa.

Prefira winsorização ou limitação de valores, que funcionam como transformador no pipeline.