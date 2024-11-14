import cv2
import numpy as np

# Carregar o gabarito fornecido como dicionário {questão: resposta correta}
gabarito = {
    1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',
    11: 'A', 12: 'C', 13: 'C', 14: 'E', 15: 'D', 16: 'B', 17: 'E', 18: 'C', 19: 'A', 20: 'E',
    21: 'D', 22: 'E', 23: 'E', 24: 'A', 25: 'C', 26: 'C', 27: 'D', 28: 'B', 29: 'D', 30: 'D',
    31: 'A', 32: 'D', 33: 'D', 34: 'B', 35: 'C', 36: 'D', 37: 'B', 38: 'D', 39: 'D', 40: 'D',
    41: 'D', 42: 'E', 43: 'C', 44: 'A', 45: 'D', 46: 'B', 47: 'C', 48: 'A', 49: 'D', 50: 'E'
}

# gabarito = {
#     1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'D', 8: 'D', 9: 'B', 10: 'B',
#     11: 'C', 12: 'C', 13: 'C', 14: 'C', 15: 'A', 16: 'A', 17: 'D', 18: 'C', 19: 'A', 20: 'D',
#     21: 'D', 22: 'B', 23: 'D', 24: 'B', 25: 'C', 26: 'B', 27: 'B', 28: 'B', 29: 'E', 30: 'B',
#     31: 'B', 32: 'B', 33: 'B', 34: 'C', 35: 'C', 36: 'D', 37: 'B', 38: 'C', 39: 'C', 40: 'B',
#     41: 'B', 42: 'A', 43: 'B', 44: 'E', 45: 'E', 46: 'A', 47: 'A', 48: 'B', 49: 'D', 50: 'C'
# }

aluno = {1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'D', 8: 'D', 9: 'B', 10: 'B',
    11: 'C', 12: 'C', 13: 'C', 14: 'C', 15: 'A', 16: 'A', 17: 'D', 18: 'C', 19: 'A', 20: 'D',
    21: 'D', 22: 'B', 23: 'D', 24: 'B', 25: 'C', 26: 'B', 27: 'B', 28: 'B', 29: 'E', 30: 'B',
    31: 'B', 32: 'B', 33: 'B', 34: 'C', 35: 'C', 36: 'D', 37: 'B', 38: 'C', 39: 'C', 40: 'B', 
    41: 'B', 42: 'A', 43: 'B', 44: 'E', 45: 'E', 46: 'A', 47: 'B', 48: 'B', 49: 'D', 50: 'C'}

def processar_imagem_cartao(imagem_caminho):
    # Carregar a imagem e converter para escala de cinza
    imagem = cv2.imread(imagem_caminho)
    imagem = corrigir_orientacao(imagem)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar limiarização para destacar as marcações
    _, thresh = cv2.threshold(imagem_cinza, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos das áreas de marcação
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    # Assumindo que o maior contorno é o cartão-resposta
    cartao_contorno = contornos[0]
    return imagem, thresh, cartao_contorno


def corrigir_orientacao(imagem_cartao):
    # Converter para escala de cinza e aplicar um threshold
    imagem_cinza = cv2.cvtColor(imagem_cartao, cv2.COLOR_BGR2GRAY)
    _, imagem_bin = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detectar contornos
    contornos, _ = cv2.findContours(imagem_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Procurar o maior contorno que poderia corresponder ao bloco de questões
    maior_contorno = max(contornos, key=cv2.contourArea)
    retangulo = cv2.minAreaRect(maior_contorno)
    angulo = retangulo[-1]
    
    # Ajustar o ângulo (se o retângulo está "deitado" ou "em pé")
    if angulo < -45:
        angulo += 90
    
    # Verificar se o ângulo está dentro do intervalo aceitável
    if angulo < 88 or angulo > 92:
        # Calcular a rotação necessária para corrigir a orientação
        (h, w) = imagem_cartao.shape[:2]
        centro = (w // 2, h // 2)
        matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        imagem_corrigida = cv2.warpAffine(imagem_cartao, matriz_rotacao, (w, h), flags=cv2.INTER_LINEAR)
        print(f"Imagem rotacionada em {angulo} graus para corrigir a orientação.")
        return imagem_corrigida
    else:
        print("Imagem já está na orientação correta.")
        return imagem_cartao



def visualizar_blocos_questoes(imagem_cartao, largura_alternativa, altura_alternativa, num_colunas, num_linhas, esp_horizontal, esp_vertical):
    altura_imagem, largura_imagem = imagem_cartao.shape[:2]
    
    for linha in range(num_linhas):
        # Calcular a posição inicial de y apenas uma vez por linha
        questao_y = int(linha * (altura_alternativa + esp_vertical))
        
        for coluna in range(num_colunas):
            # Calcular a posição x para cada coluna
            questao_x = int(coluna * (largura_alternativa + esp_horizontal))
            questao_y = questao_y + int(coluna * (esp_horizontal * 0.009))
            # Extrair a área da questão
            questao_area = imagem_cartao[questao_y:questao_y + altura_alternativa, questao_x:questao_x + largura_alternativa ]
            
            # Verificar se o bloco não está vazio antes de exibir
            if questao_area.size > 0:
                cv2.imshow(f"Linha {linha + 1} - Coluna {coluna + 1}", questao_area)
            else:
                print(f"Bloco vazio na Linha {linha + 1} - Coluna {coluna + 1}")
    
    # Aguardar que o usuário pressione uma tecla para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectar_respostas(imagem_thresh):
    # Obter as dimensões da imagem completa
    altura_imagem, largura_imagem = imagem_thresh.shape[:2]

    # Coordenadas do cartão-resposta em relação à imagem completa
    x_inicio = int(largura_imagem * 0.092)  
    y_inicio = int(altura_imagem * 0.355)  
    x_fim = int(largura_imagem * 0.9107)     
    y_fim = int(altura_imagem * 0.8248)     

    # Calculo do espacamento vertical e horizontal
    espacamento_horizontal = 0.016 * largura_imagem 
    espacamento_vertical = 0.01 * altura_imagem

    # Extrair apenas a área do cartão-resposta
    imagem_respostas = imagem_thresh[y_inicio:y_fim, x_inicio:x_fim]
    
    # Parâmetros de segmentação para as questões no cartão
    num_linhas = 4  # 4 linhas de questões
    num_colunas = 15  # 15 colunas de questões por linha
    largura_alternativa = int((x_fim - x_inicio - (14 * espacamento_horizontal)) / num_colunas)
    altura_alternativa = int((y_fim - y_inicio - (3 * espacamento_vertical)) / num_linhas)



    # Visualizar os blocos das questões inteiras
    visualizar_blocos_questoes(imagem_respostas, largura_alternativa, altura_alternativa, num_colunas, num_linhas, espacamento_horizontal, espacamento_vertical)

    respostas = {}

    # Laço para detectar as marcações em cada bloco
    for linha in range(num_linhas):
        questao_y = int(linha * (altura_alternativa + espacamento_vertical))
        for coluna in range(num_colunas):
            # Calcular a posição do bloco da questão com base nas posições relativas
            questao_x = int(coluna * (largura_alternativa + espacamento_horizontal))
            
            questao_y = questao_y + int(coluna * (espacamento_horizontal * 0.009))
            # Extrair a área da questão
            questao_area = imagem_respostas[questao_y:questao_y + altura_alternativa, questao_x:questao_x + largura_alternativa]
            
            # Definir um número de questão sequencial, de 1 a 50
            numero_questao = linha * num_colunas + coluna + 1
            if numero_questao > 50:  # Limita o número máximo de questões
                break

            # Dividir o bloco da questão em 6 partes verticais
            altura_parte = int(altura_alternativa / 6)
            
            alternativas = {}
            for i in range(1, 6):  # Ignora o primeiro bloco (i=0) e usa os 5 restantes para A-E
                opcao_y = i * altura_parte
                opcao_area = questao_area[opcao_y:opcao_y + altura_parte, :]
                
                # Contagem de pixels não-brancos (marcados) para detectar uma marcação
                contagem_pixels = cv2.countNonZero(opcao_area)
                if contagem_pixels > 90:  # Threshold para considerar uma marcação
                    alternativas[chr(65 + i - 1)] = contagem_pixels  # A=65, B=66, etc.

            # Determinar a alternativa com mais pixels, assumindo como a marcada
            if alternativas:
                resposta_marcada = max(alternativas, key=alternativas.get)
                respostas[numero_questao] = resposta_marcada

    return respostas


def corrigir_respostas(respostas_aluno, gabarito):
    # Comparar respostas detectadas com o gabarito fornecido e calcular a pontuação
    pontuacao = 0
    for questao, resposta_correta in gabarito.items():
        if respostas_aluno.get(questao) == resposta_correta:
            pontuacao += 1
    return pontuacao

# Caminho da imagem do cartão-resposta
imagem_caminho = 'imagens/cartao1.jpg'


# Processamento da imagem e detecção das respostas
imagem, imagem_thresh, contorno_cartao = processar_imagem_cartao(imagem_caminho)
respostas_aluno = detectar_respostas(imagem_thresh)



print('essas foram as respostas do aluno: ')
print(respostas_aluno)
# Correção e pontuação final
pontuacao = corrigir_respostas(respostas_aluno, gabarito)

# Exibir e salvar o resultado
print(f"Pontuação final: {pontuacao}/{len(gabarito)}")

# Opcional: salvar o resultado em um arquivo de texto
with open("resultado.txt", "w") as arquivo:
    arquivo.write(f"Pontuação final: {pontuacao}/{len(gabarito)}\n")
    arquivo.write("Respostas detectadas:\n")
    for questao, resposta in sorted(respostas_aluno.items()):
        arquivo.write(f"Questão {questao}: {resposta}\n")






