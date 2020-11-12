import numpy as np

def de_tempo_para_deformacao(taxa_def:float, tempos:np.ndarray) -> np.ndarray:
    "Converte de um array de tempos para deformacao, dada uma taxa de deformacao"
    return taxa_def * tempos

def de_tempo_para_elongamento(taxa_def_eng:float, tempos:np.ndarray) -> np.ndarray:
    """Converte de um array de tempos para elongamento, dada uma taxa de deformação de engenharia."""
    deformacao = de_tempo_para_deformacao(taxa_def_eng, tempos)
    return deformacao + 1.0

def de_tempo_para_elongamento_vdd(taxa_def_vdd:float, tempos:np.ndarray) -> np.ndarray:
    """Converte de um array de tempos para elongamento, dada uma taxa de deformação de engenharia."""
    deformacao = de_tempo_para_deformacao(taxa_def_vdd, tempos)
    return np.exp(deformacao)

def de_tempo_para_deslocamento(taxa_def:float, tempos:np.ndarray, tamanho_inicial=6.0) -> np.ndarray:
    "Converte uma lista de tempos dada uma taxa de deformação verdadeira e o tamanho inicial do corpo de prova."
    deformacao = de_tempo_para_deformacao(taxa_def,tempos)
    return tamanho_inicial *  deformacao  #tamanho_inicial * ( np.exp(deformacao) - 1)

def taxa_nominal_com_compliance(tempo_total, tamanho_inicial=6.0, deslocamento_real=-1.7) -> float :

    deformacao_nominal = deslocamento_real / tamanho_inicial
    return deformacao_nominal/tempo_total