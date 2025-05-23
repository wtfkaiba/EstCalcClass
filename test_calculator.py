# test_calculator.py
import unittest
import pandas as pd
import numpy as np
from calculator import calcular_estatisticas # Assumindo que a função está em calculator.py

class TestStatisticsCalculator(unittest.TestCase):

    def test_basic_case_3_classes(self):
        # Dados de entrada
        lim_inf = 0
        amp = 10
        qtd_classes = 3
        freqs = [5, 10, 5] # N = 20
        # Valores esperados (calculados manualmente ou com outra ferramenta confiável)
        # Classes: 0-10 (PM=5), 10-20 (PM=15), 20-30 (PM=25)
        # Média = (5*5 + 15*10 + 25*5) / 20 = 300/20 = 15
        # Variância = [5*(5-15)^2 + 10*(15-15)^2 + 5*(25-15)^2] / 19 = (500 + 0 + 500)/19 = 1000/19 = 52.6315789
        # Desvio Padrão = sqrt(52.6315789) = 7.2547625
        # CV = (7.2547625 / 15) * 100 = 48.36508
        # Mediana: N/2=10. Classe mediana 10-20. L=10, Fi_ant=5, f_md=10, h=10. Md = 10 + ((10-5)/10)*10 = 15
        # Moda Bruta: PM da classe 10-20 = 15
        # Moda Czuber: L=10, f_mo=10, f_ant=5, f_pos=5, h=10. Cz = 10 + (5/(5+5))*10 = 15

        df, stats = calcular_estatisticas(lim_inf, amp, qtd_classes, freqs)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), qtd_classes)
        self.assertAlmostEqual(stats["Média"], 15.0, places=5)
        self.assertAlmostEqual(stats["Variância"], 52.6315789, places=5)
        self.assertAlmostEqual(stats["Desvio Padrão"], 7.2547625, places=5)
        self.assertAlmostEqual(stats["Coeficiente de Variação (%)"], 48.3650833, places=5)
        self.assertAlmostEqual(stats["Mediana"], 15.0, places=5)
        self.assertAlmostEqual(stats["Moda Bruta (Ponto Médio da Classe Modal)"], 15.0, places=5)
        self.assertAlmostEqual(stats["Moda de Czuber"], 15.0, places=5)
        self.assertEqual(df['Frequência Acumulada (Fi)'].iloc[-1], sum(freqs))

    def test_case_5_classes_assymetric(self):
        lim_inf = 50
        amp = 5
        qtd_classes = 5
        freqs = [3, 8, 15, 7, 2] # N = 35
        # Classes: 50-55 (52.5), 55-60 (57.5), 60-65 (62.5), 65-70 (67.5), 70-75 (72.5)
        # Fi: 3, 11, 26, 33, 35
        # Média: (3*52.5 + 8*57.5 + 15*62.5 + 7*67.5 + 2*72.5) / 35 = (157.5 + 460 + 937.5 + 472.5 + 145) / 35 = 2172.5 / 35 = 62.0714
        # Mediana: N/2 = 17.5. Classe mediana 60-65. L=60, Fi_ant=11, f_md=15, h=5. Md = 60 + ((17.5-11)/15)*5 = 60 + (6.5/15)*5 = 60 + 2.1666 = 62.1666
        # Moda Bruta: PM classe 60-65 = 62.5
        # Moda Czuber: L=60, f_mo=15, f_ant=8, f_pos=7, h=5. Cz = 60 + ((15-8)/((15-8)+(15-7)))*5 = 60 + (7/(7+8))*5 = 60 + (7/15)*5 = 60 + 2.3333 = 62.3333

        df, stats = calcular_estatisticas(lim_inf, amp, qtd_classes, freqs)
        self.assertAlmostEqual(stats["Média"], 62.071428, places=5)
        self.assertAlmostEqual(stats["Mediana"], 62.166666, places=5)
        self.assertAlmostEqual(stats["Moda Bruta (Ponto Médio da Classe Modal)"], 62.5, places=5)
        self.assertAlmostEqual(stats["Moda de Czuber"], 62.333333, places=5)
        # Variância: (sum((xi-media)^2 * fi)) / (N-1)
        # (52.5-62.0714)^2*3 = 274.719
        # (57.5-62.0714)^2*8 = 166.964
        # (62.5-62.0714)^2*15 = 2.739
        # (67.5-62.0714)^2*7 = 206.556
        # (72.5-62.0714)^2*2 = 217.576
        # Sum = 868.554 / 34 = 25.5457
        # Variância correta: 25.54621848739496
        self.assertAlmostEqual(stats["Variância"], 25.546218, places=5)
        self.assertAlmostEqual(stats["Desvio Padrão"], np.sqrt(25.546218), places=5)

    def test_all_frequencies_zero(self):
        lim_inf = 0
        amp = 10
        qtd_classes = 3
        freqs = [0, 0, 0]
        df, stats = calcular_estatisticas(lim_inf, amp, qtd_classes, freqs)
        
        self.assertTrue(all(np.isnan(value) for value in stats.values()))
        self.assertEqual(df['Frequência Absoluta (fi)'].sum(), 0)
        self.assertEqual(df['Frequência Acumulada (Fi)'].iloc[-1], 0)

    def test_single_class(self):
        lim_inf = 100
        amp = 20
        qtd_classes = 1
        freqs = [10] # N = 10
        # Classe: 100-120 (PM=110)
        # Média = 110
        # Variância = 0 (N-1 não se aplica bem, mas a função retorna 0.0 para N>1, se N=1 o comportamento é 0 por causa de (N-1))
        # O código atual retornará variância = 0.0 para N=10. Se fosse N=1, variância = 0.0
        # Desvio Padrão = 0
        # CV = NaN (divisão por média, se média fosse 0) ou 0 se desvio padrão é 0
        # Mediana = 100 + ((5-0)/10)*20 = 100 + 10 = 110
        # Moda Bruta = 110
        # Moda Czuber = 100 + ( (10-0) / ((10-0)+(10-0)) ) * 20 -> L+h/2 = 110 (pois f_ant e f_pos são 0)

        df, stats = calcular_estatisticas(lim_inf, amp, qtd_classes, freqs)
        self.assertAlmostEqual(stats["Média"], 110.0, places=5)
        # Para N=10, variância será calculada com N-1.
        # (110-110)^2 * 10 / 9 = 0
        self.assertAlmostEqual(stats["Variância"], 0.0, places=5)
        self.assertAlmostEqual(stats["Desvio Padrão"], 0.0, places=5)
        self.assertTrue(np.isnan(stats["Coeficiente de Variação (%)"]) or stats["Coeficiente de Variação (%)"] == 0.0) # Se DP é 0, CV é 0
        self.assertAlmostEqual(stats["Mediana"], 110.0, places=5)
        self.assertAlmostEqual(stats["Moda Bruta (Ponto Médio da Classe Modal)"], 110.0, places=5)
        self.assertAlmostEqual(stats["Moda de Czuber"], 110.0, places=5) # L + h/2

    def test_czuber_mode_flat_top(self):
        # Teste para quando delta1 + delta2 == 0 (Ex: platô de frequências)
        # Classe modal no meio de um platô, ou classe modal isolada.
        lim_inf = 0
        amp = 10
        qtd_classes = 3
        freqs = [5, 10, 10] # Classe modal pode ser a segunda ou terceira. idxmax() pega a primeira.
                            # Se modal é a segunda: L=10, f_mo=10, f_ant=5, f_pos=10. delta1=5, delta2=0
                            # Cz = 10 + (5/(5+0))*10 = 10 + 10 = 20. (Ponto superior da classe modal)
                            # O ajuste min(max(...)) deve trazê-la para L_mo + h = 20.

        # Se a classe modal for a última (70-80)
        # 50-60 (5), 60-70 (5), 70-80 (10)
        # L_mo=70, f_mo=10, f_ant=5, f_pos=0. delta1=5, delta2=10. Cz = 70 + (5/(5+10))*10 = 70 + 3.33 = 73.33

        df, stats = calcular_estatisticas(0, 10, 3, [5, 10, 5]) # Caso já testado, Czuber = 15
        self.assertAlmostEqual(stats["Moda de Czuber"], 15.0, places=5)

        # Caso onde delta1+delta2 = 0 porque é um platô, e a classe modal é a primeira do platô
        # A função idxmax() pegará a primeira classe com a frequência máxima.
        # Classes: 0-10 (5), 10-20 (10), 20-30 (10)
        # Classe modal (pelo idxmax): 10-20. L_mo=10, f_mo=10, f_ant=5, f_pos=10.
        # delta1 = 10-5=5, delta2 = 10-10=0.
        # Moda Czuber = 10 + (5 / (5+0)) * 10 = 10 + 1*10 = 20.
        # O max(L_mo, min(moda_czuber, L_mo + h)) garante que fica em [10, 20].
        df_plateau_start, stats_plateau_start = calcular_estatisticas(0, 10, 3, [5, 10, 10])
        self.assertAlmostEqual(stats_plateau_start["Moda de Czuber"], 20.0, places=5)

        # Caso onde a classe modal está isolada
        # Classes: 0-10 (0), 10-20 (10), 20-30 (0)
        # L_mo=10, f_mo=10, f_ant=0, f_pos=0. delta1=10, delta2=10
        # Cz = 10 + (10/(10+10))*10 = 10 + 5 = 15
        df_isolated, stats_isolated = calcular_estatisticas(0, 10, 3, [0, 10, 0])
        self.assertAlmostEqual(stats_isolated["Moda de Czuber"], 15.0, places=5)

        # Caso onde f_mo - f_ant + f_mo - f_pos = 0, mas não por f_ant e f_pos serem iguais f_mo
        # Isso acontece se delta1 = -delta2. Ex: f_mo=10, f_ant=5 (delta1=5), f_pos=15 (delta2=-5)
        # A classe modal seria a de f_pos=15 nesse caso, não f_mo=10.
        # A lógica do delta1+delta2=0 para moda de Czuber é mais para quando (f_mo-f_ant) e (f_mo-f_post)
        # são ambos zero ou um deles é zero e o outro também, o que leva a (PM da classe modal)
        # No código: se delta1+delta2 == 0, moda_czuber = L_mo + (amplitude / 2)
        # Ex: 0-10 (10), 10-20 (5), 20-30 (10). idxmax() pega a primeira.
        # L_mo=0, f_mo=10, f_ant=0, f_pos=5. delta1=10, delta2=5. Cz = 0 + (10/(10+5))*10 = 6.66
        df_dip, stats_dip = calcular_estatisticas(0,10,3, [10,5,10])
        self.assertAlmostEqual(stats_dip["Moda de Czuber"], 6.666666, places=5)

    def test_median_f_md_zero(self):
        # Caso onde a frequência da classe mediana (f_md) é zero.
        # Isso não deveria ocorrer se as frequências são >= 0 e N > 0,
        # pois a classe mediana é determinada pela frequência acumulada.
        # Mas se ocorresse, a função retorna L_md.
        # Ex: 0-10 (5), 10-20 (0), 20-30 (5). N=10, N/2=5.
        # Fi: 5, 5, 10. Classe Mediana é 0-10 (primeira que Fi >= 5).
        # L_md=0, f_md=5, Fi_ant=0. Md = 0 + ((5-0)/5)*10 = 10.
        
        # Para forçar f_md=0 na classe onde N/2 teoricamente cairia:
        # 0-10 (0), 10-20 (10), 20-30 (0). N=10, N/2=5.
        # Fi: 0, 10, 10. Classe Mediana é 10-20.
        # L_md=10, f_md=10, Fi_ant=0. Md = 10 + ((5-0)/10)*10 = 15.
        # O cenário onde f_md=0 para a classe mediana real é difícil de construir
        # sem que N seja 0 para aquela classe. A lógica da função é um fallback.
        # Não há um teste simples para esse "else L_md" sem manipular artificialmente o df.

        # Teste onde a mediana cai exatamente no limite de uma classe
        # 0-10 (5), 10-20 (5), 20-30 (10). N=20, N/2=10
        # Fi: 5, 10, 20. Classe Mediana é 10-20.
        # L_md=10, Fi_ant=5, f_md=5. Md = 10 + ((10-5)/5)*10 = 10 + 10 = 20.
        df, stats = calcular_estatisticas(0, 10, 3, [5, 5, 10])
        self.assertAlmostEqual(stats["Mediana"], 20.0, places=5)

    def test_empty_frequencies_input(self):
        # Teste com lista de frequências vazia
        df, stats = calcular_estatisticas(0, 10, 3, [])
        # A função agora retorna df vazio e stats com NaN
        self.assertTrue(df.empty)
        self.assertTrue(all(np.isnan(value) for value in stats.values()))

    def test_invalid_frequencies_input(self):
        # Teste com lista de frequências inválida (não numérica)
        df, stats = calcular_estatisticas(0, 10, 3, [5, "a", 5])
        self.assertTrue(df.empty)
        self.assertTrue(all(np.isnan(value) for value in stats.values()))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)